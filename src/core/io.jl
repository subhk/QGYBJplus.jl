"""Manager for a simulation-owned NetCDF output stream."""
mutable struct ModelOutputManager{T, S}
    specification::S
    counter::Int
    last_time::Union{Nothing, T}
    last_iteration::Union{Nothing, Int}
    next_time::Union{Nothing, T}
    closed::Bool
end

@inline _to_xyz(array::AbstractArray) = permutedims(array, (2, 3, 1))
@inline _from_xyz(array::AbstractArray) = permutedims(array, (3, 1, 2))

function ModelOutputManager(specification::S, ::Type{T}, start_time=zero(T)) where {S, T}
    next_time = specification.schedule isa TimeInterval ?
                T(_next_deadline(start_time, specification.schedule.interval)) : nothing
    return ModelOutputManager{T, S}(
        specification, 1, nothing, nothing, next_time, false)
end

function _prepare_simulation_output!(simulation::Simulation)
    specification = simulation.run_options.output
    if specification === false
        simulation.output_manager = nothing
        return simulation
    end

    specification isa NetCDFOutput ||
        throw(ArgumentError("simulation output must be NetCDFOutput or false"))
    manager = simulation.output_manager
    if manager !== nothing && !manager.closed
        return simulation
    end

    mkpath(specification.path)
    manager = ModelOutputManager(specification, typeof(simulation.clock.time),
                                 simulation.clock.time)
    _resume_state_files!(manager, simulation.clock)
    simulation.output_manager = manager
    return simulation
end

"""Global attribute recording that output carries the model's own units."""
const _UNITS_CONVENTION =
    "Quantities are in the units of the model configuration " *
    "(for example m, s, and m/s for a dimensional setup)."

"""
The state files (`state####.nc`, regular files only) already in `path`, in
numbering order, with the time and iteration each records. A file that cannot
be read gets iteration `typemax(Int)`, so only `overwrite=true` removes it.
"""
function _existing_state_files(path::AbstractString)
    files = NamedTuple{(:index, :path, :time, :iteration),
                       Tuple{Int, String, Float64, Int}}[]
    isdir(path) || return files
    for name in readdir(path)
        matched = match(r"^state(\d+)\.nc$", name)
        file = joinpath(path, name)
        (matched === nothing || !isfile(file)) && continue
        time, iteration = try
            NCDataset(file, "r") do dataset
                (Float64(dataset["time"][1]), Int(dataset.attrib["iteration"]))
            end
        catch exception
            exception isa InterruptException && rethrow()
            (NaN, typemax(Int))
        end
        push!(files, (index=parse(Int, matched[1]), path=file, time, iteration))
    end
    return sort!(files; by=file -> file.index)
end

"""
The existing state files a run would replace: all of them for a run from
iteration 0, those after the restart iteration for a restored run.
"""
_stale_state_files(existing, clock::Clock) =
    filter(file -> clock.iteration == 0 || file.iteration > clock.iteration, existing)

"""Error, unless `overwrite=true`, when `stale` state files are in the way."""
function _check_stale_state_files(specification::NetCDFOutput, stale, clock::Clock)
    isempty(stale) || specification.overwrite || throw(ArgumentError(
        "$(specification.path) already holds state files " *
        "($(basename(first(stale).path)) to $(basename(last(stale).path))) " *
        (clock.iteration > 0 ? "after this run's restart iteration $(clock.iteration)" :
                               "from another run") *
        "; choose a new path or pass NetCDFOutput(...; overwrite=true) " *
        "to replace them"))
    return nothing
end

"""Check, changing no file, that the run's state files can be written."""
function _check_simulation_output(simulation::Simulation)
    specification = simulation.run_options.output
    specification isa NetCDFOutput || return nothing
    manager = simulation.output_manager
    (manager !== nothing && !manager.closed) && return nothing
    existing = _existing_state_files(specification.path)
    _check_stale_state_files(specification,
        _stale_state_files(existing, simulation.clock), simulation.clock)
    return nothing
end

"""
Number this run's state files after those already in its output directory.
A restored simulation (clock past iteration 0) appends after the files up to
its restart iteration, and does not write the restart state again. Other
existing files are an error unless `overwrite=true`, which removes them: all
of them for a run from iteration 0, those after the restart iteration for a
restored run.
"""
function _resume_state_files!(manager::ModelOutputManager, clock::Clock)
    specification = manager.specification
    existing = _existing_state_files(specification.path)
    isempty(existing) && return manager

    stale = _stale_state_files(existing, clock)
    if !isempty(stale)
        _check_stale_state_files(specification, stale, clock)
        foreach(file -> rm(file.path), stale)
        filter!(file -> !(file in stale), existing)
    end
    isempty(existing) && return manager

    latest = last(existing)
    manager.counter = latest.index + 1
    if latest.iteration == clock.iteration
        # The restart state is already on disk.
        manager.last_time = clock.time
        manager.last_iteration = clock.iteration
    end
    return manager
end

function _output_due(manager::ModelOutputManager, clock::Clock;
    initial::Bool=false)

    manager.closed && return false
    manager.last_iteration === clock.iteration && return false
    initial && return true

    schedule = manager.specification.schedule
    schedule === nothing && return false
    schedule isa IterationInterval &&
        return clock.iteration % schedule.interval == 0
    if schedule isa TimeInterval
        tolerance = 8eps(max(abs(clock.time), abs(schedule.interval), one(clock.time)))
        return manager.next_time !== nothing &&
               clock.time >= manager.next_time - tolerance
    end
    error("unsupported output schedule $(typeof(schedule))")
end

function _advance_output_deadline!(manager::ModelOutputManager, time)
    schedule = manager.specification.schedule
    schedule isa TimeInterval || return manager
    manager.next_time = _next_deadline(time, schedule.interval)
    return manager
end

_spectral_LA(model::QGYBJModel) =
    _spectral_wave_velocity!(similar(model.fields.B), model.fields, model)

"""Host copy of a model array, for writing to disk."""
_host_array(field) = Array(field)

_wave_formulation_name(model::QGYBJModel) =
    string(nameof(typeof(model.physics.formulation)))
_feedback_mode_name(model::QGYBJModel) =
    string(nameof(typeof(model.physics.feedback)))
_generalized_pv_convention(model::QGYBJModel) =
    _wave_feedback_enabled(ETDModelOptions(model.physics, model.numerics)) ?
    "total_with_wave_pv" : "balanced_only"
_flow_evolution_name(model::QGYBJModel) =
    string(nameof(typeof(model.physics.flow)))

function _refresh_output_diagnostics!(model::QGYBJModel,
    specification::NetCDFOutput)

    write_psi = _saves_psi(specification.fields)
    write_waves = _saves_waves(specification.fields)
    write_velocities = specification.velocities ||
                       :velocities in specification.fields

    options = ETDModelOptions(model.physics, model.numerics)
    (write_psi || write_velocities) && !_fixed_flow(options) &&
        invert_q_to_psi!(model)
    write_waves && _refresh_wave_diagnostics!(model)
    write_velocities && compute_velocities!(model; compute_w=true)
    return model
end

function _define_field!(dataset, name::AbstractString, values,
    long_name::AbstractString; dimensions=("x", "y", "z"))

    variable = NCDatasets.defVar(dataset, name, Float64, dimensions)
    variable[:, :, :] = _to_xyz(values)
    variable.attrib["long_name"] = long_name
    return variable
end

"""Define a one-dimensional variable with its values and description."""
function _define_vector!(dataset, name::AbstractString, dimension, values,
    long_name::AbstractString)

    variable = NCDatasets.defVar(dataset, name, Float64, (dimension,))
    variable[:] = values
    variable.attrib["long_name"] = long_name
    return variable
end

function _write_model_state_file!(manager::ModelOutputManager,
    simulation::Simulation)

    model = simulation.model
    runtime = model.runtime
    fields = model.fields
    grid = model.grid
    specification = manager.specification
    write_psi = _saves_psi(specification.fields)
    write_waves = _saves_waves(specification.fields)
    write_velocities = specification.velocities ||
                       :velocities in specification.fields

    _refresh_output_diagnostics!(model, specification)

    # Physical fields go through one scratch buffer, one at a time, straight
    # to the host, so writing output adds no device memory.
    psi_host, A_physical_host, LA_physical_host = with_scratch(runtime.workspace) do
        buffer = scratch_like(runtime.workspace, fields.B)
        physical_host(spectral) = _host_array(fft_backward!(buffer, spectral, runtime))
        psi_host = write_psi ? physical_host(fields.psi) : nothing
        A_host = write_waves ? physical_host(fields.A) : nothing
        LA_host = if write_waves
            _spectral_wave_velocity!(buffer, fields, model)
            physical_host(buffer)
        else
            nothing
        end
        (psi_host, A_host, LA_host)
    end

    q_host = _host_array(fields.q)
    B_host = _host_array(fields.B)
    # A fixed flow is prescribed by ψ, not inverted from q, so a restart needs it.
    fixed_flow = model.physics.flow isa FixedFlow
    psi_spectral_host = fixed_flow ? _host_array(fields.psi) : nothing
    u_host = write_velocities ? _host_array(fields.u) : nothing
    v_host = write_velocities ? _host_array(fields.v) : nothing
    w_host = write_velocities ? _host_array(fields.w) : nothing

    filename = @sprintf("state%04d.nc", manager.counter)
    filepath = joinpath(specification.path, filename)
    spectral = ("kx_index", "ky_index", "z")
    fourier = "horizontal Fourier coefficients (unnormalised forward FFT, " *
              "FFTW order; wavenumbers in kx and ky)"
    NCDataset(filepath, "c") do dataset
        nx, ny, nz = grid.size
        dataset.dim["x"] = nx
        dataset.dim["y"] = ny
        dataset.dim["z"] = nz
        dataset.dim["z_face"] = nz
        dataset.dim["kx_index"] = nx
        dataset.dim["ky_index"] = ny
        dataset.dim["time"] = 1

        _define_vector!(dataset, "x", "x", grid.x, "x coordinate of the cell centres")
        _define_vector!(dataset, "y", "y", grid.y, "y coordinate of the cell centres")
        _define_vector!(dataset, "z", "z", grid.z, "z coordinate of the cell centres")
        _define_vector!(dataset, "z_face", "z_face", grid.z_faces[2:end],
            "z coordinate of the upper face of each cell")
        _define_vector!(dataset, "kx", "kx_index", grid.kx,
            "x wavenumber of each Fourier coefficient (FFTW order)")
        _define_vector!(dataset, "ky", "ky_index", grid.ky,
            "y wavenumber of each Fourier coefficient (FFTW order)")
        _define_vector!(dataset, "time", "time", [simulation.clock.time], "time")

        if write_psi
            _define_field!(dataset, "psi", real.(psi_host),
                "balanced streamfunction (the Lagrangian-mean flow)")
        end
        if write_waves
            _define_field!(dataset, "A_real", real.(A_physical_host),
                "real part of the wave amplitude A")
            _define_field!(dataset, "A_imag", imag.(A_physical_host),
                "imaginary part of the wave amplitude A")
            _define_field!(dataset, "LA_real", real.(LA_physical_host),
                "real part of LA, the back-rotated wave velocity u + iv")
            _define_field!(dataset, "LA_imag", imag.(LA_physical_host),
                "imaginary part of LA, the back-rotated wave velocity u + iv")
        end

        _define_field!(dataset, "q_hat_real", real.(q_host),
            "real part of the potential vorticity q: " * fourier;
            dimensions=spectral)
        _define_field!(dataset, "q_hat_imag", imag.(q_host),
            "imaginary part of the potential vorticity q: " * fourier;
            dimensions=spectral)
        _define_field!(dataset, "B_hat_real", real.(B_host),
            "real part of the wave envelope B: " * fourier;
            dimensions=spectral)
        _define_field!(dataset, "B_hat_imag", imag.(B_host),
            "imaginary part of the wave envelope B: " * fourier;
            dimensions=spectral)
        if fixed_flow
            _define_field!(dataset, "psi_hat_real", real.(psi_spectral_host),
                "real part of the prescribed streamfunction: " * fourier;
                dimensions=spectral)
            _define_field!(dataset, "psi_hat_imag", imag.(psi_spectral_host),
                "imaginary part of the prescribed streamfunction: " * fourier;
                dimensions=spectral)
        end

        if write_velocities
            _define_field!(dataset, "u", u_host,
                "x velocity of the balanced flow (Lagrangian mean)")
            _define_field!(dataset, "v", v_host,
                "y velocity of the balanced flow (Lagrangian mean)")
            _define_field!(dataset, "w", w_host,
                "vertical velocity of the balanced flow (omega equation, " *
                "averaged from the faces to the centres)")
        end

        _define_vector!(dataset, "N2", "z", runtime.coefficients.N²,
            "squared buoyancy frequency at the cell centres")
        _define_vector!(dataset, "N2_face", "z_face",
            runtime.coefficients.N²_face,
            "squared buoyancy frequency at the upper cell faces")
        _define_vector!(dataset, "a_ell", "z_face", runtime.coefficients.a_ell,
            "f0^2 / N2 at the upper cell faces")

        dataset.attrib["title"] = "QG-YBJ model state"
        dataset.attrib["units_convention"] = _UNITS_CONVENTION
        dataset.attrib["model_time"] = simulation.clock.time
        dataset.attrib["iteration"] = simulation.clock.iteration
        dataset.attrib["f0"] = model.physics.coriolis.f
        dataset.attrib["Lx"] = grid.extent[1]
        dataset.attrib["Ly"] = grid.extent[2]
        dataset.attrib["Lz"] = grid.extent[3]
        dataset.attrib["wave_formulation"] = _wave_formulation_name(model)
        dataset.attrib["flow_evolution"] = _flow_evolution_name(model)
        dataset.attrib["feedback_mode"] = _feedback_mode_name(model)
        dataset.attrib["generalized_pv"] =
            _generalized_pv_convention(model)
    end

    manager.counter += 1
    manager.last_time = simulation.clock.time
    manager.last_iteration = simulation.clock.iteration
    _advance_output_deadline!(manager, simulation.clock.time)
    return filepath
end

function _maybe_write_simulation_output!(simulation::Simulation;
    initial::Bool=false)

    manager = simulation.output_manager
    manager === nothing && return simulation
    _output_due(manager, simulation.clock; initial) || return simulation
    _write_model_state_file!(manager, simulation)
    return simulation
end

"""Simulation-owned accumulator for scheduled energy diagnostics."""
mutable struct EnergyDiagnosticsManager{T, S}
    specification::S
    # Field snapshot for the original YBJ formulation, whose wave diagnostics
    # need a full field set; allocated on the first record.
    fields::Any
    last_time::Union{Nothing, T}
    last_iteration::Union{Nothing, Int}
    next_time::Union{Nothing, T}
    time::Vector{T}
    wave_KE::Vector{T}
    wave_PE::Vector{T}
    wave_CE::Vector{T}
    mean_flow_KE::Vector{T}
    mean_flow_PE::Vector{T}
    # Wall-clock time (`Base.time()`) the files were last written.
    last_write::Float64
    closed::Bool
end

function EnergyDiagnosticsManager(specification::S, ::Type{T},
    start_time=zero(T)) where {S, T}

    next_time = specification.schedule isa TimeInterval ?
                T(_next_deadline(start_time, specification.schedule.interval)) : nothing
    return EnergyDiagnosticsManager{T, S}(
        specification, nothing, nothing, nothing, next_time,
        T[], T[], T[], T[], T[], T[], 0.0, false)
end

"""Wall-clock seconds between rewrites of the energy files during a run."""
const _ENERGY_WRITE_INTERVAL = 10.0

const _ENERGY_COMPONENTS = (
    (:wave_KE, "wave kinetic energy"),
    (:wave_PE, "wave potential energy"),
    (:wave_CE, "wave correction energy"),
    (:mean_flow_KE, "mean-flow kinetic energy"),
    (:mean_flow_PE, "mean-flow potential energy"),
)

_energy_files(path) = [joinpath(path, "$(name).nc")
                       for name in (first.(_ENERGY_COMPONENTS)..., :total_energy)]

function _diagnostics_specification(simulation::Simulation)
    configured = simulation.run_options.diagnostics
    configured === false && return false
    configured isa EnergyDiagnosticsOutput && return configured
    configured isa AbstractSchedule || throw(ArgumentError(
        "simulation diagnostics must be an EnergyDiagnosticsOutput, schedule, or false"))
    output = simulation.run_options.output
    base_path = output isa NetCDFOutput ? output.path :
                simulation.run_options.output_dir
    return EnergyDiagnosticsOutput(
        path=joinpath(base_path, "diagnostic"), schedule=configured,
        overwrite=output isa NetCDFOutput && output.overwrite)
end

function _prepare_simulation_diagnostics!(simulation::Simulation)
    specification = _diagnostics_specification(simulation)
    if specification === false
        simulation.diagnostics_manager = nothing
        return simulation
    end

    manager = simulation.diagnostics_manager
    if manager !== nothing && !manager.closed
        return simulation
    end
    mkpath(specification.path)
    manager = EnergyDiagnosticsManager(
        specification, typeof(simulation.clock.time), simulation.clock.time)
    _resume_energy_series!(manager, simulation.clock)
    simulation.diagnostics_manager = manager
    return simulation
end

"""
Continue the energy series already in the diagnostic directory. A restored
simulation (clock past iteration 0) keeps the stored records up to its
restart time and does not record that time again. Other existing series are
an error unless `overwrite=true`, which drops them (for a restored run, only
the records after the restart time).
"""
function _resume_energy_series!(manager::EnergyDiagnosticsManager, clock::Clock)
    specification = manager.specification
    existing, stored, keep, tolerance = _check_energy_series(specification, clock)
    isempty(existing) && return manager
    if stored === nothing
        foreach(rm, existing)
        return manager
    end

    T = eltype(manager.time)
    append!(manager.time, T.(stored.time[1:keep]))
    for (name, _) in _ENERGY_COMPONENTS
        append!(getfield(manager, name), T.(getfield(stored, name)[1:keep]))
    end
    if keep > 0 && abs(stored.time[keep] - clock.time) <= tolerance
        # The restart time is already recorded.
        manager.last_time = T(clock.time)
        manager.last_iteration = clock.iteration
    end
    return manager
end

"""
The existing energy files of `specification`, the series stored in them (for a
restored run), how many of its records precede the restart, and the time
tolerance; an error unless `overwrite=true` when records would be replaced.
Changes no file.
"""
function _check_energy_series(specification::EnergyDiagnosticsOutput, clock::Clock)
    existing = filter(isfile, _energy_files(specification.path))
    tolerance = 8eps(max(abs(clock.time), one(clock.time)))
    isempty(existing) && return existing, nothing, 0, tolerance

    summary = joinpath(specification.path, "total_energy.nc")
    restored = clock.iteration > 0
    stored = restored && isfile(summary) ? _read_energy_series(summary) : nothing
    keep = stored === nothing ? 0 :
           count(time -> time <= clock.time + tolerance, stored.time)
    if stored === nothing || keep < length(stored.time)
        specification.overwrite || throw(ArgumentError(
            "$(specification.path) already holds energy diagnostics " *
            (restored ? "after this run's restart time $(clock.time)" :
                        "from another run") *
            "; choose a new path or pass overwrite=true to replace them"))
    end
    return existing, stored, keep, tolerance
end

"""Check, changing no file, that the run's energy series can be written."""
function _check_simulation_diagnostics(simulation::Simulation)
    specification = _diagnostics_specification(simulation)
    specification === false && return nothing
    manager = simulation.diagnostics_manager
    (manager !== nothing && !manager.closed) && return nothing
    _check_energy_series(specification, simulation.clock)
    return nothing
end

"""Times and energy components stored in a `total_energy.nc` file."""
function _read_energy_series(path::AbstractString)
    return NCDataset(path, "r") do dataset
        components = map(_ENERGY_COMPONENTS) do (name, _)
            name => Vector{Float64}(dataset[String(name)][:])
        end
        (; time=Vector{Float64}(dataset["time"][:]), components...)
    end
end

function _diagnostics_due(manager::EnergyDiagnosticsManager, clock::Clock;
    initial::Bool=false)

    manager.closed && return false
    manager.last_iteration === clock.iteration && return false
    initial && return true
    schedule = manager.specification.schedule
    schedule isa IterationInterval &&
        return clock.iteration % schedule.interval == 0
    if schedule isa TimeInterval
        tolerance = 8eps(max(abs(clock.time), abs(schedule.interval), one(clock.time)))
        return manager.next_time !== nothing &&
               clock.time >= manager.next_time - tolerance
    end
    error("unsupported diagnostic schedule $(typeof(schedule))")
end

function _advance_diagnostics_deadline!(manager::EnergyDiagnosticsManager,
    time)

    schedule = manager.specification.schedule
    schedule isa TimeInterval || return manager
    manager.next_time = _next_deadline(time, schedule.interval)
    return manager
end

@kernel function _energy_column_kernel!(energies, @Const(psi), @Const(A), @Const(C),
                                        @Const(kx), @Const(ky), mask, @Const(a_ell),
                                        dz, nz, include_wave_correction)
    i, j = @index(Global, NTuple)
    mean_flow_KE = zero(eltype(energies))
    mean_flow_PE = zero(eltype(energies))
    wave_KE = zero(eltype(energies))
    wave_PE = zero(eltype(energies))
    wave_CE = zero(eltype(energies))
    if keep_mode(mask, i, j)
        @inbounds kh² = kx[i]^2 + ky[j]^2
        for k in 1:nz
            @inbounds begin
                a = a_ell[k]
                # Horizontal fields use a full complex FFT, so Parseval assigns
                # the zero mode the same weight as every other stored coefficient.
                mean_flow_KE += kh² * abs2(psi[k, i, j])

                if k < nz
                    ψz = (psi[k + 1, i, j] - psi[k, i, j]) / dz
                    mean_flow_PE += a * abs2(ψz)
                end

                LA = if nz == 1
                    zero(eltype(C))
                elseif k == 1
                    a * C[k, i, j] / dz
                elseif k == nz
                    -a_ell[k - 1] * C[k - 1, i, j] / dz
                else
                    (a * C[k, i, j] - a_ell[k - 1] * C[k - 1, i, j]) / dz
                end
                # WKE is an informational phase-averaged kinetic-energy
                # diagnostic; unlike the following two terms, it is not part
                # of the coupled invariant in Asselin & Young (2019), (3.7).
                wave_KE += abs2(LA)

                # The shared outer factor 1/2 turns these accumulators into
                # (a/4)|∇A_z|² and (1/16)|ΔA|², respectively, as in (3.7).
                wave_PE += a / 2 * kh² * abs2(C[k, i, j])
                if include_wave_correction
                    # The correction results from L → L⁺ and is absent from
                    # the original YBJ formulation. PassiveWave keeps the L⁺
                    # diagnostic convention used by `_refresh_wave_diagnostics!`.
                    wave_CE += kh²^2 * abs2(A[k, i, j]) / 8
                end
            end
        end
    end
    @inbounds begin
        energies[1, i, j] = wave_KE
        energies[2, i, j] = wave_PE
        energies[3, i, j] = wave_CE
        energies[4, i, j] = mean_flow_KE
        energies[5, i, j] = mean_flow_PE
    end
end

"""
Domain-integrated energy components of `fields`, in the order wave KE, wave
PE, wave correction energy, mean-flow KE, mean-flow PE. Each horizontal mode
column is summed by one kernel work item, then the columns are reduced.

Nonlinear runs keep only the modes inside the two-thirds cutoff, so their
energy is summed over those; linear runs evolve every mode, and all count.
"""
_local_energy_components(model::QGYBJModel, fields::ModelFields) =
    _local_energy_components(model, fields.psi, fields.A, fields.C, fields.u)

function _local_energy_components(model::QGYBJModel, psi, A, C, real_template)
    runtime = model.runtime
    geometry = runtime.geometry
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dz = nz > 1 ? abs(geometry.z[2] - geometry.z[1]) : one(eltype(geometry.z))
    mask = _linear(ETDModelOptions(model.physics, model.numerics)) ?
           KeepAllModes() : runtime.dealias_mask

    energies = similar(real_template, 5, nx, ny)
    launch!(_energy_column_kernel!, energies, (nx, ny),
            energies, psi, A, C, geometry.kx, geometry.ky,
            mask, runtime.device_coefficients.a_ell, dz, nz,
            !(model.physics.formulation isa YBJ))
    totals = on_host(vec(sum(energies; dims=(2, 3))))

    normalization = 0.5 / ((nx * ny)^2 * nz)
    return Tuple(totals[index] * normalization for index in 1:5)
end

_energy_components(model::QGYBJModel, fields::ModelFields) =
    _local_energy_components(model, fields)

"""
Energy components rebuilt from the model's prognostic `q` and `B` (and a
fixed flow's prescribed ψ), so that a record never relies on, or mutates, the
live diagnostic fields: ETD-RK2 keeps only q and B prognostically.

ψ (balanced, from q - qʷ) and A, C (from B) are rebuilt in scratch buffers.
The original YBJ formulation recovers A through the full wave tendency, which
needs a complete field set, so it works on a snapshot of the fields instead.
"""
function _current_energy_components(model::QGYBJModel, manager)
    fields = model.fields
    runtime = model.runtime
    workspace = runtime.workspace
    coefficients = runtime.device_coefficients
    options = ETDModelOptions(model.physics, model.numerics)

    if model.physics.formulation isa YBJ
        if manager.fields === nothing
            manager.fields = copy_fields(fields)
        else
            copy_fields!(manager.fields, fields)
        end
        snapshot = manager.fields
        # FixedFlow owns a prescribed streamfunction independently of q.
        _fixed_flow(options) || _invert_total_q_to_psi!(snapshot, runtime.geometry,
            options, runtime.plans, coefficients.a_ell, runtime.dealias_mask;
            workspace)
        _refresh_wave_diagnostics!(snapshot, model)
        return _energy_components(model, snapshot)
    end

    return with_scratch(workspace) do
        psi = if _fixed_flow(options)
            fields.psi
        else
            _invert_total_q_to_psi!(scratch_like(workspace, fields.psi), fields,
                runtime.geometry, options, runtime.plans, coefficients.a_ell,
                runtime.dealias_mask; workspace)
        end
        A = scratch_like(workspace, fields.A)
        C = scratch_like(workspace, fields.C)
        Elliptic._invert_B_to_A!(A, C, fields.B, runtime.geometry,
            coefficients.a_ell, workspace)
        _local_energy_components(model, psi, A, C, fields.u)
    end
end

function _record_energy_diagnostics!(manager::EnergyDiagnosticsManager,
    simulation::Simulation)

    values = _current_energy_components(simulation.model, manager)
    T = eltype(manager.time)
    push!(manager.time, T(simulation.clock.time))
    push!(manager.wave_KE, T(values[1]))
    push!(manager.wave_PE, T(values[2]))
    push!(manager.wave_CE, T(values[3]))
    push!(manager.mean_flow_KE, T(values[4]))
    push!(manager.mean_flow_PE, T(values[5]))
    manager.last_time = T(simulation.clock.time)
    manager.last_iteration = simulation.clock.iteration
    _advance_diagnostics_deadline!(manager, simulation.clock.time)
    return manager
end

function _maybe_record_simulation_diagnostics!(simulation::Simulation;
    initial::Bool=false)

    manager = simulation.diagnostics_manager
    manager === nothing && return simulation
    _diagnostics_due(manager, simulation.clock; initial) || return simulation
    _record_energy_diagnostics!(manager, simulation)
    # Keep the files current, so that a run that is killed keeps its series.
    Base.time() - manager.last_write >= _ENERGY_WRITE_INTERVAL &&
        _write_energy_diagnostics!(manager, simulation.model)
    return simulation
end

"""
Write a NetCDF file through `build(dataset)` atomically: into a temporary file
that then replaces `path`, so a reader never sees a partly written file.
"""
function _write_atomically(build, path::AbstractString)
    temporary = path * ".tmp"
    try
        NCDataset(build, temporary, "c"; format=:netcdf3_classic)
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return path
end

function _define_energy_time!(dataset, times)
    dataset.dim["time"] = length(times)
    _define_vector!(dataset, "time", "time", times, "time")
    dataset.attrib["units_convention"] = _UNITS_CONVENTION
    return dataset
end

function _write_energy_diagnostics!(manager::EnergyDiagnosticsManager,
    model::QGYBJModel)

    manager.last_write = Base.time()
    isempty(manager.time) && return manager
    path = manager.specification.path
    for (name, long_name) in _ENERGY_COMPONENTS
        _write_atomically(joinpath(path, "$(name).nc")) do dataset
            _define_energy_time!(dataset, manager.time)
            _define_vector!(dataset, String(name), "time",
                            getfield(manager, name), long_name)
            dataset.attrib["title"] = "QG-YBJ energy diagnostic"
        end
    end

    total_wave = manager.wave_KE .+ manager.wave_PE .+ manager.wave_CE
    total_flow = manager.mean_flow_KE .+ manager.mean_flow_PE
    # Asselin & Young (2019), equation (3.7): wave kinetic energy is
    # informational and is not part of the conserved coupled energy.
    coupled_energy = total_flow .+ manager.wave_PE .+ manager.wave_CE
    _write_atomically(joinpath(path, "total_energy.nc")) do dataset
        _define_energy_time!(dataset, manager.time)
        for (name, long_name) in _ENERGY_COMPONENTS
            _define_vector!(dataset, String(name), "time",
                            getfield(manager, name), long_name)
        end
        _define_vector!(dataset, "total_wave_energy", "time", total_wave,
            "wave_KE + wave_PE + wave_CE")
        _define_vector!(dataset, "total_flow_energy", "time", total_flow,
            "mean_flow_KE + mean_flow_PE")
        _define_vector!(dataset, "coupled_energy", "time", coupled_energy,
            "total_flow_energy + wave_PE + wave_CE, conserved by the ideal " *
            "coupled system (Asselin & Young 2019, eq. 3.7)")
        _define_vector!(dataset, "total_energy", "time", total_wave .+ total_flow,
            "total_wave_energy + total_flow_energy")
        dataset.attrib["title"] = "QG-YBJ total energy diagnostics"
    end
    return manager
end

function _finish_simulation_diagnostics!(simulation::Simulation)
    manager = simulation.diagnostics_manager
    (manager === nothing || manager.closed) && return simulation
    try
        if simulation.state !== Failed &&
           manager.last_iteration !== simulation.clock.iteration
            _record_energy_diagnostics!(manager, simulation)
        end
        _write_energy_diagnostics!(manager, simulation.model)
    catch
        simulation.state = Failed
        rethrow()
    finally
        manager.closed = true
        # A closed manager is replaced on the next run!, so free its snapshot.
        manager.fields = nothing
    end
    return simulation
end

function _finish_simulation_output!(simulation::Simulation)
    manager = simulation.output_manager
    (manager === nothing || manager.closed) && return simulation

    try
        if simulation.state !== Failed &&
           manager.last_iteration !== simulation.clock.iteration
            _write_model_state_file!(manager, simulation)
        end
    catch
        simulation.state = Failed
        rethrow()
    finally
        manager.closed = true
    end
    return simulation
end

function _read_restart(model::QGYBJModel, path::AbstractString)
    return NCDataset(path, "r") do dataset
        if haskey(dataset.attrib, "wave_formulation")
            stored = String(dataset.attrib["wave_formulation"])
            expected = _wave_formulation_name(model)
            stored == expected || throw(ArgumentError(
                "restart wave formulation is $stored; model uses $expected"))
        end
        if haskey(dataset.attrib, "generalized_pv")
            stored = String(dataset.attrib["generalized_pv"])
            expected = _generalized_pv_convention(model)
            stored == expected || throw(ArgumentError(
                "restart generalized-PV convention is $stored; " *
                "model requires $expected"))
        end
        _check_restart_configuration(model, dataset, path)
        required = ("q_hat_real", "q_hat_imag",
                    "B_hat_real", "B_hat_imag")
        all(name -> haskey(dataset, name), required) ||
            throw(ArgumentError("restart file is missing prognostic fields"))
        q = _from_xyz(dataset["q_hat_real"][:, :, :] .+
                      im .* dataset["q_hat_imag"][:, :, :])
        B = _from_xyz(dataset["B_hat_real"][:, :, :] .+
                      im .* dataset["B_hat_imag"][:, :, :])
        size(q) == (model.grid.size[3], model.grid.size[1], model.grid.size[2]) ||
            throw(DimensionMismatch("restart q dimensions do not match model grid"))
        size(B) == size(q) ||
            throw(DimensionMismatch("restart B dimensions do not match restart q"))
        psi = model.physics.flow isa FixedFlow ?
              _read_restart_streamfunction(model, dataset, path, size(q)) : nothing
        if haskey(dataset.attrib, "flow_evolution")
            stored = String(dataset.attrib["flow_evolution"])
            expected = _flow_evolution_name(model)
            stored == expected || model.physics.flow isa FixedFlow ||
                @warn "Restart file $path was written by a $stored model; this " *
                      "model's flow evolves, so its streamfunction is inverted " *
                      "from the file's q"
        end
        time = haskey(dataset, "time") ? Float64(dataset["time"][1]) : nothing
        iteration = haskey(dataset.attrib, "iteration") ?
                    Int(dataset.attrib["iteration"]) : nothing
        (; q, B, psi, time, iteration)
    end
end

"""
The streamfunction of a restart file, for a `FixedFlow` model, whose flow is
prescribed rather than inverted from q: the Fourier coefficients `psi_hat`,
or else the physical `psi` of an older file, transformed.
"""
function _read_restart_streamfunction(model::QGYBJModel, dataset, path, dims)
    psi = if haskey(dataset, "psi_hat_real") && haskey(dataset, "psi_hat_imag")
        _from_xyz(dataset["psi_hat_real"][:, :, :] .+
                  im .* dataset["psi_hat_imag"][:, :, :])
    elseif haskey(dataset, "psi")
        physical = ComplexF64.(_from_xyz(dataset["psi"][:, :, :]))
        FFTW.fft!(physical, (2, 3))
    else
        throw(ArgumentError(
            "restart file $path holds no streamfunction, which a FixedFlow " *
            "model needs: its flow is prescribed, not inverted from q"))
    end
    size(psi) == dims ||
        throw(DimensionMismatch("restart ψ dimensions do not match restart q"))
    return psi
end

"""
Compare the domain, f, and N² recorded in a restart file with `model`. The
Fourier coefficients only mean the same field on a domain of the same size,
so a different extent is an error. A different f or N² changes how q is
inverted, which may be intended, so it only warns.
"""
function _check_restart_configuration(model::QGYBJModel, dataset, path)
    attributes = dataset.attrib
    for (name, value) in zip(("Lx", "Ly", "Lz"), model.grid.extent)
        haskey(attributes, name) || continue
        stored = Float64(attributes[name])
        isapprox(stored, value; rtol=1e-6) || throw(ArgumentError(
            "restart file $path is for a domain with $name = $stored; " *
            "the model has $name = $value"))
    end
    f = model.physics.coriolis.f
    if haskey(attributes, "f0") && !isapprox(Float64(attributes["f0"]), f; rtol=1e-6)
        @warn "Restart file $path was written with f0 = $(attributes["f0"]); " *
              "the model uses f0 = $f, which changes the flow inverted from q"
    end
    N² = model.runtime.coefficients.N²
    if haskey(dataset, "N2")
        stored = dataset["N2"][:]
        length(stored) == length(N²) && isapprox(stored, N²; rtol=1e-6) ||
            @warn "Restart file $path was written with a different N² profile, " *
                  "which changes the flow inverted from q"
    end
    return nothing
end

"""
    restore!(model, path)
    restore!(simulation, path)

Restore the prognostic `q` and `B` fields from a simulation state file and
rebuild the model's diagnostic fields. The file must be for a domain of the
same size and resolution; a different f or N² warns.

A `FixedFlow` model takes its prescribed streamfunction from the file; an
evolving flow is inverted from the restored q.

Restoring a `Simulation` also sets its clock to the file's time and
iteration, so output, diagnostics, and particles continue from there, and
`stop_time` and `stop_iteration` keep counting from the original start.
"""
function restore!(model::QGYBJModel, path::AbstractString)
    _restore_fields!(model, path)
    return model
end

function restore!(simulation::Simulation, path::AbstractString)
    _assert_mutable(simulation)
    restart = _restore_fields!(simulation.model, path)
    T = typeof(simulation.clock.time)
    restart.time === nothing || (simulation.clock.time = T(restart.time))
    restart.iteration === nothing ||
        (simulation.clock.iteration = restart.iteration)
    return simulation
end

function _restore_fields!(model::QGYBJModel, path::AbstractString)
    model.runtime.finalized && error("cannot restore a finalized model")
    restart = _read_restart(model, path)
    copy_from_host!(model.fields.q, restart.q)
    copy_from_host!(model.fields.B, restart.B)
    if restart.psi === nothing
        invert_q_to_psi!(model)
    else
        copy_from_host!(model.fields.psi, restart.psi)
    end
    compute_velocities!(model; compute_w=false)
    _refresh_wave_diagnostics!(model)
    return restart
end

"""Holder of the field snapshot that the energies of a `YBJ()` model need."""
mutable struct _EnergySnapshot
    fields::Any
end

"""
The domain-mean energies of `model`'s current state, named as the energy
diagnostics record them, rebuilt from the prognostic fields without changing
any model field.
"""
function _model_energies(model::QGYBJModel)
    _operator_context(model)   # errors for a finalized model
    values = _current_energy_components(model, _EnergySnapshot(nothing))
    return NamedTuple{first.(_ENERGY_COMPONENTS)}(Float64.(values))
end

"""
    flow_kinetic_energy(model)

The domain-mean kinetic energy of the balanced flow, ½⟨u² + v²⟩ (m² s⁻² in a
dimensional setup): the `mean_flow_KE` that the energy diagnostics record. It
is computed from the prognostic q (or a `FixedFlow` model's prescribed ψ),
changes no model field, and is a host scalar, also on a GPU.
"""
Diagnostics.flow_kinetic_energy(model::QGYBJModel) =
    _model_energies(model).mean_flow_KE

"""
    wave_energy(model) -> (; kinetic, potential, correction)

The domain-mean wave energies of `model`, as the energy diagnostics record
them (`wave_KE`, `wave_PE`, and `wave_CE`): the kinetic energy ½⟨|LA|²⟩ of
the wave velocity, the potential energy, and the YBJ⁺ correction (zero for
`YBJ()`). With the balanced flow's energy, the potential and correction
energies make up the energy that the ideal coupled system conserves
(Asselin & Young 2019, eq. 3.7); the kinetic energy is informational. They
are rebuilt from the prognostic B and change no model field.
"""
function Diagnostics.wave_energy(model::QGYBJModel)
    energies = _model_energies(model)
    return (kinetic=energies.wave_KE, potential=energies.wave_PE,
            correction=energies.wave_CE)
end
