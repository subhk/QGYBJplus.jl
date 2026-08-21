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
                T(start_time + specification.schedule.interval) : nothing
    return ModelOutputManager{T, S}(
        specification, 1, nothing, nothing, next_time, false)
end

function _run_on_root(operation, model::QGYBJModel)
    runtime = model.runtime
    result = nothing
    failure = nothing
    if runtime.mpi.is_root
        try
            result = operation()
        catch exception
            failure = sprint(showerror, exception, catch_backtrace())
        end
    end
    failure = MPI.bcast(failure, 0, runtime.mpi.comm)
    failure === nothing || error(failure)
    return result
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

    _run_on_root(simulation.model) do
        mkpath(specification.path)
    end
    simulation.output_manager =
        ModelOutputManager(specification, typeof(simulation.clock.time),
                           simulation.clock.time)
    return simulation
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
    manager.next_time === nothing &&
        (manager.next_time = time + schedule.interval)
    tolerance = 8eps(max(abs(time), abs(schedule.interval), one(time)))
    while manager.next_time <= time + tolerance
        manager.next_time += schedule.interval
    end
    return manager
end

function _physical_field(field, runtime::ModelRuntime)
    physical = allocate_fft_backward_dst(field, runtime)
    fft_backward!(physical, field, runtime)
    return physical
end

function _gather_array(field, model::QGYBJModel)
    runtime = model.runtime
    gathered = gather_to_root(
        field, runtime.geometry, runtime.mpi)
    return runtime.mpi.is_root ? Array(parent(gathered)) : nothing
end

function _refresh_output_diagnostics!(model::QGYBJModel,
    specification::NetCDFOutput)

    write_psi = _saves_psi(specification.fields)
    write_waves = _saves_waves(specification.fields)
    write_velocities = specification.velocities ||
                       :velocities in specification.fields

    (write_psi || write_velocities) && invert_q_to_psi!(model)
    if write_waves &&
       (model.physics.formulation isa YBJPlus ||
        model.physics.formulation isa PassiveWave)
        invert_B_to_A!(model)
    end
    write_velocities && compute_velocities!(model; compute_w=true)
    return model
end

function _define_field!(dataset, name::AbstractString, values)
    variable = NCDatasets.defVar(dataset, name, Float64, ("x", "y", "z"))
    variable[:, :, :] = _to_xyz(values)
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

    psi_physical = write_psi ? _physical_field(fields.psi, runtime) : nothing
    B_physical = write_waves ? _physical_field(fields.B, runtime) : nothing
    A_physical = write_waves ? _physical_field(fields.A, runtime) : nothing

    q_global = _gather_array(fields.q, model)
    B_global = _gather_array(fields.B, model)
    psi_global = write_psi ? _gather_array(psi_physical, model) : nothing
    B_physical_global = write_waves ? _gather_array(B_physical, model) : nothing
    A_physical_global = write_waves ? _gather_array(A_physical, model) : nothing
    u_global = write_velocities ? _gather_array(fields.u, model) : nothing
    v_global = write_velocities ? _gather_array(fields.v, model) : nothing
    w_global = write_velocities ? _gather_array(fields.w, model) : nothing

    filename = @sprintf("state%04d.nc", manager.counter)
    filepath = joinpath(specification.path, filename)
    _run_on_root(model) do
        NCDataset(filepath, "c") do dataset
            nx, ny, nz = grid.size
            dataset.dim["x"] = nx
            dataset.dim["y"] = ny
            dataset.dim["z"] = nz
            dataset.dim["time"] = 1

            x = NCDatasets.defVar(dataset, "x", Float64, ("x",))
            y = NCDatasets.defVar(dataset, "y", Float64, ("y",))
            z = NCDatasets.defVar(dataset, "z", Float64, ("z",))
            time = NCDatasets.defVar(dataset, "time", Float64, ("time",))
            x[:] = grid.x
            y[:] = grid.y
            z[:] = grid.z
            time[1] = simulation.clock.time

            if write_psi
                _define_field!(dataset, "psi", real.(psi_global))
            end
            if write_waves
                _define_field!(dataset, "LAr", real.(B_physical_global))
                _define_field!(dataset, "LAi", imag.(B_physical_global))
                _define_field!(dataset, "Ar", real.(A_physical_global))
                _define_field!(dataset, "Ai", imag.(A_physical_global))
            end

            _define_field!(dataset, "q_real", real.(q_global))
            _define_field!(dataset, "q_imag", imag.(q_global))
            _define_field!(dataset, "B_real", real.(B_global))
            _define_field!(dataset, "B_imag", imag.(B_global))

            if write_velocities
                _define_field!(dataset, "u", u_global)
                _define_field!(dataset, "v", v_global)
                _define_field!(dataset, "w", w_global)
            end

            N² = NCDatasets.defVar(dataset, "N2", Float64, ("z",))
            a_ell = NCDatasets.defVar(dataset, "a_ell", Float64, ("z",))
            N²[:] = runtime.coefficients.N²
            a_ell[:] = runtime.coefficients.a_ell

            dataset.attrib["title"] = "QG-YBJ model state"
            dataset.attrib["model_time"] = simulation.clock.time
            dataset.attrib["iteration"] = simulation.clock.iteration
            dataset.attrib["f0"] = model.physics.coriolis.f
        end
        filepath
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
    last_time::Union{Nothing, T}
    last_iteration::Union{Nothing, Int}
    next_time::Union{Nothing, T}
    time::Vector{T}
    wave_KE::Vector{T}
    wave_PE::Vector{T}
    wave_CE::Vector{T}
    mean_flow_KE::Vector{T}
    mean_flow_PE::Vector{T}
    closed::Bool
end

function EnergyDiagnosticsManager(specification::S, ::Type{T},
    start_time=zero(T)) where {S, T}

    next_time = specification.schedule isa TimeInterval ?
                T(start_time + specification.schedule.interval) : nothing
    return EnergyDiagnosticsManager{T, S}(
        specification, nothing, nothing, next_time,
        T[], T[], T[], T[], T[], T[], false)
end

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
        path=joinpath(base_path, "diagnostic"), schedule=configured)
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
    _run_on_root(simulation.model) do
        mkpath(specification.path)
    end
    simulation.diagnostics_manager = EnergyDiagnosticsManager(
        specification, typeof(simulation.clock.time), simulation.clock.time)
    return simulation
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
    manager.next_time === nothing &&
        (manager.next_time = time + schedule.interval)
    tolerance = 8eps(max(abs(time), abs(schedule.interval), one(time)))
    while manager.next_time <= time + tolerance
        manager.next_time += schedule.interval
    end
    return manager
end

function _local_energy_components(model::QGYBJModel, fields::ModelFields)
    runtime = model.runtime
    geometry = runtime.geometry
    coefficients = runtime.coefficients
    mask = runtime.dealias_mask
    psi_z = allocate_z_pencil(geometry, eltype(fields.psi))
    A_z = allocate_z_pencil(geometry, eltype(fields.A))
    C_z = allocate_z_pencil(geometry, eltype(fields.C))
    transpose_to_z_pencil!(psi_z, fields.psi, geometry)
    transpose_to_z_pencil!(A_z, fields.A, geometry)
    transpose_to_z_pencil!(C_z, fields.C, geometry)
    psi = parent(psi_z)
    A = parent(A_z)
    C = parent(C_z)
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dz = nz > 1 ? abs(geometry.z[2] - geometry.z[1]) : 1.0

    mean_flow_KE = 0.0
    mean_flow_PE = 0.0
    wave_KE = 0.0
    wave_PE = 0.0
    wave_CE = 0.0
    for j in axes(psi, 3), i in axes(psi, 2)
        i_global = local_to_global_z(i, 2, geometry)
        j_global = local_to_global_z(j, 3, geometry)
        mask[i_global, j_global] || continue
        kh² = geometry.kx[i_global]^2 + geometry.ky[j_global]^2
        horizontal_weight = (i_global == 1 && j_global == 1) ? 0.5 : 1.0
        for k in axes(psi, 1)
            k_global = local_to_global_z(k, 1, geometry)
            a = coefficients.a_ell[k_global]
            mean_flow_KE += horizontal_weight * kh² * abs2(psi[k, i, j])

            if k < last(axes(psi, 1))
                ψz = (psi[k + 1, i, j] - psi[k, i, j]) / dz
                mean_flow_PE += horizontal_weight * a * abs2(ψz)
            end

            LA = if nz == 1
                zero(eltype(C))
            elseif k_global == 1
                a * C[k, i, j] / dz
            elseif k_global == nz
                -coefficients.a_ell[k_global - 1] * C[k - 1, i, j] / dz
            else
                (a * C[k, i, j] -
                 coefficients.a_ell[k_global - 1] * C[k - 1, i, j]) / dz
            end
            wave_KE += horizontal_weight * abs2(LA)
            wave_PE += horizontal_weight * (0.5 / a) * kh² * abs2(C[k, i, j])
            wave_CE += horizontal_weight * (1 / (8a^2)) * kh²^2 * abs2(A[k, i, j])
        end
    end

    normalization = 0.5 / ((nx * ny)^2 * nz)
    return (
        wave_KE * normalization,
        wave_PE * normalization,
        wave_CE * normalization,
        mean_flow_KE * normalization,
        mean_flow_PE * normalization,
    )
end

function _energy_components(model::QGYBJModel, fields::ModelFields)
    local_values = _local_energy_components(model, fields)
    comm = model.runtime.mpi.comm
    global_values = ntuple(length(local_values)) do index
        MPI.Allreduce(local_values[index], +, comm)
    end
    return global_values
end

function _record_energy_diagnostics!(manager::EnergyDiagnosticsManager,
    simulation::Simulation)

    # ETD-RK2 stores only q and B prognostically. Rebuild the diagnostic fields
    # required by the energy formulas, including for passive/no-dispersion runs
    # whose stepping path intentionally clears A and C. Work on a copy so
    # scheduled observation never mutates the live model's diagnostic state.
    model = simulation.model
    fields = copy_fields(model.fields)
    runtime = model.runtime
    coefficients = runtime.coefficients
    density = coefficients.stratification
    invert_q_to_psi!(fields, runtime.geometry;
        a=coefficients.a_ell,
        rho_u=density.rho_u,
        rho_s=density.rho_s,
        workspace=runtime.workspace)
    if model.physics.formulation isa YBJPlus ||
       model.physics.formulation isa PassiveWave
        invert_B_to_A!(fields, runtime.geometry, coefficients.a_ell;
            rho_u=density.rho_u,
            rho_s=density.rho_s,
            workspace=runtime.workspace)
    end
    values = _energy_components(model, fields)
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
    return simulation
end

function _write_energy_file(path::AbstractString, variable_name::AbstractString,
    times, values, long_name::AbstractString)

    NCDataset(path, "c"; format=:netcdf3_classic) do dataset
        dataset.dim["time"] = length(times)
        time = NCDatasets.defVar(dataset, "time", Float64, ("time",))
        energy = NCDatasets.defVar(
            dataset, variable_name, Float64, ("time",))
        time[:] = times
        energy[:] = values
        time.attrib["units"] = "seconds"
        energy.attrib["long_name"] = long_name
        dataset.attrib["title"] = "QG-YBJ energy diagnostic"
    end
    return path
end

function _write_energy_diagnostics!(manager::EnergyDiagnosticsManager,
    model::QGYBJModel)

    isempty(manager.time) && return manager
    path = manager.specification.path
    _run_on_root(model) do
        _write_energy_file(joinpath(path, "wave_KE.nc"), "wave_KE",
            manager.time, manager.wave_KE, "wave kinetic energy")
        _write_energy_file(joinpath(path, "wave_PE.nc"), "wave_PE",
            manager.time, manager.wave_PE, "wave potential energy")
        _write_energy_file(joinpath(path, "wave_CE.nc"), "wave_CE",
            manager.time, manager.wave_CE, "wave correction energy")
        _write_energy_file(joinpath(path, "mean_flow_KE.nc"), "mean_flow_KE",
            manager.time, manager.mean_flow_KE, "mean-flow kinetic energy")
        _write_energy_file(joinpath(path, "mean_flow_PE.nc"), "mean_flow_PE",
            manager.time, manager.mean_flow_PE, "mean-flow potential energy")

        NCDataset(joinpath(path, "total_energy.nc"), "c";
                  format=:netcdf3_classic) do dataset
            dataset.dim["time"] = length(manager.time)
            time = NCDatasets.defVar(dataset, "time", Float64, ("time",))
            time[:] = manager.time
            time.attrib["units"] = "seconds"
            series = (
                wave_KE=manager.wave_KE,
                wave_PE=manager.wave_PE,
                wave_CE=manager.wave_CE,
                mean_flow_KE=manager.mean_flow_KE,
                mean_flow_PE=manager.mean_flow_PE,
            )
            for (name, values) in pairs(series)
                variable = NCDatasets.defVar(
                    dataset, String(name), Float64, ("time",))
                variable[:] = values
            end
            total_wave = manager.wave_KE .+ manager.wave_PE .+
                         manager.wave_CE
            total_flow = manager.mean_flow_KE .+ manager.mean_flow_PE
            total_wave_variable = NCDatasets.defVar(
                dataset, "total_wave_energy", Float64, ("time",))
            total_wave_variable[:] = total_wave
            total_flow_variable = NCDatasets.defVar(
                dataset, "total_flow_energy", Float64, ("time",))
            total_flow_variable[:] = total_flow
            total_variable = NCDatasets.defVar(
                dataset, "total_energy", Float64, ("time",))
            total_variable[:] = total_wave .+ total_flow
            dataset.attrib["title"] = "QG-YBJ total energy diagnostics"
        end
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

function _read_restart_on_root(model::QGYBJModel, path::AbstractString)
    return _run_on_root(model) do
        NCDataset(path, "r") do dataset
            required = ("q_real", "q_imag", "B_real", "B_imag")
            all(name -> haskey(dataset, name), required) ||
                throw(ArgumentError("restart file is missing prognostic fields"))
            q = _from_xyz(dataset["q_real"][:, :, :] .+
                          im .* dataset["q_imag"][:, :, :])
            B = _from_xyz(dataset["B_real"][:, :, :] .+
                          im .* dataset["B_imag"][:, :, :])
            size(q) == (model.grid.size[3], model.grid.size[1], model.grid.size[2]) ||
                throw(DimensionMismatch("restart q dimensions do not match model grid"))
            size(B) == size(q) ||
                throw(DimensionMismatch("restart B dimensions do not match restart q"))
            (q, B)
        end
    end
end

"""
    restore!(model, path)

Restore the prognostic `q` and `B` fields from a simulation state file and
rebuild the model's diagnostic fields.
"""
function restore!(model::QGYBJModel, path::AbstractString)
    model.runtime.finalized && error("cannot restore a finalized model")
    root_fields = _read_restart_on_root(model, path)
    q_root = is_root(model) ? first(root_fields) : nothing
    B_root = is_root(model) ? last(root_fields) : nothing
    runtime = model.runtime
    q = scatter_from_root(q_root, runtime.geometry, runtime.mpi;
                          plans=runtime.plans)
    B = scatter_from_root(B_root, runtime.geometry, runtime.mpi;
                          plans=runtime.plans)
    copyto!(parent(model.fields.q), parent(q))
    copyto!(parent(model.fields.B), parent(B))
    invert_q_to_psi!(model)
    if model.physics.formulation isa YBJPlus ||
       model.physics.formulation isa PassiveWave
        invert_B_to_A!(model)
    end
    compute_velocities!(model; compute_w=false)
    return model
end

function restore!(simulation::Simulation, path::AbstractString)
    _assert_mutable(simulation)
    restore!(simulation.model, path)
    return simulation
end

"""Return globally reduced balanced-flow kinetic energy for `model`."""
function Diagnostics.flow_kinetic_energy(model::QGYBJModel)
    compute_velocities!(model; compute_w=false)
    return Diagnostics.flow_kinetic_energy_global(
        model.fields.u, model.fields.v, model.runtime.mpi)
end

"""Return globally reduced envelope and amplitude energies for `model`."""
function Diagnostics.wave_energy(model::QGYBJModel)
    if model.physics.formulation isa YBJPlus ||
       model.physics.formulation isa PassiveWave
        invert_B_to_A!(model)
    end
    return Diagnostics.wave_energy_global(
        model.fields.B, model.fields.A, model.runtime.mpi)
end
