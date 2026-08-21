"""Manager for a simulation-owned NetCDF output stream."""
mutable struct ModelOutputManager{T, S}
    specification::S
    counter::Int
    last_time::Union{Nothing, T}
    last_iteration::Union{Nothing, Int}
    closed::Bool
end

@inline _to_xyz(array::AbstractArray) = permutedims(array, (2, 3, 1))
@inline _from_xyz(array::AbstractArray) = permutedims(array, (3, 1, 2))

ModelOutputManager(specification::S, ::Type{T}) where {S, T} =
    ModelOutputManager{T, S}(specification, 1, nothing, nothing, false)

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
        ModelOutputManager(specification, typeof(simulation.clock.time))
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
        manager.last_time === nothing && return true
        tolerance = 8eps(max(abs(clock.time), abs(schedule.interval), one(clock.time)))
        return clock.time - manager.last_time >= schedule.interval - tolerance
    end
    error("unsupported output schedule $(typeof(schedule))")
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
