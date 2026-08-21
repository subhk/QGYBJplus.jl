"""High-level composition-first model initialization and simulation lifecycle."""

using MPI
using Printf

_model(model::QGYBJModel) = model
_model(simulation::Simulation) = simulation.model

_runtime(model::QGYBJModel) = model.runtime

is_root(simulation::Simulation) = is_root(simulation.model)
nprocs(simulation::Simulation) = nprocs(simulation.model)
_time_step(simulation::Simulation) = simulation.timestepper.Δt
_advect_model_particles!(::Nothing, model::QGYBJModel, Δt, time) = nothing

function _before_stop_time(simulation::Simulation)
    stop_time = simulation.stop_time
    stop_time === nothing && return true
    time = simulation.clock.time
    tolerance = 8eps(max(
        abs(time), abs(stop_time), abs(_time_step(simulation)), one(time)))
    return time < stop_time - tolerance
end

function _check_termination_conditions!(simulation::Simulation)
    model = simulation.model
    runtime = model.runtime
    fields = model.fields
    local_bad = any(value -> !isfinite(value), parent(fields.q)) ||
                any(value -> !isfinite(value), parent(fields.B)) ||
                any(value -> !isfinite(value), parent(fields.psi))
    bad_count = MPI.Allreduce(local_bad ? 1 : 0, +, runtime.mpi.comm)
    bad_count == 0 || error("non-finite value detected in the model state")

    # Reuse the runtime-owned FFT input buffer: this check runs every step and
    # must not allocate a grid-sized physical field each time.
    psi_physical = runtime.transform_destinations.input
    fft_backward!(psi_physical, fields.psi, runtime)
    local_maximum = maximum(abs, parent(psi_physical))
    global_maximum = MPI.Allreduce(local_maximum, MPI.MAX, runtime.mpi.comm)
    global_maximum <= 1e10 || error(
        "solution appears to be blowing up (max |psi| = $global_maximum)")
    return simulation
end

function _assert_mutable(owner::QGYBJModel)
    owner.runtime.finalized && error("cannot modify a finalized model")
    return owner
end

function _assert_mutable(simulation::Simulation)
    state = simulation.state
    if state === Running
        throw(InvalidStateException(
            "cannot modify a running simulation", :running))
    elseif state === Failed
        throw(InvalidStateException(
            "cannot modify a failed simulation", :failed))
    elseif state === Finalized
        throw(InvalidStateException(
            "cannot modify a finalized simulation", :finalized))
    end
    simulation.model.runtime.finalized &&
        throw(InvalidStateException(
            "model runtime has been finalized", :finalized))
    return simulation
end

function _configure_time_stepping!(simulation::Simulation;
    Δt=nothing, stop_time=nothing, stop_iteration=nothing)

    if Δt !== nothing
        value = typeof(_time_step(simulation))(Δt)
        isfinite(value) && value > zero(value) ||
            throw(ArgumentError("Δt must be finite and positive (got $Δt)"))
        simulation.timestepper.Δt = value
    end

    if stop_iteration !== nothing
        stop_iteration > 0 || throw(ArgumentError("stop_iteration must be positive"))
        simulation.stop_iteration = Int(stop_iteration)
        simulation.stop_time = nothing
    elseif stop_time !== nothing
        value = typeof(_time_step(simulation))(stop_time)
        isfinite(value) && value > zero(value) ||
            throw(ArgumentError(
                "stop_time must be finite and positive (got $stop_time)"))
        simulation.stop_time = value
        simulation.stop_iteration = nothing
    elseif simulation.stop_iteration === nothing && simulation.stop_time === nothing
        simulation.stop_iteration = 1000
    end

    return simulation
end

_saves_psi(fields) = any(field -> field in (:ψ, :psi, :q, :flow), fields)
_saves_waves(fields) = any(field -> field in (:waves, :wave, :B, :A), fields)

function _schedule_in_seconds(schedule, Δt)
    schedule === nothing && return nothing
    schedule isa TimeInterval && return schedule.interval
    schedule isa IterationInterval && return schedule.interval * Δt
    throw(ArgumentError("schedule must be a TimeInterval or IterationInterval"))
end

function _configure_output!(simulation::Simulation; output=nothing,
    diagnostics=nothing, verbose=nothing)

    options = simulation.run_options
    verbose !== nothing && (options.verbose = verbose)

    if output === false
        options.output = false
    elseif output !== nothing
        output isa NetCDFOutput ||
            throw(ArgumentError("output must be a NetCDFOutput or false"))
        options.output = output
        options.output_dir = output.path
        interval = _schedule_in_seconds(output.schedule, _time_step(simulation))
        options.save_interval = interval === nothing ? options.save_interval : interval
        options.save_psi = _saves_psi(output.fields)
        options.save_waves = _saves_waves(output.fields)
        options.save_velocities = output.velocities || :velocities in output.fields
    end

    diagnostic_schedule = diagnostics isa EnergyDiagnosticsOutput ?
                          diagnostics.schedule : diagnostics
    if diagnostics === false
        options.diagnostics = false
    elseif diagnostics isa EnergyDiagnosticsOutput
        options.diagnostics = diagnostics
    elseif diagnostics isa AbstractSchedule
        options.diagnostics = diagnostics
    elseif diagnostics isa Integer && !(diagnostics isa Bool)
        diagnostics > 0 || throw(ArgumentError("diagnostics interval must be positive"))
        options.diagnostics = IterationInterval(diagnostics)
    elseif diagnostics !== nothing
        throw(ArgumentError(
            "diagnostics must be an EnergyDiagnosticsOutput, schedule, integer, or false"))
    end

    if diagnostic_schedule isa IterationInterval
        options.diagnostics_interval = diagnostic_schedule.interval
    elseif diagnostic_schedule isa TimeInterval
        options.diagnostics_interval =
            max(1, round(Int, diagnostic_schedule.interval / _time_step(simulation)))
    elseif diagnostic_schedule isa Integer && !(diagnostic_schedule isa Bool)
        options.diagnostics_interval = Int(diagnostic_schedule)
    end
    return simulation
end

function Simulation(model::QGYBJModel; Δt::Real=1.0, stop_time=nothing,
    stop_iteration=nothing, output=nothing, diagnostics=nothing,
    particle_output=nothing, verbose=nothing)

    model.runtime.finalized && error("cannot create a simulation from a finalized model")
    value = float(Δt)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("Δt must be finite and positive (got $Δt)"))
    options = default_run_options(typeof(value))
    simulation = Simulation(
        model,
        Clock(typeof(value)),
        ExponentialRungeKutta2(Δt=value),
        nothing,
        nothing,
        options,
        nothing,
        nothing,
        particle_output,
        Ready,
    )
    _configure_time_stepping!(simulation; stop_time, stop_iteration)
    return _configure_output!(simulation; output, diagnostics, verbose)
end

"""Set the balanced streamfunction and derive potential vorticity."""
function set_mean_flow!(owner::Union{QGYBJModel, Simulation};
    psi_func=nothing,
    method::Symbol=:function,
    pv_method::Symbol=:qg,
    amplitude::Real=1.0,
    spectral_slope::Real=-3.0,
    seed::Int=0,
    verbose::Bool=true)

    _assert_mutable(owner)
    model = _model(owner)
    runtime = model.runtime
    grid = runtime.geometry
    geometry = model.grid
    fields = model.fields
    plans = runtime.plans

    if method in (:function, :analytical)
        psi_func === nothing &&
            throw(ArgumentError("psi_func must be provided when method=:function"))
        is_root(model) && verbose && println("Setting mean flow from analytical ψ(x, y, z)")

        local_range = get_local_range_physical(runtime)
        psi_phys = allocate_fft_backward_dst(fields.psi, runtime)
        psi_array = parent(psi_phys)
        T = eltype(psi_array)
        for k_local in axes(psi_array, 1)
            k_global = local_range[1][k_local]
            z = geometry.z[k_global]
            for j_local in axes(psi_array, 3)
                j_global = local_range[3][j_local]
                y = geometry.y[j_global]
                for i_local in axes(psi_array, 2)
                    i_global = local_range[2][i_local]
                    x = geometry.x[i_global]
                    psi_array[k_local, i_local, j_local] = T(psi_func(x, y, z))
                end
            end
        end
        fft_forward!(fields.psi, psi_phys, plans)
    elseif method === :random
        is_root(model) && verbose &&
            println("Setting random mean flow: amplitude=$amplitude, slope=$spectral_slope")
        init_mpi_random_psi!(fields.psi, grid, amplitude;
                             slope=spectral_slope, seed, seed_offset=0)
    else
        throw(ArgumentError("method must be :function or :random"))
    end

    if pv_method in (:qg, :balanced)
        coefficients = runtime.coefficients
        density = coefficients.stratification
        add_balanced_component!(fields, grid, coefficients.a_ell,
            density.rho_u, density.rho_s)
    elseif pv_method in (:barotropic, :asselin)
        compute_barotropic_q_from_psi!(fields.q, fields.psi, grid)
    elseif pv_method !== :none
        throw(ArgumentError("pv_method must be :qg, :barotropic, or :none"))
    end
    compute_velocities!(model; compute_w=false)
    return owner
end

"""Set a horizontally uniform, surface-confined wave field."""
function set_surface_waves!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real,
    surface_depth::Real,
    uniform::Bool=true,
    profile::Symbol=:gaussian,
    verbose::Bool=true)

    surface_depth > 0 ||
        throw(ArgumentError("surface_depth must be positive (got $surface_depth)"))
    profile in (:gaussian, :exponential) ||
        throw(ArgumentError("profile must be :gaussian or :exponential"))

    _assert_mutable(owner)
    model = _model(owner)
    runtime = model.runtime
    grid = runtime.geometry
    geometry = model.grid
    fields = model.fields
    plans = runtime.plans
    is_root(model) && verbose &&
        println("Setting surface waves: amplitude=$amplitude, scale=$surface_depth")

    local_range = get_local_range_physical(runtime)
    B_phys = allocate_fft_backward_dst(fields.B, runtime)
    B_array = parent(B_phys)
    T = typeof(real(zero(eltype(B_array))))
    dz = geometry.dz
    for k_local in axes(B_array, 1)
        k_global = local_range[1][k_local]
        depth = max(zero(T), -geometry.z[k_global] - dz / 2)
        vertical_profile = profile === :gaussian ?
            exp(-(depth^2) / surface_depth^2) : exp(-depth / surface_depth)
        B_array[k_local, :, :] .= complex(T(amplitude) * vertical_profile)
    end
    fft_forward!(fields.B, B_phys, plans)
    _refresh_wave_diagnostics!(model)
    return owner
end

function set_exponential_surface_waves!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real, efold_depth::Real, uniform::Bool=true, verbose::Bool=true)
    return set_surface_waves!(owner; amplitude, surface_depth=efold_depth,
                              uniform, profile=:exponential, verbose)
end

_first_notnothing(values...) = begin
    for value in values
        value !== nothing && return value
    end
    return nothing
end

_field_initializer(source::AbstractArray) = FieldArray(source)
_field_initializer(source::Union{FieldArray, FieldFile}) = source

function _read_field_values(source::FieldArray)
    return Array(source.values)
end

function _read_field_values(source::FieldFile)
    isfile(source.path) ||
        throw(ArgumentError("field file does not exist: $(source.path)"))
    return NCDataset(source.path, "r") do dataset
        haskey(dataset, source.variable) || throw(ArgumentError(
            "field file $(source.path) has no variable $(source.variable)"))
        Array(dataset[source.variable][:, :, :])
    end
end

function _field_values_zxy(model::QGYBJModel,
    source::Union{FieldArray, FieldFile})

    values = _run_on_root(model) do
        raw = _read_field_values(source)
        zxy = source.layout === :zxy ? raw : permutedims(raw, (3, 1, 2))
        expected = (model.grid.size[3], model.grid.size[1], model.grid.size[2])
        size(zxy) == expected || throw(DimensionMismatch(
            "initial field has size $(size(zxy)); expected $expected in z-x-y layout"))
        ComplexF64.(zxy)
    end
    return values
end

function _assign_initial_field!(destination, model::QGYBJModel,
    source::Union{AbstractArray, FieldArray, FieldFile})

    initializer = _field_initializer(source)
    runtime = model.runtime
    root_values = _field_values_zxy(model, initializer)
    if initializer.space === :physical
        physical = scatter_from_root(
            root_values, runtime.geometry, runtime.mpi;
            pencil=runtime.plans.input_pencil)
        fft_forward!(destination, physical, runtime.plans)
    else
        spectral = scatter_from_root(
            root_values, runtime.geometry, runtime.mpi; plans=runtime.plans)
        copyto!(parent(destination), parent(spectral))
    end
    return destination
end

function _refresh_flow_diagnostics!(model::QGYBJModel, pv_method::Symbol)
    fields = model.fields
    runtime = model.runtime
    if pv_method in (:qg, :balanced)
        coefficients = runtime.coefficients
        density = coefficients.stratification
        add_balanced_component!(fields, runtime.geometry, coefficients.a_ell,
            density.rho_u, density.rho_s)
    elseif pv_method in (:barotropic, :asselin)
        compute_barotropic_q_from_psi!(fields.q, fields.psi, runtime.geometry)
    elseif pv_method !== :none
        throw(ArgumentError("pv_method must be :qg, :barotropic, or :none"))
    end
    compute_velocities!(model; compute_w=false)
    return model
end

function _refresh_normal_ybj_diagnostics!(fields::ModelFields,
    model::QGYBJModel)

    context = _operator_context(model)
    options = ETDModelOptions(model.physics, model.numerics)
    mask = context.mask
    sumB!(fields.B, context.grid;
          Lmask=mask, workspace=context.workspace)
    _diagnose_flow!(fields, context.grid, options, context.plans,
        context.a, mask;
        workspace=context.workspace,
        N2_profile=context.N2,
        rho_u=context.rho_u,
        rho_s=context.rho_s,
        compute_w=false,
        use_wave_feedback=true)

    arrays = _etdrk2_arrays(fields, nothing)
    split_B_to_real_imag!(arrays.BRk, arrays.BIk, fields.B)
    if _linear(options)
        fill!(parent(arrays.nBRk), zero(eltype(parent(arrays.nBRk))))
        fill!(parent(arrays.nBIk), zero(eltype(parent(arrays.nBIk))))
    else
        convol_waqg!(arrays.nqk, arrays.nBRk, arrays.nBIk,
            fields.u, fields.v, fields.q, arrays.BRk, arrays.BIk,
            context.grid, context.plans; Lmask=mask)
    end
    refraction_waqg!(arrays.rBRk, arrays.rBIk,
        arrays.BRk, arrays.BIk, fields.psi,
        context.grid, context.plans; Lmask=mask)
    sigma = compute_sigma(context.f, context.grid,
        arrays.nBRk, arrays.nBIk, arrays.rBRk, arrays.rBIk;
        Lmask=mask, workspace=context.workspace,
        N2_profile=context.N2)
    compute_A!(fields.A, fields.C, arrays.BRk, arrays.BIk,
        sigma, context.grid;
        Lmask=mask, workspace=context.workspace,
        N2_profile=context.N2)
    return fields
end

function _refresh_wave_diagnostics!(fields::ModelFields,
    model::QGYBJModel)

    if model.physics.formulation isa YBJ
        _refresh_normal_ybj_diagnostics!(fields, model)
    else
        context = _operator_context(model)
        invert_B_to_A!(fields, context.grid, context.a;
            rho_u=context.rho_u,
            rho_s=context.rho_s,
            workspace=context.workspace)
    end
    return fields
end

function _refresh_wave_diagnostics!(model::QGYBJModel)
    _refresh_wave_diagnostics!(model.fields, model)
    return model
end

"""Set declarative initial conditions on a model or its simulation."""
function set!(owner::Union{QGYBJModel, Simulation};
    ψ=nothing, psi=nothing, mean_flow=nothing,
    pv_method::Symbol=:qg,
    waves=nothing, B=nothing,
    particles=nothing,
    verbose::Bool=false)

    _assert_mutable(owner)
    model = _model(owner)
    flow = _first_notnothing(mean_flow, ψ, psi)
    wave = _first_notnothing(waves, B)
    if flow !== nothing
        if flow isa Function
            set_mean_flow!(owner; psi_func=flow, pv_method, verbose)
        elseif flow isa RandomStreamfunction
            set_mean_flow!(owner; method=:random, pv_method,
                amplitude=flow.amplitude, spectral_slope=flow.spectral_slope,
                seed=flow.seed, verbose)
        elseif flow isa AbstractArray || flow isa FieldArray || flow isa FieldFile
            _assign_initial_field!(model.fields.psi, model, flow)
            _refresh_flow_diagnostics!(model, pv_method)
        else
            throw(ArgumentError(
                "mean_flow/ψ must be a function, RandomStreamfunction, " *
                "array, FieldArray, or FieldFile"))
        end
    end
    if wave !== nothing
        if wave isa SurfaceWave
            set_surface_waves!(owner; amplitude=wave.amplitude,
                               surface_depth=wave.scale,
                               profile=wave.profile, verbose)
        elseif wave isa AbstractArray || wave isa FieldArray || wave isa FieldFile
            _assign_initial_field!(model.fields.B, model, wave)
            _refresh_wave_diagnostics!(model)
        else
            throw(ArgumentError(
                "waves/B must be a SurfaceWave, array, FieldArray, or FieldFile"))
        end
    end
    particles !== nothing && initialize_particles!(model, particles)
    return owner
end

"""Set a localized spectral wave packet."""
function set_wave_packet!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real, kx::Int, ky::Int, sigma_k::Real,
    z_center::Union{Real, Nothing}=nothing,
    z_width::Union{Real, Nothing}=nothing)

    _assert_mutable(owner)
    model = _model(owner)
    runtime = model.runtime
    grid = runtime.geometry
    z_c = z_center === nothing ? grid.Lz / 2 : z_center
    z_w = z_width === nothing ? grid.Lz / 4 : z_width
    0 <= z_c <= grid.Lz ||
        throw(ArgumentError("z_center must lie between 0 and Lz=$(grid.Lz)"))
    z_w > 0 || throw(ArgumentError("z_width must be positive"))

    packet = create_wave_packet(grid, kx, ky, sigma_k, amplitude;
                                z_center=z_c, z_width=z_w)
    model.fields.B .= scatter_from_root(packet, grid, runtime.mpi;
                                        plans=runtime.plans)
    _refresh_wave_diagnostics!(model)
    return owner
end

function _replace_output_field(fields::Tuple, aliases::Tuple,
    canonical::Symbol, enabled::Bool)

    filtered = Tuple(field for field in fields if !(field in aliases))
    return enabled ? (filtered..., canonical) : filtered
end

function _apply_output_overrides!(simulation::Simulation;
    output_dir=nothing, save_interval=nothing,
    save_psi=nothing, save_waves=nothing, save_velocities=nothing)

    any(value -> value !== nothing,
        (output_dir, save_interval, save_psi, save_waves, save_velocities)) ||
        return simulation

    options = simulation.run_options
    current = options.output
    current isa NetCDFOutput || throw(ArgumentError(
        "run-time output overrides require output=NetCDFOutput(...)"))

    path = output_dir === nothing ? current.path : String(output_dir)
    schedule = current.schedule
    if save_interval !== nothing
        save_interval > 0 ||
            throw(ArgumentError("save_interval must be positive"))
        schedule = TimeInterval(save_interval)
    end
    fields = current.fields
    save_psi !== nothing &&
        (fields = _replace_output_field(
            fields, (:ψ, :psi, :q, :flow), :ψ, save_psi))
    save_waves !== nothing &&
        (fields = _replace_output_field(
            fields, (:waves, :wave, :B, :A), :waves, save_waves))
    velocities = save_velocities === nothing ?
                 current.velocities : save_velocities
    specification = NetCDFOutput(; path, schedule, fields, velocities)
    return _configure_output!(simulation; output=specification)
end

"""Advance a configured simulation with the current ETD-RK2 driver."""
function run!(simulation::Simulation;
    output_dir::Union{String, Nothing}=nothing,
    Δt=nothing,
    stop_time=nothing,
    stop_iteration=nothing,
    output=nothing,
    diagnostics=nothing,
    save_interval::Union{Real, Nothing}=nothing,
    diagnostics_interval::Union{Int, Nothing}=nothing,
    verbose::Union{Bool, Nothing}=nothing,
    progress::Union{Bool, Nothing}=nothing,
    save_psi::Union{Bool, Nothing}=nothing,
    save_waves::Union{Bool, Nothing}=nothing,
    save_velocities::Union{Bool, Nothing}=nothing)

    simulation.state === Running &&
        throw(InvalidStateException("simulation is already running", :running))
    simulation.state === Finalized &&
        throw(InvalidStateException("simulation has been finalized", :finalized))
    simulation.state === Failed &&
        throw(InvalidStateException("simulation is in a failed state", :failed))
    simulation.state === Stopped &&
        throw(InvalidStateException(
            "simulation has stopped; construct a new simulation to continue", :stopped))

    model = simulation.model
    runtime = model.runtime
    runtime.finalized &&
        throw(InvalidStateException("model runtime has been finalized", :finalized))
    _configure_time_stepping!(simulation; Δt, stop_time, stop_iteration)
    _configure_output!(simulation; output, diagnostics, verbose)
    _apply_output_overrides!(simulation;
        output_dir, save_interval, save_psi, save_waves, save_velocities)

    options = simulation.run_options
    progress !== nothing && (options.verbose = progress)
    diagnostics_interval !== nothing &&
        (options.diagnostics_interval = diagnostics_interval)
    options.diagnostics_interval > 0 ||
        throw(ArgumentError("diagnostics_interval must be positive"))

    simulation.state = Running
    try
        _prepare_simulation_output!(simulation)
        _prepare_simulation_diagnostics!(simulation)
        _prepare_particle_output!(simulation)
        _maybe_write_simulation_output!(simulation; initial=true)
        _maybe_record_simulation_diagnostics!(simulation; initial=true)
        _maybe_write_particle_output!(simulation; initial=true)

        while (simulation.stop_iteration === nothing ||
               simulation.clock.iteration < simulation.stop_iteration) &&
              _before_stop_time(simulation)
            step!(model, simulation.timestepper)
            _check_termination_conditions!(simulation)
            _advect_model_particles!(model.particles, model,
                                     _time_step(simulation),
                                     simulation.clock.time)
            simulation.clock.iteration += 1
            simulation.clock.time += _time_step(simulation)
            _maybe_write_simulation_output!(simulation)
            _maybe_record_simulation_diagnostics!(simulation)
            _maybe_write_particle_output!(simulation)

            if options.verbose && runtime.mpi.is_root &&
               simulation.clock.iteration % options.diagnostics_interval == 0
                @info "Simulation progress" iteration=simulation.clock.iteration time=simulation.clock.time
            end
        end
        simulation.state = Stopped
    catch
        simulation.state = Failed
        rethrow()
    finally
        try
            _finish_simulation_output!(simulation)
        finally
            try
                _finish_simulation_diagnostics!(simulation)
            finally
                _finish_particle_output!(simulation)
            end
        end
    end
    return simulation
end

function finalize_simulation!(simulation::Simulation)
    simulation.state === Finalized && return simulation
    simulation.state === Running &&
        throw(InvalidStateException("cannot finalize a running simulation", :running))
    try
        _finish_simulation_output!(simulation)
    finally
        try
            _finish_simulation_diagnostics!(simulation)
        finally
            _finish_particle_output!(simulation)
        end
    end
    finalize_model!(simulation.model)
    simulation.state = Finalized
    return simulation
end
finalize_simulation!(model::QGYBJModel) = finalize_model!(model)

get_time(simulation::Simulation, step::Int) = step * _time_step(simulation)
get_inertial_period(model::QGYBJModel) = 2π / model.physics.coriolis.f
get_inertial_period(simulation::Simulation) = get_inertial_period(simulation.model)
inertial_period(model::QGYBJModel) = get_inertial_period(model)
inertial_period(simulation::Simulation) = get_inertial_period(simulation)
get_duration(simulation::Simulation) = simulation.stop_time === nothing ?
    simulation.stop_iteration * _time_step(simulation) : simulation.stop_time
get_duration_ip(simulation::Simulation) =
    get_duration(simulation) / get_inertial_period(simulation)

function Base.summary(io::IO, simulation::Simulation)
    model = simulation.model
    grid = model.grid
    is_root(model) || return
    println(io, "QGYBJplus Simulation")
    println(io, "="^40)
    @printf(io, "Resolution: %d × %d × %d\n", grid.size...)
    @printf(io, "Domain: %.1f km × %.1f km × %.1f m\n",
            grid.extent[1] / 1e3, grid.extent[2] / 1e3, grid.extent[3])
    @printf(io, "Coriolis: f₀ = %.2e s⁻¹\n", model.physics.coriolis.f)
    if simulation.stop_iteration === nothing
        @printf(io, "Time step: Δt = %.2f s, stop time = %.2f s\n",
                _time_step(simulation), simulation.stop_time)
    else
        @printf(io, "Time step: Δt = %.2f s, iterations = %d\n",
                _time_step(simulation), simulation.stop_iteration)
    end
    println(io, "MPI processes: $(nprocs(model))")
    println(io, "="^40)
end

function Base.show(io::IO, simulation::Simulation)
    print(io, "Simulation(model=$(simulation.model.grid.size), " *
              "Δt=$(_time_step(simulation)), stop_iteration=$(simulation.stop_iteration), " *
              "state=$(simulation.state))")
end
