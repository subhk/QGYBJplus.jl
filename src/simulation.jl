"""High-level composition-first model initialization and simulation lifecycle."""

using MPI
using Printf

mutable struct SimulationRunOptions{T}
    output_dir::String
    save_interval::Union{Nothing, T}
    diagnostics_interval::Int
    verbose::Bool
    save_psi::Bool
    save_waves::Bool
    save_velocities::Bool
    output
end

default_run_options(::Type{T}) where T = SimulationRunOptions{T}(
    "output", nothing, 10, true, true, true, false, nothing)

"""
    Simulation(model; Δt=1, stop_time=nothing, stop_iteration=nothing, ...)

Own run configuration for a `QGYBJModel`. The model and simulation are
deliberately distinct objects: model fields and runtime resources stay with
the model, while time-step and lifecycle configuration stay here.
"""
mutable struct Simulation{M, T, R}
    model::M
    Δt::T
    stop_time::Union{Nothing, T}
    stop_iteration::Union{Nothing, Int}
    run_options::R
end

_model(model::QGYBJModel) = model
_model(simulation::Simulation) = simulation.model

_runtime(model::QGYBJModel) = model.runtime
_computational_grid(model::QGYBJModel) = model.runtime.computational_grid
_parameters(model::QGYBJModel) = model.runtime.parameters

is_root(simulation::Simulation) = is_root(simulation.model)
nprocs(simulation::Simulation) = nprocs(simulation.model)

function _configure_time_stepping!(simulation::Simulation;
    Δt=nothing, stop_time=nothing, stop_iteration=nothing)

    if Δt !== nothing
        Δt > 0 || throw(ArgumentError("Δt must be positive (got $Δt)"))
        simulation.Δt = typeof(simulation.Δt)(Δt)
    end

    if stop_iteration !== nothing
        stop_iteration > 0 || throw(ArgumentError("stop_iteration must be positive"))
        simulation.stop_iteration = Int(stop_iteration)
        simulation.stop_time = nothing
    elseif stop_time !== nothing
        stop_time > 0 || throw(ArgumentError("stop_time must be positive"))
        simulation.stop_time = typeof(simulation.Δt)(stop_time)
        simulation.stop_iteration = max(1, ceil(Int, stop_time / simulation.Δt))
    elseif simulation.stop_iteration === nothing && simulation.stop_time === nothing
        simulation.stop_iteration = 1000
    end

    parameters = _parameters(simulation.model)
    parameters.dt = typeof(parameters.dt)(simulation.Δt)
    parameters.nt = simulation.stop_iteration
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
        interval = _schedule_in_seconds(output.schedule, simulation.Δt)
        options.save_interval = interval === nothing ? options.save_interval : interval
        options.save_psi = _saves_psi(output.fields)
        options.save_waves = _saves_waves(output.fields)
        options.save_velocities = output.velocities || :velocities in output.fields
    end

    if diagnostics isa IterationInterval
        options.diagnostics_interval = diagnostics.interval
    elseif diagnostics isa TimeInterval
        options.diagnostics_interval =
            max(1, round(Int, diagnostics.interval / simulation.Δt))
    elseif diagnostics isa Integer
        diagnostics > 0 || throw(ArgumentError("diagnostics interval must be positive"))
        options.diagnostics_interval = Int(diagnostics)
    elseif diagnostics !== nothing
        throw(ArgumentError("diagnostics must be an IterationInterval, TimeInterval, or integer"))
    end
    return simulation
end

function Simulation(model::QGYBJModel; Δt::Real=1.0, stop_time=nothing,
    stop_iteration=nothing, output=nothing, diagnostics=nothing,
    verbose=nothing)

    model.runtime.finalized && error("cannot create a simulation from a finalized model")
    value = float(Δt)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("Δt must be finite and positive (got $Δt)"))
    options = default_run_options(typeof(value))
    simulation = Simulation(model, value, nothing, nothing, options)
    _configure_time_stepping!(simulation; stop_time, stop_iteration)
    return _configure_output!(simulation; output, diagnostics, verbose)
end

"""
    initialize_simulation(; kwargs...)

Temporary keyword boundary for the pre-composition setup vocabulary. It now
returns a distinct `Simulation` containing a real `QGYBJModel`.
"""
function initialize_simulation(;
    nx::Int, ny::Int, nz::Int,
    Lx::Real, Ly::Real, Lz::Real,
    centered::Bool=false,
    x0::Union{Real, Nothing}=nothing,
    y0::Union{Real, Nothing}=nothing,
    f₀::Real=1e-4,
    N²::Real=1e-5,
    stratification_profile=nothing,
    dt::Real=1.0,
    nt::Int=1000,
    ybj_plus::Bool=true,
    fixed_flow::Bool=false,
    no_feedback::Union{Bool, Nothing}=nothing,
    no_wave_feedback::Bool=false,
    νₕ₁::Real=0.01,
    νₕ₂::Real=10.0,
    ilap1::Int=2,
    ilap2::Int=6,
    νₕ₁ʷ::Real=0.0,
    νₕ₂ʷ::Real=10.0,
    ilap1w::Int=2,
    ilap2w::Int=6,
    topology=nothing,
    parallel_io::Bool=false,
    verbose::Bool=true)

    if centered && (x0 !== nothing || y0 !== nothing)
        throw(ArgumentError("centered=true cannot be combined with x0 or y0"))
    end
    x_bounds = x0 === nothing ? nothing : (x0, x0 + Lx)
    y_bounds = y0 === nothing ? nothing : (y0, y0 + Ly)
    grid = RectilinearGrid(size=(nx, ny, nz), extent=(Lx, Ly, Lz),
                           x=x_bounds, y=y_bounds, centered=centered)

    feedback = if no_feedback === true ||
                  (no_feedback === nothing && no_wave_feedback)
        NoFeedback()
    elseif no_wave_feedback
        NoWaveFeedback()
    else
        WaveMeanFeedback()
    end
    stratification = stratification_profile === nothing ?
                     ConstantStratification(N²=N²) : stratification_profile
    closure = HorizontalHyperdiffusivity(
        flow=νₕ₁, flow2=νₕ₂,
        flow_laplacian_order=ilap1, flow_laplacian_order2=ilap2,
        waves=νₕ₁ʷ, waves2=νₕ₂ʷ,
        wave_laplacian_order=ilap1w, wave_laplacian_order2=ilap2w)

    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=f₀),
        stratification=stratification,
        closure=closure,
        flow=fixed_flow ? FixedFlow() : EvolvingFlow(),
        feedback=feedback,
        formulation=ybj_plus ? YBJPlus() : YBJ(),
        topology=topology,
        parallel_io=parallel_io,
        verbose=verbose)
    return Simulation(model; Δt=dt, stop_iteration=nt, verbose=verbose)
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

    model = _model(owner)
    runtime = model.runtime
    runtime.finalized && error("cannot modify a finalized model")
    grid = runtime.computational_grid
    fields = model.fields
    plans = runtime.plans

    if method in (:function, :analytical)
        psi_func === nothing &&
            throw(ArgumentError("psi_func must be provided when method=:function"))
        is_root(model) && verbose && println("Setting mean flow from analytical ψ(x, y, z)")

        local_range = get_local_range_physical(plans)
        psi_phys = allocate_fft_backward_dst(fields.psi, plans)
        psi_array = parent(psi_phys)
        T = eltype(psi_array)
        for k_local in axes(psi_array, 1)
            k_global = local_range[1][k_local]
            z = grid.z[k_global]
            for j_local in axes(psi_array, 3)
                j_global = local_range[3][j_local]
                y = grid.y0 + (j_global - 1) * grid.dy
                for i_local in axes(psi_array, 2)
                    i_global = local_range[2][i_local]
                    x = grid.x0 + (i_global - 1) * grid.dx
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
        add_balanced_component!(fields, grid, runtime.parameters, plans;
                                N2_profile=runtime.coefficients.N²)
    elseif pv_method in (:barotropic, :asselin)
        compute_barotropic_q_from_psi!(fields.q, fields.psi, grid)
    elseif pv_method !== :none
        throw(ArgumentError("pv_method must be :qg, :barotropic, or :none"))
    end
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

    model = _model(owner)
    runtime = model.runtime
    runtime.finalized && error("cannot modify a finalized model")
    grid = runtime.computational_grid
    fields = model.fields
    plans = runtime.plans
    is_root(model) && verbose &&
        println("Setting surface waves: amplitude=$amplitude, scale=$surface_depth")

    local_range = get_local_range_physical(plans)
    B_phys = allocate_fft_backward_dst(fields.B, plans)
    B_array = parent(B_phys)
    T = typeof(real(zero(eltype(B_array))))
    dz = grid.Lz / grid.nz
    for k_local in axes(B_array, 1)
        k_global = local_range[1][k_local]
        depth = max(zero(T), -grid.z[k_global] - dz / 2)
        vertical_profile = profile === :gaussian ?
            exp(-(depth^2) / surface_depth^2) : exp(-depth / surface_depth)
        B_array[k_local, :, :] .= complex(T(amplitude) * vertical_profile)
    end
    fft_forward!(fields.B, B_phys, plans)
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

"""Set declarative initial conditions on a model or its simulation."""
function set!(owner::Union{QGYBJModel, Simulation};
    ψ=nothing, psi=nothing, mean_flow=nothing,
    pv_method::Symbol=:qg,
    waves=nothing, B=nothing,
    verbose::Bool=false)

    flow = _first_notnothing(mean_flow, ψ, psi)
    wave = _first_notnothing(waves, B)
    if flow !== nothing
        flow isa Function ||
            throw(ArgumentError("mean_flow/ψ must be a function of (x, y, z)"))
        set_mean_flow!(owner; psi_func=flow, pv_method, verbose)
    end
    if wave !== nothing
        wave isa SurfaceWave || throw(ArgumentError("waves/B must be a SurfaceWave"))
        set_surface_waves!(owner; amplitude=wave.amplitude,
                           surface_depth=wave.scale,
                           profile=wave.profile, verbose)
    end
    return owner
end

"""Set a localized spectral wave packet."""
function set_wave_packet!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real, kx::Int, ky::Int, sigma_k::Real,
    z_center::Union{Real, Nothing}=nothing,
    z_width::Union{Real, Nothing}=nothing)

    model = _model(owner)
    runtime = model.runtime
    grid = runtime.computational_grid
    z_c = z_center === nothing ? grid.Lz / 2 : z_center
    z_w = z_width === nothing ? grid.Lz / 4 : z_width
    0 <= z_c <= grid.Lz ||
        throw(ArgumentError("z_center must lie between 0 and Lz=$(grid.Lz)"))
    z_w > 0 || throw(ArgumentError("z_width must be positive"))

    packet = create_wave_packet(grid, kx, ky, sigma_k, amplitude;
                                z_center=z_c, z_width=z_w)
    model.fields.B .= scatter_from_root(packet, grid, runtime.mpi;
                                        plans=runtime.plans)
    return owner
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

    model = simulation.model
    runtime = model.runtime
    runtime.finalized && error("cannot run a finalized model")
    _configure_time_stepping!(simulation; Δt, stop_time, stop_iteration)
    _configure_output!(simulation; output, diagnostics, verbose)

    options = simulation.run_options
    progress !== nothing && (options.verbose = progress)
    output_dir !== nothing && (options.output_dir = output_dir)
    save_interval !== nothing &&
        (options.save_interval = typeof(simulation.Δt)(save_interval))
    diagnostics_interval !== nothing &&
        (options.diagnostics_interval = diagnostics_interval)
    save_psi !== nothing && (options.save_psi = save_psi)
    save_waves !== nothing && (options.save_waves = save_waves)
    save_velocities !== nothing && (options.save_velocities = save_velocities)
    options.diagnostics_interval > 0 ||
        throw(ArgumentError("diagnostics_interval must be positive"))

    parameters = runtime.parameters
    write_output = options.output !== false
    if write_output && runtime.mpi.is_root
        mkpath(options.output_dir)
    end
    MPI.Barrier(runtime.mpi.comm)

    interval = options.save_interval === nothing ?
               inertial_period(model) : options.save_interval
    interval > 0 || throw(ArgumentError("save_interval must be positive"))
    output_config = write_output ? OutputConfig(
        output_dir=options.output_dir,
        state_file_pattern="state%04d.nc",
        psi_interval=interval,
        wave_interval=interval,
        diagnostics_interval=interval,
        save_psi=options.save_psi,
        save_waves=options.save_waves,
        save_velocities=options.save_velocities,
        save_vorticity=false,
        save_diagnostics=false) : nothing

    run_simulation!(model.fields, runtime.computational_grid, parameters,
                    runtime.plans;
        output_config,
        mpi_config=runtime.mpi,
        workspace=runtime.workspace,
        N2_profile=runtime.coefficients.N²,
        print_progress=runtime.mpi.is_root && options.verbose,
        diagnostics_interval=options.diagnostics_interval)
    return simulation
end

finalize_simulation!(simulation::Simulation) =
    (finalize_model!(simulation.model); simulation)
finalize_simulation!(model::QGYBJModel) = finalize_model!(model)

get_time(simulation::Simulation, step::Int) = step * simulation.Δt
get_inertial_period(model::QGYBJModel) = 2π / model.physics.coriolis.f
get_inertial_period(simulation::Simulation) = get_inertial_period(simulation.model)
inertial_period(model::QGYBJModel) = get_inertial_period(model)
inertial_period(simulation::Simulation) = get_inertial_period(simulation)
get_duration(simulation::Simulation) = simulation.stop_iteration * simulation.Δt
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
    @printf(io, "Time step: Δt = %.2f s, iterations = %d\n",
            simulation.Δt, simulation.stop_iteration)
    println(io, "MPI processes: $(nprocs(model))")
    println(io, "="^40)
end

function Base.show(io::IO, simulation::Simulation)
    print(io, "Simulation(model=$(simulation.model.grid.size), " *
              "Δt=$(simulation.Δt), stop_iteration=$(simulation.stop_iteration))")
end
