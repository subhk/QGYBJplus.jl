"""High-level composition-first model initialization and simulation lifecycle."""

using Printf

_model(model::QGYBJModel) = model
_model(simulation::Simulation) = simulation.model

_runtime(model::QGYBJModel) = model.runtime

architecture(simulation::Simulation) = architecture(simulation.model)
_time_step(simulation::Simulation) = simulation.timestepper.Δt
_advect_model_particles!(::Nothing, model::QGYBJModel, Δt, time) = nothing
_record_particle_velocity!(::Nothing, model::QGYBJModel, time) = nothing

_maybe_integer(value) = isinteger(value) ? Int(value) : value

function _prettytime_units(seconds)
    seconds < 1e-9 && return seconds, "seconds"
    seconds < 1e-6 && return seconds * 1e9, "ns"
    seconds < 1e-3 && return seconds * 1e6, "μs"
    seconds < 1 && return seconds * 1e3, "ms"
    seconds < 60 && return seconds, seconds == 1 ? "second" : "seconds"
    if seconds < 3600
        value = _maybe_integer(seconds / 60)
        return value, value == 1 ? "minute" : "minutes"
    elseif seconds < 86400
        value = _maybe_integer(seconds / 3600)
        return value, value == 1 ? "hour" : "hours"
    else
        value = _maybe_integer(seconds / 86400)
        return value, value == 1 ? "day" : "days"
    end
end

function _prettytime(seconds::Real)
    iszero(seconds) && return "0 seconds"
    seconds < 1e-9 && return @sprintf("%.3e seconds", seconds)

    value, units = _prettytime_units(_maybe_integer(seconds))
    return isinteger(value) ?
           @sprintf("%d %s", value, units) :
           @sprintf("%.3f %s", value, units)
end

function _before_stop_time(simulation::Simulation)
    stop_time = simulation.stop_time
    stop_time === nothing && return true
    time = simulation.clock.time
    tolerance = 8eps(max(
        abs(time), abs(stop_time), abs(_time_step(simulation)), one(time)))
    return time < stop_time - tolerance
end

_nonfinite(value) = !isfinite(value)

"""Sum of the absolute real and imaginary parts; non-finite when an entry is."""
_l1_norm(array) = sum(value -> abs(real(value)) + abs(imag(value)), array)

"""Error when q, B, or ψ holds a NaN or an Inf."""
function _check_finite_state(fields::ModelFields)
    bad = any(_nonfinite, fields.q) ||
          any(_nonfinite, fields.B) ||
          any(_nonfinite, fields.psi)
    bad && error("non-finite value detected in the model state")
    return nothing
end

function _check_termination_conditions!(simulation::Simulation)
    model = simulation.model
    runtime = model.runtime
    fields = model.fields
    # This runs every step, so a healthy state costs one reduction per field.
    # A NaN or Inf makes its field's sum non-finite, and since the inverse
    # transform is ψ = Σ ψ̂ e^{ik·x} / (nx ny), Σ|ψ̂| / (nx ny) bounds max |ψ|.
    # The exact checks, with a transform of ψ, run only when these allow a
    # failure.
    nx, ny, _ = model.grid.size
    sums = (_l1_norm(fields.q), _l1_norm(fields.B), _l1_norm(fields.psi))
    all(isfinite, sums) || _check_finite_state(fields)
    last(sums) / (nx * ny) <= 1e10 && return simulation

    # Borrow from the runtime scratch pool rather than writing into the FFT
    # plan's own work arrays: this check must not allocate a grid-sized
    # physical field.
    maximum_psi = with_scratch(runtime.workspace) do
        psi_physical = scratch_physical(runtime.workspace, fields.psi, runtime.plans)
        fft_backward!(psi_physical, fields.psi, runtime)
        maximum(abs, psi_physical)
    end
    maximum_psi <= 1e10 || error(
        "solution appears to be blowing up (max |psi| = $maximum_psi)")
    return simulation
end

"""
Spectral wave velocity `LA` of `fields` stored in `LA`: `B` itself for the
original YBJ formulation, `B + kₕ²A/4` otherwise (`B = L⁺A = LA - kₕ²A/4`).
"""
function _spectral_wave_velocity!(LA, fields::ModelFields, model::QGYBJModel)
    formulation = model.physics.formulation
    if formulation isa YBJ
        copyto!(LA, fields.B)
    else
        geometry = model.runtime.geometry
        # The weight in the fields' precision keeps single-precision kernels single.
        weight = real(eltype(LA))(Operators._wave_velocity_weight(formulation))
        launch!(Operators._wave_velocity_kernel!, LA, size(LA),
                LA, fields.B, fields.A, geometry.kx, geometry.ky, weight)
    end
    return LA
end

function _progress_maxima(model::QGYBJModel)
    runtime = model.runtime

    # Recover the physical wave velocity LA from pooled scratch.
    return with_scratch(runtime.workspace) do
        _progress_maxima(model, runtime)
    end
end

function _progress_maxima(model::QGYBJModel, runtime)
    fields = model.fields
    LA = scratch_like(runtime.workspace, fields.B)
    _spectral_wave_velocity!(LA, fields, model)
    fft_backward!(LA, LA, runtime)
    return (
        wave_speed=maximum(abs, LA),
        flow_speed=mapreduce(hypot, max, fields.u, fields.v),
    )
end

function _print_detailed_progress(
    simulation::Simulation, run_wall_start::UInt64)

    maxima = _progress_maxima(simulation.model)
    wall_time = 1e-9 * (time_ns() - run_wall_start)
    @printf(stdout,
        "Iteration: %04d, time: %s, Δt: %s, max(|LA|) = %.3e m s⁻¹, max(|uₕ|) = %.3e m s⁻¹, wall time: %s\n",
        simulation.clock.iteration,
        _prettytime(simulation.clock.time),
        _prettytime(_time_step(simulation)),
        maxima.wave_speed,
        maxima.flow_speed,
        _prettytime(wall_time),
    )
    flush(stdout)
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

    if stop_iteration !== nothing && stop_time !== nothing
        throw(ArgumentError(
            "pass stop_time or stop_iteration, not both " *
            "(got stop_time=$stop_time, stop_iteration=$stop_iteration)"))
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
        verbose && println("Setting mean flow from analytical ψ(x, y, z)")

        # User functions are evaluated on the host, then copied to the
        # model's architecture and transformed there.
        nx, ny, nz = geometry.size
        T = eltype(fields.psi)
        psi_host = Array{T}(undef, nz, nx, ny)
        for j in 1:ny, i in 1:nx, k in 1:nz
            psi_host[k, i, j] = T(psi_func(geometry.x[i], geometry.y[j], geometry.z[k]))
        end
        copyto!(fields.psi, psi_host)
        fft_forward!(fields.psi, fields.psi, plans)
    elseif method === :random
        verbose &&
            println("Setting random mean flow: amplitude=$amplitude, slope=$spectral_slope")
        init_random_psi!(fields.psi, grid, amplitude;
                         slope=spectral_slope, seed, seed_offset=0)
    else
        throw(ArgumentError("method must be :function or :random"))
    end

    _rebuild_total_q_from_psi!(fields, model, pv_method)
    compute_velocities!(model; compute_w=false)
    return owner
end

"""
Set a horizontally uniform, surface-confined wave field. `surface_depth` is
measured downward from the top of the domain, its rigid lid.
"""
function set_surface_waves!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real,
    surface_depth::Real,
    uniform::Bool=true,
    profile::Symbol=:gaussian,
    pv_method::Symbol=:qg,
    verbose::Bool=true)

    surface_depth > 0 ||
        throw(ArgumentError("surface_depth must be positive (got $surface_depth)"))
    profile in (:gaussian, :exponential) ||
        throw(ArgumentError("profile must be :gaussian or :exponential"))

    _assert_mutable(owner)
    model = _model(owner)
    runtime = model.runtime
    geometry = model.grid
    fields = model.fields
    plans = runtime.plans
    verbose &&
        println("Setting surface waves: amplitude=$amplitude, scale=$surface_depth")

    nx, ny, nz = geometry.size
    B_host = Array{eltype(fields.B)}(undef, nz, nx, ny)
    T = real(eltype(B_host))
    dz = geometry.dz
    # Depth below the lid, which need not be at z = 0.
    top = last(geometry.z_faces)
    for k in 1:nz
        depth = max(zero(T), top - geometry.z[k] - dz / 2)
        vertical_profile = profile === :gaussian ?
            exp(-(depth^2) / surface_depth^2) : exp(-depth / surface_depth)
        B_host[k, :, :] .= complex(T(amplitude) * vertical_profile)
    end
    copyto!(fields.B, B_host)
    fft_forward!(fields.B, fields.B, plans)
    _refresh_wave_diagnostics!(model;
        rebuild_total_q=true, pv_method, project=true)
    return owner
end

function set_exponential_surface_waves!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real, efold_depth::Real, uniform::Bool=true,
    pv_method::Symbol=:qg, verbose::Bool=true)
    return set_surface_waves!(owner; amplitude, surface_depth=efold_depth,
                              uniform, profile=:exponential, pv_method, verbose)
end

_first_notnothing(values...) = begin
    for value in values
        value !== nothing && return value
    end
    return nothing
end

_field_initializer(source::AbstractArray) = FieldArray(source)
_field_initializer(source::Union{FieldArray, FieldFile}) = source

_field_description(source::FieldArray) = "the FieldArray"
_field_description(source::FieldFile) =
    "$(repr(source.variable)) in $(_source_description(source.path, source.group))"

"""
The values of `source` on the model's grid in z-x-y layout, interpolated from
the coordinates of its points when they are given and differ from the grid.
"""
function _field_values_zxy(model::QGYBJModel,
    source::Union{FieldArray, FieldFile})

    raw, coordinates = _read_field(source)
    zxy = source.layout === :zxy ? raw : permutedims(raw, (3, 1, 2))
    grid = model.grid
    if coordinates !== nothing && !_on_model_grid(coordinates, grid)
        source.space === :physical || throw(ArgumentError(
            "$(_field_description(source)) is spectral and not on the model's " *
            "grid; only physical fields can be interpolated"))
        return _interpolate_to_grid(zxy, coordinates, grid, _field_description(source))
    end
    expected = (grid.size[3], grid.size[1], grid.size[2])
    size(zxy) == expected || throw(DimensionMismatch(
        "initial field has size $(size(zxy)); expected $expected in z-x-y " *
        "layout. A physical field on another grid is interpolated when the " *
        "coordinates x, y, and z of its points are given."))
    return ComplexF64.(zxy)
end

function _assign_initial_field!(destination, model::QGYBJModel,
    source::Union{AbstractArray, FieldArray, FieldFile})

    initializer = _field_initializer(source)
    values = _field_values_zxy(model, initializer)
    copyto!(destination, values)
    if initializer.space === :physical
        fft_forward!(destination, destination, model.runtime.plans)
    end
    return destination
end

function _rebuild_total_q_from_psi!(fields::ModelFields,
    model::QGYBJModel, pv_method::Symbol)

    runtime = model.runtime
    if pv_method in (:qg, :balanced)
        coefficients = runtime.device_coefficients
        add_balanced_component!(fields, runtime.geometry, coefficients.a_ell;
                                workspace=runtime.workspace)
    elseif pv_method in (:barotropic, :asselin)
        compute_barotropic_q_from_psi!(fields.q, fields.psi, runtime.geometry)
    elseif pv_method !== :none
        throw(ArgumentError("pv_method must be :qg, :barotropic, or :none"))
    end

    pv_method === :none && return fields

    options = ETDModelOptions(model.physics, model.numerics)
    if _wave_feedback_enabled(options)
        with_scratch(runtime.workspace) do
            qwave = scratch_like(runtime.workspace, fields.q)
            compute_qw_complex!(qwave, fields.B, runtime.geometry,
                runtime.plans; f=options.f, Lmask=runtime.dealias_mask,
                workspace=runtime.workspace)
            kernel_map!(+, fields.q, fields.q, qwave)
        end
    end
    return fields
end

function _refresh_flow_diagnostics!(model::QGYBJModel, pv_method::Symbol)
    _rebuild_total_q_from_psi!(model.fields, model, pv_method)
    compute_velocities!(model; compute_w=false)
    return model
end

function _refresh_normal_ybj_diagnostics!(fields::ModelFields,
    model::QGYBJModel; rebuild_total_q::Bool=false,
    pv_method::Symbol=:qg, project::Bool=false)

    context = _operator_context(model)
    options = ETDModelOptions(model.physics, model.numerics)
    mask = context.mask
    recovery_mask = _linear(options) ? nothing : mask

    project && _project_initial_envelope!(fields, context.grid, options, mask,
                                          context.workspace)

    rebuild_total_q && _wave_feedback_enabled(options) &&
        _rebuild_total_q_from_psi!(fields, model, pv_method)

    _diagnose_flow!(fields, context.grid, options, context.plans,
        context.a, mask;
        workspace=context.workspace,
        N2_profile=context.N2,
        compute_w=false,
        use_wave_feedback=true)

    workspace = context.workspace
    with_scratch(workspace) do
        nBk = scratch_like(workspace, fields.B)
        rBk = scratch_like(workspace, fields.B)
        if _linear(options)
            fill!(nBk, zero(eltype(nBk)))
        else
            convol_waqg_B!(nBk, fields.u, fields.v, fields.B,
                context.grid, context.plans; Lmask=mask, workspace)
        end
        refraction_waqg_B!(rBk, fields.B, fields.psi,
            context.grid, context.plans; Lmask=mask, workspace)
        sigma = compute_sigma(context.f, context.grid, nBk, rBk;
            Lmask=mask, workspace)
        compute_A!(fields.A, fields.C, fields.B, sigma, context.grid;
            f=context.f, Lmask=recovery_mask, workspace,
            N2_profile=context.N2_face)
    end
    return fields
end

"""
Rebuild the wave amplitude `A` (and `C = A_z`) from `B`. Only a new initial
condition (`project=true`) is restricted to the envelopes a wave amplitude can
produce; a refresh for diagnostics, output, or a restart leaves the prognostic
`B` untouched, so that it never changes the solution.
"""
function _refresh_wave_diagnostics!(fields::ModelFields,
    model::QGYBJModel; rebuild_total_q::Bool=false,
    pv_method::Symbol=:qg, project::Bool=false)

    if model.physics.formulation isa YBJ
        _refresh_normal_ybj_diagnostics!(fields, model;
            rebuild_total_q, pv_method, project)
    else
        context = _operator_context(model)
        options = ETDModelOptions(model.physics, model.numerics)
        project && _project_initial_envelope!(fields, context.grid, options,
                                              context.mask, context.workspace)
        rebuild_total_q && _wave_feedback_enabled(options) &&
            _rebuild_total_q_from_psi!(fields, model, pv_method)
        invert_B_to_A!(fields, context.grid, context.a;
            workspace=context.workspace)
    end
    return fields
end

"""
Restrict `fields.B` to what a wave amplitude can produce (see
`_project_wave_envelope!`), warning if nothing is left: a depth-uniform
envelope carries no wave velocity in either YBJ formulation.
"""
function _project_initial_envelope!(fields::ModelFields, grid, options, mask,
                                    workspace)
    before = maximum(abs, fields.B)
    _project_wave_envelope!(fields.B, grid, options, mask, workspace)
    if before > 0 && maximum(abs, fields.B) <= 8eps(before)
        @warn "The wave envelope is uniform in depth and was removed: B = L⁺A " *
              "(or LA) has zero depth mean in the horizontally uniform column " *
              "(every column for YBJ()), so a depth-uniform B carries no wave " *
              "velocity. Give the waves vertical structure." maxlog=1
    end
    return fields
end

function _refresh_wave_diagnostics!(model::QGYBJModel; kwargs...)
    _refresh_wave_diagnostics!(model.fields, model; kwargs...)
    return model
end

"""
    set!(owner; ψ, ζ, mean_flow, pv_method=:qg, waves, B, particles)

Set declarative initial conditions on a model or its simulation.

The mean flow is given by its streamfunction `ψ` (or `psi`, or
`mean_flow`, which also takes a `RandomStreamfunction`) or by its relative
vorticity `ζ` (or `zeta`), from which ψ follows by inverting `∇ₕ²ψ = ζ` at
each level. `pv_method` selects how the potential vorticity follows from ψ:
`:qg`, `:barotropic`, or `:none`. `B` (or `waves`, which also takes a
`SurfaceWave`) sets the wave envelope. `ψ` and `ζ` may be functions of
`(x, y, z)`; all three may be arrays on the model's grid,
[`FieldArray`](@ref)s, or [`FieldFile`](@ref)s.
"""
function set!(owner::Union{QGYBJModel, Simulation};
    ψ=nothing, psi=nothing, mean_flow=nothing,
    ζ=nothing, zeta=nothing,
    pv_method::Symbol=:qg,
    waves=nothing, B=nothing,
    particles=nothing,
    verbose::Bool=false)

    _assert_mutable(owner)
    model = _model(owner)
    flow = _first_notnothing(mean_flow, ψ, psi)
    vorticity = _first_notnothing(ζ, zeta)
    wave = _first_notnothing(waves, B)
    flow !== nothing && vorticity !== nothing &&
        throw(ArgumentError("give the mean flow by ψ or by ζ, not both"))
    if vorticity !== nothing
        vorticity isa Union{Function, AbstractArray, FieldArray, FieldFile} ||
            throw(ArgumentError(
                "ζ must be a function, array, FieldArray, or FieldFile"))
        _set_vorticity!(model, vorticity, pv_method)
    end
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
                               profile=wave.profile, pv_method, verbose)
        elseif wave isa AbstractArray || wave isa FieldArray || wave isa FieldFile
            _assign_initial_field!(model.fields.B, model, wave)
            _refresh_wave_diagnostics!(model;
                rebuild_total_q=true, pv_method, project=true)
        else
            throw(ArgumentError(
                "waves/B must be a SurfaceWave, array, FieldArray, or FieldFile"))
        end
    end
    particles !== nothing && initialize_particles!(model, particles)
    return owner
end

"""
Set ψ from the relative vorticity `source` by inverting `∇ₕ²ψ = ζ` level by
level, then rebuild q and the velocities as for a streamfunction.
"""
function _set_vorticity!(model::QGYBJModel, source, pv_method::Symbol)
    grid = model.grid
    initializer = if source isa Function
        nx, ny, nz = grid.size
        FieldArray([source(grid.x[i], grid.y[j], grid.z[k])
                    for k in 1:nz, i in 1:nx, j in 1:ny])
    else
        _field_initializer(source)
    end
    values = _field_values_zxy(model, initializer)
    _warn_horizontal_mean(values, initializer.space)
    psi = model.fields.psi
    copyto!(psi, values)
    initializer.space === :physical &&
        fft_forward!(psi, psi, model.runtime.plans)
    compute_psi_from_vorticity!(psi, model.runtime.geometry)
    _refresh_flow_diagnostics!(model, pv_method)
    return model
end

"""
Warn when the vorticity `values` (z-x-y layout) has a horizontal mean above 1%
of its rms at some level: the inversion drops it, since the circulation
around a doubly periodic domain vanishes.
"""
function _warn_horizontal_mean(values, space::Symbol)
    nz, nx, ny = size(values)
    points = nx * ny
    if space === :physical
        means = [abs(sum(view(values, k, :, :))) / points for k in 1:nz]
        rms = sqrt(sum(abs2, values) / length(values))
    else
        # Unnormalized FFT coefficients: the mean is the (0, 0) coefficient
        # over the number of points, and Parseval gives the rms.
        means = [abs(values[k, 1, 1]) / points for k in 1:nz]
        rms = sqrt(sum(abs2, values) / nz) / points
    end
    largest = maximum(means)
    largest > 0.01 * rms &&
        @warn "ζ has a horizontal mean of up to $(round(largest; sigdigits=3)) " *
              "(rms $(round(rms; sigdigits=3))), which is " *
              "removed: the circulation around a doubly periodic domain, and " *
              "with it the mean vorticity, vanishes."
    return nothing
end

"""Set a localized spectral wave packet."""
function set_wave_packet!(owner::Union{QGYBJModel, Simulation};
    amplitude::Real, kx::Int, ky::Int, sigma_k::Real,
    z_center::Union{Real, Nothing}=nothing,
    z_width::Union{Real, Nothing}=nothing,
    pv_method::Symbol=:qg)

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
    copyto!(model.fields.B, packet)
    _refresh_wave_diagnostics!(model;
        rebuild_total_q=true, pv_method, project=true)
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
    specification = NetCDFOutput(; path, schedule, fields, velocities,
                                 overwrite=current.overwrite)
    return _configure_output!(simulation; output=specification)
end

"""
Set up the output, energy diagnostics, and particle output of a run. Nothing
is written yet, so a failure here, such as output that already exists, leaves
the simulation ready to run again with other settings: the managers this call
created are dropped and the error is rethrown.
"""
function _prepare_run_output!(simulation::Simulation)
    output_manager = simulation.output_manager
    diagnostics_manager = simulation.diagnostics_manager
    try
        _prepare_simulation_output!(simulation)
        _prepare_simulation_diagnostics!(simulation)
        _prepare_particle_output!(simulation)
    catch
        simulation.output_manager = output_manager
        simulation.diagnostics_manager = diagnostics_manager
        rethrow()
    end
    return simulation
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

    detailed_progress = progress === true
    options = simulation.run_options
    progress !== nothing && (options.verbose = progress)
    diagnostics_interval !== nothing &&
        (options.diagnostics_interval = diagnostics_interval)
    options.diagnostics_interval > 0 ||
        throw(ArgumentError("diagnostics_interval must be positive"))

    run_wall_start = time_ns()
    _prepare_run_output!(simulation)
    simulation.state = Running
    try
        _maybe_write_simulation_output!(simulation; initial=true)
        _maybe_record_simulation_diagnostics!(simulation; initial=true)
        # The velocity at the start of the first particle step, also stored
        # with the initial particle positions.
        _record_particle_velocity!(model.particles, model, simulation.clock.time)
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

            if options.verbose &&
               simulation.clock.iteration % options.diagnostics_interval == 0
                if detailed_progress
                    _print_detailed_progress(simulation, run_wall_start)
                else
                    @info "Simulation progress" iteration=simulation.clock.iteration time=simulation.clock.time
                end
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
    # The ETD-RK2 workspace is the largest device allocation after the fields.
    simulation.timestepper.workspace = nothing
    simulation.timestepper.workspace_owner = nothing
    simulation.state = Finalized
    return simulation
end
finalize_simulation!(model::QGYBJModel) = finalize_model!(model)

get_time(simulation::Simulation, step::Int) = step * _time_step(simulation)
get_inertial_period(model::QGYBJModel) = 2π / abs(model.physics.coriolis.f)
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
    println(io, "Architecture: $(architecture(model))")
    println(io, "="^40)
end

function Base.show(io::IO, simulation::Simulation)
    print(io, "Simulation(model=$(simulation.model.grid.size), " *
              "Δt=$(_time_step(simulation)), stop_iteration=$(simulation.stop_iteration), " *
              "state=$(simulation.state))")
end
