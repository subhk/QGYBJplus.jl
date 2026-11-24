"""Build a particle tracker from model-owned geometry and runtime resources."""
function UnifiedParticleAdvection.ParticleTracker(
    configuration::ParticleConfig{T}, model::QGYBJModel) where T

    runtime = model.runtime
    runtime.finalized && error("cannot attach particles to a finalized model")
    return UnifiedParticleAdvection.ParticleTracker(
        configuration,
        runtime.geometry;
        plans=runtime.plans,
        model=model,
    )
end

"""
    initialize_particles!(model, config)
    initialize_particles!(simulation, config)

Install particles on `model` (or on the simulation's model), placed by `config`:
a [`ParticleConfig`](@ref) or [`ParticleConfig3D`](@ref), usually from a helper
such as [`particles_in_box`](@ref). They replace any particles the model had,
and are advanced by `run!` with the model's own coefficients; read them from
`model.particles`. Particles installed through a simulation start at its
clock's time, which is the restart time after `restore!`.
"""
function UnifiedParticleAdvection.initialize_particles!(
    model::QGYBJModel, configuration::Union{ParticleConfig, ParticleConfig3D})

    _assert_mutable(model)
    model.particles = _new_particle_tracker(model, configuration)
    return model
end

function UnifiedParticleAdvection.initialize_particles!(
    simulation::Simulation, configuration::Union{ParticleConfig, ParticleConfig3D})

    _assert_mutable(simulation)
    UnifiedParticleAdvection.initialize_particles!(simulation.model, configuration)
    _align_particle_time!(simulation.model.particles, simulation.clock.time,
                          _time_step(simulation))
    return simulation
end

"""A tracker for `model` holding the particles of `configuration`, not yet installed."""
function _new_particle_tracker(model::QGYBJModel, configuration::ParticleConfig)
    tracker = ParticleTracker(configuration, model)
    UnifiedParticleAdvection.initialize_particles!(tracker, configuration)
    return tracker
end

function _new_particle_tracker(model::QGYBJModel, configuration::ParticleConfig3D)
    basic_configuration =
        UnifiedParticleAdvection.EnhancedParticleConfig.convert_to_basic_config(
            configuration)
    tracker = ParticleTracker(basic_configuration, model)
    UnifiedParticleAdvection.initialize_particles!(tracker, configuration)
    return tracker
end

"""
    advect_particles!(model, Δt; current_time=nothing)

Advance the particles installed on `model` from `current_time` (default: the
particles' own time) to `current_time + Δt`. The model's state is taken as the
velocity at the end of that interval, so call this after advancing the model
over the same interval, as `run!` does. The velocity at the start is the one
kept from the previous call that ended at `current_time`; integration stages
interpolate linearly in time between the two. Without it (on the first call,
or after a gap) the end-of-interval velocity is used for the whole step.
"""
UnifiedParticleAdvection.advect_particles!(
    model::QGYBJModel, Δt::Real; current_time=nothing) =
    _advect_particles!(model, Δt, current_time)

"""
`advect_particles!(model, Δt; current_time)`. `as_stepped`, which only `run!`
passes, says that the model's fields are as the step just taken left them, so
that their velocity is not diagnosed again.
"""
function _advect_particles!(model::QGYBJModel, Δt::Real, current_time;
                            as_stepped::Bool=false)
    tracker = model.particles
    tracker isa ParticleTracker ||
        throw(ArgumentError("initialize model particles before advection"))
    # Particle times are Float64 whatever the particles' precision.
    value = _time_value(Δt)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("particle time step must be finite and positive"))

    runtime = model.runtime
    runtime.finalized && error("cannot advect particles on a finalized model")
    UnifiedParticleAdvection.advect_particles!(
        tracker, model.fields, runtime.geometry, value, current_time;
        as_stepped, _particle_velocity_options(model)...)
    return model
end

"""Coefficients and buffers the particle velocity diagnostic takes from `model`."""
function _particle_velocity_options(model::QGYBJModel)
    runtime = model.runtime
    return (; f=model.physics.coriolis.f,
            N2=first(runtime.coefficients.N²),
            N2_profile=runtime.device_coefficients.N²,
            N2_face_profile=runtime.device_coefficients.N²_face,
            formulation=model.physics.formulation,
            workspace=runtime.workspace)
end

function _advect_model_particles!(tracker::ParticleTracker,
    model::QGYBJModel, Δt, time; as_stepped::Bool=false)

    tracker === model.particles ||
        error("model particle ownership is inconsistent")
    _advect_particles!(model, Δt, time; as_stepped)
    return tracker
end

"""
Keep the velocity of the model's current state, at simulation time `time`, as
the start of the next particle step, and store each particle's velocity at
that time (unless the particles are released later, and so stay at rest).
Diagnose it from the same projected state as the time step, in scratch
storage so the model's fields remain available to the integrator.
"""
function _record_particle_velocity!(tracker::ParticleTracker,
    model::QGYBJModel, time)

    tracker === model.particles ||
        error("model particle ownership is inconsistent")
    time = _time_value(time)
    UnifiedParticleAdvection._released(tracker, time) || return tracker

    context = _operator_context(model)
    (; fields, grid, plans, workspace, mask, a, N2, N2_face) = context
    options = ETDModelOptions(model.physics, model.numerics)
    instantaneous = tracker.config.wave_motion === :instantaneous
    # These fields are only needed at startup. Keep them out of the workspace,
    # whose high-water mark is retained for every step.
    B = fields.B
    if instantaneous || _wave_feedback_enabled(options)
        B = copy(fields.B)
        _project_wave_envelope!(B, grid, options, mask, workspace)
    end
    fixed_flow = _fixed_flow(options)
    psi = fixed_flow ? fields.psi : similar(fields.psi)
    A = instantaneous ? similar(fields.A) : fields.A
    C = instantaneous ? similar(fields.C) : fields.C
    state = ModelFields(fields.q, B, psi, A, C,
                        similar(fields.u), similar(fields.v), similar(fields.w))
    fixed_flow || _invert_total_q_to_psi!(state, grid, options, plans, a, mask;
        workspace, project_nonlinear_state=true)
    instantaneous && _recover_wave_amplitude!(state, grid, options, plans, a,
        mask, workspace, N2, N2_face; from_psi=true)
    UnifiedParticleAdvection.update_velocity_fields!(tracker, state, grid;
        time, _particle_velocity_options(model)...)

    UnifiedParticleAdvection._store_particle_velocities!(tracker, time)
    tracker.released = true
    return tracker
end


_align_particle_time!(tracker::ParticleTracker, time, Δt) =
    UnifiedParticleAdvection._align_particle_time!(tracker, time, Δt)

"""Error unless `particle_output` is `nothing`, `false`, or a `ParticleOutputManager`."""
function _check_particle_output_type(particle_output)
    particle_output === nothing || particle_output === false ||
        particle_output isa ParticleOutputManager || throw(ArgumentError(
            "particle_output must be a ParticleOutputManager or false " *
            "(got $(typeof(particle_output)))"))
    return nothing
end

"""
Check, changing no file, that the run can write its particle output: the
model has particles, the manager has not been closed by an earlier run, and
no file it would write is in the way unless `overwrite=true`.
"""
function _check_particle_output(simulation::Simulation)
    manager = simulation.particle_output_manager
    _check_particle_output_type(manager)
    (manager === nothing || manager === false) && return nothing
    manager.closed && throw(ArgumentError(
        "this ParticleOutputManager was closed at the end of an earlier run; " *
        "give the new simulation a new ParticleOutputManager"))
    simulation.model.particles isa ParticleTracker || throw(ArgumentError(
        "particle output requires particles initialized on the model"))
    manager.initialized || ParticleIO._check_existing_particle_files(manager)
    return nothing
end

function _prepare_particle_output!(simulation::Simulation)
    manager = simulation.particle_output_manager
    (manager === nothing || manager === false) && return simulation
    manager.time_step = _time_step(simulation)
    if !manager.initialized
        setup_particle_output!(manager, simulation.model.particles)
        time = simulation.clock.time
        manager.last_save_iter = simulation.clock.iteration
        manager.last_save_time = time
        manager.save_interval_time > 0 && (manager.next_save_time =
            _next_deadline(time, manager.save_interval_time, _time_step(simulation)))
    end
    return simulation
end

function _maybe_write_particle_output!(simulation::Simulation;
    initial::Bool=false)

    manager = simulation.particle_output_manager
    (manager === nothing || manager === false || manager.closed) &&
        return simulation
    clock = simulation.clock
    due = initial ? manager.save_count == 0 :
          should_save_particles(manager, clock.iteration, clock.time)
    due || return simulation
    save_particle_positions!(
        manager, simulation.model.particles, clock.iteration, clock.time)
    return simulation
end

"""
Write the final particle state and close the particle output the run opened.
Output that was never opened, because the run failed to start, is left
untouched, so the manager can still be given to another simulation.
"""
function _finish_particle_output!(simulation::Simulation)
    manager = simulation.particle_output_manager
    (manager === nothing || manager === false || manager.closed ||
     !manager.initialized) && return simulation
    tracker = simulation.model.particles
    try
        if simulation.state !== Failed && tracker isa ParticleTracker &&
           manager.last_save_iter != simulation.clock.iteration
            save_particle_positions!(
                manager, tracker, simulation.clock.iteration,
                simulation.clock.time)
        end
        finalize_particle_output!(manager, tracker)
    catch
        simulation.state = Failed
        rethrow()
    end
    return simulation
end
