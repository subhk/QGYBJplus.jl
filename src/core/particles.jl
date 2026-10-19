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

function UnifiedParticleAdvection.initialize_particles!(
    model::QGYBJModel, configuration::ParticleConfig)

    tracker = ParticleTracker(configuration, model)
    UnifiedParticleAdvection.initialize_particles!(tracker, configuration)
    model.particles = tracker
    return model
end

function UnifiedParticleAdvection.initialize_particles!(
    simulation::Simulation, configuration::Union{ParticleConfig, ParticleConfig3D})

    _assert_mutable(simulation)
    UnifiedParticleAdvection.initialize_particles!(simulation.model, configuration)
    return simulation
end

function UnifiedParticleAdvection.initialize_particles!(
    model::QGYBJModel, configuration::ParticleConfig3D)

    basic_configuration =
        UnifiedParticleAdvection.EnhancedParticleConfig.convert_to_basic_config(
            configuration)
    tracker = ParticleTracker(basic_configuration, model)
    UnifiedParticleAdvection.initialize_particles!(tracker, configuration)
    model.particles = tracker
    return model
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
function UnifiedParticleAdvection.advect_particles!(
    model::QGYBJModel, Δt::Real; current_time=nothing)

    tracker = model.particles
    tracker isa ParticleTracker ||
        throw(ArgumentError("initialize model particles before advection"))
    T = typeof(tracker.particles.time)
    value = T(Δt)
    isfinite(value) && value > zero(T) ||
        throw(ArgumentError("particle time step must be finite and positive"))

    runtime = model.runtime
    runtime.finalized && error("cannot advect particles on a finalized model")
    UnifiedParticleAdvection.advect_particles!(
        tracker, model.fields, runtime.geometry, value, current_time;
        _particle_velocity_options(model)...)
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
    model::QGYBJModel, Δt, time)

    tracker === model.particles ||
        error("model particle ownership is inconsistent")
    advect_particles!(model, Δt; current_time=time)
    return tracker
end

"""
Keep the velocity of the model's current state, at simulation time `time`, as
the start of the next particle step, and store each particle's velocity at
that time (unless the particles are released later, and so stay at rest).
"""
function _record_particle_velocity!(tracker::ParticleTracker,
    model::QGYBJModel, time)

    tracker === model.particles ||
        error("model particle ownership is inconsistent")
    T = typeof(tracker.particles.time)
    T(time) < tracker.config.particle_advec_time && return tracker
    UnifiedParticleAdvection.update_velocity_fields!(
        tracker, model.fields, model.runtime.geometry; time=T(time),
        _particle_velocity_options(model)...)
    UnifiedParticleAdvection._store_particle_velocities!(tracker, T(time))
    return tracker
end


function _prepare_particle_output!(simulation::Simulation)
    manager = simulation.particle_output_manager
    (manager === nothing || manager === false) && return simulation
    manager isa ParticleOutputManager || throw(ArgumentError(
        "particle_output must be a ParticleOutputManager or false"))
    tracker = simulation.model.particles
    tracker isa ParticleTracker || throw(ArgumentError(
        "particle output requires particles initialized on the model"))
    if !manager.initialized
        setup_particle_output!(manager, tracker)
        T = typeof(manager.last_save_time)
        manager.last_save_iter = simulation.clock.iteration
        manager.last_save_time = T(simulation.clock.time)
        manager.next_save_time = T(simulation.clock.time) +
                                 manager.save_interval_time
    end
    return simulation
end

function _maybe_write_particle_output!(simulation::Simulation;
    initial::Bool=false)

    manager = simulation.particle_output_manager
    (manager === nothing || manager === false || manager.closed) &&
        return simulation
    tracker = simulation.model.particles
    T = typeof(manager.last_save_time)
    time = T(simulation.clock.time)
    due = initial ? manager.save_count == 0 :
          should_save_particles(manager, simulation.clock.iteration, time)
    due || return simulation
    save_particle_positions!(
        manager, tracker, simulation.clock.iteration, time)
    return simulation
end

function _finish_particle_output!(simulation::Simulation)
    manager = simulation.particle_output_manager
    (manager === nothing || manager === false || manager.closed) &&
        return simulation
    tracker = simulation.model.particles
    try
        if simulation.state !== Failed &&
           manager.last_save_iter != simulation.clock.iteration
            T = typeof(manager.last_save_time)
            save_particle_positions!(
                manager, tracker, simulation.clock.iteration,
                T(simulation.clock.time))
        end
        finalize_particle_output!(manager, tracker)
    catch
        simulation.state = Failed
        rethrow()
    end
    return simulation
end
