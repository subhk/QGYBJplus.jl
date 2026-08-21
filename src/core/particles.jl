"""Build a particle tracker from model-owned geometry and runtime resources."""
function UnifiedParticleAdvection.ParticleTracker(
    configuration::ParticleConfig{T}, model::QGYBJModel) where T

    runtime = model.runtime
    runtime.finalized && error("cannot attach particles to a finalized model")
    return UnifiedParticleAdvection.ParticleTracker(
        configuration,
        runtime.geometry,
        runtime.mpi;
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
    model::QGYBJModel, configuration::ParticleConfig3D)

    basic_configuration =
        UnifiedParticleAdvection.EnhancedParticleConfig.convert_to_basic_config(
            configuration)
    tracker = ParticleTracker(basic_configuration, model)
    UnifiedParticleAdvection.initialize_particles!(tracker, configuration)
    model.particles = tracker
    return model
end

"""Advect the particle state installed on `model` using model-owned fields."""
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
        tracker,
        model.fields,
        runtime.geometry,
        value,
        current_time;
        f=model.physics.coriolis.f,
        N2=first(runtime.coefficients.N²),
        N2_profile=runtime.coefficients.N²,
    )
    return model
end

function _advect_model_particles!(tracker::ParticleTracker,
    model::QGYBJModel, Δt, time)

    tracker === model.particles ||
        error("model particle ownership is inconsistent")
    advect_particles!(model, Δt; current_time=time)
    return tracker
end
