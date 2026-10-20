# [Particle API](@id api-particles)

```@meta
CurrentModule = QGYBJplus
```

Particles are optional model-owned data. Initialize them after model
construction:

```@docs
ParticleConfig
ParticleConfig3D
ParticleTracker
ParticleState
ParticleDistribution
initialize_particles!
advect_particles!
interpolate_velocity_at_position
```

```julia
configuration = ParticleConfig{Float64}(
    x_min=first(model.grid.x_faces),
    x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces),
    y_max=last(model.grid.y_faces),
    z_level=-100.0,
    nx_particles=16,
    ny_particles=16,
    interpolation_method=TRILINEAR,
)
initialize_particles!(model, configuration)
```

`run!(simulation)` advances installed particles after each model step. A
manual update takes the model's current state as the velocity at the end of
the particle step, so advance the model over the same interval first:

```julia
step!(model, timestepper)
advect_particles!(model, timestepper.Δt; current_time=t)
```

## Distribution helpers

```@docs
particles_in_box
particles_in_circle
particles_in_grid_3d
particles_in_layers
particles_random_3d
particles_custom
```

## Interpolation

```@docs
InterpolationMethod
interpolate_velocity_advanced
```

Available methods are `TRILINEAR`, `TRICUBIC` (default), and `QUINTIC`;
`ADAPTIVE` is deprecated and now uses `TRICUBIC`. See
[Particle interpolation](@ref interpolation). Particles move with the
horizontal velocity and keep their depth; they are advected on the model's
architecture, where the tracker keeps its copy of the velocity.

Use `ParticleConfig3D` or the distribution helpers to release particles at
several depths, in volumes, layers, random points, or explicit positions.

## Output

Particle trajectory helpers remain separate from Eulerian NetCDF snapshots:

```@docs
ParticleOutputManager
write_particle_trajectories
read_particle_trajectories
write_particle_snapshot
write_particle_trajectories_by_zlevel
create_particle_output_file
```
