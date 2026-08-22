# [Particle API](@id api-particles)

```@meta
CurrentModule = QGYBJplus
```

Particles are optional model-owned data. Initialize them after model
construction:

```@docs
ParticleConfig
ParticleConfig3D
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

`run!(simulation)` advances installed particles after each model step. For a
manual particle-only update:

```julia
advect_particles!(model, 10.0)
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

Available methods are `TRILINEAR`, `TRICUBIC`, `ADAPTIVE`, and `QUINTIC`.
Distributed trackers obtain local domains, halo plans, and migration metadata
from `model.runtime`.

Use `ParticleConfig3D` or the distribution helpers for volumes, layers,
random points, and explicit positions.

## Output

Particle trajectory helpers remain separate from Eulerian NetCDF snapshots:

```@docs
ParticleOutputManager
write_particle_trajectories
read_particle_trajectories
write_particle_snapshot
```
