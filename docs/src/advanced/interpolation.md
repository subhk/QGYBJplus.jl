# [Particle interpolation](@id interpolation)

```@meta
CurrentModule = QGYBJplus
```

Particle interpolation evaluates model-owned physical velocities at particle
positions.

## Select a method

| Value | Behavior |
|:--|:--|
| `TRILINEAR` | lowest communication and interpolation cost |
| `TRICUBIC` | smoother interpolation with a wider stencil |
| `QUINTIC` | widest stencil and highest interpolation order |
| `ADAPTIVE` | selects trilinear or tricubic interpolation from local field smoothness |

```julia
configuration = ParticleConfig{Float64}(
    x_min=first(model.grid.x_faces),
    x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces),
    y_max=last(model.grid.y_faces),
    z_level=-100.0,
    interpolation_method=TRICUBIC,
)
initialize_particles!(model, configuration)
```

Horizontal boundaries are periodic by default. The tracker converts global
positions to the local coordinate system and applies the configured vertical
boundary behavior.

## Distributed halos

Under MPI, velocity halos are exchanged before interpolation. Trilinear,
tricubic, and quintic interpolation require one, two, and three horizontal
halo cells respectively; adaptive interpolation reserves three. Wider halos
increase communication cost.

Choose a topology that leaves enough interior cells on each rank for the
selected stencil. Particle migration occurs after the trajectory update and
periodic boundary normalization.
