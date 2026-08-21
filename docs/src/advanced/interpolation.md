# [Particle interpolation](@id interpolation)

```@meta
CurrentModule = QGYBJplus
```

Particle interpolation evaluates model-owned physical velocities at
off-lattice positions.

## Available methods

| Value | Use |
|:--|:--|
| `TRILINEAR` | low-cost eight-point interpolation |
| `TRICUBIC` | smoother, wider stencil |
| `QUINTIC` | higher-order, widest stencil |
| `ADAPTIVE` | chooses an appropriate stencil for the local resolution |

Select the method in `ParticleConfig`:

```julia
configuration = ParticleConfig{Float64}(
    x_max=model.grid.extent[1],
    y_max=model.grid.extent[2],
    z_level=-100.0,
    interpolation_method=TRICUBIC,
)
initialize_particles!(model, configuration)
```

## Direct interpolation

The exported array-level helper is useful for serial references and tests:

```julia
geometry = (
    dx=model.grid.dx,
    dy=model.grid.dy,
    dz=model.grid.dz,
    Lx=model.grid.extent[1],
    Ly=model.grid.extent[2],
    Lz=model.grid.extent[3],
)
boundaries = (periodic_x=true, periodic_y=true, periodic_z=false)

u, v, w = interpolate_velocity_advanced(
    x, y, z,
    u_array, v_array, w_array,
    geometry, boundaries, TRILINEAR,
)
```

Model-owned particle advection builds the complete geometry tuple, applies
vertical bounds, and handles horizontal periodicity automatically.

## Distributed halos

Under MPI, velocity halos are exchanged before interpolation. Wider methods
require wider halos, so accuracy and communication cost should be considered
together. Particle migration occurs after boundary normalization and the
trajectory update.

## Small local domains

`ADAPTIVE` is useful when a requested high-order stencil does not fit the local
resolution. For production runs, choose topology factors that leave multiple
interior cells beyond the halo width.
