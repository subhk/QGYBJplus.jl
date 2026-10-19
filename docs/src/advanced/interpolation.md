# [Particle interpolation](@id interpolation)

```@meta
CurrentModule = QGYBJplus
```

Particle interpolation evaluates the tracker's copy of the cell-centred
horizontal velocity at particle positions, in the advection kernel on the
model's architecture. Every scheme reproduces the grid values exactly.

## Select a method

| Value | Stencil | Error for smooth fields | Interpolant |
|:--|:--|:--|:--|
| `TRILINEAR` | 2 points per direction | O(h²) | continuous |
| `TRICUBIC` (default) | 4 points per direction, Catmull-Rom | O(h³) | continuously differentiable |
| `QUINTIC` | 6 points per direction, degree-5 Lagrange | O(h⁶) | continuous |

A three-dimensional evaluation reads 8 grid points per field for `TRILINEAR`,
64 for `TRICUBIC`, and 216 for `QUINTIC`. `ADAPTIVE` is deprecated and now
uses `TRICUBIC`.

```julia
configuration = ParticleConfig{Float64}(
    x_min=first(model.grid.x_faces),
    x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces),
    y_max=last(model.grid.y_faces),
    z_level=-100.0,
    interpolation_method=QUINTIC,
)
initialize_particles!(model, configuration)
```

## Boundaries

Horizontally the stencils wrap periodically. Vertically, each column is
extended past the lid and the bottom by its mirror image, evenly for the
horizontal velocity, which has zero vertical derivative there
(``\psi_z = 0``), so the interpolation keeps its order of accuracy up to the
boundaries. Positions outside the column take the boundary value. The
lower-level `interpolate_velocity_advanced` also interpolates a vertical
velocity, mirrored oddly so that it vanishes at both boundaries; it reads
host arrays.
