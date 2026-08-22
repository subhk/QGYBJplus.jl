# [Geometry and fields](@id api-grid-state)

```@meta
CurrentModule = QGYBJplus
```

## Global geometry

```julia
grid = RectilinearGrid(size=(64, 48, 24),
                       x=(-π, π), y=(-2π, 2π), z=(-1.0, 0.0))
```

Useful properties are `size`, `extent`, `origin`, `z_bounds`, `x`, `y`, `z`,
`x_faces`, `y_faces`, `z_faces`, scalar spacings, `kx`, `ky`, and `kh2`.

## Model-owned arrays

`model.fields` is a [`ModelFields`](@ref) value. Arrays use `(z, x, y)` order:

| Field | Space | Role |
|:--|:--|:--|
| `q` | complex spectral | prognostic generalized potential vorticity |
| `B` | complex spectral | prognostic wave envelope |
| `psi` | complex spectral | diagnosed streamfunction |
| `A`, `C` | complex spectral | diagnosed wave quantities |
| `u`, `v`, `w` | real physical | diagnosed velocities |

```julia
fields_copy = copy_fields(model.fields)
```

`copy_fields` preserves distributed pencil layouts and copies every array.

## Distributed metadata

The public geometry has no MPI decomposition. Query the runtime or model:

```julia
get_local_range(model)
get_local_range_physical(model)
get_local_range_spectral(model)
local_to_global(1, 2, model)
```

`get_local_range(model)` returns the spectral range. Query physical and
spectral layouts explicitly when a custom operation crosses transform space.

Use `gather_to_root` and `scatter_from_root` for explicit
global-array boundaries. Model-level operators generally handle distribution
without requiring these helpers.
