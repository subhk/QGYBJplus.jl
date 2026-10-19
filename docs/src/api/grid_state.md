# [Geometry and fields](@id api-grid-state)

```@meta
CurrentModule = QGYBJplus
```

## Global geometry

```julia
grid = RectilinearGrid(size=(64, 48, 24),
                       x=(-π, π), y=(-2π, 2π), z=(-1.0, 0.0))
kx, ky = grid.kx, grid.ky
architecture(grid)   # CPU()
```

Useful properties are `size`, `extent`, `origin`, `z_bounds`, `x`, `y`, `z`,
`x_faces`, `y_faces`, `z_faces`, scalar spacings, `kx`, `ky`, and `kh2`.
The wavenumbers follow FFTW ordering and can be used directly with `fft`.
These coordinate vectors always live on the host, whatever the architecture.

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

`copy_fields` copies every array and keeps the copies on the fields'
architecture.
Time steps update the arrays in place, so both a cached `fields = model.fields`
and a cached array such as `q = model.fields.q` remain live handles. Use
`copy_fields` when a fixed snapshot is needed.

## Arrays on a GPU

On a GPU, model arrays are device arrays such as `CuArray`. Copy one to the
host with `Array` before indexing elements, and move host data onto the
model's architecture with [`on_architecture`](@ref):

```julia
q_host = Array(model.fields.q)
copyto!(model.fields.q, on_architecture(architecture(model), q_host))
```
