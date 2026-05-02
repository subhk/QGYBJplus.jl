# [Stratification](@id stratification)

```@meta
CurrentModule = QGYBJplus
```

Stratification enters the elliptic operators through the dimensional
coefficient `f²/N²`. QGYBJ+.jl evaluates profiles on the model's vertical grid.

The vertical coordinate is physical `z`: `z = 0` at the surface and negative
below the surface.

## Constant Stratification

```julia
model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = ConstantStratification(N² = 1e-5))
```

Use this for idealized experiments or when matching constant-`N` reference
solutions.

## Analytic Profiles

Use a `StratificationProfile` when `N²` varies with depth:

```julia
profile = TanhProfile{Float64}(0.01,    # upper-ocean N
                               0.025,   # pycnocline N
                               500.0,   # pycnocline depth [m]
                               100.0)   # width [m]

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = profile)
```

For custom formulas:

```julia
N = z -> 1e-3 + 2e-3 * exp(z / 500)
profile = AnalyticalProfile{Float64}(N, false) # false means N(z), not N²(z)
```

## File-Backed Profiles

Load `z` and `N` from NetCDF or JLD2:

```julia
profile = FileStratification("stratification.nc")

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = profile)
```

If the file stores `N²` directly:

```julia
profile = FileStratification("stratification.nc"; N2 = "N2")
```

The file coordinate may be physical `z <= 0` or positive depth below the
surface. The profile is interpolated to the numerical grid during model
construction.

JLD2 files use the same keys:

```julia
using JLD2

jldsave("stratification.jld2"; z = z_data, N = N_data)
profile = FileStratification("stratification.jld2")
```

## Low-Level Use

```julia
params = default_params(Lx = 500e3, Ly = 500e3, Lz = 4000.0,
                        nx = 64, ny = 64, nz = 32)

grid = init_grid(params)
profile = FileStratification("stratification.nc")
N2_profile = compute_stratification_profile(profile, grid)
a_ell = a_ell_from_N2(N2_profile, params)
```

## Practical Checks

- `N²` must be positive everywhere.
- Sharp pycnoclines need enough vertical resolution.
- Use dimensional values: seconds, meters, and `s⁻²`.

## See Also

- [Configuration](@ref configuration)
- [QG Equations](@ref qg-equations)
- [YBJ+ Wave Model](@ref ybj-plus)
