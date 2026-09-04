# [Stratification](@id stratification)

```@meta
CurrentModule = QGYBJplus
```

The vertical coordinate spans negative depth to zero and is cell centered.
Runtime coefficients contain `N²` evaluated at the cell centers
`model.grid.z`, together with `N²_face` evaluated at each cell's upper face
`model.grid.z_faces[2:end]`. The Boussinesq elliptic coefficient is the
face-based array `a_ell = f²/N²_face`.

## Constant profile

```julia
stratification = ConstantStratification(N²=1e-5)
model = QGYBJModel(grid=grid, stratification=stratification)
```

## Analytical profile

```julia
N²_function = z -> 1e-5 + 4e-5 * exp(-((z + 100.0) / 40.0)^2)
profile = AnalyticalProfile(N²_function; returns=:N²)
model = QGYBJModel(grid=grid, stratification=profile)
```

Use `returns=:N` when the function returns buoyancy frequency instead of its
square. `precision=Float32` selects a different profile precision.

## Built-in profiles

The exported profile families include `ConstantN`, `SkewedGaussian`,
`TanhProfile`, `ExponentialProfile`, `PiecewiseProfile`, and `FileProfile`.
They can be evaluated independently:

```julia
values = compute_stratification_profile(profile, grid)
value_at_depth = evaluate_N2(profile, -100.0)
```

`compute_stratification_profile` returns the cell-center samples used by
pointwise physics.

## File input

```julia
profile = load_stratification_from_file("stratification.nc")
model = QGYBJModel(grid=grid, stratification=profile)
```

Inspect external values before model construction:

```julia
errors, warnings = validate_stratification(values)
isempty(errors) || error(join(errors, "\n"))
```

Model construction requires finite, positive values. The validation helper
also reports unusually weak or strong stratification.
