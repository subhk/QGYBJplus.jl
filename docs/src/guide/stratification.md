# [Stratification](@id stratification)

```@meta
CurrentModule = QGYBJplus
```

The vertical coordinate spans negative depth to zero and is cell centered.
Stratification is evaluated on `model.grid.z`; runtime coefficients contain
the resulting `N²`, `f²/N²`, and density factors.

## Constant profile

```julia
stratification = ConstantStratification(N²=1e-5)
model = QGYBJModel(grid=grid, stratification=stratification)
```

## Analytical profile

```julia
N²_function = z -> 1e-5 + 4e-5 * exp(-((z + 100.0) / 40.0)^2)
profile = AnalyticalProfile{Float64, typeof(N²_function)}(N²_function, true)
model = QGYBJModel(grid=grid, stratification=profile)
```

The second constructor value declares whether the function returns `N²`
(`true`) or `N` (`false`).

## Built-in profiles

The exported profile families include `ConstantN`, `SkewedGaussian`,
`TanhProfile`, `ExponentialProfile`, `PiecewiseProfile`, and `FileProfile`.
They can be evaluated independently:

```julia
values = compute_stratification_profile(profile, grid)
value_at_depth = evaluate_N2(profile, -100.0)
```

## File input

```julia
profile = load_stratification_from_file("stratification.nc")
model = QGYBJModel(grid=grid, stratification=profile)
```

Validate external data before model construction:

```julia
errors, warnings = validate_stratification(values)
isempty(errors) || error(join(errors, "\n"))
```

All values must be finite and positive. The operator coefficient helper
[`a_ell_from_N2`](@ref) applies the same validation.
