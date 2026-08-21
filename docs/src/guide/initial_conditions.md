# [Initial conditions](@id initial-conditions)

```@meta
CurrentModule = QGYBJplus
```

Model-level setters update model-owned arrays and use runtime transform plans
automatically.

## Analytical mean flow

```julia
set!(
    model;
    ψ=(x, y, z) -> 1e3 * sin(2π * x / model.grid.extent[1]),
    pv_method=:qg,
)
```

Use `pv_method=:barotropic` for an imposed vertically uniform flow, as in the
Asselin dipole example. Use `:none` only when `q` will be assigned separately.

## Deterministic random mean flow

```julia
set!(
    model;
    mean_flow=RandomStreamfunction(
        amplitude=0.4,
        spectral_slope=-3,
        seed=42,
    ),
)
```

The MPI initializer is decomposition independent for a fixed seed.

## Surface-confined waves

```julia
set!(
    model;
    waves=SurfaceWave(
        amplitude=0.1,
        scale=30.0,
        profile=:gaussian,
    ),
)
```

The profile may be `:gaussian` or `:exponential`.

## Wave packets

```julia
set_wave_packet!(
    model;
    amplitude=0.1,
    kx=2,
    ky=1,
    sigma_k=0.5,
)
```

## Direct array access

Pass a global `(z, x, y)` array directly to initialize a physical field:

```julia
set!(model; ψ=psi_values, pv_method=:qg)
set!(model; B=wave_values)
```

Use [`FieldArray`](@ref) to declare spectral data or `(x, y, z)` layout:

```julia
set!(model; B=FieldArray(B_hat; space=:spectral, layout=:zxy))
```

NetCDF-backed fields use [`FieldFile`](@ref). Its default layout is the
file-oriented `(x, y, z)` ordering:

```julia
set!(model; B=FieldFile("initial.nc", "B"))
```

All public setters rebuild dependent `q`, `A`, `C`, and velocity diagnostics.
Advanced applications may still assign distributed prognostic arrays directly,
but then must refresh diagnostic operators themselves.
