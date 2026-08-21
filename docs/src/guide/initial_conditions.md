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

Advanced applications may assign distributed spectral arrays through
`model.fields.q` and `model.fields.B`. Use `scatter_from_root` with
`model.runtime.geometry`, `model.runtime.mpi`, and `plans=model.runtime.plans`
when the global array exists only on rank zero.

After direct assignment, call model-level diagnostic operators as needed to
refresh `ψ`, `A`, and velocities.
