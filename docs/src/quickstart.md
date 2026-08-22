# [Quick start](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

This example evolves waves in an evolving balanced flow and writes scheduled
snapshots and energy diagnostics.

## 1. Create the model

```julia
using QGYBJplus

grid = RectilinearGrid(
    size=(32, 32, 16),
    x=(-250e3, 250e3),
    y=(-250e3, 250e3),
    z=(-4000.0, 0.0),
)

model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure=HorizontalHyperdiffusivity(
        flow=0, flow2=0, waves=0, waves2=0),
    flow=EvolvingFlow(),
    feedback=NoWaveFeedback(),
    formulation=YBJPlus(),
)
```

The model owns its fields and computational runtime. Global geometry remains
available as `model.grid`.

## 2. Set initial conditions

```julia
set!(
    model;
    ψ=(x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3),
    pv_method=:barotropic,
    waves=SurfaceWave(amplitude=0.1, scale=30.0),
)
```

## 3. Run and clean up

```julia
simulation = Simulation(
    model;
    Δt=20.0,
    stop_iteration=10,
    output=NetCDFOutput(
        path="output",
        schedule=IterationInterval(5),
    ),
    diagnostics=IterationInterval(5),
)

try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

Snapshots are written under `output/`; energy time series are written under
`output/diagnostic/`. See [I/O and restart](@ref io-output) for the file schema
and [configuration](@ref configuration) for other physical choices.
