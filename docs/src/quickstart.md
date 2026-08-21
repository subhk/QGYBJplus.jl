# [Quick start](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

## 1. Define immutable geometry

```julia
grid = RectilinearGrid(
    size=(64, 64, 32),
    x=(-250e3, 250e3),
    y=(-250e3, 250e3),
    z=(-4000.0, 0.0),
)
```

Horizontal nodes are periodic and vertical nodes are cell centered. Global
coordinates and wavenumbers remain on this object; local decomposition data
belongs to the model runtime.

## 2. Compose a model

```julia
model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure=HorizontalHyperdiffusivity(
        flow=0,
        flow2=0,
        waves=1e5,
        waves2=0,
        wave_laplacian_order=2,
    ),
    flow=FixedFlow(),
    feedback=NoFeedback(),
    formulation=YBJPlus(),
)
```

The model owns `model.fields`, `model.physics`, `model.numerics`, and
`model.runtime`. Prognostic arrays are `model.fields.q` and `model.fields.B`.

## 3. Initialize fields

```julia
set!(
    model;
    ψ=(x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3),
    pv_method=:barotropic,
    waves=SurfaceWave(amplitude=0.1, scale=30.0, profile=:gaussian),
)
```

Use `RandomStreamfunction` for deterministic spectral initialization or the
model-level setters described in [Initial conditions](@ref initial-conditions).

## 4. Configure and run

```julia
simulation = Simulation(
    model;
    Δt=20.0,
    stop_iteration=100,
    output=NetCDFOutput(
        path="output",
        schedule=IterationInterval(20),
        fields=(:ψ, :waves),
        velocities=true,
    ),
    diagnostics=IterationInterval(10),
)

try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

Inspect `simulation.clock.time`, `simulation.clock.iteration`, and
`simulation.state`. Model arrays remain under `simulation.model.fields`.

## 5. Restart

Create a fresh, compatible model and restore its prognostic fields before
starting a new simulation:

```julia
restore!(model, "output/state0006.nc")
```

See [I/O and restart](@ref io-output) for the file schema and scheduling rules.
