# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

Configuration is expressed by composing focused Julia values. There is no
catch-all parameter record or secondary builder layer.

## Geometry

```julia
grid = RectilinearGrid(
    size=(128, 128, 64),
    x=(-250e3, 250e3),
    y=(-250e3, 250e3),
    z=(-4000.0, 0.0),
)
```

Alternatively, provide `extent=(Lx, Ly, Lz)` and optionally
`centered=true`.

## Physics choices

```julia
model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    flow=EvolvingFlow(),
    feedback=WaveMeanFeedback(),
    formulation=YBJPlus(),
)
```

Available focused choices include:

| Concern | Values |
|:--|:--|
| balanced flow | `FixedFlow()`, `EvolvingFlow()` |
| coupling | `NoFeedback()`, `NoWaveFeedback()`, `WaveMeanFeedback()` |
| waves | `YBJPlus()`, `YBJ()`, `PassiveWave()` |
| nonlinear terms | `NonlinearDynamics()`, `LinearDynamics()` |
| dispersion | `Dispersive()`, `NoDispersion()` |
| dissipation | `Dissipative()`, `Inviscid()` |

Constructor shorthands such as `flow=:fixed` remain accepted at the public
boundary, but typed values make intent and ownership clearest.

## Numerical choices

```julia
closure = HorizontalHyperdiffusivity(
    flow=1e7,
    flow2=0,
    flow_laplacian_order=2,
    waves=1e5,
    waves2=0,
    wave_laplacian_order=2,
)

model = QGYBJModel(
    grid=grid,
    closure=closure,
    vertical_diffusion=VerticalDiffusivity(coefficient=0),
)
```

Set unused closure coefficients explicitly to zero in dimensional examples.

## Execution choices

MPI topology and parallel I/O are runtime construction options:

```julia
model = QGYBJModel(
    grid=grid,
    topology=(2, 4),
    parallel_io=false,
)
```

Time integration and output belong to a separate simulation:

```julia
simulation = Simulation(
    model;
    Δt=10.0,
    stop_iteration=1000,
    output=NetCDFOutput(
        path="output",
        schedule=IterationInterval(100),
    ),
)
```

Only ETD-RK2 is available; constructing `Simulation` installs an
[`ExponentialRungeKutta2`](@ref) timestepper automatically.
