# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

Build a model by composing typed physical and numerical choices.

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

Available choices include:

| Constructor keyword | Values |
|:--|:--|
| `flow` | `FixedFlow()`, `EvolvingFlow()` |
| `feedback` | `NoFeedback()`, `NoWaveFeedback()`, `WaveMeanFeedback()` |
| `formulation` | `YBJPlus()`, `YBJ()`, `PassiveWave()` |
| `linear` | `NonlinearDynamics()`, `LinearDynamics()` |
| `no_dispersion` | `Dispersive()`, `NoDispersion()` |
| `inviscid` | `Dissipative()`, `Inviscid()` |

See [key concepts](@ref concepts) for the recommended flow and feedback
combinations.

## Numerical choices

```julia
closure = WaveHyperdiffusivity(coefficient=1e5)

model = QGYBJModel(
    grid=grid,
    closure=closure,
    vertical_diffusion=VerticalDiffusivity(coefficient=0),
)
```

This applies a single fourth-order damping term to waves. Use
[`HorizontalHyperdiffusivity`](@ref) when an evolving flow or two damping
orders need separate coefficients.

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
