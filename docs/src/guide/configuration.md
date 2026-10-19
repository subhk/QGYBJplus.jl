# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

Build a model by composing typed physical and numerical choices.

## Geometry

```julia
grid = RectilinearGrid(
    CPU();
    size=(128, 128, 64),
    x=(-250e3, 250e3),
    y=(-250e3, 250e3),
    z=(-4000.0, 0.0),
)
```

Alternatively, provide `extent=(Lx, Ly, Lz)` and optionally
`centered=true`. The first argument selects the architecture: `CPU()` (the
default when omitted) or `GPU()` after `using CUDA`; see [CPU and GPU
execution](@ref architectures).

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
closure = HorizontalHyperdiffusivity(
    flow=FlowHyperdiffusivity(coefficient=0),
    wave=WaveHyperdiffusivity(coefficient=1e5),
)

model = QGYBJModel(
    grid=grid,
    closure=closure,
    vertical_diffusion=VerticalDiffusivity(coefficient=0),
)
```

This leaves the balanced flow undamped and applies one fourth-order damping
term to waves. Each `order` is the total derivative order.

The default closure, `HorizontalHyperdiffusivity()`, keeps the coefficients
of nondimensional setups, which damp nothing at dimensional wavenumbers; a
nonlinear model warns when that is so. For a dimensional run, choose the
coefficients from the grid spacing and time step:

```julia
ν = compute_hyperdiff_coeff(dx=grid.dx, dy=grid.dy, dt=Δt, order=8)
closure = HorizontalHyperdiffusivity(
    flow=FlowHyperdiffusivity(coefficient=ν, order=8),
    wave=WaveHyperdiffusivity(coefficient=ν, order=8),
)
```

`compute_hyperdiff_coeff` damps the largest wavenumber the two-thirds
dealiasing keeps by one e-fold every `efold_steps=10` steps; pass
`inviscid=Inviscid()` to run without horizontal damping deliberately.

## Execution choices

Where the model runs is fixed by the grid's architecture. Time integration and
output belong to a separate simulation:

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
