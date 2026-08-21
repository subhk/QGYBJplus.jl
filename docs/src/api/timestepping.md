# [Time integration](@id api-timestepping)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl uses second-order exponential Runge–Kutta integration exclusively.
Horizontal hyperdiffusion is handled by exact integrating factors; the
remaining tendency is evaluated in two stages.

## Simulation-owned stepping

```julia
simulation = Simulation(model; Δt=10.0, stop_iteration=100, output=false)
run!(simulation)
```

The timestepper and reusable workspace live at
`simulation.timestepper`.

## Manual stepping

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
step!(model, timestepper)
```

The first step allocates an [`ExponentialRungeKutta2Workspace`](@ref); later
steps reuse it.

```@docs
ExponentialRungeKutta2
ExponentialRungeKutta2Workspace
step!
run!
```
