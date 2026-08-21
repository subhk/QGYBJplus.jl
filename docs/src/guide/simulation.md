# [Running simulations](@id simulation-guide)

```@meta
CurrentModule = QGYBJplus
```

[`Simulation`](@ref) owns execution concerns separately from
[`QGYBJModel`](@ref).

```julia
simulation = Simulation(
    model;
    Δt=10.0,
    stop_time=86400.0,
    output=false,
    diagnostics=IterationInterval(100),
    verbose=true,
)
```

Specify either `stop_time` or `stop_iteration`. ETD-RK2 is created
automatically and stored in `simulation.timestepper`.

## Lifecycle

The lifecycle progresses through `Ready`, `Running`, and `Stopped`. An
exception moves it to `Failed`; explicit cleanup moves it to `Finalized`.

```julia
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

Repeated finalization is safe. A finalized simulation cannot be run again.

## Overrides at run time

`run!` accepts scoped overrides such as `Δt`, `stop_iteration`, `output`, and
`diagnostics`. Constructor configuration is preferred for reproducible runs:

```julia
run!(simulation; stop_iteration=20, verbose=false)
```

## Clock access

```julia
simulation.clock.time
simulation.clock.iteration
get_time(simulation, 10)
inertial_period(simulation)
```

## Manual model stepping

Low-level applications may construct [`ExponentialRungeKutta2`](@ref) and call
`step!(model, timestepper)`. Ordinary runs should use `Simulation` so clock,
particle, output, and failure lifecycles stay synchronized.
