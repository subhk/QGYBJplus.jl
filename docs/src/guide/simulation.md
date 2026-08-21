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

The diagnostics schedule records energy time series under the state-output
directory's `diagnostic/` subdirectory. Use
`EnergyDiagnosticsOutput(path=..., schedule=...)` to choose that directory
explicitly, or `diagnostics=false` to disable diagnostic files.

## Lifecycle

The lifecycle progresses through `Ready`, `Running`, and `Stopped`. An
exception moves it to `Failed`; explicit cleanup moves it to `Finalized`.
Each `Simulation` is a one-shot execution owner: after it reaches `Stopped`,
construct a new simulation around the same model to continue from the current
fields. This prevents closed output managers from overwriting earlier files.
After every ETD-RK2 step, non-finite fields and excessive streamfunction
growth are checked collectively, so all MPI ranks transition to `Failed`
before scheduled output is written.

```julia
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

Repeated finalization is safe. A finalized simulation cannot be run again.

## Overrides at run time

Before the initial run, `run!` accepts scoped overrides such as `Δt`,
`stop_iteration`, `output`, and `diagnostics`. Constructor configuration is
preferred for reproducible runs:

```julia
run!(simulation; stop_iteration=20, verbose=false)
```

Changing `Δt` does not alter an existing `stop_time`; the simulation advances
until the clock first reaches or crosses that time. Output-path, interval, and
field-selection overrides rebuild the effective `NetCDFOutput` configuration
before its manager opens.

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
