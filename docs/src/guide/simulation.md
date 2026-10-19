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
fields. A new simulation's clock starts at zero; to continue the time axis
and the output files of the previous run, restore its last state file into
the new simulation with `restore!(simulation, path)`, which also sets the
clock (see [I/O and restart](@ref io-output)).
After every step the simulation checks for non-finite values and for a
blowing-up streamfunction, and enters `Failed` if either occurs.

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

Set `progress=true` to print the maximum wave and flow speeds at the
`diagnostics_interval` cadence:

```julia
run!(simulation; progress=true, diagnostics_interval=100)
```

```text
Iteration: 0100, time: 3.333 minutes, Δt: 2 seconds, max(|LA|) = 9.998e-02 m s⁻¹, max(|uₕ|) = 3.349e-01 m s⁻¹, wall time: 12.345 seconds
```

Here the wave speed is ``\max |LA|`` and the flow speed is
``\max \sqrt{u^2+v^2}``. Wall time is the elapsed time for the current
`run!` call.

By default, model construction also reports the architecture and resolution:

```text
┌ Info: QGYBJModel runtime initialized
│   architecture = GPU(CUDABackend)
└   size = (256, 256, 128)
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
`step!(model, timestepper)`. That call advances model fields only; it does not
update a clock, particles, output, or diagnostics. Ordinary runs should use
`Simulation`.
