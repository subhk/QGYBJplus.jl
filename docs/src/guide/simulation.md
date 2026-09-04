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
Termination checks are collective, so all MPI ranks enter `Failed` together.

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

Set `progress=true` to print global maximum wave and flow speeds at the
`diagnostics_interval` cadence:

```julia
run!(simulation; progress=true, diagnostics_interval=100)
```

```text
Iteration: 0100, time: 3.333 minutes, Δt: 2 seconds, max(|LA|) = 9.998e-02 m s⁻¹, max(|uₕ|) = 3.349e-01 m s⁻¹, wall time: 12.345 seconds
```

Here the wave speed is ``\max |LA|`` and the flow speed is
``\max \sqrt{u^2+v^2}``. In MPI runs, the maxima are global and only rank zero
prints. Wall time is the elapsed time for the current `run!` call.

By default, model construction also reports the MPI and runtime setup on rank
zero:

```text
┌ Info: MPI initialized with 2D decomposition
│   nprocs = 16
└   topology = (4, 4)
┌ Info: Topology validation passed
│   nx = 256
│   ny = 256
│   nz = 128
│   topology = (4, 4)
└   decomp_dims = (2, 3)
┌ Info: Pencil decompositions created
│   xy_decomp = (2, 3)
│   xz_decomp = (1, 3)
└   z_decomp = (2, 3)
┌ Info: QGYBJModel runtime initialized
│   size = (256, 256, 128)
└   ranks = 16
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
