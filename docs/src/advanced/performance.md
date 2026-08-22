# [Performance](@id performance)

```@meta
CurrentModule = QGYBJplus
```

## Reuse the simulation timestepper

`Simulation` reuses one [`ExponentialRungeKutta2Workspace`](@ref) after the
first step. For manual field-only stepping, reuse the timestepper object:

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
for _ in 1:100
    step!(model, timestepper)
end
```

Constructing a new timestepper inside the loop defeats workspace reuse.
Manual stepping does not advance particles, output, diagnostics, or a clock.

## Avoid unnecessary communication

- Avoid gathering full fields during the time loop.
- Write only the fields required for analysis.
- Measure with output disabled before diagnosing solver performance.
- Finalize simulations promptly so owned runtime resources are released.

## Resolution and topology

Choose topology factors that divide the horizontal resolution and leave each
rank with enough cells for nonlinear transforms and particle halos. Measure
several factorizations for communication-heavy workloads.

## Limit output

Writing physical `ψ`, waves, and velocities requires inverse transforms and a
gather. Select only the fields required for analysis:

```julia
NetCDFOutput(path="output",
             schedule=IterationInterval(100),
             fields=(:waves,),
             velocities=false)
```

## Benchmark reproducibly

Warm up compilation before measuring. For MPI runs, report the maximum elapsed
time across ranks and include resolution, rank count, topology, Julia version,
and output settings with every result.
