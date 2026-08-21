# [Performance](@id performance)

```@meta
CurrentModule = QGYBJplus
```

## Reuse the ETD-RK2 workspace

`Simulation` reuses one [`ExponentialRungeKutta2Workspace`](@ref) after the
first step. For manual stepping, reuse the timestepper object:

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
for _ in 1:100
    step!(model, timestepper)
end
```

Constructing a new timestepper inside the loop defeats workspace reuse.

## Keep ownership boundaries intact

- Reuse `model.runtime.plans` and `model.runtime.workspace`.
- Use `copy_fields` instead of generic deep copying for distributed arrays.
- Avoid gathering full fields during the time loop.
- Schedule output less frequently than diagnostics.
- Finalize models promptly so MPI and transform resources are released.

## Resolution and topology

Choose topology factors that divide the horizontal resolution and leave each
rank with enough cells for nonlinear transforms and particle halos. Measure
several factorizations for communication-heavy workloads.

## Output

Writing physical `ψ`, waves, and velocities requires inverse transforms and a
gather. Select only the fields required for analysis:

```julia
NetCDFOutput(path="output",
             schedule=IterationInterval(100),
             fields=(:waves,),
             velocities=false)
```

## Benchmarking

Warm up compilation before measuring. For MPI runs, report the maximum elapsed
time across ranks and include resolution, rank count, topology, Julia version,
and output settings with every result.
