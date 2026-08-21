# [I/O and restart](@id io-output)

```@meta
CurrentModule = QGYBJplus
```

## Scheduled NetCDF snapshots

```julia
output = NetCDFOutput(
    path="output",
    schedule=IterationInterval(100),
    fields=(:ψ, :waves),
    velocities=true,
)
simulation = Simulation(model; Δt=10.0, stop_iteration=1000,
                        output=output)
```

`TimeInterval(seconds)` provides time-based scheduling instead. The initial
condition is written when output is enabled, and finalization writes the last
iteration if it was not already scheduled.

## Snapshot schema

Every file contains `x`, `y`, `z`, `time`, iteration metadata, spectral
`q_real`, `q_imag`, `B_real`, and `B_imag`, plus `N2` and `a_ell`.

Selecting `:ψ` adds physical `psi`. Selecting `:waves` adds physical `LAr`,
`LAi`, `Ar`, and `Ai`. `velocities=true` adds `u`, `v`, and `w`.

Array dimensions in NetCDF are `(x, y, z)`; model arrays use `(z, x, y)`.

## Restart

Build a compatible model, then restore the prognostic arrays:

```julia
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))
restore!(model, "output/state0011.nc")
simulation = Simulation(model; Δt=10.0, stop_iteration=100)
```

The restart dimensions must match `model.grid.size`. Diagnostic arrays are
reconstructed after the distributed scatter.

## Failure behavior

Output exceptions move the simulation to `Failed`, close the output manager,
and are rethrown on every rank. Use `try`/`finally` to guarantee cleanup:

```julia
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```
