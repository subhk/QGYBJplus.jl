# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

Diagnostics operate on the model that owns the fields and runtime metadata.

## Flow kinetic energy

```julia
kinetic_energy = flow_kinetic_energy(model)
```

The result is reduced across the model communicator.

## Wave energy

```julia
B_energy, A_energy = wave_energy(model)
```

The returned tuple reports globally reduced envelope and diagnosed-amplitude
energies. The diagnostic reconstructs fields required by the selected wave
formulation before reducing them.

## Runtime inspection

```julia
simulation.clock.time
simulation.clock.iteration
simulation.state
is_root(simulation)
nprocs(simulation)
```

Use an [`IterationInterval`](@ref) or [`TimeInterval`](@ref) when constructing
the simulation to schedule energy diagnostics. `verbose=true` reports progress
at the corresponding approximate iteration cadence.

For scheduled component time series, configure
[`EnergyDiagnosticsOutput`](@ref). See [I/O and restart](@ref io-output) for
the files written and `examples/compute_energy.jl` for spatial
post-processing.
