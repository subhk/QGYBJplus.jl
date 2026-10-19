# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

Diagnostics operate on the model that owns the fields and runtime metadata.

## Flow kinetic energy

```julia
kinetic_energy = flow_kinetic_energy(model)
```

The result is a host scalar, also when the model runs on a GPU.

## Wave energy

```julia
B_energy, A_energy = wave_energy(model)
```

The returned tuple reports the envelope and diagnosed-amplitude energies
summed over the domain. The diagnostic reconstructs fields required by the selected wave
formulation before reducing them.

## Runtime inspection

```julia
simulation.clock.time
simulation.clock.iteration
simulation.state
architecture(simulation)
```

Use an [`IterationInterval`](@ref) or [`TimeInterval`](@ref) when constructing
the simulation to schedule energy diagnostics. `verbose=true` reports progress
at the corresponding approximate iteration cadence.

For scheduled component time series, configure
[`EnergyDiagnosticsOutput`](@ref). See [I/O and restart](@ref io-output) for
the saved operator fields and FFTW analysis, or run
`examples/compute_energy.jl output/` to write spatial kinetic-energy fields.
