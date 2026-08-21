# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

Diagnostics operate on the model that owns the fields and runtime metadata.

## Flow kinetic energy

```julia
compute_velocities!(model)
kinetic_energy = flow_kinetic_energy(model)
```

The result is reduced across the model communicator.

## Wave energy

```julia
B_energy, A_energy = wave_energy(model)
```

The returned tuple reports the global squared norms of the prognostic wave
envelope and diagnosed wave amplitude. `wave_energy` reconstructs amplitude
and vertical-derivative fields with the model's selected `YBJ`, `YBJPlus`, or
`PassiveWave` formulation, including no-dispersion runs whose timestep does
not otherwise retain those diagnostic fields.

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

## Output-based analysis

[`NetCDFOutput`](@ref) writes the vertical `N2` and `a_ell` profiles alongside
the fields needed for reproducible diagnostics. Simulation-owned energy
managers write component time series configured by
[`EnergyDiagnosticsOutput`](@ref). The repository also includes
`examples/compute_energy.jl` for spatial kinetic-energy post-processing.
