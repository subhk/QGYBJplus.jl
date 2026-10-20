# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

Diagnostics operate on the model that owns the fields and runtime metadata.

## Flow kinetic energy

```julia
kinetic_energy = flow_kinetic_energy(model)
```

The domain-mean kinetic energy of the balanced flow, ``\tfrac12\langle u^2 +
v^2\rangle`` (m² s⁻² in a dimensional setup): the `mean_flow_KE` that the
energy diagnostics record. The result is a host scalar, also when the model
runs on a GPU.

## Wave energy

```julia
energy = wave_energy(model)
energy.kinetic      # ½⟨|LA|²⟩
energy.potential
energy.correction   # zero for YBJ()
```

The domain-mean wave energies, as the energy diagnostics record them
(`wave_KE`, `wave_PE`, and `wave_CE`). With the balanced flow's energy, the
potential and correction energies make up the energy that the ideal coupled
system conserves (Asselin & Young 2019, eq. 3.7); the kinetic energy is
informational. Both functions rebuild what they need from the prognostic
fields and change no model field.

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
