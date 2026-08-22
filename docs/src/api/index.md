# [API map](@id api-index)

```@meta
CurrentModule = QGYBJplus
```

Frequently used exported entry points are grouped below. Detailed signatures
and examples are on the linked API pages.

## Construction

- [`RectilinearGrid`](@ref)
- [`QGYBJModel`](@ref)
- [`Simulation`](@ref)
- [`NetCDFOutput`](@ref)
- [`EnergyDiagnosticsOutput`](@ref)

## Initialization

- `set!`
- `set_mean_flow!`
- `set_surface_waves!`
- `set_exponential_surface_waves!`
- `set_wave_packet!`
- [`FieldArray`](@ref)
- [`FieldFile`](@ref)
- `restore!`

## Integration and lifecycle

- [`ExponentialRungeKutta2`](@ref)
- [`step!`](@ref)
- [`run!`](@ref)
- `finalize_model!`
- `finalize_simulation!`

## Operators and diagnostics

- `invert_q_to_psi!`
- `invert_B_to_A!`
- `compute_velocities!`
- `compute_vertical_velocity!`
- `compute_ybj_vertical_velocity!`
- `compute_total_velocities!`
- `compute_wave_velocities!`
- `flow_kinetic_energy`
- `wave_energy`

## Parallel helpers

- `get_local_range`
- `get_local_range_physical`
- `get_local_range_spectral`
- `local_to_global`
- `gather_to_root`
- `scatter_from_root`

## Particles

- [`ParticleConfig`](@ref)
- [`ParticleConfig3D`](@ref)
- `initialize_particles!`
- `advect_particles!`
- [`particles_in_box`](@ref)
- [`particles_in_circle`](@ref)
- [`particles_in_grid_3d`](@ref)
- [`particles_in_layers`](@ref)
- [`particles_random_3d`](@ref)
- [`particles_custom`](@ref)
- [`ParticleOutputManager`](@ref)
