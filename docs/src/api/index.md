# [API index](@id api-index)

```@meta
CurrentModule = QGYBJplus
```

## Construction

- [`RectilinearGrid`](@ref)
- [`QGYBJModel`](@ref)
- [`Simulation`](@ref)
- [`NetCDFOutput`](@ref)

## Initialization

- `set!`
- `set_mean_flow!`
- `set_surface_waves!`
- `set_exponential_surface_waves!`
- `set_wave_packet!`
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
- `compute_total_velocities!`
- `flow_kinetic_energy`
- `wave_energy`

## Parallel helpers

- `get_local_range`
- `local_to_global`
- `gather_to_root`
- `scatter_from_root`

## Particles

- `ParticleConfig`
- `initialize_particles!`
- `advect_particles!`
- [`particles_in_layers`](@ref)
- [`particles_random_3d`](@ref)
