# [Physics and operator API](@id api-physics)

```@meta
CurrentModule = QGYBJplus
```

Public operator entry points accept a model so they can use its fields,
geometry, coefficients, transform plans, and architecture consistently.

## Elliptic inversions

```julia
invert_q_to_psi!(model)
invert_B_to_A!(model)
```

```@docs
invert_q_to_psi!
invert_B_to_A!
invert_helmholtz!
```

## Velocities

```julia
compute_velocities!(model)
compute_vertical_velocity!(model)
compute_ybj_vertical_velocity!(model)
compute_total_velocities!(model; t)   # instantaneous velocity at time t
compute_total_velocities!(model; include_wave_velocity=false)   # Eulerian mean
```

```@docs
compute_velocities!
compute_vertical_velocity!
compute_ybj_vertical_velocity!
compute_total_velocities!
compute_wave_velocities!
```

## Diagnostics

```julia
flow_energy = flow_kinetic_energy(model)   # ½⟨u² + v²⟩
wave = wave_energy(model)                  # (; kinetic, potential, correction)
```

```@docs
flow_kinetic_energy
wave_energy
```

## Coefficient helpers

```@docs
a_ell_from_N2
dealias_mask
is_dealiased
compute_hyperdiff_coeff
compute_hyperdiff_params
dimensional_hyperdiff_params
```

Lower-level array kernels are implementation details. Model-level methods are
the supported boundary for applications.
