# [Physics and operator API](@id api-physics)

```@meta
CurrentModule = QGYBJplus
```

Public operator entry points accept a model so they can use its fields,
geometry, coefficients, transform plans, and communicator consistently.

## Elliptic inversions

```julia
invert_q_to_psi!(model)
invert_B_to_A!(model)
```

## Velocities

```julia
compute_velocities!(model)
compute_vertical_velocity!(model)
compute_ybj_vertical_velocity!(model)
compute_total_velocities!(model)
compute_wave_velocities!(model)
```

## Diagnostics

```julia
flow_energy = flow_kinetic_energy(model)
B_energy, A_energy = wave_energy(model)
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
