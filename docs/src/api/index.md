# [API Reference](@id api-index)

```@meta
CurrentModule = QGYBJplus
```

This section is a compact map of the public API. Detailed docstrings live on
the topic pages linked below.

## Main User Interface

- `RectilinearGrid` defines the physical domain.
- `QGYBJModel` stores the grid, physics, closures, state, transforms, and MPI
  setup.
- `set!` initializes the flow and waves.
- `Simulation` defines the run clock, output, and diagnostics.
- `run!` advances the model.
- `NetCDFOutput` configures state-file output.

See [Quick Start](@ref quickstart), [Configuration](@ref configuration), and
[I/O and Output](@ref io-output) for examples.

## Topic Pages

- [Core Types](types.md): `QGParams`, `Grid`, `State`, and configuration types.
- [Grid & State](grid_state.md): grid creation, state allocation, and FFT plans.
- [Physics Functions](physics.md): elliptic inversions, velocities, diagnostics,
  and transforms.
- [Time Stepping](timestepping.md): exponential RK2 integration.
- [Particles](particles.md): Lagrangian particle tracking.

## Low-Level Entry Points

These are useful for solver development and tests:

- `default_params`
- `setup_model`
- `dealias_mask`
- `exp_rk2_step!`
- `invert_q_to_psi!`
- `invert_L⁺A_to_A!`
- `compute_velocities!`
- `write_state_file`

Most user scripts should not need to call the low-level MPI, pencil-array, or
transpose helpers directly.
