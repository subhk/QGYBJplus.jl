## Simulation

### High‑Level API

```julia
using QGYBJ

config = create_simple_config(dt=1e-3, total_time=2.0)
sim = setup_simulation(config)
run_simulation!(sim)
```

The `QGYBJSimulation` object holds:
- `params::QGParams`, `grid::Grid`, `state::State`
- `plans` (FFTs), `output_manager`, `stratification_profile`, `N2_profile`
- `current_time`, `time_step`, and `diagnostics`

### Time Stepping

The time stepper performs:
- First step: projection method (`first_projection_step!`)
- Main loop: leapfrog with Robert–Asselin filter (`leapfrog_step!`)

Flags that affect integration (in `QGParams`):
- `ybj_plus::Bool` selects YBJ+ vs. normal YBJ recovery
- `no_wave_feedback::Bool` disables `q^w` feedback on mean flow
- `fixed_flow::Bool` freezes mean flow (q does not evolve)
- `linear`, `inviscid`, `no_dispersion`, `passive_scalar`

### Velocities and Diagnostics

- `compute_velocities!` computes QG u, v and optionally w (QG omega
  or YBJ vertical velocity)
- `compute_total_velocities!` adds wave‑induced velocities (Stokes‑like)
- Diagnostics (module `Diagnostics`) include wave energy, slices, and the
  RHS of the omega equation.

