## Worked Example

This walkthrough sets up a small QG–YBJ+ simulation, runs it for a short time,
and inspects some outputs.

### 1) Create a configuration

```julia
using QGYBJ

domain = create_domain_config(nx=64, ny=64, nz=32, Lx=4π, Ly=4π, Lz=2π)
strat  = create_stratification_config(:constant_N, N0=1.0)
init   = create_initial_condition_config(psi_type=:random, wave_type=:random,
                                         psi_amplitude=0.1, wave_amplitude=0.01)
output = create_output_config(output_dir="./output_example",
                              psi_interval=1.0, wave_interval=1.0,
                              save_velocities=true, save_vertical_velocity=true)

config = create_model_config(domain, strat, init, output; dt=1e-3, total_time=2.0)
```

### 2) Setup and run the simulation

```julia
sim = setup_simulation(config)
run_simulation!(sim)
```

### 3) Inspect fields

```julia
# Compute velocities (if not already)
compute_velocities!(sim.state, sim.grid; plans=sim.plans, params=sim.params, compute_w=true)

# Simple diagnostics
using QGYBJ: wave_energy
EB, EA = wave_energy(sim.state.B, sim.state.A)
@info "Wave energies" EB EA

# Extract a horizontal slice of ψ
using QGYBJ: slice_horizontal
ψxy = slice_horizontal(sim.state.psi, sim.grid, sim.plans; k=sim.grid.nz ÷ 2)
@info "psi slice stats" minimum(ψxy) maximum(ψxy)
```

### 4) Optional: NetCDF output

If you've installed NCDatasets.jl, `run_simulation!` writes NetCDF files in
`output_example/` at your configured intervals. You can open them with tools
like Panoply, MATLAB, xarray, or NCDatasets.jl.

