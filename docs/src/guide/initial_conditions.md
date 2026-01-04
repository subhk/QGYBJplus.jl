# [Initial Conditions](@id initial-conditions)

```@meta
CurrentModule = QGYBJplus
```

This page describes how to set up initial conditions for QGYBJ+.jl simulations.

## Overview

Initial conditions must be specified for:
- **Potential vorticity** ``q`` (or streamfunction ``\psi``)
- **Wave envelope** ``B`` (or wave amplitude ``A``)

## Random Initialization

### Random Streamfunction

The most common way to initialize a simulation is with random streamfunction:

```julia
using QGYBJplus

# Create parameters and setup
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a_ell = setup_model(par)

# Random streamfunction
init_random_psi!(S, G; amplitude=0.1, seed=12345)

# Compute q from psi (required before time stepping)
compute_q_from_psi!(S, G, plans, a_ell)
```

The random field is band-limited to resolved wavenumbers.

## Analytical Initial Conditions

### Analytical Streamfunction

```julia
# Initialize with analytical pattern (e.g., dipole)
init_analytical_psi!(S, G; mode=:dipole, amplitude=1.0)

# Or single mode
init_analytical_psi!(S, G; mode=:single, kx=2, ky=2, amplitude=0.5)

# Compute q from psi
compute_q_from_psi!(S, G, plans, a_ell)
```

### Analytical Waves

```julia
# Initialize wave envelope with analytical pattern
init_analytical_waves!(S, G; amplitude=0.01, vertical_mode=1)
```

## Balanced Initialization

To ensure the flow starts in geostrophic balance:

```julia
# First set up streamfunction
init_random_psi!(S, G; amplitude=0.1)

# Add balanced component
add_balanced_component!(S, G, plans, a_ell)

# Compute q from the balanced psi
compute_q_from_psi!(S, G, plans, a_ell)
```

## From Configuration

Using the high-level API, initial conditions can be specified in the configuration:

```julia
using QGYBJplus

# Create initial condition configuration
init_config = create_initial_condition_config(
    psi_type=:random,     # :analytical, :from_file, :random
    wave_type=:random,    # :zero, :analytical, :from_file, :random
    wave_amplitude=1e-3,
    random_seed=1234,
)

# Use in simulation setup
domain = create_domain_config(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)
strat = create_stratification_config(type=:constant_N)

sim = setup_simulation(domain, strat; initial_conditions=init_config)
```

### Exponentially Decaying Surface Waves (Config)

```julia
using QGYBJplus

init_config = create_initial_condition_config(
    psi_type=:analytical,
    wave_type=:surface_exponential,
    wave_amplitude=0.1,      # u₀ [m/s]
    wave_surface_depth=50.0, # e-folding depth [m]
    wave_uniform=true
)

domain = create_domain_config(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)
strat = create_stratification_config(type=:constant_N)

sim = setup_simulation(domain, strat; initial_conditions=init_config)
```

## From Data Files

### From NetCDF

```julia
using QGYBJplus

par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a_ell = setup_model(par)

# Read initial streamfunction from file
psi_init = read_initial_psi("initial_conditions.nc", G)
S.psi .= psi_init

# Read initial waves from file (optional)
B_init = read_initial_waves("initial_conditions.nc", G)
S.B .= B_init

# Compute derived quantities
compute_q_from_psi!(S, G, plans, a_ell)
compute_velocities!(S, G, plans)
```

### From NetCDF with MPI

For MPI runs, prefer the config-based path so rank 0 reads and the field is
scattered automatically:

```julia
using QGYBJplus

init_config = create_initial_condition_config(
    psi_type=:from_file,
    psi_filename="initial_conditions.nc"
)

domain = create_domain_config(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)
strat = create_stratification_config(type=:constant_N)

sim = setup_simulation(domain, strat; initial_conditions=init_config)
```

Run with `mpirun -n 4 julia --project your_script.jl` to enable MPI.

### Using ncread Functions

For legacy compatibility:

```julia
# Read streamfunction
ncread_psi!(S, G, "psi_file.nc")

# Read wave envelope
ncread_la!(S, G, "waves_file.nc")
```

## Direct Assignment

You can directly assign values in spectral space:

```julia
using QGYBJplus

par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a_ell = setup_model(par)

# Direct assignment (in spectral space)
S.psi .= 0.0  # Zero everywhere
S.B .= 0.0    # No waves

# Or set specific modes
# S.psi[kx_idx, ky_idx, kz] = amplitude

# Always compute q from psi after modifying psi
compute_q_from_psi!(S, G, plans, a_ell)
```

## Complete Example: Spin-Up

For realistic simulations, start with random initialization and spin up:

```julia
using QGYBJplus

# Setup
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.001, nt=10000
)
G, S, plans, a_ell = setup_model(par)

# Initialize flow randomly
init_random_psi!(S, G; amplitude=0.1)
compute_q_from_psi!(S, G, plans, a_ell)

# Spin-up phase (develop turbulence) - no waves
spinup_steps = 1000
first_projection_step!(S, G, par, plans, a_ell)
for step = 2:spinup_steps
    leapfrog_step!(S, G, par, plans, a_ell)
end

# Now add waves
init_analytical_waves!(S, G; amplitude=0.01)

# Production run
for step = 1:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

## Verification

### Check Initial Energy

```julia
# Compute velocities first
compute_velocities!(S, G, plans)

# Check flow kinetic energy
KE = flow_kinetic_energy(S.u, S.v)
println("Initial KE: $KE")

# Check wave energy
WE_B, WE_A = wave_energy(S.B, S.A)
println("Initial Wave Energy (B): $WE_B")
println("Initial Wave Energy (A): $WE_A")
```

### Visualize

```julia
using Plots

# Get horizontal slice
psi_slice = slice_horizontal(S.psi, G, plans; k=G.nz)

heatmap(real(psi_slice), title="Initial Surface ψ", aspect_ratio=1)
```

## API Reference

The following functions are available for initial conditions:

| Function | Description |
|:---------|:------------|
| `init_random_psi!` | Random streamfunction field |
| `init_analytical_psi!` | Analytical streamfunction pattern |
| `init_analytical_waves!` | Analytical wave envelope |
| `add_balanced_component!` | Add balanced component to flow |
| `compute_q_from_psi!` | Compute PV from streamfunction |
| `initialize_from_config` | Initialize from configuration object |
| `read_initial_psi` | Read ψ from NetCDF file |
| `read_initial_waves` | Read B from NetCDF file |
| `ncread_psi!` | Legacy NetCDF read for ψ |
| `ncread_la!` | Legacy NetCDF read for waves |

See the [Grid & State API](../api/grid_state.md) for more details on state initialization.
