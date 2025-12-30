# [API Reference](@id api-index)

```@meta
CurrentModule = QGYBJplus
```

Complete API reference for QGYBJ+.jl.

## Quick Links

- [Core Types](types.md): `QGParams`, `Grid`, `State`
- [Grid & State](grid_state.md): Initialization and utilities
- [Physics Functions](physics.md): Inversions, operators, diagnostics
- [Time Stepping](timestepping.md): Leapfrog and IMEX-CNAB integration
- [Particles](particles.md): Lagrangian particle tracking

## Module Structure

```
QGYBJplus/src/
├── QGYBJplus.jl       # Main module, exports
├── Core Types
│   ├── parameters.jl  # QGParams struct
│   ├── grid.jl        # Grid struct, wavenumbers
│   └── config.jl      # Configuration types
├── Physics
│   ├── physics.jl     # Stratification N², a_ell coefficients
│   ├── elliptic.jl    # Tridiagonal inversions (q→ψ, B→A)
│   ├── nonlinear.jl   # Jacobians, refraction, qw feedback
│   ├── operators.jl   # Velocity computation
│   └── ybj_normal.jl  # Normal YBJ operators
├── Transforms
│   ├── transforms.jl  # FFT planning (serial)
│   └── parallel_mpi.jl # MPI 2D decomposition, transposes
├── Time Stepping
│   ├── timestep.jl       # Leapfrog with Robert-Asselin filter
│   └── timestep_imex.jl  # IMEX-CNAB with Strang splitting
├── Initialization
│   ├── initconds.jl       # Random/analytic initial conditions
│   ├── initialization.jl  # Field initialization helpers
│   └── stratification.jl  # Stratification profiles
├── Diagnostics
│   ├── diagnostics.jl        # Energy, omega equation
│   └── energy_diagnostics.jl # Separate energy output files
├── I/O
│   └── netcdf_io.jl   # NetCDF read/write
├── High-Level Interface
│   ├── model_interface.jl # QGYBJSimulation, run_simulation!
│   ├── simulation.jl      # Simulation API (initialize_simulation, run!)
│   └── runtime.jl         # Setup helpers
├── Particles
│   ├── particle_advection.jl  # Core advection
│   ├── particle_config.jl     # Configuration types
│   ├── particle_io.jl         # Trajectory I/O
│   ├── interpolation_schemes.jl # Trilinear, tricubic, etc.
│   └── halo_exchange.jl       # MPI halo exchange
└── Utilities
    └── pretty_printing.jl # Display formatting
```

## Naming Conventions

| Suffix | Meaning | Example |
|:-------|:--------|:--------|
| `!` | In-place modification | `compute_velocities!` |
| `_spectral` | Operates in spectral space | `jacobian_spectral!` |
| `_waqg` | Wave-related | `convol_waqg!` |
| `_mpi` | MPI-enabled version | `init_mpi_grid` |

## Main Entry Points

### Setup

```julia
# Create parameters (domain size REQUIRED)
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)

# Initialize everything at once
G, S, plans, a = setup_model(par)
```

### Time Stepping

```julia
# Leapfrog (default, explicit)
first_projection_step!(S, G, par, plans; a=a, dealias_mask=L)
leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a, dealias_mask=L)

# IMEX-CNAB (implicit dispersion, 10x larger timestep)
imex_ws = init_imex_workspace(S, G)
first_imex_step!(S, G, par, plans, imex_ws; a=a, dealias_mask=L)
imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a=a, dealias_mask=L)
```

### MPI Parallel Mode

```julia
using MPI, PencilArrays, PencilFFTs, QGYBJplus
MPI.Init()
mpi_config = setup_mpi_environment()
G = init_mpi_grid(par, mpi_config)
plans = plan_mpi_transforms(G, mpi_config)
S = init_mpi_state(G, plans, mpi_config)
workspace = init_mpi_workspace(G, mpi_config)
```

## Key Functions by Category

### Initialization
| Function | Description |
|:---------|:------------|
| `default_params` | Create model parameters |
| `init_grid` / `init_mpi_grid` | Initialize grid |
| `init_state` / `init_mpi_state` | Initialize state arrays |
| `plan_transforms!` / `plan_mpi_transforms` | Create FFT plans |
| `init_mpi_workspace` | Allocate z-pencil workspace |

### Physics
| Function | Description |
|:---------|:------------|
| `invert_q_to_psi!` | Solve elliptic PV inversion |
| `invert_B_to_A!` | Solve YBJ+ wave inversion |
| `compute_velocities!` | Compute u, v from ψ |
| `jacobian_spectral!` | Compute Jacobian J(a,b) |

### Time Stepping
| Function | Description |
|:---------|:------------|
| `first_projection_step!` | Forward Euler initialization (Leapfrog) |
| `leapfrog_step!` | Leapfrog with Robert-Asselin filter |
| `init_imex_workspace` | Allocate IMEX workspace arrays |
| `first_imex_step!` | Forward Euler initialization (IMEX) |
| `imex_cn_step!` | IMEX-CNAB with Strang splitting |

### Parallel Utilities
| Function | Description |
|:---------|:------------|
| `allocate_fft_backward_dst` | Allocate physical array for FFT output |
| `get_local_range_physical` | Get physical array local ranges |
| `get_local_range_spectral` | Get spectral array local ranges |
| `transpose_to_z_pencil!` | Transpose to z-local layout |
| `transpose_to_xy_pencil!` | Transpose back to xy layout |

### Diagnostics
| Function | Description |
|:---------|:------------|
| `flow_kinetic_energy` | Compute mean flow KE |
| `wave_energy` | Compute wave energy |
| `slice_horizontal` | Extract horizontal slice |
| `omega_eqn_rhs!` | Compute omega equation RHS |

See individual pages for detailed API documentation.
