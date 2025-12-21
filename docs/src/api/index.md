# [API Reference](@id api-index)

```@meta
CurrentModule = QGYBJplus
```

Complete API reference for QGYBJ+.jl.

## Quick Links

- [Core Types](types.md): `QGParams`, `Grid`, `State`
- [Grid & State](grid_state.md): Initialization and utilities
- [Physics Functions](physics.md): Inversions, operators, diagnostics
- [Time Stepping](timestepping.md): Leapfrog integration
- [Particles](particles.md): Lagrangian particle tracking

## Module Structure

```
QGYBJplus
├── Core Types
│   ├── QGParams      # Model parameters
│   ├── Grid          # Spatial grid and wavenumbers
│   └── State         # Prognostic/diagnostic fields
├── Physics
│   ├── elliptic.jl   # Tridiagonal inversions
│   ├── nonlinear.jl  # Jacobians, refraction, qw
│   ├── operators.jl  # Velocities
│   └── transforms.jl # FFT wrappers
├── Time Stepping
│   └── timestep.jl   # Leapfrog with Robert-Asselin
├── YBJ Normal Mode
│   └── ybj_normal.jl # sumB!, compute_sigma, compute_A!
├── Diagnostics
│   └── diagnostics.jl # Energy, omega equation
├── Particles
│   └── particles/    # Lagrangian tracking
└── I/O
    └── netcdf_io.jl  # NetCDF output
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
# Initial projection step
first_projection_step!(S, G, par, plans; a=a, dealias_mask=L)

# Leapfrog steps
leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a, dealias_mask=L)
```

### MPI Parallel Mode

```julia
using MPI, PencilArrays, PencilFFTs, QGYBJplus
MPI.Init()
mpi_config = setup_mpi_environment()
G = init_mpi_grid(par, mpi_config)
S = init_mpi_state(G, mpi_config)
```

See individual pages for detailed API documentation.
