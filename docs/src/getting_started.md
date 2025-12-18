# [Installation & Getting Started](@id getting_started)

```@meta
CurrentModule = QGYBJ
```

This page walks you through installing QGYBJ.jl, running a quick example, and understanding the core concepts.

## Installation

### Basic Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJ.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/subhk/QGYBJ.jl
cd QGYBJ.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### Dependencies

| Package | Purpose | Required |
|:--------|:--------|:---------|
| FFTW.jl | FFT transforms | Yes |
| LinearAlgebra | Matrix operations | Yes (stdlib) |
| NCDatasets.jl | NetCDF I/O | Yes (bundled) |

### MPI Support (Optional)

For parallel execution with 2D pencil decomposition:

```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

The MPI extension is automatically loaded when these packages are imported.

## Quick Example

### Serial Mode

```julia
using QGYBJ

# Create configuration (Lx, Ly, Lz are REQUIRED)
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # 500km × 500km × 4km
    nx=64, ny=64, nz=32,
    dt=0.001,
    total_time=1.0,
    output_interval=100
)

# Run simulation
result = run_simple_simulation(config)

# Check results
println("Final KE: ", flow_kinetic_energy(result.state.u, result.state.v))
```

### Parallel Mode (MPI)

```julia
# run_parallel.jl
using MPI, PencilArrays, PencilFFTs, QGYBJ

MPI.Init()
mpi_config = QGYBJ.setup_mpi_environment()

# Setup distributed simulation (Lx, Ly, Lz are REQUIRED)
params = default_params(
    nx=256, ny=256, nz=128,
    Lx=1000e3, Ly=1000e3, Lz=5000.0  # 1000km × 1000km × 5km
)
grid = QGYBJ.init_mpi_grid(params, mpi_config)
state = QGYBJ.init_mpi_state(grid, mpi_config)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# Run time stepping...

MPI.Finalize()
```

Run with:
```bash
mpiexec -n 16 julia --project run_parallel.jl
```

## Core Concepts

### QGParams

All model parameters in one struct:

```julia
params = default_params(
    nx=64, ny=64, nz=32,       # Grid dimensions
    Lx=500e3, Ly=500e3,        # Horizontal domain size [m] (REQUIRED)
    Lz=4000.0,                 # Vertical domain depth [m] (REQUIRED)
    f₀=1.0,                    # Coriolis parameter
    N²=1.0,                    # Buoyancy frequency squared
    dt=0.001,                  # Time step
    νₕ₂=10.0,                  # Hyperdiffusion
    ybj_plus=true              # Use YBJ+ formulation
)
```

Note: Parameter names use Unicode subscripts (e.g., `f₀` instead of `f0`). Type `f\_0<tab>` in Julia REPL to get `f₀`.

### Grid

Spatial coordinates and spectral wavenumbers:

```julia
grid = init_grid(params)

# Physical coordinates
grid.x, grid.y, grid.z     # Coordinate arrays
grid.dx, grid.dy           # Grid spacings

# Spectral wavenumbers
grid.kx, grid.ky           # Wavenumber vectors
grid.kh2                   # Horizontal wavenumber squared

# Parallel decomposition (if using MPI)
grid.decomp                # PencilDecomp or nothing
```

### State

Prognostic and diagnostic fields:

```julia
state = init_state(grid)

# Prognostic (time-stepped)
state.q      # QG potential vorticity (spectral)
state.B      # Wave envelope B = L⁺A (spectral)

# Diagnostic (computed)
state.psi    # Streamfunction (spectral)
state.A      # Wave amplitude (spectral)
state.C      # Vertical derivative dA/dz (spectral)

# Velocities (real space)
state.u, state.v, state.w
```

### FFT Transforms

```julia
# Serial mode
plans = plan_transforms!(grid)
fft_forward!(dst, src, plans)   # Physical → Spectral
fft_backward!(dst, src, plans)  # Spectral → Physical

# Parallel mode (automatic with PencilFFTs)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)
```

## What's Next?

- [Quick Start Tutorial](@ref quickstart) - Hands-on introduction
- [Configuration Guide](@ref configuration) - All parameters explained
- [MPI Parallelization](@ref parallel) - Scale to large domains
- [Physics Overview](@ref physics-overview) - Understand the equations
