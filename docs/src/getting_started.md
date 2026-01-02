# [Installation & Getting Started](@id getting_started)

```@meta
CurrentModule = QGYBJplus
```

This page walks you through installing QGYBJ+.jl, running a quick example, and understanding the core concepts.

## Installation

### Basic Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/subhk/QGYBJplus.jl
cd QGYBJ+.jl
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
using QGYBJplus

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
using MPI, PencilArrays, PencilFFTs, QGYBJplus

MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()

# Setup distributed simulation (Lx, Ly, Lz are REQUIRED)
params = default_params(
    nx=256, ny=256, nz=128,
    Lx=1000e3, Ly=1000e3, Lz=5000.0  # 1000km × 1000km × 5km
)
grid = QGYBJplus.init_mpi_grid(params, mpi_config)
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
state = QGYBJplus.init_mpi_state(grid, plans, mpi_config)
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)

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
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
```

## Codebase Overview

Understanding the code organization helps when debugging or extending the model:

```
src/
├── QGYBJplus.jl       # Main module (all exports)
├── parameters.jl      # QGParams struct (all configuration)
├── grid.jl            # Grid struct (coordinates, wavenumbers)
├── transforms.jl      # FFT planning and execution
├── physics.jl         # Stratification, N² profiles
├── elliptic.jl        # q→ψ and B→A inversions
├── operators.jl       # Velocity computation
├── nonlinear.jl       # Jacobians, wave feedback
├── timestep.jl        # Leapfrog time stepping
├── timestep_imex.jl   # IMEX-CNAB (implicit dispersion)
├── simulation.jl      # High-level Simulation API
├── parallel_mpi.jl    # MPI 2D decomposition
├── netcdf_io.jl       # NetCDF I/O
├── diagnostics.jl     # Energy diagnostics
└── particles/         # Lagrangian particle tracking
    ├── particle_advection.jl
    ├── particle_config.jl
    └── interpolation_schemes.jl
```

**Key entry points:**
- `default_params()` → Create model parameters
- `setup_model()` → Initialize Grid, State, FFT plans
- `Simulation` / `run!()` → High-level simulation API
- `leapfrog_step!()` / `imex_cn_step!()` → Low-level time stepping

See [API Reference](@ref api-index) for the complete module structure.

## What's Next?

- [Quick Start Tutorial](@ref quickstart) - Hands-on introduction
- [Configuration Guide](@ref configuration) - All parameters explained
- [MPI Parallelization](@ref parallel) - Scale to large domains
- [Physics Overview](@ref physics-overview) - Understand the equations

---

## Troubleshooting

### Installation Issues

**Problem**: `Package not found` error
```
ERROR: Package QGYBJplus not found in registry
```

**Solution**: Install directly from GitHub:
```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

**Problem**: MPI packages fail to install
```
ERROR: MPI.jl requires a working MPI installation
```

**Solution**: Install system MPI first:
- **macOS**: `brew install open-mpi`
- **Ubuntu**: `sudo apt install libopenmpi-dev openmpi-bin`
- **HPC**: Load the MPI module: `module load openmpi`

Then reinstall:
```julia
Pkg.build("MPI")
```

### Runtime Issues

**Problem**: `MethodError: no method matching default_params(...)`

**Solution**: Check that you're providing `Lx`, `Ly`, `Lz` (required):
```julia
# Wrong - missing domain size
par = default_params(nx=64, ny=64, nz=32)

# Correct - include domain size
par = default_params(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)
```

**Problem**: Simulation blows up (NaN values)

**Solutions**:
1. Reduce time step: `dt = dt / 2`
2. Increase hyperdiffusion: `νₕ₁ = νₕ₁ * 10`
3. Check initial conditions for extreme values
4. Use IMEX time stepping for wave-dominated problems

**Problem**: Out of memory

**Solutions**:
1. Reduce grid size
2. Use MPI parallelization for distributed memory
3. Use Float32 precision: `default_params(..., T=Float32)`

### Unicode Characters

**Problem**: Can't type Greek letters like `f₀` or `ν`

**Solution**: In Julia REPL, type the LaTeX name and press Tab:
- `f\_0<Tab>` → `f₀`
- `\nu<Tab>` → `ν`
- `N\^2<Tab>` → `N²`

Or use ASCII alternatives:
```julia
par = default_params(f0=1e-4, N2=1e-5, ...)  # Some functions accept ASCII
```

### Performance Issues

**Problem**: Simulation is slow

**Solutions**:
1. Use IMEX time stepping (10x faster for waves)
2. Run with multiple threads: `julia -t auto`
3. Use MPI for large grids: `mpiexec -n 16 julia script.jl`
4. Check you're not running in debug mode

See [Performance Tips](@ref performance) for detailed optimization strategies.
