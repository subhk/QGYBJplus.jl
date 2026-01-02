# [Installation & Getting Started](@id getting_started)

```@meta
CurrentModule = QGYBJplus
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

Or develop locally:
```bash
git clone https://github.com/subhk/QGYBJplus.jl
cd QGYBJ+.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### MPI Support (Optional)

```julia
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

Requires system MPI: `brew install open-mpi` (macOS) or `apt install libopenmpi-dev` (Ubuntu).

## Quick Example

```julia
using QGYBJplus

config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # Domain (REQUIRED)
    nx=64, ny=64, nz=32,
    dt=0.001, total_time=1.0
)

result = run_simple_simulation(config)
println("KE: ", flow_kinetic_energy(result.state.u, result.state.v))
```

## Core Types

### QGParams
```julia
params = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # REQUIRED
    nx=64, ny=64, nz=32,
    f₀=1.0, N²=1.0,
    ybj_plus=true
)
```

Unicode: type `f\_0<tab>` → `f₀`, `\nu<tab>` → `ν`

### Grid & State
```julia
grid = init_grid(params)     # Coordinates, wavenumbers
state = init_state(grid)     # Fields: q, B, psi, A, u, v
plans = plan_transforms!(grid)  # FFT plans
```

## Code Structure

```
src/
├── parameters.jl      # QGParams
├── grid.jl            # Grid struct
├── elliptic.jl        # q→ψ, B→A inversions
├── timestep.jl        # Leapfrog
├── timestep_imex.jl   # IMEX-CN (faster)
├── simulation.jl      # High-level API
└── parallel_mpi.jl    # MPI support
```

## What's Next?

- [Quick Start](@ref quickstart) - Tutorial
- [Configuration](@ref configuration) - All parameters
- [MPI Parallelization](@ref parallel) - Large-scale runs
