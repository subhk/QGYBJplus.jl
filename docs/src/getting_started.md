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

### MPI Support

MPI parallel packages (MPI.jl, PencilArrays.jl, PencilFFTs.jl) are included as dependencies and installed automatically.

To run MPI simulations, you need a system MPI library:
- **macOS**: `brew install open-mpi`
- **Ubuntu**: `apt install libopenmpi-dev`

## Quick Example

```julia
using QGYBJplus

grid = RectilinearGrid(size=(64, 64, 32),
                       extent=(500e3, 500e3, 4000.0), centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))

set!(model;
     ψ=(x, y, z) -> 1e3 * sin(2π*x/500e3) * cos(2π*y/500e3),
     waves=SurfaceWave(amplitude=0.1, scale=30.0))

simulation = Simulation(model; Δt=20.0, stop_time=86400.0,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(3600.0)))
run!(simulation)
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
├── loop_macros.jl     # Spectral loop helpers
├── timestep.jl        # ETD-RK2
├── simulation.jl      # High-level API
└── parallel_mpi.jl    # MPI support
```

## What's Next?

- [Quick Start](@ref quickstart) - Tutorial
- [Configuration](@ref configuration) - All parameters
- [MPI Parallelization](@ref parallel) - Large-scale runs
