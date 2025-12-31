# QGYBJ+.jl

```@meta
CurrentModule = QGYBJplus
```

## A Julia Model for Wave-Eddy Interactions in the Ocean

**QGYBJ+.jl** is a high-performance Julia implementation of the Quasi-Geostrophic Young-Ben Jelloul Plus (QG-YBJ+) model for simulating the interaction between near-inertial waves and mesoscale ocean eddies.

!!! tip "New to QGYBJ+.jl?"
    Start with the [Quick Start Tutorial](@ref quickstart) for a hands-on introduction.

## What This Model Does

The ocean contains two important types of motion at different scales:

1. **Mesoscale eddies** (~100 km): Long-lived rotating vortices that dominate ocean kinetic energy
2. **Near-inertial waves** (~10 km): Wind-generated internal waves oscillating near the Coriolis frequency

These two types of motion **interact strongly**: eddies refract and focus waves, while waves can feed energy back into the mean flow. QGYBJ+.jl simulates this coupled system using:

- **Quasi-geostrophic (QG) dynamics** for the balanced eddy flow
- **YBJ+ equations** for the near-inertial wave envelope
- **Two-way coupling** capturing wave-mean flow energy exchange

## Key Features

| Feature | Description |
|:--------|:------------|
| **Spectral Methods** | Pseudo-spectral horizontal derivatives with FFTW |
| **Vertical Solvers** | Efficient tridiagonal solvers for elliptic inversions |
| **2D Pencil Decomposition** | Scalable MPI parallelization with PencilArrays/PencilFFTs |
| **Dual Pencil Configurations** | Automatic xy-pencil ↔ z-pencil transposes for vertical operations |
| **Particle Tracking** | Lagrangian advection with multiple interpolation schemes |
| **Flexible Physics** | Configurable stratification, dissipation, and wave feedback |
| **NetCDF I/O** | Standard output format for analysis |

## Quick Example

```julia
using QGYBJplus

# Create configuration (Lx, Ly, Lz are REQUIRED)
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # 500km × 500km × 4km (required)
    nx=64, ny=64, nz=32,             # Grid size
    dt=0.001,                         # Time step
    total_time=1.0,                   # Simulation duration
    output_interval=100               # Output frequency
)

# Run simulation
result = run_simple_simulation(config)

# Access results
psi = result.state.psi   # Streamfunction
B = result.state.B       # Wave envelope
```

## Documentation Sections

### Getting Started
- [Installation](@ref getting_started) - How to install QGYBJ+.jl
- [Quick Start Tutorial](@ref quickstart) - Your first simulation in 5 minutes
- [Worked Example](@ref worked_example) - Detailed walkthrough with explanations

### Physics & Theory
- [Model Overview](@ref physics-overview) - Physical background and equations
- [QG Equations](@ref qg-equations) - Quasi-geostrophic dynamics
- [YBJ+ Wave Model](@ref ybj-plus) - Near-inertial wave formulation
- [Numerical Methods](@ref numerical-methods) - Algorithms and discretization

### User Guide
- [Configuration](@ref configuration) - Setting up simulations
- [Stratification](@ref stratification) - Ocean density profiles
- [I/O and Output](@ref io-output) - Saving and loading data
- [Diagnostics](@ref diagnostics) - Energy and analysis tools

### Advanced Topics
- [MPI Parallelization](@ref parallel) - Running on clusters
- [Particle Advection](@ref particles) - Lagrangian tracking
- [Performance Tips](@ref performance) - Optimization strategies

### API Reference
- [Core Types](@ref api-types) - QGParams, Grid, State
- [Physics Functions](@ref api-physics) - Operators and solvers
- [Full Index](@ref api-index) - Complete function listing and codebase structure

## Installation

### Basic Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

### With MPI Support

```julia
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

### NetCDF Support

NCDatasets.jl is included as a dependency for NetCDF I/O functionality. No additional installation is required.

## Citation

If you use QGYBJ+.jl in your research, please cite:

```bibtex
@software{qgybj_jl,
  author = {Kar, Subhajit},
  title = {QGYBJ+.jl: A Julia Implementation of the QG-YBJ+ Model},
  year = {2024},
  url = {https://github.com/subhk/QGYBJplus.jl}
}
```

## Key References

- **Asselin & Young (2019)**: YBJ+ formulation for penetration of near-inertial waves
- **Xie & Vanneste (2015)**: Wave feedback mechanism (qʷ term)
- **Young & Ben Jelloul (1997)**: Original YBJ wave envelope equation

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/subhk/QGYBJplus.jl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/subhk/QGYBJplus.jl/discussions)

## License

QGYBJ+.jl is released under the MIT License.
