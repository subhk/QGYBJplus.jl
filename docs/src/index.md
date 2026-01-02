# QGYBJ+.jl

```@meta
CurrentModule = QGYBJplus
```

## A Julia Model for Wave-Eddy Interactions in the Ocean

**QGYBJ+.jl** is a high-performance Julia implementation of the Quasi-Geostrophic Young-Ben Jelloul Plus (QG-YBJ+) model for simulating the interaction between near-inertial waves and mesoscale ocean eddies.

---

## Start Here

New to QGYBJ+.jl? Follow this learning path:

| Step | Page | Time | What You'll Learn |
|:-----|:-----|:-----|:------------------|
| 1 | [Key Concepts](@ref concepts) | 15 min | What QG-YBJ+ models, why it matters, core terminology |
| 2 | [Quick Start](@ref quickstart) | 5 min | Run your first simulation with copy-paste code |
| 3 | [Worked Example](@ref worked_example) | 30 min | Build a real simulation step-by-step with explanations |

---

## What This Model Does

The ocean contains two important types of motion at different scales:

**Mesoscale Eddies** (~100 km scale)
- Giant rotating vortices lasting weeks to months
- Contain ~90% of ocean kinetic energy
- Move slowly (~10 cm/s)

**Near-Inertial Waves** (~10 km scale)
- Wind-generated internal waves oscillating every ~17 hours
- Key driver of ocean mixing
- Propagate through the water column

These two types of motion **interact strongly**: eddies refract and focus waves, while waves can feed energy back into the mean flow. QGYBJ+.jl simulates this coupled system using:

- **Quasi-geostrophic (QG) dynamics** for the balanced eddy flow
- **YBJ+ equations** for the near-inertial wave envelope
- **Two-way coupling** capturing wave-mean flow energy exchange

## Key Features

| Feature | Description |
|:--------|:------------|
| **Spectral Methods** | Pseudo-spectral horizontal derivatives with FFTW |
| **Vertical Solvers** | Efficient tridiagonal solvers for elliptic inversions |
| **Two Time Steppers** | Leapfrog (explicit) or IMEX-CN (10x faster for waves) |
| **MPI Parallel** | 2D pencil decomposition for large domains |
| **Particle Tracking** | Lagrangian advection with multiple interpolation schemes |
| **Flexible Physics** | Configurable stratification, dissipation, and wave feedback |
| **NetCDF I/O** | Standard output format for analysis |

## Minimal Example

```julia
using QGYBJplus

# Create configuration (domain size is REQUIRED)
config = create_simple_config(
    Lx = 500e3, Ly = 500e3, Lz = 4000.0,  # 500km × 500km × 4km
    nx = 64, ny = 64, nz = 32,
    dt = 0.001, total_time = 1.0
)

# Run simulation
result = run_simple_simulation(config)

# Check energy
println("Kinetic Energy: ", flow_kinetic_energy(result.state.u, result.state.v))
```

---

## Documentation Guide

### For Beginners
- [Key Concepts](@ref concepts) - **Start here!** Core ideas without code
- [Installation](@ref getting_started) - How to install QGYBJ+.jl
- [Quick Start](@ref quickstart) - Your first simulation in 5 minutes
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

### Reference
- [Core Types](@ref api-types) - QGParams, Grid, State
- [Physics Functions](@ref api-physics) - Operators and solvers
- [Full Index](@ref api-index) - Complete function listing

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

For MPI parallel support:
```julia
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

See [Installation Guide](@ref getting_started) for detailed instructions.

---

## How Simulations Work

The typical simulation workflow is:

1. **Configure** - Create `QGParams` with grid size, domain, physics options
2. **Setup** - Initialize `Grid`, `State`, FFT plans, and elliptic coefficients
3. **Run** - Time-step the prognostic fields (q, B) with inversions each step
4. **Output** - Save fields and diagnostics to NetCDF files

See [Worked Example](@ref worked_example) for a complete walkthrough.

---

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
