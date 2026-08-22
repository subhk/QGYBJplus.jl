# QGYBJ+.jl

```@meta
CurrentModule = QGYBJplus
```

**QGYBJ+.jl** implements the Quasi-Geostrophic Young–Ben Jelloul Plus (QG-YBJ+) model for the interaction between near-inertial waves and mesoscale ocean eddies.

## Start Here

1. [Key Concepts](@ref concepts) — what the model simulates, no code
2. [Installation](@ref getting_started) — install and MPI setup
3. [Quick Start](@ref quickstart) — first simulation, copy-paste
4. [Worked Example](@ref worked_example) — full simulation, step by step

## What This Model Does

Mesoscale eddies (~100 km, weeks to months) and near-inertial waves (~10 km, ~17 hours) interact strongly: eddies refract and focus waves, and waves feed energy back into the mean flow. QGYBJ+.jl solves this coupled system with quasi-geostrophic dynamics for the balanced flow, the YBJ+ equations for the wave envelope, and two-way wave–mean coupling.

## Features

- Pseudo-spectral horizontal derivatives (FFTW)
- ETD-RK2 time stepping with exact horizontal hyperdiffusion
- MPI parallelism via 2D pencil decomposition
- Lagrangian particle tracking with several interpolation schemes
- Configurable stratification, dissipation, and wave feedback
- NetCDF output with energy diagnostics

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

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

MPI support (MPI.jl, PencilArrays.jl, PencilFFTs.jl) comes as a dependency and installs automatically; see [Installation](@ref getting_started) for the system MPI library.

## Key References

- **Asselin & Young (2019)** — YBJ+ formulation for near-inertial wave penetration
- **Xie & Vanneste (2015)** — wave feedback mechanism (qʷ term)
- **Young & Ben Jelloul (1997)** — original YBJ wave envelope equation

## Citation

```bibtex
@software{qgybj_jl,
  author = {Kar, Subhajit},
  title = {QGYBJ+.jl: A Julia Implementation of the QG-YBJ+ Model},
  year = {2025},
  url = {https://github.com/subhk/QGYBJplus.jl}
}
```

## Getting Help

[GitHub Issues](https://github.com/subhk/QGYBJplus.jl/issues) · [GitHub Discussions](https://github.com/subhk/QGYBJplus.jl/discussions)

## License

MIT.
