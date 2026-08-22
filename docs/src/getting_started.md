# [Installation](@id getting_started)

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
cd QGYBJplus.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### MPI Support

MPI parallel packages (MPI.jl, PencilArrays.jl, PencilFFTs.jl) are included as dependencies and installed automatically.

To run MPI simulations, you need a system MPI library:
- **macOS**: `brew install open-mpi`
- **Ubuntu**: `apt install libopenmpi-dev`

## What's Next?

- [Quick Start](@ref quickstart) — first simulation
- [Configuration](@ref configuration) — all parameters
- [Core Types](@ref api-types) — `QGParams`, `Grid`, `State`
- [MPI Parallelization](@ref parallel) — large-scale runs
