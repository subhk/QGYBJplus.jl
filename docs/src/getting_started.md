# [Installation](@id getting_started)

```@meta
CurrentModule = QGYBJplus
```

## Install

After registration, install with:

```julia
using Pkg
Pkg.add("QGYBJplus")
```

For development from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/QGYBJplus.jl")
```

Or clone the repository:

```bash
git clone https://github.com/subhk/QGYBJplus.jl
cd QGYBJplus.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Check the Installation

```julia
using QGYBJplus

grid = RectilinearGrid(size = (16, 16, 8),
                       x = (-50e3, 50e3),
                       y = (-50e3, 50e3),
                       z = (-1000.0, 0.0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = ConstantStratification(N² = 1e-5),
                   flow = :fixed,
                   feedback = :none,
                   Δt = 300.0,
                   stop_iteration = 1,
                   verbose = false)
```

If this runs without error, the package and its Julia dependencies are
available.

## MPI

MPI support is included through `MPI.jl`, `PencilArrays.jl`, and
`PencilFFTs.jl`. To run distributed simulations you also need an MPI
implementation installed on the system, such as OpenMPI or MPICH.

Typical system installs are:

```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt install libopenmpi-dev
```

Then run scripts with `mpiexec`, for example:

```bash
mpiexec -n 4 julia --project=. examples/my_simulation.jl
```

## Where to Go Next

- [Quick Start](@ref quickstart) for the recommended script structure.
- [Configuration](@ref configuration) for model options.
- [I/O and Output](@ref io-output) for NetCDF output.
