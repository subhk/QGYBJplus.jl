# [Installation and setup](@id getting_started)

```@meta
CurrentModule = QGYBJplus
```

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

For a local checkout:

```bash
git clone https://github.com/subhk/QGYBJplus.jl
cd QGYBJplus.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

MPI.jl, PencilArrays.jl, and PencilFFTs.jl are package dependencies. Use the
MPI.jl launcher so the configured MPI implementation is selected correctly:

```bash
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl
```

## Verify the installation

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Minimal construction

```julia
using QGYBJplus

grid = RectilinearGrid(size=(32, 32, 16),
                       extent=(100e3, 100e3, 1000.0),
                       centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5),
                   verbose=false)
simulation = Simulation(model; Δt=10.0, stop_iteration=1, output=false)

try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

Model construction initializes MPI when necessary and records that ownership.
Finalization only closes MPI when the model initialized it; an MPI session
created by an application or test runner remains externally owned.

## Next steps

- [Quick start](@ref quickstart)
- [Configuration](@ref configuration)
- [Asselin dipole walkthrough](@ref worked_example)
