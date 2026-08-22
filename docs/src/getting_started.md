# [Installation](@id getting_started)

```@meta
CurrentModule = QGYBJplus
```

## Install the package

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

Verify that Julia can load it:

```bash
julia -e 'using QGYBJplus'
```

## Work from a checkout

```bash
git clone https://github.com/subhk/QGYBJplus.jl
cd QGYBJplus.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Run with MPI

Install MPI.jl's launcher once for the selected Julia environment:

```bash
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
```

Then launch the same Julia program on multiple ranks:

```bash
mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl
```

Application code does not need a separate MPI construction path. See [MPI
parallel execution](@ref parallel) for topology and distributed-array details.

Next, follow the [quick start](@ref quickstart).
