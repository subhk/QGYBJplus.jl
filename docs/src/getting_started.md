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

## Run on several CPU threads

Kernels use Julia's threads on the CPU:

```bash
julia --threads=auto --project=. examples/asselin_jpo2020.jl
```

## Run on a GPU

Add CUDA.jl to the environment and pass `GPU()` to the grid:

```julia
using Pkg
Pkg.add("CUDA")

using CUDA, QGYBJplus
grid = RectilinearGrid(GPU(); size=(128, 128, 64), extent=(500e3, 500e3, 4000.0))
```

The rest of a script is unchanged. See [CPU and GPU execution](@ref
architectures).

Next, follow the [quick start](@ref quickstart).
