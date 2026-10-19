# [CPU and GPU execution](@id architectures)

```@meta
CurrentModule = QGYBJplus
```

Every array operation in QGYBJ+.jl is a
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernel, so the same model code runs on the CPU and on a GPU. The architecture
is chosen once, when the grid is built, in the same way as in Oceananigans.jl:

```julia
grid = RectilinearGrid(CPU(); size=(128, 128, 64), extent=(500e3, 500e3, 4000.0))
```

`CPU()` is the default, so `RectilinearGrid(size=..., extent=...)` is a CPU
grid. A model built on a grid allocates its fields, transform plans, scratch
buffers, coefficient profiles, and dealiasing mask on the grid's architecture.

## Running on a GPU

Load CUDA.jl and pass `GPU()` to the grid:

```julia
using CUDA
using QGYBJplus

grid = RectilinearGrid(GPU(); size=(256, 256, 64), extent=(500e3, 500e3, 4000.0))
model = QGYBJModel(grid=grid, coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))
```

Nothing else in a script changes: `set!`, `Simulation`, `run!`, NetCDF output,
energy diagnostics, `restore!`, and particles work as they do on the CPU.
Model arrays are `CuArray`s, the kernels run on `CUDA.CUDABackend()`, and the
horizontal transforms use cuFFT.

`GPU()` uses the current CUDA device. Choose another device before building
the grid with `CUDA.device!(n)`. One model runs on one GPU; multi-GPU
decomposition is not supported. To fit a larger grid, or to run faster on a
card with reduced double-precision throughput, build the grid in single
precision, `RectilinearGrid(GPU(), Float32; ...)` (see [Performance](@ref
performance)).

Any other KernelAbstractions backend can be passed explicitly as
`GPU(backend)`. Its arrays must support AbstractFFTs plans over dimensions
`(2, 3)`; otherwise add a method of
`QGYBJplus.Transforms.horizontal_fft_plans` for that array type, as in
[Testing without a GPU](@ref testing-without-gpu).

Kernels run on the backend that their arrays report
(`KernelAbstractions.get_backend`). For CUDA that is `CUDABackend()` with
default options, so launch options such as
`GPU(CUDABackend(; prefer_blocks=true))` are not used, and a warning says so.

KernelAbstractions also exports names `CPU` and `GPU`. A script that loads
both packages with `using` must qualify them, as in `QGYBJplus.GPU()`.

## Multithreaded CPU runs

On the CPU, kernels run on KernelAbstractions' CPU backend and use Julia's
threads. Start Julia with several threads to use them:

```bash
julia --threads=8 --project=. examples/asselin_jpo2020.jl
```

FFTW keeps its own thread count, which defaults to one. Plans are made when
the model is built, so set it first to thread the transforms as well:

```julia
using FFTW
FFTW.set_num_threads(Threads.nthreads())
model = QGYBJModel(grid=grid)
```

## What runs where

| Work | Location |
|:--|:--|
| Time stepping, elliptic solves, transforms, velocities, energy sums | architecture of the grid |
| Initial-condition functions passed to `set!`, `FieldFile` reads | host, then copied to the device |
| NetCDF output and restart files | device data copied to the host when written |
| Particle positions, the velocity that advects them, and the advection kernel | architecture of the grid; positions are copied to the host when output or the in-memory history records them |
| Grid coordinates (`grid.x`, `grid.kx`, …) | host |

User functions passed to `set!`, such as `ψ=(x, y, z) -> ...`, are evaluated
on the host, so they need not be GPU-compatible.

## Working with model arrays

Copy device data to the host with `Array` before indexing it element by
element, and use `on_architecture` to move arrays in either direction:

```julia
q_host = Array(model.fields.q)                     # host copy of a device field
mask = on_architecture(architecture(model), dealias_mask(model.grid))
```

Whole-array operations such as `maximum(abs, model.fields.B)`, `sum`, `fill!`,
and `copyto!` work directly on device arrays. Indexing a single element of a
device array from the host is disallowed by CUDA.jl outside interactive use.

## [Testing without a GPU](@id testing-without-gpu)

The test suite runs the device code path on the CPU with JLArrays, the
reference GPU array implementation, which rejects scalar indexing. JLArrays has
no FFT library, so the tests supply plans that round-trip through FFTW:

```julia
using JLArrays, FFTW, QGYBJplus

struct HostRoundTripPlan{P, B}
    plan::P
    buffer::B
end

function Base.:*(plan::HostRoundTripPlan, x::JLArray)
    copyto!(plan.buffer, x)
    plan.plan * plan.buffer
    return copyto!(x, plan.buffer)
end

function QGYBJplus.Transforms.horizontal_fft_plans(buffer::JLArray{<:Complex, 3})
    host = Array(buffer)
    return (HostRoundTripPlan(FFTW.plan_fft!(host, (2, 3)), host),
            HostRoundTripPlan(FFTW.plan_ifft!(host, (2, 3)), host))
end

grid = RectilinearGrid(GPU(JLBackend()); size=(16, 16, 8), extent=(2π, 2π, 1.0))
model = QGYBJModel(grid=grid)
```

`test/test_architectures.jl` checks that such a model reproduces the CPU model
for every wave formulation, including output, diagnostics, restart, and
particles, and that no kernel is handed a host array. It cannot exercise
cuFFT or a GPU math library, so run a short CUDA simulation before relying on
a new GPU setup.
