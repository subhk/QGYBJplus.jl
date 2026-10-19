# QGYBJ+.jl

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl simulates interactions between quasigeostrophic balanced flow and
near-inertial waves. It combines horizontally pseudo-spectral operators,
second-order vertical differences, and ETD-RK2 time integration, with every
array operation written as a KernelAbstractions.jl kernel so that the same
model runs on CPUs and GPUs.

Start with [installation](@ref getting_started), then run the
[quick-start example](@ref quickstart). The [Asselin dipole walkthrough](@ref
worked_example) is the complete research-scale example.

## Minimal workflow

```julia
using QGYBJplus

grid = RectilinearGrid(
    size=(32, 32, 16),
    extent=(500e3, 500e3, 4000.0),
    centered=true,
)
model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    inviscid=Inviscid(),
    flow=EvolvingFlow(),
    feedback=NoWaveFeedback(),
    formulation=YBJPlus(),
)
set!(
    model;
    ψ=(x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3),
    waves=SurfaceWave(amplitude=0.1, scale=30.0),
)

simulation = Simulation(
    model;
    Δt=20.0,
    stop_iteration=10,
    output=NetCDFOutput(
        path="output",
        schedule=IterationInterval(5),
    ),
)
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

To run the same script on a GPU, load CUDA.jl and build the grid with
`RectilinearGrid(GPU(); ...)`. ETD-RK2 is the sole production stepping scheme.

## What owns what

1. [`RectilinearGrid`](@ref) stores immutable global geometry and the
   architecture (`CPU()` or `GPU()`).
2. [`QGYBJModel`](@ref) owns physics, numerics, fields, runtime resources, and
   optional particles.
3. [`Simulation`](@ref) owns the clock, timestepper, stopping rules, output,
   diagnostics, and lifecycle.

## Documentation map

- [Installation](@ref getting_started)
- [Quick start](@ref quickstart)
- [Key concepts](@ref concepts)
- [Asselin dipole walkthrough](@ref worked_example)
- [Configuration](@ref configuration)
- [Physics overview](@ref physics-overview)
- [CPU and GPU execution](@ref architectures)
- [Particle advection](@ref particles)
- [Core API](@ref api-types)
