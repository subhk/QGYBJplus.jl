# QGYBJ+.jl

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl simulates interactions between quasigeostrophic balanced flow and
near-inertial waves. It combines horizontally pseudo-spectral operators,
second-order vertical differences, MPI pencil decomposition, and a
second-order exponential Runge–Kutta integrator.

## Composition-first interface

Four objects have distinct ownership:

1. [`RectilinearGrid`](@ref) stores immutable global geometry.
2. [`QGYBJModel`](@ref) owns typed physics, numerics, fields, runtime
   resources, and optional particles.
3. [`Simulation`](@ref) owns the clock, ETD-RK2 timestepper, stopping rules,
   schedules, output, and lifecycle.
4. [`NetCDFOutput`](@ref) describes snapshot output without owning model data.

```julia
using QGYBJplus

grid = RectilinearGrid(
    size=(64, 64, 32),
    extent=(500e3, 500e3, 4000.0),
    centered=true,
)
model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    flow=EvolvingFlow(),
    feedback=WaveMeanFeedback(),
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
    stop_time=86400.0,
    output=NetCDFOutput(
        path="output",
        schedule=TimeInterval(3600.0),
    ),
)
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```

ETD-RK2 is the sole stepping scheme. The same model construction works in a
single Julia process and under `mpiexecjl`.

## Documentation map

- [Key concepts](@ref concepts)
- [Installation](@ref getting_started)
- [Quick start](@ref quickstart)
- [Asselin dipole walkthrough](@ref worked_example)
- [Configuration](@ref configuration)
- [MPI parallel execution](@ref parallel)
- [Particle advection](@ref particles)
- [Core API](@ref api-types)

## References

- Asselin & Young (2019), *Journal of Physical Oceanography*, 49, 1699–1717.
- Xie & Vanneste (2015), *Journal of Fluid Mechanics*, 774, 143–169.
- Young & Ben Jelloul (1997), *Journal of Marine Research*, 55, 735–766.
