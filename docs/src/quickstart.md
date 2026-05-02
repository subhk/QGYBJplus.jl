# [Quick Start](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

This page shows the recommended high-level workflow. It follows the same style
as Oceananigans: build a grid, build a model, set initial conditions, create a
simulation, then call `run!`.

## Minimal Script

```julia
using QGYBJplus

grid = RectilinearGrid(size = (64, 64, 32),
                       x = (-250e3, 250e3),
                       y = (-250e3, 250e3),
                       z = (-4000.0, 0.0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = ConstantStratification(N² = 1e-5),
                   closure = HorizontalHyperdiffusivity(waves = 1e5),
                   flow = :fixed,
                   feedback = :none,
                   Δt = 300.0,
                   stop_iteration = 10)

ψᵢ = (x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3)

set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05, scale = 500.0))

output = NetCDFOutput(path = "output",
                      schedule = TimeInterval(inertial_period(model)),
                      fields = (:ψ, :waves))

simulation = Simulation(model;
                        stop_time = 2inertial_period(model),
                        output = output,
                        diagnostics = IterationInterval(10))

run!(simulation)
finalize_simulation!(simulation)
```

## What the Script Does

- `RectilinearGrid` defines the physical domain in meters.
- `QGYBJModel` defines the Coriolis frequency, stratification, closure, and
  coupling choices.
- `set!` initializes the balanced flow and the near-inertial wave field.
- `NetCDFOutput` controls what is written and how often.
- `Simulation` controls the run clock.

The prognostic equations are dimensional. Use physical values for lengths,
times, Coriolis frequency, stratification, and diffusivities.

## Saving Surface Output

To save only the level nearest the surface:

```julia
surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(inertial_period(model)),
                              fields = (:ψ, :waves),
                              z = 0.0)
```

To save both the full domain and the surface level:

```julia
full_output = NetCDFOutput(path = "output/full",
                           schedule = TimeInterval(inertial_period(model)),
                           fields = (:ψ, :waves))

simulation = Simulation(model;
                        output = (full_output, surface_output),
                        stop_time = 2inertial_period(model))
```

## Next Steps

- [Running Simulations](@ref running) explains runtime options.
- [I/O and Output](@ref io-output) explains NetCDF files.
- [Stratification](@ref stratification) explains constant, analytic, and
  file-backed profiles.
- [MPI Parallelization](@ref parallel) explains distributed runs.
