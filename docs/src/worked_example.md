# [Worked Example](@id worked_example)

```@meta
CurrentModule = QGYBJplus
```

This example runs a dimensional QG-YBJ+ experiment with a prescribed
barotropic flow and a surface-confined near-inertial wave field.

## Parameters

```julia
using QGYBJplus
using Printf

const Lx = 500e3
const Ly = 500e3
const Lz = 4000.0

const nx = 128
const ny = 128
const nz = 64

const f₀ = 1e-4
const N² = 1e-5
const Δt = 300.0
const inertial_periods = 5
```

## Grid and Model

```julia
grid = RectilinearGrid(size = (nx, ny, nz),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0.0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = f₀),
                   stratification = ConstantStratification(N² = N²),
                   closure = HorizontalHyperdiffusivity(waves = 1e5),
                   flow = :fixed,
                   feedback = :none,
                   ybj_plus = true,
                   Δt = Δt,
                   verbose = true)
```

This uses a fixed balanced flow so the wave response is easy to interpret.
Set `flow = :prognostic` and enable feedback for fully coupled experiments.

## Initial Conditions

Define a simple dipole-like streamfunction and surface wave:

```julia
U = 0.3
R = 50e3

ψᵢ = function (x, y, z)
    r₊² = (x - 75e3)^2 + y^2
    r₋² = (x + 75e3)^2 + y^2
    return U * R * (exp(-r₊² / R^2) - exp(-r₋² / R^2))
end

set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05,
                         scale = 500.0,
                         profile = :gaussian))
```

The streamfunction is dimensional, with units of `m²/s`. The wave amplitude is
a velocity scale in `m/s`.

## Output

Save the full domain and the nearest-surface level:

```julia
save_interval = inertial_period(model)

full_output = NetCDFOutput(path = "output/full",
                           schedule = TimeInterval(save_interval),
                           fields = (:ψ, :waves))

surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(save_interval),
                              fields = (:ψ, :waves),
                              z = 0.0)
```

The surface file stores the actual nearest native grid level in its `z`
coordinate.

## Run

```julia
simulation = Simulation(model;
                        stop_time = inertial_periods * inertial_period(model),
                        output = (full_output, surface_output),
                        diagnostics = IterationInterval(20),
                        verbose = true)

if is_root(simulation)
    @printf("Running %.1f inertial periods\n", inertial_periods)
end

run!(simulation)
finalize_simulation!(simulation)
```

## Read the Results

```julia
using NCDatasets

NCDataset("output/surface/state0001.nc", "r") do ds
    x = ds["x"][:]
    y = ds["y"][:]
    z = ds["z"][:]
    ψ = ds["psi"][:, :, 1]
    LA = ds["LAr"][:, :, 1] .+ im .* ds["LAi"][:, :, 1]

    @info "Loaded surface output" z = z[1] maximum_wave = maximum(abs, LA)
end
```

## Complete Script

```julia
using QGYBJplus
using Printf

const Lx, Ly, Lz = 500e3, 500e3, 4000.0
const nx, ny, nz = 128, 128, 64
const f₀, N² = 1e-4, 1e-5
const Δt = 300.0
const inertial_periods = 5

grid = RectilinearGrid(size = (nx, ny, nz),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0.0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = f₀),
                   stratification = ConstantStratification(N² = N²),
                   closure = HorizontalHyperdiffusivity(waves = 1e5),
                   flow = :fixed,
                   feedback = :none,
                   ybj_plus = true,
                   Δt = Δt)

U, R = 0.3, 50e3
ψᵢ = (x, y, z) -> U * R * (exp(-((x - 75e3)^2 + y^2) / R^2) -
                           exp(-((x + 75e3)^2 + y^2) / R^2))

set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05, scale = 500.0))

save_interval = inertial_period(model)
full_output = NetCDFOutput(path = "output/full",
                           schedule = TimeInterval(save_interval),
                           fields = (:ψ, :waves))
surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(save_interval),
                              fields = (:ψ, :waves),
                              z = 0.0)

simulation = Simulation(model;
                        stop_time = inertial_periods * inertial_period(model),
                        output = (full_output, surface_output),
                        diagnostics = IterationInterval(20))

run!(simulation)
finalize_simulation!(simulation)
```

## Next Steps

- [Configuration](@ref configuration) for model options.
- [I/O and Output](@ref io-output) for NetCDF output.
- [Stratification](@ref stratification) for constant, analytic, and file-backed
  `N²` profiles.
- [MPI Parallelization](@ref parallel) for distributed runs.
