# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

This page summarizes the high-level model configuration. For normal scripts,
prefer `RectilinearGrid`, `QGYBJModel`, `Simulation`, and `run!`.

## Grid

Use physical coordinate ranges:

```julia
grid = RectilinearGrid(size = (128, 128, 64),
                       x = (-250e3, 250e3),
                       y = (-250e3, 250e3),
                       z = (-4000.0, 0.0))
```

The vertical grid is cell centered. The surface is `z = 0`, the bottom is
`z = -Lz`, and saved model levels are the native cell centers.

## Model

```julia
model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = ConstantStratification(N² = 1e-5),
                   closure = HorizontalHyperdiffusivity(waves = 1e5),
                   flow = :fixed,
                   feedback = :none,
                   ybj_plus = true,
                   Δt = 300.0,
                   stop_iteration = 100)
```

Common options:

| Keyword | Meaning |
|:--------|:--------|
| `coriolis` | Usually `FPlane(f = f₀)` |
| `stratification` | Constant, analytic, or file-backed `N²` profile |
| `closure` | Horizontal hyperdiffusion settings |
| `flow` | Use `:fixed` for prescribed mean flow |
| `feedback` | Use `:none` for one-way wave evolution |
| `ybj_plus` | Use the YBJ+ wave model |
| `Δt` | Dimensional time step in seconds |

The equations are solved in dimensional form, so parameters should be given in
SI units.

## Initial Conditions

```julia
ψᵢ = (x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3)

set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05, scale = 500.0))
```

`pv_method = :barotropic` computes the barotropic PV from the streamfunction.

## Output

```julia
output = NetCDFOutput(path = "output",
                      schedule = TimeInterval(inertial_period(model)),
                      fields = (:ψ, :waves))
```

Use multiple streams to save different products:

```julia
surface = NetCDFOutput(path = "output/surface",
                       schedule = TimeInterval(inertial_period(model)),
                       fields = (:ψ, :waves),
                       z = 0.0)

simulation = Simulation(model; output = (output, surface))
```

## Low-Level Parameters

`default_params`, `setup_model`, and `exp_rk2_step!` remain available for kernel
tests and numerical-method development. They expose more internal state than
most user scripts need.

```julia
params = default_params(Lx = 500e3, Ly = 500e3, Lz = 4000.0,
                        nx = 64, ny = 64, nz = 32,
                        dt = 300.0, nt = 100)

grid, state, plans, a_ell = setup_model(params)
```

## See Also

- [Quick Start](@ref quickstart)
- [Stratification](@ref stratification)
- [I/O and Output](@ref io-output)
