# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

Diagnostics are available as Julia functions and, for normal simulations, as
NetCDF output fields.

## Flow Kinetic Energy

```julia
compute_velocities!(state, grid; plans, params)
KE = flow_kinetic_energy(state.u, state.v)
```

The balanced horizontal velocities are derived from the streamfunction:

```math
u = -\partial_y \psi, \qquad v = \partial_x \psi.
```

## Wave Energy

The wave kinetic energy uses the physical wave velocity envelope `LA`:

```julia
WE = wave_energy_vavg(state.L⁺A, state.A, grid, plans)
```

In spectral space,

```math
LA = L^+A + \frac{k_h^2}{4} A.
```

This is the same quantity written to NetCDF as `LAr` and `LAi`.

## Output Diagnostics

State files contain the fields needed for most post-processing:

```julia
output = NetCDFOutput(path = "output",
                      schedule = TimeInterval(inertial_period(model)),
                      fields = (:ψ, :waves, :velocities),
                      velocities = true)
```

Read them with `NCDatasets.jl`:

```julia
using NCDatasets

NCDataset("output/state0001.nc", "r") do ds
    ψ = ds["psi"][:, :, :]
    LA = ds["LAr"][:, :, :] .+ im .* ds["LAi"][:, :, :]
    z = ds["z"][:]
end
```

## Horizontal and Vertical Slices

For quick analysis in Julia:

```julia
ψ_surface = slice_horizontal(state.psi, grid, plans; k = grid.nz)
ψ_xz = slice_vertical_xz(state.psi, grid, plans; j = grid.ny ÷ 2)
```

Use saved z-level output when you know in advance that you only need a few
depths:

```julia
surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(inertial_period(model)),
                              fields = (:ψ, :waves),
                              z = 0.0)
```

## MPI Runs

The global diagnostic helpers reduce across MPI ranks:

```julia
KE = flow_kinetic_energy_global(state.u, state.v, grid, mpi_config)
WE = wave_energy_global(state.L⁺A, state.A, grid, plans, mpi_config)
```

Use these in custom analysis code when the state is distributed.
