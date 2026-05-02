# [I/O and Output](@id io-output)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl writes simulation snapshots as NetCDF files. The normal user-facing
entry point is `NetCDFOutput`, passed to `Simulation`.

## NetCDF Snapshots

```julia
output = NetCDFOutput(path = "output",
                      schedule = TimeInterval(3600),
                      fields = (:ψ, :waves))

simulation = Simulation(model; output)
run!(simulation)
```

This writes files named `state0001.nc`, `state0002.nc`, ... in `output/`.

The main fields are:

| Variable | Meaning | Layout |
|:---------|:--------|:-------|
| `psi` | physical-space streamfunction | `(x, y, z)` |
| `LAr`, `LAi` | real and imaginary parts of the wave velocity envelope `LA` | `(x, y, z)` |
| `u`, `v` | horizontal velocities, when requested | `(x, y, z)` |
| `w` | diagnostic vertical velocity, when requested | `(x, y, z)` |
| `vorticity` | relative vorticity, when requested | `(x, y, z)` |
| `a_ell` | vertical elliptic coefficient `f²/N²` | `(z)` |

The NetCDF files use dimensional coordinates. The vertical coordinate `z` is
negative below the surface and stores the actual model cell-center levels.

## Selecting Output Fields

```julia
output = NetCDFOutput(path = "output",
                      schedule = TimeInterval(3600),
                      fields = (:ψ, :waves, :velocities),
                      velocities = true)
```

Use `fields = (:ψ,)` for flow-only snapshots, `fields = (:waves,)` for wave
snapshots, or include both for the default coupled output.

## Saving Specific z Levels

Use `z` or `z_levels` to save only selected vertical levels:

```julia
surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(3600),
                              fields = (:ψ, :waves),
                              z = 0.0)
```

Requested levels are matched to the nearest native grid level. For example,
`z = 0.0` writes the top model cell center. The NetCDF `z` coordinate records
the actual saved level, and the requested levels are stored in coordinate
metadata.

To save both the full domain and selected levels in the same run, pass multiple
output streams:

```julia
full_output = NetCDFOutput(path = "output/full",
                           schedule = TimeInterval(3600),
                           fields = (:ψ, :waves))

surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(3600),
                              fields = (:ψ, :waves),
                              z = 0.0)

simulation = Simulation(model; output = (full_output, surface_output))
run!(simulation)
```

## Reading Output

Use `NCDatasets.jl` for analysis:

```julia
using NCDatasets

NCDataset("output/full/state0001.nc", "r") do ds
    x = ds["x"][:]
    y = ds["y"][:]
    z = ds["z"][:]
    ψ = ds["psi"][:, :, :]
    LA = ds["LAr"][:, :, :] .+ im .* ds["LAi"][:, :, :]
end
```

## Lower-Level I/O

For advanced workflows, `OutputConfig`, `OutputManager`, and `write_state_file`
are still available. These are useful when working directly with `State`,
`Grid`, and FFT plans:

```julia
output_config = OutputConfig(output_dir = "output",
                             psi_interval = 3600.0,
                             wave_interval = 3600.0,
                             save_psi = true,
                             save_waves = true,
                             z_levels = [0.0])

manager = OutputManager(output_config, params)
write_state_file(manager, state, grid, plans, time; params)
```

Most scripts should prefer the `NetCDFOutput` form shown above.
