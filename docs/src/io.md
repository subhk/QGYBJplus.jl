## I/O

QGYBJ.jl supports optional NetCDF I/O (via NCDatasets.jl). If NCDatasets
is not installed, NetCDF I/O is disabled to keep the core functionality
working in restricted environments.

### Enabling NetCDF I/O

```julia
julia --project=. -e 'using Pkg; Pkg.add("NCDatasets")'
```

### Output Configuration

Set the output directory, file pattern, variables to save, and intervals via
`OutputConfig`:

```julia
output = create_output_config(
    output_dir = "./my_run",
    state_file_pattern = "state%04d.nc",
    psi_interval = 1.0,
    wave_interval = 1.0,
    save_psi = true,
    save_waves = true,
    save_velocities = true,
    save_vertical_velocity = true,
)
```

Files are written by `write_state_file` through the `OutputManager` during
`run_simulation!` according to your intervals.

### Reading Initial Conditions

You can initialize ψ and B (L⁺A) from NetCDF files by using
`InitialConditionConfig` with `psi_type=:from_file` and/or `wave_type=:from_file`.

Low‑level helpers (when NCDatasets is available):
- `read_initial_psi(path)` and `read_initial_waves(path)`
- `read_stratification_profile(path)`

### Manual I/O

You can access fields directly from `sim.state` (ψ, A/B, u/v/w) and use your
own I/O libraries/formats.
