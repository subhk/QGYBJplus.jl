# [I/O and Output](@id io-output)

```@meta
CurrentModule = QGYBJplus
```

This page explains how to save and load simulation data in QGYBJ+.jl.

## Output Formats

QGYBJ+.jl supports multiple output formats:

| Format | Extension | Use Case | Parallel Support |
|:-------|:----------|:---------|:-----------------|
| NetCDF | `.nc` | Analysis, visualization | Yes |
| JLD2 | `.jld2` | Restart files, full state | Yes |
| HDF5 | `.h5` | Large datasets | Yes |

## NetCDF Output

### Basic Setup

```julia
using NCDatasets

# Configure output
output_config = OutputConfig(
    dir = "output",
    prefix = "qgybj",
    interval = 100,           # Steps between outputs
    variables = [:psi, :B, :q],
    format = :netcdf
)

# Initialize output file
init_output!(output_config, grid, params)
```

### Writing Data

```julia
# In time loop
for step = 1:nsteps
    timestep!(state, ...)

    # Write at specified intervals
    if step % output_config.interval == 0
        write_output!(output_config, state, grid, step, time)
    end
end

# Close file
close_output!(output_config)
```

### Output Variables

| Variable | Description | Dimensions |
|:---------|:------------|:-----------|
| `psi` | Streamfunction (physical) | (x, y, z, t) |
| `LAr`, `LAi` | Wave envelope B = L⁺A real/imag parts (physical) | (x, y, z, t) |
| `Ar`, `Ai` | Wave amplitude A real/imag parts (physical) | (x, y, z, t) |
| `q` | Potential vorticity (spectral) | (kx, ky, z, t) |
| `u`, `v` | Velocities (physical) | (x, y, z, t) |
| `energy` | Domain-integrated energy | (t) |

!!! note "Wave field naming"
    - `LAr`, `LAi`: Real and imaginary parts of the wave envelope ``B = L^+ A``
    - `Ar`, `Ai`: Real and imaginary parts of the wave amplitude ``A``
    - Both are needed to compute wave kinetic energy per equation (4.7): ``\text{WKE} = \frac{1}{2}|LA|^2`` where ``LA = B + (k_h^2/4)A``

### Custom Variables

```julia
# Add custom diagnostic
function my_diagnostic(state, grid)
    return sum(abs2.(state.psi)) * grid.dx * grid.dy * grid.dz
end

# Register custom output
add_output_variable!(output_config, "my_diag", my_diagnostic;
    dims = ("time",),
    units = "m^4/s^2"
)
```

## Reading NetCDF Data

### Basic Reading

```julia
using NCDatasets

ds = NCDataset("output/qgybj_0001.nc")

# Read variables
psi = ds["psi"][:]      # Full 4D array (kx, ky, z, time)
time = ds["time"][:]    # Time coordinate

# Read single snapshot
psi_t10 = ds["psi"][:, :, :, 10]

close(ds)
```

### Lazy Reading

For large files, read lazily:

```julia
ds = NCDataset("output/qgybj_0001.nc")

# This doesn't load data yet
psi_var = ds["psi"]

# Load only what you need
for t in 1:10
    snapshot = psi_var[:, :, :, t]
    # Process snapshot
end

close(ds)
```

### Using `do` Block

```julia
NCDataset("output/qgybj_0001.nc") do ds
    psi = ds["psi"][:]
    # Process data
end  # File automatically closed
```

## JLD2 for Restart Files

### Saving State

```julia
using JLD2

# Save full state for restart
@save "restart.jld2" state grid params step time

# Or more selectively
jldsave("restart.jld2";
    psi = state.psi,
    B = state.B,
    q = state.q,
    step = step,
    time = time
)
```

### Loading State

```julia
# Load everything
@load "restart.jld2" state grid params step time

# Or selectively
data = load("restart.jld2")
psi = data["psi"]
step = data["step"]
```

### Restarting Simulation

```julia
# Load restart file
@load "restart.jld2" state grid params step_start time_start

# Continue simulation
for step = step_start+1:nsteps
    timestep!(state, ...)
end
```

## Checkpointing

### Automatic Checkpoints

```julia
config = create_simple_config(
    # ...
    checkpoint_interval = 1000,  # Steps between checkpoints
    checkpoint_dir = "checkpoints"
)
```

### Manual Checkpointing

```julia
function save_checkpoint(state, grid, params, step, time)
    filename = "checkpoints/checkpoint_$(lpad(step, 8, '0')).jld2"
    @save filename state grid params step time
    return filename
end

# In time loop
if step % checkpoint_interval == 0
    save_checkpoint(state, grid, params, step, time)
end
```

### Checkpoint Rotation

Keep only recent checkpoints to save disk space:

```julia
function rotate_checkpoints(dir, keep_n=3)
    files = sort(glob("checkpoint_*.jld2", dir))
    while length(files) > keep_n
        rm(popfirst!(files))
    end
end
```

## Diagnostics Output

### Time Series

```julia
# Collect diagnostics during run
diagnostics = DiagnosticsTimeSeries()

for step = 1:nsteps
    timestep!(state, ...)

    # Compute and store diagnostics
    record!(diagnostics, step, time,
        KE = flow_kinetic_energy(state.u, state.v),
        PE = flow_potential_energy(state.psi, grid),
        WE = wave_energy(state.B, state.A)
    )
end

# Save to file
save_diagnostics("diagnostics.csv", diagnostics)
```

### Format

```csv
step,time,KE,PE,WE
0,0.0,0.0123,0.0045,0.0089
100,0.1,0.0121,0.0044,0.0087
...
```

## MPI Parallel I/O with 2D Decomposition

QGYBJ+.jl provides seamless I/O support for 2D pencil decomposition. The I/O functions automatically handle distributed arrays.

### Writing State Files

```julia
using MPI, PencilArrays, PencilFFTs, QGYBJplus

MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()

# Setup distributed grid and state
grid = QGYBJplus.init_mpi_grid(params, mpi_config)
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
state = QGYBJplus.init_mpi_state(grid, plans, mpi_config)

# Create output manager with parallel config
output_config = OutputConfig(
    output_dir = "output",
    state_file_pattern = "state%04d.nc",
    psi_interval = 0.1,
    wave_interval = 0.1
)
manager = OutputManager(output_config, params, mpi_config)

# Write state - automatically handles 2D decomposition
write_state_file(manager, state, grid, plans, time, mpi_config)
```

### Reading Initial Conditions

```julia
# Read psi - works in both serial and parallel mode
psi = read_initial_psi("initial_psi.nc", grid, plans; parallel_config=mpi_config)
# In parallel: rank 0 reads, then scatters to all processes

# Read wave field
B = read_initial_waves("initial_waves.nc", grid, plans; parallel_config=mpi_config)

# Or use legacy wrappers with parallel support
ncread_psi!(state, grid, plans; path="psi.nc", parallel_config=mpi_config)
ncread_la!(state, grid, plans; path="la.nc", parallel_config=mpi_config)
```

### I/O Strategy for 2D Decomposition

QGYBJ+.jl uses a **gather-to-root** strategy for parallel I/O:

```
┌─────────────────────────────────────────────────────────────┐
│                    Parallel I/O Strategy                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WRITING:                                                   │
│  ┌─────────┐   gather_to_root   ┌─────────┐   write   ┌───┐ │
│  │ Rank 0  │ ←───────────────── │ Rank 0  │ ────────→ │.nc│ │
│  │ Rank 1  │                    │ (full   │           └───┘ │
│  │ Rank 2  │                    │  array) │                 │
│  │   ...   │                    └─────────┘                 │
│  └─────────┘                                                │
│                                                             │
│  READING:                                                   │
│  ┌───┐   read    ┌─────────┐  scatter_from_root  ┌────────┐ │
│  │.nc│ ────────→ │ Rank 0  │ ──────────────────→ │ Rank 0 │ │
│  └───┘           │ (full   │                     │ Rank 1 │ │
│                  │  array) │                     │ Rank 2 │ │
│                  └─────────┘                     │   ...  │ │
│                                                  └────────┘ │
└─────────────────────────────────────────────────────────────┘
```

This approach is:
- **Simple**: No parallel NetCDF library required
- **Reliable**: Standard serial NetCDF always works
- **Portable**: Works on any system

### Local Index Ranges for Manual I/O

```julia
# Get local ranges for this process (xy-pencil)
if grid.decomp !== nothing
    local_range = grid.decomp.local_range_xy
    # local_range = (1:nx_local, y_start:y_end, z_start:z_end)
else
    local_range = (1:grid.nx, 1:grid.ny, 1:grid.nz)
end

# Use with NCDatasets for manual parallel writes
NCDatasets.Dataset("output.nc", "c"; comm=mpi_config.comm) do ds
    ds.dim["x"] = grid.nx
    ds.dim["y"] = grid.ny
    ds.dim["z"] = grid.nz

    psi_var = NCDatasets.defVar(ds, "psi", Float64, ("x", "y", "z"))

    # Each rank writes its portion
    psi_var[local_range[1], local_range[2], local_range[3]] = local_psi_data
end
```

### Gather/Scatter for I/O

```julia
# Gather distributed array to rank 0
global_psi = QGYBJplus.gather_to_root(state.psi, grid, mpi_config)
# Returns full array on rank 0, nothing on other ranks

# Scatter from rank 0 to all processes
local_psi = QGYBJplus.scatter_from_root(global_psi, grid, mpi_config)
# Each rank receives its local portion
```

## Physical Space Output

### Transform Before Writing

```julia
# Spectral → Physical
psi_phys = irfft(state.psi, grid.nx)

# Write physical space data
ds["psi_phys"][:, :, :, t] = psi_phys
```

### Vorticity

```julia
# Compute vorticity (spectral)
zeta_k = -grid.kh2 .* state.psi

# Transform to physical
zeta = irfft(zeta_k, grid.nx)

ds["vorticity"][:, :, :, t] = zeta
```

## Output Best Practices

### File Naming

```julia
# Include simulation info in filename
prefix = "qgybj_nx$(nx)_nz$(nz)"

# Timestamp outputs
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
filename = "$(prefix)_$(timestamp).nc"
```

### Compression

```julia
# Enable compression for NetCDF
defVar(ds, "psi", Float64, ("x", "y", "z", "time");
    deflatelevel = 5,  # 0-9, higher = more compression
    chunksizes = (nx, ny, nz, 1)  # Chunk by time slice
)
```

### Metadata

```julia
# Add attributes
ds.attrib["title"] = "QGYBJ+.jl simulation output"
ds.attrib["history"] = "Created $(now())"
ds.attrib["Conventions"] = "CF-1.8"

# Variable attributes
ds["psi"].attrib["long_name"] = "Streamfunction"
ds["psi"].attrib["units"] = "m^2/s"
```

## Visualization Integration

### Quick Plotting

```julia
using Plots, NCDatasets

NCDataset("output.nc") do ds
    psi = ds["psi"][:, :, end, end]  # Surface, last time
    heatmap(real(irfft(psi, nx)), title="Surface ψ")
end
```

### Animation

```julia
using Plots

NCDataset("output.nc") do ds
    anim = @animate for t in 1:size(ds["psi"], 4)
        psi = ds["psi"][:, :, end, t]
        heatmap(real(irfft(psi, nx)),
            title = "t = $(ds["time"][t])",
            clim = (-1, 1)
        )
    end
    gif(anim, "animation.gif", fps=10)
end
```

## API Reference

For I/O operations, use the following approaches:
- **NetCDF reading**: Use `NCDatasets.jl` directly as shown above
- **Initial conditions**: `ncread_psi!`, `ncread_la!` for loading spectral fields
- **Parallel I/O**: `gather_to_root`, `scatter_from_root` for 2D decomposition

See the [Grid & State API](../api/grid_state.md) for field initialization functions.
