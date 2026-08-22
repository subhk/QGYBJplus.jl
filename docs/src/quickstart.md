# [Quick Start Tutorial](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

Run your first QGYBJ+.jl simulation in 5 minutes.

## Minimal Example

```julia
using QGYBJplus

grid = RectilinearGrid(size=(64, 64, 32),
                       extent=(500e3, 500e3, 4000.0), centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))

set!(model;
     ψ=(x, y, z) -> 1e3 * sin(2π*x/500e3) * cos(2π*y/500e3),
     waves=SurfaceWave(amplitude=0.1, scale=30.0))

simulation = Simulation(model; Δt=20.0, stop_time=86400.0,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(3600.0)))
run!(simulation)
```

## Step-by-Step Breakdown

### Step 1: Create the Grid and Model

```julia
grid = RectilinearGrid(size=(64, 64, 32),
                       extent=(500e3, 500e3, 4000.0), centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5),
                   flow=:evolving,
                   feedback=:wave_mean)
```

The model allocates its MPI decomposition, FFT plans, state, and workspaces.
The same code works on one process or under `mpiexecjl`.

### Step 2: Set Initial Conditions

```julia
set!(model;
     ψ=(x, y, z) -> 1e3 * sin(2π*x/500e3) * cos(2π*y/500e3),
     waves=SurfaceWave(amplitude=0.1, scale=30.0))
```

Use `pv_method=:barotropic` in `set!` when the supplied streamfunction is a
vertically uniform imposed flow.

### Step 3: Configure and Run

```julia
simulation = Simulation(model;
                        Δt=20.0,
                        stop_time=86400.0,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(3600.0)),
                        diagnostics=IterationInterval(100))
run!(simulation)
```

ETD-RK2 is the only timestepper, so there is no method selector to configure.

### Step 4: Access Results

```julia
state = simulation.state

# Spectral fields (complex, in Fourier space)
state.psi    # Streamfunction
state.B      # Wave envelope
state.A      # Wave amplitude (diagnosed from B)
state.C      # Vertical derivative of A

# Physical fields (real, in physical space)
state.u      # Zonal velocity
state.v      # Meridional velocity
state.w      # Vertical velocity
```

### Step 5: Compute Diagnostics

```julia
# Mean flow kinetic energy
KE = flow_kinetic_energy(state.u, state.v)

# Wave energy components per YBJ+ equation (4.7)
WKE, WPE, WCE = compute_detailed_wave_energy(state, simulation.grid, simulation.params)

# Simple wave energy
WE_B, WE_A = wave_energy(state.B, state.A)
```

## Common Configuration Options

```julia
model = QGYBJModel(
    grid=grid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure=HorizontalHyperdiffusivity(flow=0.01, waves=1e5),
    flow=:evolving,       # or :fixed
    feedback=:wave_mean,  # or :none / :no_wave_feedback
    ybj_plus=true,
)
```

| Option | Default | Description |
|:-------|:--------|:------------|
| `ybj_plus` | `true` | Use YBJ+ formulation (recommended) |
| `flow` | `:evolving` | Evolve the balanced flow or hold it fixed |
| `feedback` | `:none` | Select no, one-way, or two-way wave–mean coupling |
| `closure` | `HorizontalHyperdiffusivity()` | Flow and wave dissipation |
| `stratification` | `ConstantStratification(N²=1e-5)` | Vertical stratification |

## Output Files

The `NetCDFOutput(path="output", ...)` above creates:

```
output/
├── state0001.nc          # Field snapshots
├── state0002.nc
└── diagnostic/           # Energy time series
    ├── wave_KE.nc
    ├── mean_flow_KE.nc
    └── total_energy.nc
```

## What's Next?

- [Worked Example](@ref worked_example) — step-by-step walkthrough
- [Configuration](@ref configuration) — all available parameters
- [MPI Parallelization](@ref parallel) — large-scale runs
