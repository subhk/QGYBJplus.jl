# [Quick Start Tutorial](@id quickstart)

```@meta
CurrentModule = QGYBJ
```

This tutorial will guide you through your first QGYBJ.jl simulation in about 5 minutes.

## What We'll Build

A simulation of near-inertial waves interacting with a turbulent eddy field:
- 64×64×32 grid
- Random initial streamfunction (eddies)
- Random initial wave field
- 1000 time steps

## Step 1: Load the Package

```julia
using QGYBJ
```

## Step 2: Create a Simple Configuration

The easiest way to set up a simulation is with `create_simple_config`:

```julia
config = create_simple_config(
    # Domain size (REQUIRED - in meters)
    Lx = 500e3,   # 500 km
    Ly = 500e3,   # 500 km
    Lz = 4000.0,  # 4 km depth

    # Grid dimensions
    nx = 64,
    ny = 64,
    nz = 32,

    # Time stepping
    dt = 0.001,
    total_time = 1.0,

    # Output
    output_interval = 100,
    output_dir = "output"
)
```

This creates a complete configuration with sensible defaults. Note that `Lx`, `Ly`, and `Lz` are **required** parameters - there are no default domain sizes.

## Step 3: Run the Simulation

```julia
result = run_simple_simulation(config)
```

You'll see progress output:
```
Setting up QG-YBJ simulation...
  Grid: 64 × 64 × 32
  Time step: 0.001
  Total steps: 1000
Starting simulation...
  Step 100/1000 (10.0%)
  Step 200/1000 (20.0%)
  ...
Simulation complete!
```

## Step 4: Examine Results

The result contains the final state and diagnostics:

```julia
# Access the final state
state = result.state

# Streamfunction (spectral space, complex)
psi = state.psi  # Array{ComplexF64, 3} of size (64, 64, 32)

# Wave envelope (spectral space, complex)
B = state.B      # Array{ComplexF64, 3} of size (64, 64, 32)

# Velocities (real space)
u = state.u      # Zonal velocity
v = state.v      # Meridional velocity
```

## Step 5: Compute Diagnostics

```julia
# Kinetic energy of the flow
KE = flow_kinetic_energy(state.u, state.v)
println("Kinetic Energy: $KE")

# Wave energy
E_B, E_A = wave_energy(state.B, state.A)
println("Wave Energy (B): $E_B")
println("Wave Energy (A): $E_A")
```

## Step 6: Visualize (Optional)

If you have Plots.jl installed:

```julia
using Plots

# Get plans for FFT
G = result.grid
plans = plan_transforms!(G)

# Get a horizontal slice of vorticity
zeta = slice_horizontal(state.psi, G, plans; k=G.nz)

# Multiply by -kh² to get vorticity
# (simplified - actual vorticity computation is more involved)

heatmap(zeta, title="Surface Vorticity", aspect_ratio=1)
```

## Complete Script

Here's the complete code:

```julia
using QGYBJ

# Configure (Lx, Ly, Lz are REQUIRED)
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # 500km × 500km × 4km
    nx=64, ny=64, nz=32,
    dt=0.001, total_time=1.0,
    output_interval=100
)

# Run
result = run_simple_simulation(config)

# Analyze
println("Final Kinetic Energy: ", flow_kinetic_energy(result.state.u, result.state.v))
println("Final Wave Energy: ", wave_energy(result.state.B, result.state.A))
```

## What's Next?

- [Worked Example](@ref worked_example): More detailed walkthrough
- [Configuration](@ref configuration): Customize all parameters
- [Physics Overview](@ref physics-overview): Understand the equations
- [MPI Parallelization](@ref parallel): Scale to large domains

## Common Customizations

### Change Stratification

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # Domain size required
    nx=64, ny=64, nz=32,
    stratification_type = :skewed_gaussian,  # Realistic pycnocline
    # Or use :constant_N for uniform stratification
)
```

### Enable/Disable Physics

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,

    # Disable wave feedback on mean flow
    no_wave_feedback = true,

    # Run inviscid
    inviscid = true,

    # Linear dynamics only (no advection)
    linear = true,
)
```

### Larger Domain

```julia
config = create_simple_config(
    Lx=1000e3, Ly=1000e3, Lz=5000.0,  # 1000km × 1000km × 5km
    nx=256, ny=256, nz=128,            # Larger grid
    dt=0.0005,                          # Smaller time step for stability
    total_time=10.0,
)
```

!!! tip "Memory Consideration"
    A 256×256×128 complex array uses about 1 GB of memory. For larger domains,
    consider using [MPI parallelization](@ref parallel).

## Parallel Mode (MPI)

For large-scale simulations, use MPI with 2D pencil decomposition:

```julia
# run_parallel.jl
using MPI, PencilArrays, PencilFFTs, QGYBJ

MPI.Init()
mpi_config = QGYBJ.setup_mpi_environment()

# Setup distributed simulation (Lx, Ly, Lz are REQUIRED)
params = default_params(
    nx=256, ny=256, nz=128,
    Lx=1000e3, Ly=1000e3, Lz=5000.0  # 1000km × 1000km × 5km
)
grid = QGYBJ.init_mpi_grid(params, mpi_config)
state = QGYBJ.init_mpi_state(grid, mpi_config)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# ... run simulation ...

MPI.Finalize()
```

Run with:
```bash
mpiexec -n 16 julia --project run_parallel.jl
```

See [MPI Parallelization](@ref parallel) for details on 2D pencil decomposition.
