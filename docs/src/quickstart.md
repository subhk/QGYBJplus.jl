# [Quick Start Tutorial](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

Run your first QGYBJ+.jl simulation in 5 minutes.

## Minimal Example

```julia
using QGYBJplus

# Configure (Lx, Ly, Lz are REQUIRED)
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # Domain size [m]
    nx=64, ny=64, nz=32,             # Grid points
    dt=0.001, total_time=1.0,        # Time stepping
    output_interval=100
)

# Run
result = run_simple_simulation(config)

# Check results
println("Kinetic Energy: ", flow_kinetic_energy(result.state.u, result.state.v))
```

## Step-by-Step Breakdown

### 1. Create Configuration

```julia
config = create_simple_config(
    Lx = 500e3,        # Domain length x [m] (REQUIRED)
    Ly = 500e3,        # Domain length y [m] (REQUIRED)
    Lz = 4000.0,       # Domain depth [m] (REQUIRED)
    nx = 64, ny = 64, nz = 32,  # Grid dimensions
    dt = 0.001,        # Time step
    total_time = 1.0,  # Total simulation time
)
```

!!! warning "Lx, Ly, Lz are required"
    There are no default domain sizes. Omitting them causes a `MethodError`.

### 2. Run Simulation

```julia
result = run_simple_simulation(config)
```

### 3. Access Results

```julia
state = result.state

# Spectral fields (complex)
state.psi    # Streamfunction
state.B      # Wave envelope

# Physical fields (real)
state.u, state.v   # Velocities
```

### 4. Compute Diagnostics

```julia
# Mean flow kinetic energy
KE = flow_kinetic_energy(state.u, state.v)

# Wave kinetic energy per YBJ+ equation (4.7): WKE = (1/2)|LA|²
WKE, WPE, WCE = compute_detailed_wave_energy(state, grid, params)
```

## Common Options

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,

    # Physics
    ybj_plus = true,          # YBJ+ formulation (default)
    linear = false,           # Disable nonlinear terms
    inviscid = true,          # No dissipation
    no_wave_feedback = true,  # One-way coupling

    # Stratification
    stratification_type = :constant_N,  # or :skewed_gaussian
)
```

## What's Next?

- [Worked Example](@ref worked_example) - Detailed walkthrough
- [Configuration](@ref configuration) - All parameters
- [MPI Parallelization](@ref parallel) - Large-scale runs
