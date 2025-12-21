# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

This page explains how to configure QGYBJ+.jl simulations.

## Configuration Approaches

QGYBJ+.jl offers two ways to configure simulations:

1. **Simple API**: Use `create_simple_config()` for quick setup
2. **Full Control**: Create `QGParams` directly via `default_params()`

!!! warning "Different Defaults Between APIs"
    The two APIs have different default settings for physics flags:

    | Flag | `create_simple_config()` | `default_params()` |
    |:-----|:------------------------|:-------------------|
    | `inviscid` | `true` (no dissipation) | `false` (with dissipation) |
    | `no_wave_feedback` | `false` (two-way coupling) | `true` (one-way coupling) |

    **Simple API** defaults are designed for idealized/educational runs.
    **Full Control API** defaults match the Fortran code for production runs.

    To get production-like settings with the simple API, explicitly set:
    ```julia
    config = create_simple_config(
        Lx=500e3, Ly=500e3, Lz=4000.0,
        inviscid=false,           # Enable dissipation
        no_wave_feedback=true     # One-way wave-flow coupling
    )
    ```

## Simple Configuration

### Basic Usage

```julia
config = create_simple_config(
    # Required: Domain size
    Lx = 500e3,       # Domain length in x [m]
    Ly = 500e3,       # Domain length in y [m]
    Lz = 4000.0,      # Domain depth [m]

    # Optional: Grid size (defaults: 64)
    nx = 128,
    ny = 128,
    nz = 64,

    # Required: Time stepping
    dt = 0.001,
    total_time = 10.0,

    # Optional: Output
    output_interval = 100,
    output_dir = "output"
)

# Run simulation
result = run_simple_simulation(config)
```

### All Options

```julia
config = create_simple_config(
    # Grid dimensions
    nx = 128,                    # Points in x
    ny = 128,                    # Points in y
    nz = 64,                     # Points in z

    # Domain size (REQUIRED - no defaults)
    Lx = 500e3,                  # Domain length in x [m]
    Ly = 500e3,                  # Domain length in y [m]
    Lz = 4000.0,                 # Domain depth [m]

    # Time stepping
    dt = 0.001,                  # Time step
    total_time = 10.0,           # Total simulation time

    # Stratification
    stratification_type = :constant_N,  # or :skewed_gaussian
    N² = 1.0,                    # Buoyancy frequency squared

    # Physics flags
    inviscid = true,             # Disable all dissipation (default for simple API)
    linear = false,              # Disable nonlinear terms
    no_wave_feedback = false,    # Disable wave feedback on flow
    ybj_plus = true,             # Use YBJ+ (vs normal YBJ)

    # Dissipation parameters (hyperviscosity)
    νₕ₁ = 0.01,                  # First hyperviscosity (flow)
    ilap1 = 2,                   # Laplacian power for νₕ₁
    νₕ₂ = 10.0,                  # Second hyperviscosity (flow)
    ilap2 = 6,                   # Laplacian power for νₕ₂
    νz = 0.0,                    # Vertical diffusion

    # Output
    output_interval = 100,       # Steps between output
    output_dir = "output",       # Output directory

    # Initial conditions (optional)
    init_psi = nothing,          # Initial streamfunction
    init_B = nothing,            # Initial wave envelope
)
```

## Full Configuration

For complete control, create components separately using the low-level API.

### Step 1: Create Parameters

```julia
par = default_params(;
    # Domain (REQUIRED)
    Lx = 500e3,                  # Domain length in x [m]
    Ly = 500e3,                  # Domain length in y [m]
    Lz = 4000.0,                 # Domain depth [m]

    # Grid resolution
    nx = 128, ny = 128, nz = 64,

    # Time stepping
    dt = 0.001,
    nt = 10000,

    # Physical parameters
    f₀ = 1.0,                    # Coriolis parameter (type: f\_0<tab>)
    N² = 1.0,                    # Buoyancy frequency squared (type: N\^2<tab>)

    # Model options
    ybj_plus = true,
    no_feedback = false,
    inviscid = false,
    linear = false,

    # Dissipation (hyperviscosity)
    νₕ₁ = 0.01,                  # First hyperviscosity (type: \nu\_h\_1<tab>)
    ilap1 = 2,                   # Power for νₕ₁ (biharmonic)
    νₕ₂ = 10.0,                  # Second hyperviscosity
    ilap2 = 6,                   # Power for νₕ₂ (hyper-6)
    νz = 0.0,                    # Vertical diffusion (type: \nu\_z<tab>)
)
```

### Step 2: Initialize Grid and State

```julia
# Create grid from parameters
G = init_grid(par)

# Initialize state with allocated arrays
S = init_state(G)
```

### Step 3: Set Up FFT Plans and Elliptic Coefficient

```julia
# Create FFT plans
plans = plan_transforms!(G)

# Compute elliptic coefficient for PV inversion
a_ell = a_ell_ut(par, G)
```

Or for non-constant stratification:

```julia
# Get all components including N² profile
G, S, plans, a_ell, N2_profile = setup_model_with_profile(par)
```

### Step 4: Set Initial Conditions

```julia
# Random streamfunction
init_random_psi!(S, G; amplitude=0.1, seed=12345)

# Or analytical initial condition
init_analytical_psi!(S, G; mode=:dipole, amplitude=1.0)

# Set wave envelope (optional)
init_analytical_waves!(S, G; amplitude=0.01)

# Compute initial q from ψ
compute_q_from_psi!(S, G, plans, a_ell)
```

### Step 5: Time Integration

```julia
# First step uses forward Euler
first_projection_step!(S, G, par, plans, a_ell)

# Main time loop uses leapfrog
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

### Complete Low-Level Example

```julia
using QGYBJplus

# Create parameters
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.001, nt=1000
)

# Initialize components
G, S, plans, a_ell = setup_model(par)

# Set initial conditions
init_random_psi!(S, G; amplitude=0.1)
compute_q_from_psi!(S, G, plans, a_ell)

# Time integration
first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

## Parameter Reference

### Physical Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `f₀` | Float64 | 1.0 | Coriolis parameter |
| `N²` | Float64 | 1.0 | Buoyancy frequency squared |
| `γ` | Float64 | 1e-3 | Robert-Asselin filter coefficient |

Note: Type Unicode characters in Julia REPL using `\` + name + `<tab>`, e.g., `f\_0<tab>` → `f₀`

### Domain Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `nx`, `ny` | Int | 64 | Horizontal grid points |
| `nz` | Int | 64 | Vertical grid points |
| `Lx`, `Ly` | Float64 | **REQUIRED** | Horizontal domain size [m] |
| `Lz` | Float64 | **REQUIRED** | Vertical domain depth [m] |

### Model Flags

| Flag | Type | Default | Effect |
|:-----|:-----|:--------|:-------|
| `ybj_plus` | Bool | true | Use YBJ+ formulation |
| `no_feedback` | Bool | true | Master switch: disable all wave-mean coupling |
| `no_wave_feedback` | Bool | true | Disable qʷ term specifically |
| `inviscid` | Bool | false | Disable all dissipation |
| `linear` | Bool | false | Disable nonlinear terms |
| `fixed_flow` | Bool | false | Keep mean flow ψ constant |
| `no_dispersion` | Bool | false | Disable wave dispersion |
| `passive_scalar` | Bool | false | Waves as passive tracers |

### Dissipation Parameters

The model uses two hyperdiffusion operators: `ν₁(-∇²)^ilap1 + ν₂(-∇²)^ilap2`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `νₕ₁` | Float64 | 0.01 | First hyperviscosity coefficient (flow) |
| `ilap1` | Int | 2 | Laplacian power for νₕ₁ (2=biharmonic) |
| `νₕ₂` | Float64 | 10.0 | Second hyperviscosity coefficient (flow) |
| `ilap2` | Int | 6 | Laplacian power for νₕ₂ (hyper-6) |
| `νₕ₁ʷ` | Float64 | 0.0 | First hyperviscosity coefficient (waves) |
| `νₕ₂ʷ` | Float64 | 10.0 | Second hyperviscosity coefficient (waves) |
| `νz` | Float64 | 0.0 | Vertical diffusivity |

## Configuration Validation

The package validates configurations on creation:

```julia
# This will error if invalid
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx = -1,  # Error: must be positive
)
```

Common validation checks:
- Grid dimensions must be positive integers
- Domain sizes (Lx, Ly, Lz) must be positive
- `dt` must be positive
- `N²` must be positive
- Dissipation coefficients must be non-negative
- Powers of 2 for nx, ny are recommended (warning if not)

## Saving and Loading Configurations

### Save Configuration

```julia
using JLD2

# Save parameters
@save "config.jld2" par
```

### Load Configuration

```julia
@load "config.jld2" par
```

### TOML Format

For human-readable configs:

```julia
using TOML

config_dict = Dict(
    "domain" => Dict("nx" => 128, "ny" => 128, "nz" => 64,
                     "Lx" => 500e3, "Ly" => 500e3, "Lz" => 4000.0),
    "physics" => Dict("ybj_plus" => true, "N2" => 1.0),
    "time" => Dict("dt" => 0.001, "nt" => 10000)
)

open("config.toml", "w") do io
    TOML.print(io, config_dict)
end
```

## Examples

### Minimal QG-Only Run

```julia
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.01, nt=10000,
    no_dispersion=true  # No waves
)

G, S, plans, a_ell = setup_model(par)
init_random_psi!(S, G)
compute_q_from_psi!(S, G, plans, a_ell)

first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

### High-Resolution Wave-Eddy

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=512, ny=512, nz=128,
    dt=0.0001, total_time=1.0,
    νₕ₂=1e-12, ilap2=8,  # Very selective dissipation
    output_interval=1000
)

result = run_simple_simulation(config)
```

### Linear Wave Propagation

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=128, ny=128, nz=64,
    dt=0.001, total_time=10.0,
    linear=true,           # No nonlinear terms
    inviscid=true,         # No dissipation
    no_wave_feedback=true  # One-way coupling only
)

result = run_simple_simulation(config)
```

### Using QGYBJSimulation API

```julia
using QGYBJplus

# Create configuration components
domain = create_domain_config(
    nx=64, ny=64, nz=32,
    Lx=500e3, Ly=500e3, Lz=4000.0
)

strat = create_stratification_config(type=:constant_N, N0=1.0)

model = create_model_config(
    ybj_plus=true,
    inviscid=false,
    no_wave_feedback=false  # Enable two-way coupling
)

output = create_output_config(
    output_dir="results",
    output_interval=100
)

# Setup and run
sim = setup_simulation(domain, strat; model=model, output=output)
run_simulation!(sim, dt=0.001, nsteps=1000)
```

## Next Steps

- [Stratification](@ref stratification): Configure density profiles
- [I/O and Output](@ref io-output): Saving and loading data
- [Diagnostics](@ref diagnostics): Energy and analysis tools
