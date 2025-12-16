# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJ
```

This page explains how to configure QGYBJ.jl simulations.

## Configuration Approaches

QGYBJ.jl offers two ways to configure simulations:

1. **Simple API**: Use `create_simple_config()` for quick setup
2. **Full Control**: Create `QGParams` and `Grid` directly

## Simple Configuration

### Basic Usage

```julia
config = create_simple_config(
    # Required: Grid size
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
```

### All Options

```julia
config = create_simple_config(
    # Grid dimensions
    nx = 128,                    # Points in x
    ny = 128,                    # Points in y
    nz = 64,                     # Points in z

    # Domain size (nondimensional)
    Lx = 2π,                     # Domain length in x
    Ly = 2π,                     # Domain length in y
    H = 1.0,                     # Domain depth

    # Time stepping
    dt = 0.001,                  # Time step
    total_time = 10.0,           # Total simulation time

    # Stratification
    stratification_type = :exponential,  # See Stratification section
    N0 = 1.0,                    # Reference buoyancy frequency
    pycnocline_depth = 0.1,      # Pycnocline depth (for some types)

    # Physics flags
    inviscid = false,            # Disable all dissipation
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

For complete control, create components separately:

### Step 1: Create Parameters

```julia
params = default_params(;
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

### Step 2: Create Grid

```julia
grid = Grid(
    nx = 128,
    ny = 128,
    nz = 64,
    Lx = 2π,
    Ly = 2π,
    H = 1.0
)
```

### Step 3: Set Up Stratification

```julia
# Option A: Constant N²
setup_stratification!(grid, params, :constant_N)

# Option B: Exponential profile
setup_stratification!(grid, params, :exponential; depth_scale=0.1)

# Option C: Custom profile
N2_custom = [compute_N2(z) for z in grid.z]
set_stratification!(grid, N2_custom)
```

### Step 4: Initialize State

```julia
# Create state with allocated arrays
state = create_state(grid)

# Set initial conditions
initialize_random_flow!(state, grid; energy_level=1.0)
initialize_random_waves!(state, grid; amplitude=0.1)
```

### Step 5: Run

```julia
# Create work arrays and plans
work = create_work_arrays(grid)
plans = plan_transforms!(grid)
a_ell = setup_elliptic_matrices(grid, params)

# Time loop
for step = 1:nsteps
    timestep!(state, grid, params, work, plans, a_ell, dt)
end
```

## Parameter Reference

### Physical Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `f₀` | Float64 | 1.0 | Coriolis parameter |
| `N²` | Float64 | 1.0 | Buoyancy frequency squared |
| `W2F` | Float64 | 0.01 | Wave-to-flow velocity ratio squared |
| `γ` | Float64 | 1e-3 | Robert-Asselin filter coefficient |

Note: Type Unicode characters in Julia REPL using `\` + name + `<tab>`, e.g., `f\_0<tab>` → `f₀`

### Model Flags

| Flag | Type | Default | Effect |
|:-----|:-----|:--------|:-------|
| `ybj_plus` | Bool | true | Use YBJ+ formulation |
| `no_feedback` | Bool | false | Disable wave → flow feedback |
| `inviscid` | Bool | false | Disable all dissipation |
| `linear` | Bool | false | Disable nonlinear terms |
| `no_advection_psi` | Bool | false | Disable ψ advection of PV |
| `no_advection_B` | Bool | false | Disable ψ advection of B |
| `no_refraction` | Bool | false | Disable wave refraction |
| `no_dispersion` | Bool | false | Disable wave dispersion |

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

### Grid Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `nx`, `ny` | Int | - | Horizontal grid points |
| `nz` | Int | - | Vertical grid points |
| `Lx`, `Ly` | Float64 | 2π | Horizontal domain size |
| `H` | Float64 | 1.0 | Vertical domain depth |

## Configuration Validation

The package validates configurations on creation:

```julia
# This will error if invalid
config = create_simple_config(
    nx = 65,  # Error: must be power of 2 for FFT efficiency
    ny = 64,
    nz = 32,
    dt = 0.001,
    total_time = 1.0
)
```

Common validation checks:
- Grid dimensions must be positive integers
- `dt` must be positive
- `total_time` must be greater than `dt`
- Dissipation coefficients must be non-negative

## Saving and Loading Configurations

### Save Configuration

```julia
using JLD2

# Save all parameters
@save "config.jld2" params grid
```

### Load Configuration

```julia
@load "config.jld2" params grid
```

### TOML Format

For human-readable configs:

```julia
using TOML

config_dict = Dict(
    "grid" => Dict("nx" => 128, "ny" => 128, "nz" => 64),
    "physics" => Dict("ybj_plus" => true),
    "time" => Dict("dt" => 0.001, "total" => 10.0)
)

open("config.toml", "w") do io
    TOML.print(io, config_dict)
end
```

## Examples

### Minimal QG-Only Run

```julia
config = create_simple_config(
    nx=64, ny=64, nz=32,
    dt=0.01, total_time=100.0,
    init_B = zeros(ComplexF64, 33, 64, 32)  # No waves
)
```

### High-Resolution Wave-Eddy

```julia
config = create_simple_config(
    nx=512, ny=512, nz=128,
    dt=0.0001, total_time=1.0,
    nu_h2=1e-12, p2=8,  # Very selective dissipation
    output_interval=1000
)
```

### Linear Wave Propagation

```julia
config = create_simple_config(
    nx=128, ny=128, nz=64,
    dt=0.001, total_time=10.0,
    linear=true,         # No nonlinear terms
    inviscid=true,       # No dissipation
    no_wave_feedback=true  # One-way coupling only
)
```

## Next Steps

- [Stratification](@ref stratification): Configure density profiles
- [Initial Conditions](@ref initial-conditions): Set up initial fields
- [Running Simulations](@ref running): Execute and monitor runs
