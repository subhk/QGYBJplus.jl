# [Core Types](@id api-types)

```@meta
CurrentModule = QGYBJplus
```

This page documents the core data types in QGYBJ+.jl.

## QGParams

The main parameter structure containing all simulation settings.

```@docs
QGParams
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `nx`, `ny`, `nz` | Int | Grid dimensions |
| `Lx`, `Ly`, `Lz` | Float64 | Domain sizes (REQUIRED) |
| `dt` | Float64 | Time step |
| `nt` | Int | Number of time steps |
| `f₀` | Float64 | Coriolis parameter |
| `N²` | Float64 | Buoyancy frequency squared |
| `γ` | Float64 | Robert-Asselin filter coefficient |
| `ybj_plus` | Bool | Use YBJ+ formulation |
| `no_feedback` | Bool | Master switch: disable all wave-mean coupling |
| `no_wave_feedback` | Bool | Disable wave feedback on mean flow |
| `inviscid` | Bool | Disable all dissipation |
| `linear` | Bool | Disable nonlinear terms |
| `νₕ₁` | Float64 | First hyperviscosity coefficient (flow) |
| `ilap1` | Int | Laplacian power for νₕ₁ |
| `νₕ₂` | Float64 | Second hyperviscosity coefficient (flow) |
| `ilap2` | Int | Laplacian power for νₕ₂ |
| `νₕ₁ʷ` | Float64 | First hyperviscosity coefficient (waves) |
| `νₕ₂ʷ` | Float64 | Second hyperviscosity coefficient (waves) |
| `νz` | Float64 | Vertical diffusivity |

Note: Type Unicode characters using `\` + name + `<tab>` in Julia REPL (e.g., `f\_0<tab>` → `f₀`)

### Constructors

```julia
# Domain size is REQUIRED (no defaults)
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km

# Custom parameters with domain size
params = default_params(;
    Lx = 500e3, Ly = 500e3, Lz = 4000.0,  # Domain size (REQUIRED)
    nx = 128, ny = 128, nz = 64,           # Grid dimensions
    f₀ = 1.0,
    N² = 1.0,
    ybj_plus = true,
    νₕ₂ = 10.0,
    ilap2 = 6
)
```

### Example

```julia
# High-resolution parameters (domain size REQUIRED)
params = default_params(;
    Lx = 500e3, Ly = 500e3, Lz = 4000.0,
    nx = 256, ny = 256, nz = 128,
    ybj_plus = true,
    no_feedback = false,
    νₕ₂ = 1e-12,
    ilap2 = 8
)
```

## Grid

The computational grid structure.

```@docs
Grid
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `nx`, `ny`, `nz` | Int | Grid dimensions |
| `Lx`, `Ly`, `Lz` | Float64 | Domain sizes |
| `dx`, `dy`, `dz` | Float64 | Grid spacings |
| `x`, `y`, `z` | Vector{Float64} | Coordinate arrays |
| `kx`, `ky` | Vector{Float64} | Wavenumber arrays |
| `kh2` | Array{Float64,2} | Horizontal wavenumber squared |

### Constructors

```julia
# Initialize grid from parameters (recommended)
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
grid = init_grid(params)
```

### Grid Utilities

```julia
# Get wavenumbers at local indices (works with MPI)
kx = get_kx(grid, i)   # Get kx at local index i
ky = get_ky(grid, j)   # Get ky at local index j
kh2 = get_kh2(grid, i, j)  # Get horizontal wavenumber squared
```

## State

The simulation state containing all prognostic and diagnostic fields.

```@docs
State
```

### Fields

**Prognostic Fields (evolved in time):**

| Field | Type | Description |
|:------|:-----|:------------|
| `q` | Array{ComplexF64,3} | QG potential vorticity (spectral) |
| `B` | Array{ComplexF64,3} | YBJ+ wave envelope B = L⁺A (spectral) |

**Diagnostic Fields (computed from prognostic):**

| Field | Type | Description |
|:------|:-----|:------------|
| `psi` | Array{ComplexF64,3} | Streamfunction ψ (spectral) |
| `A` | Array{ComplexF64,3} | Wave amplitude (spectral) |
| `C` | Array{ComplexF64,3} | Vertical derivative ∂A/∂z (spectral) |

**Velocity Fields (real space):**

| Field | Type | Description |
|:------|:-----|:------------|
| `u` | Array{Float64,3} | Zonal velocity u = -∂ψ/∂y |
| `v` | Array{Float64,3} | Meridional velocity v = ∂ψ/∂x |
| `w` | Array{Float64,3} | Vertical velocity (from omega equation) |

!!! note "Leapfrog Time-Stepping"
    The leapfrog scheme uses separate State objects (Snm1, Sn, Snp1) rather than
    storing previous time levels within a single State struct. This design allows
    proper handling of MPI parallel arrays (PencilArrays).

### Constructors

```julia
# Create empty state from grid
state = init_state(grid)

# Copy state (preserves PencilArray topology for MPI)
state_copy = copy_state(state)
```

### Copying States

For MPI parallel runs, always use `copy_state` instead of `deepcopy`:

```julia
# CORRECT: preserves pencil topology
Snm1 = copy_state(S)

# WRONG: breaks PencilArray transpose operations
Snm1 = deepcopy(S)  # Causes "pencil topologies must be the same" error
```

### Accessing Fields

```julia
# Spectral fields (complex)
psi_k = state.psi  # size (nz, nx, ny)

# Physical fields (real)
u = state.u        # size (nz, nx, ny)
v = state.v        # size (nz, nx, ny)
```

## FFT Plans

FFTW plan structures for efficient transforms.

### Creating Plans

```julia
# Standard plans
plans = plan_transforms!(grid)

# With optimization
plans = plan_transforms!(grid; flags=FFTW.MEASURE)

# With threading
FFTW.set_num_threads(8)
plans = plan_transforms!(grid)
```

### Using Transforms

```julia
# Forward transform: Physical → Spectral
fft_forward!(dst, src, plans)

# Backward transform: Spectral → Physical
fft_backward!(dst, src, plans)
```

## Setup Model

The `setup_model` function is the recommended way to initialize all components:

```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a_ell = setup_model(par)
```

This returns:
- `G`: Grid structure
- `S`: State structure
- `plans`: FFT plans
- `a_ell`: Elliptic coefficient array for PV inversion

For non-constant stratification, use:

```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, stratification=:skewed_gaussian)
G, S, plans, a_ell, N2_profile = setup_model_with_profile(par)
```

## Type Hierarchy

```
QGParams{T}     - Model parameters

Grid            - Spatial grid and wavenumbers

State           - Prognostic and diagnostic fields
```

## Type Stability

All core types are fully type-stable:

```julia
using Test
@inferred init_state(grid)
@inferred leapfrog_step!(state, grid, params, plans, a_ell)
```

## Serialization

Types support JLD2 serialization:

```julia
using JLD2

# Save
@save "simulation.jld2" grid params state

# Load
@load "simulation.jld2" grid params state
```
