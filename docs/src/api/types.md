# [Core Types](@id api-types)

```@meta
CurrentModule = QGYBJ
```

This page documents the core data types in QGYBJ.jl.

## QGParams

The main parameter structure containing all simulation settings.

```@docs
QGParams
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `f₀` | Float64 | Coriolis parameter |
| `N²` | Float64 | Buoyancy frequency squared |
| `W2F` | Float64 | Wave-to-flow velocity ratio squared |
| `γ` | Float64 | Robert-Asselin filter coefficient |
| `ybj_plus` | Bool | Use YBJ+ formulation |
| `no_feedback` | Bool | Disable wave feedback |
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
# Default parameters
params = default_params()

# Custom parameters
params = default_params(;
    f₀ = 1.0,
    N² = 1.0,
    ybj_plus = true,
    νₕ₂ = 10.0,
    ilap2 = 6
)
```

### Example

```julia
# High-resolution parameters
params = default_params(;
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
| `Lx`, `Ly`, `H` | Float64 | Domain sizes |
| `dx`, `dy`, `dz` | Float64 | Grid spacings |
| `x`, `y`, `z` | Vector{Float64} | Coordinate arrays |
| `kx`, `ky` | Vector{Float64} | Wavenumber arrays |
| `kh2` | Array{Float64,2} | Horizontal wavenumber squared |
| `N2` | Vector{Float64} | Buoyancy frequency squared |
| `a` | Vector{Float64} | f₀²/N² at cell centers |

### Constructors

```julia
# Basic grid
grid = Grid(nx=64, ny=64, nz=32)

# Full specification
grid = Grid(;
    nx = 128,
    ny = 128,
    nz = 64,
    Lx = 2π,
    Ly = 2π,
    H = 1.0
)
```

### Grid Utilities

```julia
# Get physical coordinates
x, y, z = get_coordinates(grid)

# Get spectral coordinates
kx, ky = get_wavenumbers(grid)

# Grid info
println(grid)  # Prints summary
```

## State

The simulation state containing all prognostic and diagnostic fields.

```@docs
State
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `q` | Array{ComplexF64,3} | Potential vorticity (spectral) |
| `psi` | Array{ComplexF64,3} | Streamfunction (spectral) |
| `B` | Array{ComplexF64,3} | Wave envelope (spectral) |
| `A` | Array{ComplexF64,3} | Wave amplitude (spectral) |
| `u` | Array{Float64,3} | Zonal velocity (physical) |
| `v` | Array{Float64,3} | Meridional velocity (physical) |

### Constructors

```julia
# Create empty state
state = create_state(grid)

# With initialization
state = create_state(grid; init=:random)
```

### Accessing Fields

```julia
# Spectral fields (complex)
psi_k = state.psi  # size (nx÷2+1, ny, nz)

# Physical fields (real)
u = state.u        # size (nx, ny, nz)
v = state.v        # size (nx, ny, nz)
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

### Plan Types

| Plan | Direction | Transform |
|:-----|:----------|:----------|
| `plan_fft` | Forward | Real → Complex |
| `plan_ifft` | Backward | Complex → Real |

## Setup Model

The `setup_model` function is the recommended way to initialize all components:

```julia
par = default_params(nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(; par)
```

This returns:
- `G`: Grid structure
- `S`: State structure
- `plans`: FFT plans
- `a`: a_ell coefficient array for elliptic inversions

## Type Hierarchy

```
AbstractParams
└── QGParams

AbstractGrid
└── Grid
    └── MPIGrid

AbstractState
└── State
    └── MPIState
```

## Type Stability

All core types are fully type-stable:

```julia
using Test
@inferred create_state(grid)
@inferred timestep!(state, grid, params, work, plans, a_ell, dt)
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
