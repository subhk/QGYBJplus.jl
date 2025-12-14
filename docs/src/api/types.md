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
| `f0` | Float64 | Coriolis parameter |
| `N0` | Float64 | Reference buoyancy frequency |
| `ybj_plus` | Bool | Use YBJ+ formulation |
| `no_feedback` | Bool | Disable wave feedback |
| `inviscid` | Bool | Disable all dissipation |
| `linear` | Bool | Disable nonlinear terms |
| `nu_h1` | Float64 | Large-scale horizontal diffusivity |
| `p1` | Int | Power for nu_h1 |
| `nu_h2` | Float64 | Small-scale horizontal hyperviscosity |
| `p2` | Int | Power for nu_h2 |
| `nu_z` | Float64 | Vertical diffusivity |

### Constructors

```julia
# Default parameters
params = QGParams()

# Custom parameters
params = QGParams(;
    f0 = 1.0,
    N0 = 1.0,
    ybj_plus = true,
    nu_h2 = 1e-8,
    p2 = 4
)
```

### Example

```julia
# High-resolution parameters
params = QGParams(;
    ybj_plus = true,
    no_feedback = false,
    nu_h2 = 1e-12,
    p2 = 8
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

## Work Arrays

Pre-allocated temporary arrays for computations.

```@docs
WorkArrays
```

### Usage

```julia
# Create work arrays
work = create_work_arrays(grid)

# Used internally by timestep
timestep!(state, grid, params, work, plans, a_ell, dt)
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

## Elliptic Matrices

Pre-computed matrices for elliptic inversions.

```@docs
EllipticMatrices
```

### Usage

```julia
# Setup once
a_ell = setup_elliptic_matrices(grid, params)

# Use in time stepping
invert_q_to_psi!(state, grid, params, a_ell)
invert_B_to_A!(state, grid, params, a_ell)
```

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
