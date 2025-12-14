# [Grid & State](@id api-grid-state)

```@meta
CurrentModule = QGYBJ
```

This page documents the Grid and State types in detail.

## Grid Type

### Definition

```julia
struct Grid{T<:AbstractFloat}
    # Dimensions
    nx::Int
    ny::Int
    nz::Int

    # Domain sizes
    Lx::T
    Ly::T
    H::T

    # Grid spacings
    dx::T
    dy::T
    dz::T

    # Physical coordinates
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}

    # Spectral coordinates
    kx::Vector{T}
    ky::Vector{T}
    kh2::Array{T,2}

    # Stratification
    N2::Vector{T}
    N2_face::Vector{T}
    a::Vector{T}
    a_face::Vector{T}

    # Masks
    dealias_mask::Array{T,2}
end
```

### Constructors

```julia
# Basic constructor
grid = Grid(nx=64, ny=64, nz=32)

# With domain size
grid = Grid(;
    nx = 128,
    ny = 128,
    nz = 64,
    Lx = 2π,
    Ly = 2π,
    H = 1.0
)

# With type parameter
grid = Grid{Float32}(nx=64, ny=64, nz=32)
```

### Grid Properties

```julia
# Total number of points
N = grid.nx * grid.ny * grid.nz

# Nyquist wavenumber
k_nyq = π / grid.dx

# Dealiased wavenumber
k_max = 2/3 * k_nyq
```

### Coordinate Access

```julia
# Physical coordinates
x = grid.x  # 1D array, length nx
y = grid.y  # 1D array, length ny
z = grid.z  # 1D array, length nz

# 3D coordinate grids
X, Y, Z = meshgrid(grid)

# Wavenumbers
kx = grid.kx  # length nx÷2+1
ky = grid.ky  # length ny
kh2 = grid.kh2  # 2D array (nx÷2+1, ny)
```

### Stratification Access

```julia
# Buoyancy frequency squared at cell centers
N2 = grid.N2  # length nz

# At cell faces
N2_face = grid.N2_face  # length nz+1

# Stretching coefficient f₀²/N²
a = grid.a  # length nz
a_face = grid.a_face  # length nz+1
```

## State Type

### Definition

```julia
struct State{T<:AbstractFloat, C<:Complex{T}}
    # Prognostic variables (spectral space)
    q::Array{C,3}    # Potential vorticity
    B::Array{C,3}    # Wave envelope

    # Diagnostic variables (spectral space)
    psi::Array{C,3}  # Streamfunction
    A::Array{C,3}    # Wave amplitude

    # Physical space fields
    u::Array{T,3}    # Zonal velocity
    v::Array{T,3}    # Meridional velocity

    # Time stepping history
    rq_old::Array{C,3}
    rq_old2::Array{C,3}
    rB_old::Array{C,3}
    rB_old2::Array{C,3}
end
```

### Constructors

```julia
# From grid
state = create_state(grid)

# With type parameter
state = create_state(grid; T=Float32)

# With initialization
state = create_state(grid; init=:random)
```

### Array Dimensions

```julia
# Spectral arrays (complex)
size(state.psi)  # (nx÷2+1, ny, nz)
size(state.q)    # (nx÷2+1, ny, nz)
size(state.B)    # (nx÷2+1, ny, nz)
size(state.A)    # (nx÷2+1, ny, nz)

# Physical arrays (real)
size(state.u)    # (nx, ny, nz)
size(state.v)    # (nx, ny, nz)
```

### Accessing Fields

```julia
# Direct access
psi_k = state.psi[i, j, k]  # Single spectral coefficient

# Slices
psi_surface = state.psi[:, :, end]  # Surface slice
psi_profile = state.psi[i, j, :]    # Vertical profile

# Full arrays
psi_all = copy(state.psi)  # Copy spectral field
```

### Modifying Fields

```julia
# Set directly
state.psi[i, j, k] = complex_value

# Set slice
state.psi[:, :, end] .= surface_values

# Set all
state.psi .= initial_psi
```

## Work Arrays

### Definition

```julia
struct WorkArrays{T<:AbstractFloat, C<:Complex{T}}
    # Spectral work arrays
    tmp_k::Array{C,3}
    tmp_k2::Array{C,3}
    tmp_k3::Array{C,3}

    # Physical work arrays
    tmp::Array{T,3}
    tmp2::Array{T,3}
    tmp3::Array{T,3}

    # 2D work arrays
    tmp_k_2d::Array{C,2}
    tmp_2d::Array{T,2}
end
```

### Usage

```julia
# Create once
work = create_work_arrays(grid)

# Pass to timestep
timestep!(state, grid, params, work, plans, a_ell, dt)
```

## Grid Utilities

### Distance Functions

```julia
# Distance between points (periodic)
d = periodic_distance(x1, y1, x2, y2, grid)

# Distance squared
d2 = periodic_distance_squared(x1, y1, x2, y2, grid)
```

### Cell Finding

```julia
# Find cell containing point
i, j, k = find_cell(x, y, z, grid)

# With fractional position
i, j, k, fx, fy, fz = find_cell_frac(x, y, z, grid)
```

### Grid Information

```julia
# Print summary
show(grid)

# Memory usage
mem = memory_usage(grid)
println("Grid memory: $(mem/1e6) MB")
```

## State Utilities

### Copying

```julia
# Deep copy
state2 = copy_state(state)

# Copy fields only
copy_fields!(state_dest, state_src)
```

### Zeroing

```julia
# Zero all fields
zero_state!(state)

# Zero spectral fields only
zero_spectral!(state)
```

### Validation

```julia
# Check for NaN/Inf
if has_nan(state)
    error("NaN detected in state")
end

# Check finite
@assert all_finite(state)
```

## Type Parameters

Both Grid and State are parameterized:

```julia
# Double precision (default)
grid64 = Grid{Float64}(nx=64, ny=64, nz=32)
state64 = create_state(grid64)

# Single precision
grid32 = Grid{Float32}(nx=64, ny=64, nz=32)
state32 = create_state(grid32)
```

## API Reference

```@docs
Grid
create_state
create_work_arrays
copy_state
zero_state!
find_cell
periodic_distance
```
