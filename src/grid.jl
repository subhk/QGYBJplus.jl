#=
================================================================================
                        grid.jl - Spatial Grid and State
================================================================================

This file defines the fundamental data structures for the QG-YBJ+ model:

1. GRID: Spatial coordinates and spectral wavenumbers
2. STATE: Prognostic and diagnostic field arrays

GRID STRUCTURE:
---------------
The model uses a doubly-periodic horizontal domain with:
- x ∈ [0, Lx) with nx points (Lx in meters - REQUIRED)
- y ∈ [0, Ly) with ny points (Ly in meters - REQUIRED)
- z ∈ [0, Lz] with nz points (Lz in meters - REQUIRED, e.g., 4000.0 for 4km depth)

SPECTRAL REPRESENTATION:
------------------------
Horizontal fields are represented in spectral space using 2D FFTs.
The wavenumber arrays follow FFTW conventions:
- kx = [0, 1, 2, ..., nx/2-1, -nx/2, ..., -2, -1] × (2π/Lx)
- ky = [0, 1, 2, ..., ny/2-1, -ny/2, ..., -2, -1] × (2π/Ly)

VERTICAL DISCRETIZATION:
------------------------
The vertical coordinate uses an unstaggered grid with second-order
finite differences, matching the Fortran implementation:
- z[k] = k × dz for k = 0, 1, ..., nz-1
- Staggered values (for derivatives) at z[k] + dz/2

STATE VARIABLES:
----------------
Prognostic (evolved in time):
- q: Quasi-geostrophic potential vorticity (spectral)
- B: YBJ+ wave envelope L⁺A (spectral)

Diagnostic (computed from prognostic):
- psi (ψ): Streamfunction, from q via elliptic inversion
- A: Wave amplitude, from B via YBJ+ inversion
- C: Vertical derivative A_z (for wave velocities)
- u, v: Horizontal velocities (real space)
- w: Vertical velocity (real space)

PARALLELIZATION:
----------------
The Grid and State structs support both serial and parallel (MPI) execution:
- Serial: Standard Julia Arrays
- Parallel: PencilArrays for distributed memory

FORTRAN CORRESPONDENCE:
----------------------
- Grid corresponds to init_arrays in init.f90
- State corresponds to the field arrays in main_waqg.f90

================================================================================
=#

#=
================================================================================
                            GRID STRUCTURE
================================================================================
=#

"""
    Grid{T, AT}

Numerical grid and spectral metadata for the QG-YBJ+ model.

# Type Parameters
- `T`: Floating point type (typically Float64)
- `AT`: Array type for 2D arrays (Array{T,2} or PencilArray{T,2})

# Fields

## Grid Dimensions
- `nx, ny, nz::Int`: Number of grid points in x, y, z directions
- `Lx, Ly, Lz::T`: Domain size in x, y, z in meters (REQUIRED - no default)
- `dx, dy::T`: Grid spacing in x, y (computed as Lx/nx, Ly/ny)

## Vertical Grid
- `z::Vector{T}`: Unstaggered vertical levels, length nz
- `dz::Vector{T}`: Layer thicknesses between levels, length nz-1

## Spectral Wavenumbers
- `kx::Vector{T}`: x-wavenumbers following FFTW convention, length nx
- `ky::Vector{T}`: y-wavenumbers following FFTW convention, length ny
- `kh2::AT`: Horizontal wavenumber squared kx² + ky², size (nx, ny)

## Parallel Decomposition
- `decomp::Any`: PencilArrays decomposition (nothing for serial)

# Wavenumber Convention
For a domain of size L with n points:
```
k[i] = (2π/L) × m  where m = i-1        for i ≤ n/2
                          m = i-1-n      for i > n/2
```

# Example
```julia
par = default_params(nx=64, ny=64, nz=32)
G = init_grid(par)
# G.kx[1] = 0 (mean mode)
# G.kx[33] = -32 × (2π/Lx) (most negative wavenumber)
```

See also: [`init_grid`](@ref), [`State`](@ref)
"""
Base.@kwdef mutable struct Grid{T, AT}
    #= Grid dimensions and spacings =#
    nx::Int                # Number of points in x (horizontal)
    ny::Int                # Number of points in y (horizontal)
    nz::Int                # Number of points in z (vertical)
    Lx::T                  # Domain size in x [m] (REQUIRED)
    Ly::T                  # Domain size in y [m] (REQUIRED)
    Lz::T                  # Domain size in z [m] (REQUIRED)
    dx::T                  # Grid spacing in x: dx = Lx/nx
    dy::T                  # Grid spacing in y: dy = Ly/ny

    #= Vertical grid (unstaggered) =#
    z::Vector{T}           # Vertical levels z[k] ∈ [0, Lz], size nz
    dz::Vector{T}          # Layer thicknesses: dz[k] = z[k+1] - z[k], size nz-1

    #= Spectral wavenumbers =#
    kx::Vector{T}          # x-wavenumbers, size nx
    ky::Vector{T}          # y-wavenumbers, size ny
    kh2::AT                # kx² + ky² on spectral grid, size (nx, ny)

    #= MPI decomposition (PencilArrays) =#
    decomp::Any            # PencilDecomposition or nothing for serial
end

"""
    init_grid(par::QGParams) -> Grid

Initialize the spatial grid and spectral wavenumbers from parameters.

# Grid Setup
- Horizontal: Uniform grid with spacing dx = Lx/nx, dy = Ly/ny
- Vertical: Uniform grid from 0 to Lz with spacing dz ≈ Lz/nz
- Domain size (Lx, Ly, Lz) is REQUIRED - specify in meters (e.g., 500e3 for 500 km)

# Wavenumber Arrays
Computes kx, ky following FFTW conventions for periodic domain:
```
kx[i] = (i-1)           for i = 1, ..., nx/2
        (i-1-nx)        for i = nx/2+1, ..., nx
```
multiplied by 2π/Lx.

# Arguments
- `par::QGParams`: Parameter struct with nx, ny, nz, Lx, Ly, Lz

# Returns
Initialized `Grid` struct with all arrays allocated.

# Example
```julia
# Domain size is REQUIRED - specify in meters
par = default_params(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km
G = init_grid(par)
```

# Fortran Correspondence
This matches `init_arrays` in init.f90.
"""
function init_grid(par::QGParams)
    T = Float64
    nx, ny, nz = par.nx, par.ny, par.nz

    # Horizontal grid spacing
    dx = par.Lx / nx
    dy = par.Ly / ny

    #= Vertical grid: z ∈ [0, Lz]
    z[k] ranges from 0 to Lz with nz points
    Lz in meters (e.g., 4000.0 for 4 km depth) =#
    # Handle nz=1 edge case: range() requires length>=2 when endpoints differ
    z = if nz == 1
        T[par.Lz / 2]  # Single point at mid-depth
    else
        T.(collect(range(0, par.Lz; length=nz)))
    end
    # Handle nz=1 edge case: diff returns empty array, so use full domain depth
    dz = nz > 1 ? diff(z) : T[par.Lz]

    #= Wavenumbers for periodic domain
    Following FFTW convention:
    - Positive wavenumbers first: 0, 1, 2, ..., n/2-1
    - Then negative: -n/2, -n/2+1, ..., -1

    k_physical = k_index × (2π/L) =#
    kx = T.([i <= nx÷2 ? (2π/par.Lx)*(i-1) : (2π/par.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/par.Ly)*(j-1) : (2π/par.Ly)*(j-1-ny) for j in 1:ny])

    # Horizontal wavenumber squared: kh² = kx² + ky²
    kh2 = Array{T}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        kh2[i,j] = kx[i]^2 + ky[j]^2
    end

    # No MPI decomposition by default (serial mode)
    decomp = nothing

    return Grid{T, typeof(kh2)}(nx, ny, nz, par.Lx, par.Ly, par.Lz, dx, dy, z, dz, kx, ky, kh2, decomp)
end

"""
    compute_wavenumbers!(G::Grid)

Recompute wavenumber arrays `kx`, `ky`, `kh2` if grid parameters changed.

This is useful after modifying grid dimensions or domain size.

# Example
```julia
G.Lx = 4π  # Change domain size
compute_wavenumbers!(G)  # Update wavenumbers
```
"""
function compute_wavenumbers!(G::Grid)
    nx, ny = G.nx, G.ny

    # Recompute wavenumbers
    G.kx .= [i <= nx÷2 ? (2π/G.Lx)*(i-1) : (2π/G.Lx)*(i-1-nx) for i in 1:nx]
    G.ky .= [j <= ny÷2 ? (2π/G.Ly)*(j-1) : (2π/G.Ly)*(j-1-ny) for j in 1:ny]

    # Recompute kh²
    @inbounds for j in 1:ny, i in 1:nx
        G.kh2[i,j] = G.kx[i]^2 + G.ky[j]^2
    end

    return G
end

"""
    init_pencil_decomposition!(G::Grid)

**DEPRECATED**: This function is a legacy stub and does not properly initialize
MPI decomposition.

For proper MPI parallelization, use the extension module instead:

```julia
using MPI
using PencilArrays
using PencilFFTs
using QGYBJ

MPI.Init()
mpi_config = QGYBJ.setup_mpi_environment()
grid = QGYBJ.init_mpi_grid(params, mpi_config)  # Creates grid with decomposition
state = QGYBJ.init_mpi_state(grid, mpi_config)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)
```

This function now only issues a deprecation warning and returns the grid unchanged.
"""
function init_pencil_decomposition!(G::Grid)
    @warn """
    init_pencil_decomposition! is deprecated and does not work correctly.

    For MPI parallelization, use the extension module:
        using MPI, PencilArrays, PencilFFTs, QGYBJ
        MPI.Init()
        mpi_config = QGYBJ.setup_mpi_environment()
        grid = QGYBJ.init_mpi_grid(params, mpi_config)

    The grid will remain in serial mode (G.decomp = nothing).
    """ maxlog=1
    return G
end

#=
================================================================================
                    LOCAL-TO-GLOBAL INDEX MAPPING
================================================================================
Helper functions for working with both serial arrays and PencilArrays.
In serial mode, local indices equal global indices. In parallel mode,
we need to map local indices to global indices for wavenumber lookups.
================================================================================
=#

"""
    get_local_range(G::Grid) -> NTuple{3, UnitRange{Int}}

Get the local index range for the current process (xy-pencil configuration).

For 2D decomposition, this returns the xy-pencil ranges where x is local
and y,z are distributed. Use `get_local_range_z` for z-pencil configuration.

# Returns
- Serial mode: `(1:nx, 1:ny, 1:nz)`
- Parallel mode: The local range from the xy-pencil decomposition

# Example
```julia
local_range = get_local_range(grid)
for k in local_range[3], j in local_range[2], i in local_range[1]
    # Access data at local indices
end
```
"""
function get_local_range(G::Grid)
    if G.decomp === nothing
        return (1:G.nx, 1:G.ny, 1:G.nz)
    else
        # For 2D decomposition, use xy-pencil ranges (default)
        return G.decomp.local_range_xy
    end
end

"""
    local_to_global(local_idx::Int, dim::Int, G::Grid) -> Int

Convert a local array index to a global index (xy-pencil configuration).

For 2D decomposition, this uses xy-pencil ranges. Use `local_to_global_z`
for operations on z-pencil arrays.

# Arguments
- `local_idx`: Local index in the array
- `dim`: Dimension (1, 2, or 3 for x, y, z)
- `G::Grid`: Grid with optional decomposition

# Returns
Global index for wavenumber lookup.

# Example
```julia
for j_local in axes(arr, 2), i_local in axes(arr, 1)
    i_global = local_to_global(i_local, 1, grid)
    j_global = local_to_global(j_local, 2, grid)
    kx = grid.kx[i_global]
    ky = grid.ky[j_global]
end
```
"""
function local_to_global(local_idx::Int, dim::Int, G::Grid)
    if G.decomp === nothing
        return local_idx
    else
        # For 2D decomposition, use xy-pencil ranges (default)
        return G.decomp.local_range_xy[dim][local_idx]
    end
end

"""
    get_kx(i_local::Int, G::Grid) -> Real

Get the x-wavenumber for a local index, handling both serial and parallel cases.
"""
@inline function get_kx(i_local::Int, G::Grid)
    i_global = local_to_global(i_local, 1, G)
    return G.kx[i_global]
end

"""
    get_ky(j_local::Int, G::Grid) -> Real

Get the y-wavenumber for a local index, handling both serial and parallel cases.
"""
@inline function get_ky(j_local::Int, G::Grid)
    j_global = local_to_global(j_local, 2, G)
    return G.ky[j_global]
end

"""
    get_kh2(i_local::Int, j_local::Int, k_local::Int, arr, G::Grid) -> Real

Get horizontal wavenumber squared for local indices.

For serial mode, accesses G.kh2 directly.
For parallel mode, accesses the local PencilArray element.
"""
@inline function get_kh2(i_local::Int, j_local::Int, k_local::Int, arr, G::Grid)
    if G.decomp === nothing
        # Serial: kh2 is a 2D array
        return G.kh2[i_local, j_local]
    else
        # Parallel: kh2 is a 3D PencilArray, same value for all k
        # Access via parent to get local data
        return real(parent(G.kh2)[i_local, j_local, k_local])
    end
end

"""
    get_local_dims(arr) -> Tuple{Int, Int, Int}

Get the local dimensions of an array (works for both Array and PencilArray).
"""
function get_local_dims(arr)
    p = parent(arr)  # Works for both Array and PencilArray
    return (size(p, 1), size(p, 2), size(p, 3))
end

"""
    is_parallel_array(arr) -> Bool

Check if an array is a PencilArray (parallel) or regular Array (serial).
"""
is_parallel_array(arr) = !(typeof(parent(arr)) === typeof(arr))

#=
================================================================================
                            STATE STRUCTURE
================================================================================
=#

"""
    State{T, RT, CT}

Container for all prognostic and diagnostic fields in the QG-YBJ+ model.

# Type Parameters
- `T`: Floating point type (Float64)
- `RT`: Real array type (Array{T,3} or PencilArray{T,3})
- `CT`: Complex array type (Array{Complex{T},3} or PencilArray{Complex{T},3})

# Prognostic Fields (evolved in time)
- `q::CT`: QG potential vorticity in spectral space
- `B::CT`: YBJ+ wave envelope B = L⁺A in spectral space

# Diagnostic Fields (computed from prognostic)
- `psi::CT`: Streamfunction ψ (from q via elliptic inversion)
- `A::CT`: Wave amplitude (from B via YBJ+ inversion)
- `C::CT`: Vertical derivative C = ∂A/∂z (for wave velocities)

# Velocity Fields (real space)
- `u::RT`: Zonal velocity u = -∂ψ/∂y
- `v::RT`: Meridional velocity v = ∂ψ/∂x
- `w::RT`: Vertical velocity (from omega equation or YBJ)

# Array Dimensions
All arrays have shape (nx, ny, nz).
- Spectral fields (q, psi, A, B, C): Complex arrays
- Real-space fields (u, v, w): Real arrays

# Physical Interpretation
The prognostic variables are:
1. q: Quasi-geostrophic potential vorticity
   - Related to ψ by: q = ∇²ψ + (f²/N²)∂²ψ/∂z²

2. B: YBJ+ wave envelope
   - Related to wave amplitude A by: B = L⁺A
   - L⁺ is an elliptic operator involving ∂²/∂z² and kh²

# Example
```julia
G = init_grid(par)
S = init_state(G)

# Access fields
q_spectral = S.q          # Complex (nx, ny, nz)
u_realspace = S.u         # Real (nx, ny, nz)
```

See also: [`init_state`](@ref), [`Grid`](@ref)
"""
Base.@kwdef mutable struct State{T, RT<:AbstractArray{T,3}, CT<:AbstractArray{Complex{T},3}}
    #= Prognostic fields (spectral space, complex)
    These are the variables that are time-stepped =#
    q::CT           # QG potential vorticity
    B::CT           # YBJ+ wave envelope (B = L⁺A)

    #= Diagnostic fields (spectral space, complex)
    These are computed from prognostic fields =#
    psi::CT         # Streamfunction (from q via inversion)
    A::CT           # Wave amplitude (from B via YBJ+ inversion)
    C::CT           # Vertical derivative A_z

    #= Velocity fields (real space, real)
    Computed from ψ and optionally A =#
    u::RT           # Zonal velocity: u = -∂ψ/∂y
    v::RT           # Meridional velocity: v = ∂ψ/∂x
    w::RT           # Vertical velocity (from omega equation)
end

"""
    allocate_field(T, G; complex=false) -> Array

Allocate a 3D field array of size (nx, ny, nz).

Uses PencilArrays when parallel decomposition is available,
otherwise standard Julia Arrays.

# Arguments
- `T::Type`: Element type (Float64)
- `G::Grid`: Grid struct (determines array size and parallel mode)
- `complex::Bool`: If true, allocate complex array

# Returns
- Serial mode: `Array{T,3}` or `Array{Complex{T},3}`
- Parallel mode: `PencilArray{T,3}` or `PencilArray{Complex{T},3}`

# Example
```julia
q = allocate_field(Float64, G; complex=true)   # Complex spectral field
u = allocate_field(Float64, G; complex=false)  # Real velocity field
```
"""
function allocate_field(::Type{T}, G::Grid; complex::Bool=false) where {T}
    sz = (G.nx, G.ny, G.nz)
    if G.decomp === nothing
        # Serial mode: use standard Arrays
        return complex ? Array{Complex{T}}(undef, sz) : Array{T}(undef, sz)
    else
        # Parallel mode: use PencilArrays
        if complex
            return PencilArrays.PencilArray{Complex{T}}(G.decomp, sz)
        else
            return PencilArrays.PencilArray{T}(G.decomp, sz)
        end
    end
end

"""
    init_state(G::Grid; T=Float64) -> State

Allocate and initialize a State with all fields set to zero.

# Arguments
- `G::Grid`: Grid struct (determines array sizes)
- `T::Type`: Floating point type (default Float64)

# Returns
State struct with:
- Spectral fields (q, psi, A, B, C): Complex arrays, initialized to 0
- Real fields (u, v, w): Real arrays, initialized to 0

# Example
```julia
G = init_grid(par)
S = init_state(G)

# All fields are zero - use init_random_psi! or similar to set ICs
init_random_psi!(S, G, par, plans)
```

See also: [`State`](@ref), [`init_random_psi!`](@ref)
"""
function init_state(G::Grid; T=Float64)
    # Allocate spectral (complex) fields
    q   = allocate_field(T, G; complex=true);    fill!(q, 0)
    psi = allocate_field(T, G; complex=true);    fill!(psi, 0)
    A   = allocate_field(T, G; complex=true);    fill!(A, 0)
    B   = allocate_field(T, G; complex=true);    fill!(B, 0)
    C   = allocate_field(T, G; complex=true);    fill!(C, 0)

    # Allocate real-space (real) fields
    u   = allocate_field(T, G; complex=false);   fill!(u, 0)
    v   = allocate_field(T, G; complex=false);   fill!(v, 0)
    w   = allocate_field(T, G; complex=false);   fill!(w, 0)

    return State{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end
