# [Grid & State](@id api-grid-state)

```@meta
CurrentModule = QGYBJplus
```

This page documents the core data structures: `Grid` and `State`.

## Grid Type

The `Grid` struct contains spatial coordinates, spectral wavenumbers, and parallel decomposition information.

### Definition

```julia
mutable struct Grid{T, AT}
    # Grid dimensions
    nx::Int                # Number of points in x (horizontal)
    ny::Int                # Number of points in y (horizontal)
    nz::Int                # Number of points in z (vertical)

    # Domain sizes
    Lx::T                  # Domain size in x
    Ly::T                  # Domain size in y

    # Grid spacings
    dx::T                  # Grid spacing in x: dx = Lx/nx
    dy::T                  # Grid spacing in y: dy = Ly/ny

    # Vertical grid (staggered, cell-centered)
    z::Vector{T}           # Vertical levels z[k], size nz
    dz::Vector{T}          # Layer thicknesses: dz[k] = z[k+1] - z[k], size nz-1

    # Spectral wavenumbers
    kx::Vector{T}          # x-wavenumbers, size nx
    ky::Vector{T}          # y-wavenumbers, size ny
    kh2::AT                # kx² + ky² on spectral grid

    # MPI decomposition (PencilArrays)
    decomp::Any            # PencilDecomp or nothing for serial
end
```

### Type Parameters

- `T`: Floating point type (typically `Float64`)
- `AT`: Array type for `kh2` (`Array{T,2}` for serial, `PencilArray{T,3}` for parallel)

### Constructors

```julia
# Initialize from parameters (serial mode)
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
grid = init_grid(params)

# Initialize with MPI decomposition
using MPI, PencilArrays, PencilFFTs
MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()
grid = QGYBJplus.init_mpi_grid(params, mpi_config)
```

### Grid Properties

```julia
# Dimensions
nx, ny, nz = grid.nx, grid.ny, grid.nz

# Domain size
Lx, Ly = grid.Lx, grid.Ly

# Grid spacings
dx, dy = grid.dx, grid.dy

# Vertical levels (staggered grid: z runs from -Lz+dz/2 to -dz/2 with dz = Lz/nz)
z = grid.z       # Vector of length nz
dz = grid.dz     # Vector of length nz-1
# Unstaggered (face) levels used for coefficients: z_face = z .- dz/2

# Check if parallel
is_parallel = grid.decomp !== nothing
```

### Wavenumber Access

```julia
# Wavenumber vectors (global, same on all processes)
kx = grid.kx     # Vector of length nx
ky = grid.ky     # Vector of length ny

# Horizontal wavenumber squared
# Serial: 2D array (nx, ny)
# Parallel: 3D PencilArray (local_nz, local_nx, local_ny)
kh2 = grid.kh2

# Convenience functions (handle serial/parallel automatically)
kx_val = get_kx(i_local, grid)    # Get kx for local index
ky_val = get_ky(j_local, grid)    # Get ky for local index
kh2_val = get_kh2(i, j, k, arr, grid)  # Get kh² for local indices
```

### Index Mapping (Parallel)

When using parallel decomposition, local indices must be mapped to global indices:

```julia
# Get local index ranges (xy-pencil / FFT input)
local_range = get_local_range(grid)   # (k_range, i_range, j_range)

# Map local to global indices for a given array's pencil
i_global = local_to_global(i_local, 2, field)  # x dimension
j_global = local_to_global(j_local, 3, field)  # y dimension
k_global = local_to_global(k_local, 1, field)  # z dimension

# Map local to global indices (z-pencil)
i_global = local_to_global_z(i_local, 2, grid)
j_global = local_to_global_z(j_local, 3, grid)

# Get local dimensions of any array
nz_local, nx_local, ny_local = get_local_dims(arr)

# Check if array is parallel
is_distributed = is_parallel_array(arr)
```

### Decomposition Access

```julia
# Access decomposition (parallel mode only)
if grid.decomp !== nothing
    decomp = grid.decomp

    # Pencil configurations
    pencil_xy = decomp.pencil_xy   # For horizontal FFTs
    pencil_z = decomp.pencil_z     # For vertical operations

    # Local ranges
    range_xy = decomp.local_range_xy
    range_z = decomp.local_range_z

    # Global dimensions
    global_dims = decomp.global_dims   # (nz, nx, ny)

    # Process topology
    topology = decomp.topology   # (px, py)
end
```

## State Type

The `State` struct contains all prognostic and diagnostic fields.

### Definition

```julia
mutable struct State{T, RT<:AbstractArray{T,3}, CT<:AbstractArray{Complex{T},3}}
    # Prognostic fields (spectral space, complex)
    q::CT           # QG potential vorticity
    B::CT           # YBJ+ wave envelope (B = L⁺A)

    # Diagnostic fields (spectral space, complex)
    psi::CT         # Streamfunction (from q via inversion)
    A::CT           # Wave amplitude (from B via YBJ+ inversion)
    C::CT           # Vertical derivative A_z

    # Velocity fields (real space, real)
    u::RT           # Zonal velocity: u = -dψ/dy
    v::RT           # Meridional velocity: v = dψ/dx
    w::RT           # Vertical velocity (from omega equation)
end
```

### Type Parameters

- `T`: Floating point type (`Float64`)
- `RT`: Real array type (`Array{T,3}` or `PencilArray{T,3}`)
- `CT`: Complex array type (`Array{Complex{T},3}` or `PencilArray{Complex{T},3}`)

### Constructors

```julia
# Initialize from grid (serial mode)
state = init_state(grid)

# Initialize with MPI (parallel mode)
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
state = QGYBJplus.init_mpi_state(grid, plans, mpi_config)
```

### Field Access

```julia
# Prognostic fields (time-stepped)
q = state.q      # QG potential vorticity (spectral)
B = state.B      # Wave envelope (spectral)

# Diagnostic fields (computed)
psi = state.psi  # Streamfunction (spectral)
A = state.A      # Wave amplitude (spectral)
C = state.C      # Vertical derivative dA/dz (spectral)

# Velocity fields (real space)
u = state.u      # Zonal velocity
v = state.v      # Meridional velocity
w = state.w      # Vertical velocity
```

### Working with Arrays

```julia
# Get underlying data (works for both Array and PencilArray)
psi_data = parent(state.psi)

# Get local dimensions
nz_local, nx_local, ny_local = size(parent(state.psi))

# Access single element
val = state.psi[k, i, j]

# Access slice
top_level = state.psi[end, :, :]  # Closest level to the surface
profile = state.psi[:, i, j]

# Set values
state.psi[k, i, j] = complex_value
state.psi[end, :, :] .= top_values

# Copy all of one field
state.psi .= initial_psi
```

### Physical Interpretation

| Field | Symbol | Physical Meaning |
|:------|:-------|:-----------------|
| `q` | q | QG potential vorticity: q = nabla²psi + (f²/N²)d²psi/dz² |
| `B` | B | YBJ+ wave envelope: B = L⁺A |
| `psi` | psi | Streamfunction |
| `A` | A | Wave amplitude |
| `C` | dA/dz | Vertical derivative of wave amplitude |
| `u` | u | Zonal velocity: u = -dpsi/dy |
| `v` | v | Meridional velocity: v = dpsi/dx |
| `w` | w | Vertical velocity (from omega equation or YBJ) |

## MPI Workspace

For 2D parallel decomposition, workspace arrays store z-pencil data:

### Definition

```julia
struct MPIWorkspace{T, PA}
    q_z::PA      # q in z-pencil configuration
    psi_z::PA    # psi in z-pencil configuration
    B_z::PA      # B in z-pencil configuration
    A_z::PA      # A in z-pencil configuration
    C_z::PA      # C in z-pencil configuration
    work_z::PA   # General workspace
end
```

### Constructor

```julia
# Initialize workspace (parallel mode only)
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)
```

### Usage

```julia
# Pass workspace to functions requiring vertical operations
invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
invert_B_to_A!(state, grid, params, a_vec; workspace=workspace)
compute_vertical_velocity!(state, grid, plans, params; workspace=workspace)
```

## Allocating Arrays

### Serial Mode

```julia
# Allocate using grid
q = allocate_field(Float64, grid; complex=true)   # Complex spectral
u = allocate_field(Float64, grid; complex=false)  # Real physical
```

### Parallel Mode

```julia
# Allocate in xy-pencil (for FFTs, horizontal operations)
arr_xy = QGYBJplus.allocate_xy_pencil(grid, ComplexF64)

# Allocate in z-pencil (for vertical operations)
arr_z = QGYBJplus.allocate_z_pencil(grid, ComplexF64)

# Allocate FFT backward destination (handles spectral→physical pencil difference)
phys_arr = QGYBJplus.allocate_fft_backward_dst(spectral_arr, plans)
```

### FFT Backward Destination Allocation

In 2D MPI decomposition, spectral arrays (FFT output pencil) and physical arrays (FFT input pencil) may have different local dimensions. Use `allocate_fft_backward_dst` to correctly allocate the destination for `fft_backward!`:

```julia
# Allocate physical-space destination for backward FFT
phys = allocate_fft_backward_dst(spectral_arr, plans)

# Now safe to transform
fft_backward!(phys, spectral_arr, plans)

# Loop over physical array with correct dimensions
nz_phys, nx_phys, ny_phys = size(parent(phys))
for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
    # Access phys[k, i, j]
end
```

## Utility Functions

### Grid Utilities

```julia
# Compute wavenumbers (after changing Lx, Ly)
compute_wavenumbers!(grid)

# Get dealiasing mask
mask = dealias_mask(grid)  # 2D Bool array (nx, ny)
```

### State Utilities

```julia
# Zero all fields
fill!(state.q, 0)
fill!(state.B, 0)
fill!(state.psi, 0)

# Check for NaN
has_nan = any(isnan, parent(state.psi))
```

## Serial vs Parallel Comparison

| Operation | Serial | Parallel |
|:----------|:-------|:---------|
| Grid initialization | `init_grid(params)` | `init_mpi_grid(params, mpi_config)` |
| State initialization | `init_state(grid)` | `init_mpi_state(grid, plans, mpi_config)` |
| Array type | `Array{T,3}` | `PencilArray{T,3}` |
| Index access | Direct `arr[k,i,j]` | Via `parent(arr)[k,i,j]` |
| Wavenumber lookup | Direct `grid.kx[i]` | `grid.kx[local_to_global(i,2,grid)]` |
| `grid.decomp` | `nothing` | `PencilDecomp` struct |

## API Reference

### Grid Initialization

```@docs
init_grid
compute_wavenumbers!
```

### State Initialization

```@docs
init_state
```

Field allocation is handled internally by `init_state`. For manual array creation, use standard Julia array allocation or `allocate_xy_pencil`/`allocate_z_pencil` for parallel mode.

### Index Mapping Functions

```@docs
get_local_range
local_to_global
get_local_dims
get_kx
get_ky
get_kh2
```

### FFT Array Allocation

```@docs
allocate_fft_backward_dst
```
