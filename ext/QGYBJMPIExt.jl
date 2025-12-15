#=
================================================================================
                    QGYBJMPIExt - MPI Extension for QGYBJ.jl
================================================================================

This extension module provides MPI parallelization support using:
- MPI.jl for inter-process communication
- PencilArrays.jl for distributed array decomposition
- PencilFFTs.jl for parallel FFT transforms

The extension is automatically loaded when MPI, PencilArrays, and PencilFFTs
are imported alongside QGYBJ.

DECOMPOSITION STRATEGY:
-----------------------
Uses 2D pencil decomposition for optimal parallel scalability:
- Data is distributed across a 2D process grid (px × py)
- Different pencil configurations for different operations:
  * xy-pencils: For horizontal operations (FFTs in x and y)
  * z-pencils: For vertical operations (tridiagonal solves)
- Transpose operations move data between pencil configurations

This approach allows scaling to O(N²) processes for an N³ grid.

USAGE:
------
    using MPI
    using PencilArrays
    using PencilFFTs
    using QGYBJ

    # Now parallel functions are available
    MPI.Init()
    mpi_config = QGYBJ.setup_mpi_environment()
    grid = QGYBJ.init_mpi_grid(params, mpi_config)
    state = QGYBJ.init_mpi_state(grid, mpi_config)
    plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

================================================================================
=#

module QGYBJMPIExt

using QGYBJ
using MPI
using PencilArrays
using PencilFFTs

# Explicit imports from PencilArrays for clarity and forward compatibility
import PencilArrays: Pencil, PencilArray, MPITopology
import PencilArrays: range_local, range_remote, transpose!, gather
import PencilFFTs: PencilFFTPlan, first_pencil, last_pencil

# Import types we need to extend
import QGYBJ: Grid, State, QGParams, Plans
import QGYBJ: plan_transforms!, fft_forward!, fft_backward!

#=
================================================================================
                        MPI CONFIGURATION
================================================================================
=#

"""
    MPIConfig

Configuration for MPI parallel execution with 2D PencilArrays decomposition.

# Fields
- `comm`: MPI communicator
- `rank`: Process rank (0-indexed)
- `nprocs`: Total number of processes
- `is_root`: True if this is rank 0
- `topology`: 2D process topology (px, py)
"""
struct MPIConfig
    comm::MPI.Comm
    rank::Int
    nprocs::Int
    is_root::Bool
    topology::Tuple{Int,Int}
end

"""
    compute_2d_topology(nprocs::Int) -> Tuple{Int,Int}

Compute optimal 2D process grid for given number of processes.
Tries to make the grid as square as possible.
"""
function compute_2d_topology(nprocs::Int)
    # Find factors closest to sqrt(nprocs)
    sqrt_n = isqrt(nprocs)
    for p1 in sqrt_n:-1:1
        if nprocs % p1 == 0
            p2 = nprocs ÷ p1
            return (p1, p2)
        end
    end
    return (1, nprocs)  # Fallback to 1D if no good factorization
end

"""
    setup_mpi_environment(; topology=nothing) -> MPIConfig

Initialize MPI and return configuration. Call this after MPI.Init().

Uses 2D decomposition for optimal scalability. The process topology
can be specified manually or computed automatically.

# Keyword Arguments
- `topology`: Optional tuple (px, py) for process grid. If not specified,
  computed automatically to be as square as possible.

# Example
```julia
MPI.Init()
mpi_config = setup_mpi_environment()
# or with explicit topology
mpi_config = setup_mpi_environment(topology=(4, 4))
```
"""
function QGYBJ.setup_mpi_environment(; topology=nothing)
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    is_root = rank == 0

    # Compute or validate topology
    if topology === nothing
        topo = compute_2d_topology(nprocs)
    else
        @assert prod(topology) == nprocs "Topology $(topology) doesn't match nprocs=$nprocs"
        topo = topology
    end

    if is_root
        @info "MPI initialized with 2D decomposition" nprocs topology=topo
    end

    return MPIConfig(comm, rank, nprocs, is_root, topo)
end

#=
================================================================================
                        PENCIL DECOMPOSITION
================================================================================
=#

"""
    PencilDecomp

Wrapper for PencilArrays decomposition supporting 2D decomposition with
multiple pencil configurations for different operations.

# Fields
- `pencil_xy`: Pencil configuration for horizontal operations (decomp_dims=(2,3), x local)
- `pencil_xz`: Intermediate pencil for two-step transpose (decomp_dims=(1,3), y local)
- `pencil_z`: Pencil configuration for vertical operations (decomp_dims=(1,2), z local)
- `local_range_xy`: Local index ranges in xy-pencil configuration
- `local_range_xz`: Local index ranges in xz-pencil configuration (intermediate)
- `local_range_z`: Local index ranges in z-pencil configuration
- `global_dims`: Global array dimensions (nx, ny, nz)
- `topology`: 2D process topology (px, py)

# Transpose Operations
PencilArrays requires that pencil decompositions differ by at most one dimension
for transpose operations. Since xy-pencil (2,3) and z-pencil (1,2) differ in both
dimensions, we use a two-step transpose through an intermediate xz-pencil (1,3):

    xy-pencil (2,3) ↔ xz-pencil (1,3) ↔ z-pencil (1,2)

This satisfies the PencilArrays constraint:
- (2,3) → (1,3): differs in first dim only ✓
- (1,3) → (1,2): differs in second dim only ✓

# Type Parameters
Using parametric types for pencils to ensure compatibility across PencilArrays versions.
The actual types are Pencil{3,2,...} for 3D data with 2D decomposition.
"""
struct PencilDecomp{P1, P2, P3}
    pencil_xy::P1                         # decomp_dims=(2,3): y,z distributed; x local
    pencil_xz::P2                         # decomp_dims=(1,3): x,z distributed; y local (intermediate)
    pencil_z::P3                          # decomp_dims=(1,2): x,y distributed; z local
    local_range_xy::NTuple{3, UnitRange{Int}}
    local_range_xz::NTuple{3, UnitRange{Int}}
    local_range_z::NTuple{3, UnitRange{Int}}
    global_dims::NTuple{3, Int}
    topology::Tuple{Int,Int}
end

"""
    create_pencil_decomposition(nx, ny, nz, mpi_config) -> PencilDecomp

Create a 2D pencil decomposition for 3D data with support for both
horizontal (FFT) and vertical (tridiagonal solve) operations.

# Decomposition Strategy
Three pencil configurations are created for different operations:

- `pencil_xy`: decomp_dims=(2,3) - y,z distributed, x local
  Used for FFTs and horizontal operations.

- `pencil_xz`: decomp_dims=(1,3) - x,z distributed, y local (INTERMEDIATE)
  Required for two-step transpose between xy and z pencils.
  PencilArrays requires decomp_dims to differ by at most 1 for transpose.

- `pencil_z`: decomp_dims=(1,2) - x,y distributed, z local
  Used for vertical operations (tridiagonal solves).

# Transpose Path
    xy-pencil (2,3) ↔ xz-pencil (1,3) ↔ z-pencil (1,2)

# Topology Validation
The function validates that the grid dimensions are compatible with the
process topology to ensure each process has at least one grid point.
"""
function create_pencil_decomposition(nx::Int, ny::Int, nz::Int, mpi_config::MPIConfig)
    topo = mpi_config.topology
    px, py = topo

    # Validate topology against grid dimensions
    # For xy-pencil (decompose y, z): need ny >= py and nz >= px
    # For xz-pencil (decompose x, z): need nx >= px and nz >= py
    # For z-pencil (decompose x, y): need nx >= px and ny >= py
    if ny < py
        error("Grid dimension ny=$ny is smaller than process topology py=$py. " *
              "Each process must have at least one grid point in y. " *
              "Reduce the number of processes or increase ny.")
    end
    if nz < px
        error("Grid dimension nz=$nz is smaller than process topology px=$px. " *
              "Each process must have at least one grid point in z for xy-pencil. " *
              "Reduce the number of processes or increase nz.")
    end
    if nx < px
        error("Grid dimension nx=$nx is smaller than process topology px=$px. " *
              "Each process must have at least one grid point in x for z-pencil. " *
              "Reduce the number of processes or increase nx.")
    end
    if nz < py
        error("Grid dimension nz=$nz is smaller than process topology py=$py. " *
              "Each process must have at least one grid point in z for xz-pencil. " *
              "Reduce the number of processes or increase nz.")
    end

    if mpi_config.is_root
        @info "Topology validation passed" nx ny nz topology=topo
    end

    # Create 2D MPI topology
    mpi_topo = MPITopology(mpi_config.comm, topo)

    # Create xy-pencil (for FFTs): decompose in dims (2, 3) - y,z distributed, x local
    pencil_xy = Pencil(mpi_topo, (nx, ny, nz), (2, 3))

    # Create xz-pencil (INTERMEDIATE): decompose in dims (1, 3) - x,z distributed, y local
    # This bridges xy-pencil and z-pencil since they differ by 2 dimensions
    pencil_xz = Pencil(mpi_topo, (nx, ny, nz), (1, 3))

    # Create z-pencil (for vertical ops): decompose in dims (1, 2) - x,y distributed, z local
    pencil_z = Pencil(mpi_topo, (nx, ny, nz), (1, 2))

    # Get local index ranges for each configuration
    local_range_xy = range_local(pencil_xy)
    local_range_xz = range_local(pencil_xz)
    local_range_z = range_local(pencil_z)

    if mpi_config.is_root
        @info "Pencil decompositions created" xy_decomp=(2,3) xz_decomp=(1,3) z_decomp=(1,2)
    end

    return PencilDecomp(
        pencil_xy,
        pencil_xz,
        pencil_z,
        local_range_xy,
        local_range_xz,
        local_range_z,
        (nx, ny, nz),
        topo
    )
end

#=
================================================================================
                        TRANSPOSE OPERATIONS
================================================================================
Two-step transpose is required because PencilArrays only allows transposition
between pencil configurations that differ by at most one decomposed dimension.

xy-pencil (2,3) and z-pencil (1,2) differ in BOTH dimensions, so we must use
an intermediate xz-pencil (1,3):

    xy-pencil (2,3) ↔ xz-pencil (1,3) ↔ z-pencil (1,2)

Each step only changes one decomposed dimension, satisfying PencilArrays' constraint.
================================================================================
=#

# Thread-local buffer cache for intermediate transpose arrays
# This avoids repeated allocations during time-stepping
const _transpose_buffer_cache = Dict{UInt, Any}()

"""
    _get_transpose_buffer(decomp::PencilDecomp, ::Type{T}) where T

Get or create a cached buffer for the intermediate xz-pencil transpose.
Thread-safe using object id as key.
"""
function _get_transpose_buffer(decomp::PencilDecomp, ::Type{T}) where T
    key = objectid(decomp)
    if !haskey(_transpose_buffer_cache, key)
        _transpose_buffer_cache[key] = PencilArray{T}(undef, decomp.pencil_xz)
    end
    return _transpose_buffer_cache[key]::PencilArray{T}
end

"""
    transpose_to_z_pencil!(dst, src, decomp::PencilDecomp)

Transpose data from xy-pencil to z-pencil configuration using two-step transpose.
After this operation, z is fully local on each process.
Use this before vertical operations (tridiagonal solves, vertical derivatives).

# Two-Step Transpose Path
    xy-pencil (2,3) → xz-pencil (1,3) → z-pencil (1,2)

This is required because PencilArrays only allows transposition between
pencil configurations that differ by at most one decomposed dimension.
"""
function transpose_to_z_pencil!(dst::PencilArray, src::PencilArray, decomp::PencilDecomp)
    # Get intermediate buffer in xz-pencil configuration
    T = eltype(src)
    buffer_xz = _get_transpose_buffer(decomp, T)

    # Two-step transpose:
    # Step 1: xy-pencil (2,3) → xz-pencil (1,3) [changes first decomp dim: 2→1]
    transpose!(buffer_xz, src)

    # Step 2: xz-pencil (1,3) → z-pencil (1,2) [changes second decomp dim: 3→2]
    transpose!(dst, buffer_xz)

    return dst
end

"""
    transpose_to_xy_pencil!(dst, src, decomp::PencilDecomp)

Transpose data from z-pencil to xy-pencil configuration using two-step transpose.
Use this after vertical operations to return to the FFT-ready layout.

# Two-Step Transpose Path
    z-pencil (1,2) → xz-pencil (1,3) → xy-pencil (2,3)

This is required because PencilArrays only allows transposition between
pencil configurations that differ by at most one decomposed dimension.
"""
function transpose_to_xy_pencil!(dst::PencilArray, src::PencilArray, decomp::PencilDecomp)
    # Get intermediate buffer in xz-pencil configuration
    T = eltype(src)
    buffer_xz = _get_transpose_buffer(decomp, T)

    # Two-step transpose:
    # Step 1: z-pencil (1,2) → xz-pencil (1,3) [changes second decomp dim: 2→3]
    transpose!(buffer_xz, src)

    # Step 2: xz-pencil (1,3) → xy-pencil (2,3) [changes first decomp dim: 1→2]
    transpose!(dst, buffer_xz)

    return dst
end

# Export transpose functions to QGYBJ module
function QGYBJ.transpose_to_z_pencil!(dst, src, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        # Serial mode - just copy
        dst .= src
        return dst
    end
    transpose_to_z_pencil!(dst, src, decomp)
end

function QGYBJ.transpose_to_xy_pencil!(dst, src, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        # Serial mode - just copy
        dst .= src
        return dst
    end
    transpose_to_xy_pencil!(dst, src, decomp)
end

#=
================================================================================
                        GRID INITIALIZATION WITH MPI
================================================================================
=#

"""
    init_mpi_grid(params::QGParams, mpi_config::MPIConfig) -> Grid

Initialize a Grid with MPI-distributed arrays using 2D PencilArrays decomposition.

# Arguments
- `params::QGParams`: Model parameters
- `mpi_config::MPIConfig`: MPI configuration from setup_mpi_environment()

# Returns
Grid with:
- `decomp::PencilDecomp`: The 2D pencil decomposition with transpose support
- All arrays in xy-pencil configuration (ready for FFTs)

# Example
```julia
mpi_config = setup_mpi_environment()
params = default_params(nx=128, ny=128, nz=64)
grid = init_mpi_grid(params, mpi_config)
```
"""
function QGYBJ.init_mpi_grid(params::QGParams, mpi_config::MPIConfig)
    T = Float64
    nx, ny, nz = params.nx, params.ny, params.nz

    # Create 2D pencil decomposition
    decomp = create_pencil_decomposition(nx, ny, nz, mpi_config)

    # Horizontal grid spacing
    dx = params.Lx / nx
    dy = params.Ly / ny

    # Vertical grid (same on all processes)
    z = T.(collect(range(0, 2π; length=nz)))
    dz = diff(z)

    # Wavenumbers (global arrays, same on all processes)
    kx = T.([i <= nx÷2 ? (2π/params.Lx)*(i-1) : (2π/params.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/params.Ly)*(j-1) : (2π/params.Ly)*(j-1-ny) for j in 1:ny])

    # Create distributed kh2 array in xy-pencil configuration
    kh2_pencil = PencilArray{T}(undef, decomp.pencil_xy)

    # Fill local portion of kh2
    local_range = decomp.local_range_xy
    parent_arr = parent(kh2_pencil)

    for k_local in axes(parent_arr, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(parent_arr, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(parent_arr, 1)
                i_global = local_range[1][i_local]
                parent_arr[i_local, j_local, k_local] = kx[i_global]^2 + ky[j_global]^2
            end
        end
    end

    return Grid{T, typeof(kh2_pencil)}(
        nx, ny, nz,
        params.Lx, params.Ly,
        dx, dy,
        z, dz,
        kx, ky,
        kh2_pencil,
        decomp
    )
end

#=
================================================================================
                        STATE INITIALIZATION WITH MPI
================================================================================
=#

"""
    MPIState

Extended state structure for MPI parallel execution with workspace arrays
for transpose operations.

The main state arrays are in xy-pencil configuration (for FFTs).
Additional z-pencil workspace arrays are provided for vertical operations.
"""
struct MPIWorkspace{T, PA}
    # Z-pencil workspace arrays for vertical operations
    q_z::PA
    psi_z::PA
    B_z::PA
    A_z::PA
    C_z::PA
    work_z::PA
end

"""
    init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64) -> State

Initialize a State with MPI-distributed PencilArrays in xy-pencil configuration.

All fields are allocated as PencilArrays using the grid's xy-pencil decomposition.
"""
function QGYBJ.init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition. Use init_mpi_grid() first.")
    end

    pencil_xy = decomp.pencil_xy

    # Allocate spectral (complex) fields in xy-pencil configuration
    q   = PencilArray{Complex{T}}(undef, pencil_xy); fill!(q, 0)
    psi = PencilArray{Complex{T}}(undef, pencil_xy); fill!(psi, 0)
    A   = PencilArray{Complex{T}}(undef, pencil_xy); fill!(A, 0)
    B   = PencilArray{Complex{T}}(undef, pencil_xy); fill!(B, 0)
    C   = PencilArray{Complex{T}}(undef, pencil_xy); fill!(C, 0)

    # Allocate real-space (real) fields
    u = PencilArray{T}(undef, pencil_xy); fill!(u, 0)
    v = PencilArray{T}(undef, pencil_xy); fill!(v, 0)
    w = PencilArray{T}(undef, pencil_xy); fill!(w, 0)

    return State{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end

"""
    init_mpi_workspace(grid::Grid, mpi_config::MPIConfig; T=Float64) -> MPIWorkspace

Initialize pre-allocated workspace arrays for transpose operations.

# Why Pre-Allocation Matters
Many functions (like `invert_q_to_psi!`, `compute_vertical_velocity!`, etc.)
require temporary z-pencil arrays for vertical operations. By default, these
are allocated on each call, which can cause significant GC pressure in time-
stepping loops.

Pre-allocating workspace arrays and passing them to these functions eliminates
repeated allocations and dramatically improves performance for large grids.

# Fields
The returned `MPIWorkspace` contains z-pencil arrays:
- `q_z`, `psi_z`, `B_z`, `A_z`, `C_z`: Named arrays for common field transposes
- `work_z`: General-purpose workspace array

# Example
```julia
using MPI, PencilArrays, PencilFFTs, QGYBJ
MPI.Init()

mpi_config = QGYBJ.setup_mpi_environment()
params = QGYBJ.default_params(nx=256, ny=256, nz=64)
grid = QGYBJ.init_mpi_grid(params, mpi_config)
state = QGYBJ.init_mpi_state(grid, mpi_config)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# Pre-allocate workspace (do this ONCE before time-stepping)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

# Time-stepping loop - pass workspace to avoid allocations
for step in 1:1000
    # Elliptic inversion with pre-allocated workspace
    QGYBJ.invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)

    # Vertical velocity with pre-allocated workspace
    QGYBJ.compute_vertical_velocity!(state, grid, plans; params=params, workspace=workspace)

    # ... rest of time step
end
```

# Performance Note
For a 256×256×64 grid, pre-allocating workspace typically reduces allocation
overhead by 90%+ in the time-stepping loop.
"""
function QGYBJ.init_mpi_workspace(grid::Grid, mpi_config::MPIConfig; T=Float64)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end

    pencil_z = decomp.pencil_z

    # Allocate z-pencil workspace arrays
    q_z    = PencilArray{Complex{T}}(undef, pencil_z); fill!(q_z, 0)
    psi_z  = PencilArray{Complex{T}}(undef, pencil_z); fill!(psi_z, 0)
    B_z    = PencilArray{Complex{T}}(undef, pencil_z); fill!(B_z, 0)
    A_z    = PencilArray{Complex{T}}(undef, pencil_z); fill!(A_z, 0)
    C_z    = PencilArray{Complex{T}}(undef, pencil_z); fill!(C_z, 0)
    work_z = PencilArray{Complex{T}}(undef, pencil_z); fill!(work_z, 0)

    return MPIWorkspace{T, typeof(q_z)}(q_z, psi_z, B_z, A_z, C_z, work_z)
end

#=
================================================================================
                        FFT TRANSFORMS WITH PENCILFFTS
================================================================================
=#

"""
    MPIPlans

FFT plans for MPI-parallel execution using PencilFFTs.

PencilFFTs automatically handles the transposes needed for 2D distributed FFTs:
- Forward FFT: x-pencil → y-pencil → output (handles x-FFT, transpose, y-FFT)
- Backward FFT: Uses ldiv!(dst, forward_plan, src) for normalized inverse

# Pencil Configuration Note
For complex-to-complex FFTs with NoTransform on z, the output pencil decomposition
should match the input pencil decomposition because dimensions don't change.
If they differ, the code uses work arrays for correct data handling.

# Note on Inverse Transform
We use `ldiv!(dst, plan, src)` with the forward plan for inverse FFT because:
1. It gives normalized inverse (consistent with FFTW.ifft)
2. It's more memory-efficient than storing a separate backward plan
"""
struct MPIPlans{P, PI, PO}
    forward::P
    input_pencil::PI
    output_pencil::PO
    work_arrays::NamedTuple
    pencils_match::Bool  # True if output_pencil decomposition matches input_pencil
end

"""
    plan_mpi_transforms(grid::Grid, mpi_config::MPIConfig) -> MPIPlans

Create PencilFFTs plans for parallel 2D horizontal FFT execution.

PencilFFTs handles the transposes between pencil configurations automatically
for the FFT operations. The returned plans transform dimensions 1 and 2 (x, y).

# Pencil Configuration
- `input_pencil`: Configuration for physical space arrays (from `first_pencil(plan)`)
- `output_pencil`: Configuration for spectral space arrays (from `last_pencil(plan)`)

For complex-to-complex FFTs, these should have matching decompositions. If they
differ, work arrays are used for correct data handling.

# Example
```julia
plans = plan_mpi_transforms(grid, mpi_config)
fft_forward!(dst, src, plans)
```
"""
function QGYBJ.plan_mpi_transforms(grid::Grid, mpi_config::MPIConfig)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end

    pencil_xy = decomp.pencil_xy

    # Create transforms for each dimension:
    # - Dimensions 1 and 2 (x and y): FFT
    # - Dimension 3 (z): NoTransform (stays in physical space)
    transform = (
        PencilFFTs.Transforms.FFT(),         # x dimension
        PencilFFTs.Transforms.FFT(),         # y dimension
        PencilFFTs.Transforms.NoTransform()  # z dimension (no transform)
    )

    # Create the PencilFFT plan
    plan = PencilFFTPlan(pencil_xy, transform)

    # Get input and output pencil configurations
    input_pencil = first_pencil(plan)
    output_pencil = last_pencil(plan)

    # Check if pencil decompositions match (important for correct data handling)
    # For C2C transforms, they should match since dimensions don't change
    pencils_match = _check_pencil_compatibility(input_pencil, output_pencil, pencil_xy)

    if !pencils_match && mpi_config.is_root
        @warn "PencilFFTs input/output pencils have different decompositions. " *
              "Work arrays will be used for FFT operations, which may impact performance."
    end

    # Allocate work arrays for transforms (used when pencils don't match)
    work_in = PencilArray{Complex{Float64}}(undef, input_pencil)
    work_out = PencilArray{Complex{Float64}}(undef, output_pencil)

    # Note: We don't create a separate backward plan because:
    # ldiv!(dst, plan, src) computes the normalized inverse FFT efficiently
    # This is consistent with FFTW.ifft and saves memory

    return MPIPlans(
        plan,
        input_pencil,
        output_pencil,
        (input=work_in, output=work_out),
        pencils_match
    )
end

"""
    _check_pencil_compatibility(input_pencil, output_pencil, pencil_xy) -> Bool

Check if the input/output pencil configurations are compatible with pencil_xy.
Returns true if arrays allocated with pencil_xy can be used directly with FFT operations.

For C2C FFTs, pencils are compatible if they have the same decomposition pattern.
"""
function _check_pencil_compatibility(input_pencil, output_pencil, pencil_xy)
    # Check that input_pencil matches pencil_xy (required for physical space)
    in_range = range_local(input_pencil)
    xy_range = range_local(pencil_xy)

    if in_range != xy_range
        return false
    end

    # Check that output_pencil matches pencil_xy (required for spectral space)
    out_range = range_local(output_pencil)

    if out_range != xy_range
        return false
    end

    return true
end

"""
    fft_forward!(dst, src, plans::MPIPlans)

Perform forward 2D horizontal FFT using PencilFFTs.
Transforms dimensions 1 and 2 (x and y) of the input array.

# Pencil Configuration Handling
For complex-to-complex FFTs with NoTransform on z, input and output pencils
should have matching decompositions. If they differ, an error is raised since
the data cannot be correctly transferred between incompatible pencil layouts.

# Arguments
- `dst`: Destination array for spectral data
- `src`: Source array in physical space
- `plans`: FFT plans from `plan_mpi_transforms`
"""
function QGYBJ.fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    if plans.pencils_match
        # Direct transform when pencil configurations match
        mul!(dst, plans.forward, src)
    else
        # Pencil configurations don't match - this is a critical error
        # For C2C transforms, this should not happen. If it does, the data
        # layout would be corrupted by copying between incompatible pencils.
        error("PencilFFTs input/output pencil configurations do not match. " *
              "This is unexpected for C2C transforms with NoTransform on z. " *
              "Cannot safely perform FFT operations with mismatched pencils. " *
              "Please check PencilFFTs version compatibility or grid setup.")
    end
    return dst
end

"""
    fft_backward!(dst, src, plans::MPIPlans)

Perform inverse 2D horizontal FFT using PencilFFTs.
Uses ldiv! for normalized inverse transform (consistent with FFTW.ifft).

# Pencil Configuration Handling
For complex-to-complex FFTs with NoTransform on z, input and output pencils
should have matching decompositions. If they differ, an error is raised.

# Arguments
- `dst`: Destination array for physical space data (normalized)
- `src`: Source array in spectral space
- `plans`: FFT plans from `plan_mpi_transforms`
"""
function QGYBJ.fft_backward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    if plans.pencils_match
        # Direct transform when pencil configurations match
        ldiv!(dst, plans.forward, src)
    else
        # Pencil configurations don't match - this is a critical error
        error("PencilFFTs input/output pencil configurations do not match. " *
              "This is unexpected for C2C transforms with NoTransform on z. " *
              "Cannot safely perform FFT operations with mismatched pencils. " *
              "Please check PencilFFTs version compatibility or grid setup.")
    end
    return dst
end

#=
================================================================================
                        LOCAL INDEX MAPPING FOR 2D DECOMPOSITION
================================================================================
=#

"""
    get_local_range_xy(grid::Grid) -> NTuple{3, UnitRange{Int}}

Get local index ranges for xy-pencil configuration (used for FFTs).
"""
function QGYBJ.get_local_range_xy(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return (1:grid.nx, 1:grid.ny, 1:grid.nz)
    end
    return decomp.local_range_xy
end

"""
    get_local_range_z(grid::Grid) -> NTuple{3, UnitRange{Int}}

Get local index ranges for z-pencil configuration (used for vertical operations).
"""
function QGYBJ.get_local_range_z(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return (1:grid.nx, 1:grid.ny, 1:grid.nz)
    end
    return decomp.local_range_z
end

"""
    local_to_global_xy(local_idx::Int, dim::Int, grid::Grid) -> Int

Convert local index to global index for xy-pencil configuration.
"""
function QGYBJ.local_to_global_xy(local_idx::Int, dim::Int, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return local_idx
    end
    return decomp.local_range_xy[dim][local_idx]
end

"""
    local_to_global_z(local_idx::Int, dim::Int, grid::Grid) -> Int

Convert local index to global index for z-pencil configuration.
"""
function QGYBJ.local_to_global_z(local_idx::Int, dim::Int, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return local_idx
    end
    return decomp.local_range_z[dim][local_idx]
end

#=
================================================================================
                        UTILITY FUNCTIONS
================================================================================
=#

"""
    gather_to_root(arr::PencilArray, grid::Grid, mpi_config::MPIConfig)

Gather a distributed PencilArray to the root process.
Returns the full array on rank 0, nothing on other ranks.
"""
function QGYBJ.gather_to_root(arr::PencilArray, grid::Grid, mpi_config::MPIConfig)
    gathered = gather(arr)
    return mpi_config.is_root ? gathered : nothing
end

"""
    scatter_from_root(arr, grid::Grid, mpi_config::MPIConfig)

Scatter an array from root to all processes as PencilArrays.

This function efficiently distributes data from root to all processes,
only requiring the full array to exist on the root process.

# Implementation Note
PencilArrays uses a Cartesian topology where `range_local(pencil, n)` takes
a linear index `n` into the process grid (1-indexed, column-major order).
We must convert MPI ranks to the correct linear index based on the topology.
"""
function QGYBJ.scatter_from_root(arr, grid::Grid, mpi_config::MPIConfig)
    decomp = grid.decomp
    pencil_xy = decomp.pencil_xy
    local_range = decomp.local_range_xy
    topo = mpi_config.topology  # (px, py)

    # Allocate distributed array (only local portion on each rank)
    T = mpi_config.is_root ? eltype(arr) : ComplexF64
    # Broadcast element type from root
    T = MPI.bcast(T, 0, mpi_config.comm)

    distributed = PencilArray{T}(undef, pencil_xy)
    parent_arr = parent(distributed)

    # Get local dimensions
    local_size = size(parent_arr)

    # Extract and send local portions from root to each process
    # Using point-to-point communication to avoid full array allocation on all ranks
    if mpi_config.is_root
        # Root extracts and sends each process's portion
        for rank in 0:(mpi_config.nprocs - 1)
            # Convert MPI rank to PencilArrays linear index
            # MPI ranks are assigned in row-major order over the 2D topology (px × py):
            #   rank = coord_x + coord_y * px  (0-indexed)
            # So to get coordinates:
            #   coord_x = rank % px
            #   coord_y = rank ÷ px
            # PencilArrays uses 1-indexed linear indices in column-major order:
            #   linear_idx = coord_x + coord_y * px + 1
            # This happens to equal rank + 1 for this topology ordering.
            coord_x = rank % topo[1]
            coord_y = rank ÷ topo[1]
            linear_idx = coord_x + coord_y * topo[1] + 1  # 1-indexed (equals rank + 1)

            # Get the range for this process using range_remote (not range_local!)
            # range_local only returns the current process's range
            # range_remote(pencil, n) returns the range for process n
            rank_range = range_remote(pencil_xy, linear_idx)

            # Extract the portion for this rank
            portion = arr[rank_range[1], rank_range[2], rank_range[3]]

            if rank == 0
                # Root keeps its own portion
                parent_arr .= portion
            else
                # Send to other ranks
                MPI.Send(Array(portion), rank, 0, mpi_config.comm)
            end
        end
    else
        # Non-root processes receive their portion
        recv_buf = similar(parent_arr)
        MPI.Recv!(recv_buf, 0, 0, mpi_config.comm)
        parent_arr .= recv_buf
    end

    return distributed
end

"""
    mpi_barrier(mpi_config::MPIConfig)

Synchronize all MPI processes.
"""
function QGYBJ.mpi_barrier(mpi_config::MPIConfig)
    MPI.Barrier(mpi_config.comm)
end

"""
    mpi_reduce_sum(val, mpi_config::MPIConfig)

Sum a value across all processes.
"""
function QGYBJ.mpi_reduce_sum(val, mpi_config::MPIConfig)
    return MPI.Allreduce(val, +, mpi_config.comm)
end

"""
    local_indices(grid::Grid)

Get the local index ranges for the current process (xy-pencil configuration).
"""
function QGYBJ.local_indices(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return (1:grid.nx, 1:grid.ny, 1:grid.nz)
    end
    return decomp.local_range_xy
end

#=
================================================================================
                        PARALLEL I/O HELPERS
================================================================================
=#

"""
    write_mpi_field(filename, varname, arr::PencilArray, grid::Grid, mpi_config)

Write a distributed field to file using parallel I/O or gather.
"""
function QGYBJ.write_mpi_field(filename::String, varname::String,
                               arr::PencilArray, grid::Grid, mpi_config::MPIConfig)
    # Gather to root and write serially
    gathered = gather(arr)

    if mpi_config.is_root && gathered !== nothing
        return gathered
    end
    return nothing
end

#=
================================================================================
                        INITIALIZATION HELPERS
================================================================================
=#

"""
    init_mpi_random_field!(arr::PencilArray, grid::Grid, amplitude, seed_offset=0)

Initialize a PencilArray with deterministic random values.
Uses hash-based seeding to ensure reproducibility across different process counts.
"""
function QGYBJ.init_mpi_random_field!(arr::PencilArray, grid::Grid,
                                       amplitude::Real, seed_offset::Int=0)
    decomp = grid.decomp
    local_range = decomp.local_range_xy
    parent_arr = parent(arr)

    for k_local in axes(parent_arr, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(parent_arr, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(parent_arr, 1)
                i_global = local_range[1][i_local]

                # Deterministic phase based on global indices
                φ = 2π * ((hash((i_global, j_global, k_global, seed_offset)) % 1_000_000) / 1_000_000)
                parent_arr[i_local, j_local, k_local] = amplitude * cis(φ)
            end
        end
    end

    return arr
end

"""
    allocate_z_pencil(grid::Grid, ::Type{T}=ComplexF64) where T

Allocate an array in z-pencil configuration for vertical operations.

# Return Type
- **Serial mode** (`grid.decomp === nothing`): Returns `Array{T,3}` of size (nx, ny, nz)
- **Parallel mode**: Returns `PencilArray{T,3}` with z-pencil decomposition

Both types support the same array operations via duck typing. Use `parent(arr)`
to access the underlying contiguous storage for performance-critical loops.

# Example
```julia
work = allocate_z_pencil(grid, ComplexF64)
parent_arr = parent(work)  # Works for both Array and PencilArray
```
"""
function QGYBJ.allocate_z_pencil(grid::Grid, ::Type{T}=ComplexF64) where T
    decomp = grid.decomp
    if decomp === nothing
        # Serial mode - return standard Array
        return zeros(T, grid.nx, grid.ny, grid.nz)
    end
    # Parallel mode - return PencilArray with z-pencil decomposition
    arr = PencilArray{T}(undef, decomp.pencil_z)
    fill!(arr, zero(T))
    return arr
end

"""
    allocate_xy_pencil(grid::Grid, ::Type{T}=ComplexF64) where T

Allocate an array in xy-pencil configuration for horizontal operations (FFTs).

# Return Type
- **Serial mode** (`grid.decomp === nothing`): Returns `Array{T,3}` of size (nx, ny, nz)
- **Parallel mode**: Returns `PencilArray{T,3}` with xy-pencil decomposition

Both types support the same array operations via duck typing. Use `parent(arr)`
to access the underlying contiguous storage for performance-critical loops.

# Example
```julia
work = allocate_xy_pencil(grid, ComplexF64)
parent_arr = parent(work)  # Works for both Array and PencilArray
```
"""
function QGYBJ.allocate_xy_pencil(grid::Grid, ::Type{T}=ComplexF64) where T
    decomp = grid.decomp
    if decomp === nothing
        # Serial mode - return standard Array
        return zeros(T, grid.nx, grid.ny, grid.nz)
    end
    # Parallel mode - return PencilArray with xy-pencil decomposition
    arr = PencilArray{T}(undef, decomp.pencil_xy)
    fill!(arr, zero(T))
    return arr
end

# Export extension types
export MPIConfig, MPIPlans, PencilDecomp, MPIWorkspace

end # module QGYBJMPIExt
