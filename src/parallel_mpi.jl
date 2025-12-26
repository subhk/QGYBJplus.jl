"""
MPI Parallel Interface for QG-YBJ model using PencilArrays and PencilFFTs.

This module provides MPI parallelization support using:
- MPI.jl for inter-process communication
- PencilArrays.jl for distributed array decomposition
- PencilFFTs.jl for parallel FFT transforms

DECOMPOSITION STRATEGY:
-----------------------
Uses 2D pencil decomposition for optimal parallel scalability:
- Data is distributed across a 2D process grid (px × py)
- THREE pencil configurations for different operations:
  * xy-pencil (2,3): y,z distributed, x local - for horizontal FFTs
  * xz-pencil (1,3): x,z distributed, y local - INTERMEDIATE for transposes
  * z-pencil (1,2):  x,y distributed, z local - for vertical operations

TWO-STEP TRANSPOSE:
-------------------
PencilArrays requires that pencil decompositions differ by at most ONE
dimension for transpose operations. Since xy-pencil and z-pencil differ
in BOTH decomposed dimensions, we use a two-step transpose through the
intermediate xz-pencil:

    xy-pencil (2,3) ↔ xz-pencil (1,3) ↔ z-pencil (1,2)

This approach allows scaling to O(N²) processes for an N³ grid.

USAGE:
------
    using QGYBJplus

    MPI.Init()
    mpi_config = QGYBJplus.setup_mpi_environment()
    grid = QGYBJplus.init_mpi_grid(params, mpi_config)
    state = QGYBJplus.init_mpi_state(grid, mpi_config)
    plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
"""

using MPI
using PencilArrays
using PencilFFTs

# Explicit imports from PencilArrays for clarity
import PencilArrays: Pencil, PencilArray, MPITopology
import PencilArrays: range_local, range_remote, transpose!, gather, pencil
import PencilFFTs: PencilFFTPlan, allocate_input, allocate_output

# Note: Grid, State, QGParams, Plans are already in scope since we're included in QGYBJplus
# init_analytical_psi!, init_analytical_waves!, add_balanced_component! also already available

# Import fft_forward! and fft_backward! from Transforms submodule to extend with MPI methods
# Must import from the defining module (Transforms), not from QGYBJplus which re-exports them
import .Transforms: fft_forward!, fft_backward!

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
- `use_mpi`: Always true (for compatibility with legacy code)
- `parallel_io`: Whether to use parallel I/O
"""
struct MPIConfig
    comm::MPI.Comm
    rank::Int
    nprocs::Int
    is_root::Bool
    topology::Tuple{Int,Int}
    use_mpi::Bool
    parallel_io::Bool
end

# Alias for backward compatibility
const ParallelConfig = MPIConfig

"""
    compute_2d_topology(nprocs::Int) -> Tuple{Int,Int}

Compute optimal 2D process grid for given number of processes.
Tries to make the grid as square as possible.
"""
function compute_2d_topology(nprocs::Int)
    sqrt_n = isqrt(nprocs)
    for p1 in sqrt_n:-1:1
        if nprocs % p1 == 0
            p2 = nprocs ÷ p1
            return (p1, p2)
        end
    end
    return (1, nprocs)
end

"""
    setup_mpi_environment(; topology=nothing, parallel_io=true) -> MPIConfig

Initialize MPI and return configuration. Automatically calls MPI.Init() if needed.

Uses 2D decomposition for optimal scalability. The process topology
can be specified manually or computed automatically.

# Keyword Arguments
- `topology`: Optional tuple (px, py) for process grid. If not specified,
  computed automatically to be as square as possible.
- `parallel_io`: Whether to use parallel I/O (default: true)

# Example
```julia
using QGYBJplus
MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()
# or with explicit topology
mpi_config = QGYBJplus.setup_mpi_environment(topology=(4, 4))
```
"""
function setup_mpi_environment(; topology=nothing, parallel_io=true)
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

    # Register atexit handler to ensure MPI.Finalize() is called
    atexit() do
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end

    return MPIConfig(comm, rank, nprocs, is_root, topo, true, parallel_io)
end

# Alias for backward compatibility
const setup_parallel_environment = setup_mpi_environment

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
- `pencil_xy`: Pencil configuration for horizontal operations (decomp_dims=(2,3), z local)
- `pencil_xz`: Same layout as `pencil_xy` (kept for compatibility)
- `pencil_z`: Same layout as `pencil_xy` (kept for compatibility)
- `local_range_xy`: Local index ranges in xy-pencil configuration
- `local_range_xz`: Local index ranges in xz-pencil configuration
- `local_range_z`: Local index ranges in z-pencil configuration
- `global_dims`: Global array dimensions (nz, nx, ny)
- `topology`: 2D process topology (px, py)
"""
struct PencilDecomp{P1, P2, P3}
    pencil_xy::P1
    pencil_xz::P2
    pencil_z::P3
    local_range_xy::NTuple{3, UnitRange{Int}}
    local_range_xz::NTuple{3, UnitRange{Int}}
    local_range_z::NTuple{3, UnitRange{Int}}
    global_dims::NTuple{3, Int}
    topology::Tuple{Int,Int}
end

"""
    create_pencil_decomposition(nx, ny, nz, mpi_config; decomp_dims=(2,3)) -> PencilDecomp

Create a 2D pencil decomposition for 3D data.

`decomp_dims` selects which dimensions are distributed across the 2D MPI
topology. For the internal `(z, x, y)` storage:
- `(2, 3)` distributes x and y (z local)
Note: `(2, 3)` is required by the current PencilFFTs backend.

Note: `decomp_dims=(1,2)` is not compatible with the current PencilFFTs
backend, which requires decomposition on the last two dimensions.
"""
function create_pencil_decomposition(nx::Int, ny::Int, nz::Int, mpi_config::MPIConfig; decomp_dims=(2,3))
    topo = mpi_config.topology
    px, py = topo
    # Internal storage order is (z, x, y)
    dims = (nz, nx, ny)

    length(decomp_dims) == 2 || error("decomp_dims must have length 2 (got $decomp_dims)")
    all(1 .<= decomp_dims .<= 3) || error("decomp_dims entries must be in 1:3 (got $decomp_dims)")
    decomp_dims[1] != decomp_dims[2] || error("decomp_dims entries must be distinct (got $decomp_dims)")

    # Validate topology against grid dimensions
    if dims[decomp_dims[1]] < px
        error("Grid dimension for dim=$(decomp_dims[1]) is smaller than process topology px=$px.")
    end
    if dims[decomp_dims[2]] < py
        error("Grid dimension for dim=$(decomp_dims[2]) is smaller than process topology py=$py.")
    end

    if mpi_config.is_root
        @info "Topology validation passed" nx ny nz topology=topo decomp_dims=decomp_dims
    end

    # Create 2D MPI topology
    mpi_topo = MPITopology(mpi_config.comm, topo)

    if decomp_dims == (2, 3)
        # z local, x/y distributed. Vertical operations are local so no z-pencil needed.
        pencil_xy = Pencil(mpi_topo, (nz, nx, ny), (2, 3))
        pencil_xz = pencil_xy
        pencil_z = pencil_xy
    else
        error("Unsupported decomp_dims=$decomp_dims. Supported: (2,3).")
    end

    # Get local index ranges
    local_range_xy = range_local(pencil_xy)
    local_range_xz = range_local(pencil_xz)
    local_range_z = range_local(pencil_z)

    if mpi_config.is_root
        @info "Pencil decompositions created" xy_decomp=decomp_dims xz_decomp=decomp_dims z_decomp=decomp_dims
    end

    return PencilDecomp(
        pencil_xy, pencil_xz, pencil_z,
        local_range_xy, local_range_xz, local_range_z,
        (nz, nx, ny), topo
    )
end

#=
================================================================================
                        TRANSPOSE OPERATIONS
================================================================================
=#

# Two-step transpose using PencilArrays' built-in MPI transpose.
# We keep a cached xz-pencil buffer to satisfy the "one-dimension-at-a-time" rule.
const _transpose_buffer_cache = Dict{Tuple{UInt, DataType}, Any}()
const _plan_transpose_buffer_cache = Dict{Tuple{UInt, DataType}, Any}()

function _get_transpose_buffer(decomp::PencilDecomp, ::Type{T}) where T
    key = (objectid(decomp), T)
    if !haskey(_transpose_buffer_cache, key)
        _transpose_buffer_cache[key] = PencilArray{T}(undef, decomp.pencil_xz)
    end
    return _transpose_buffer_cache[key]::PencilArray{T}
end

# Note: _get_plan_transpose_buffer is defined after MPIPlans struct (see below)

function _copy_if_ranges_match!(dst::PencilArray, src::PencilArray, context::AbstractString)
    if range_local(pencil(dst)) != range_local(pencil(src))
        error("Local ranges do not match for $context. " *
              "Ensure plans and arrays are created from the same MPI grid/topology.")
    end
    copyto!(parent(dst), parent(src))
    return dst
end

"""
    transpose_to_z_pencil!(dst, src, decomp::PencilDecomp)

Transpose data from xy-pencil to z-pencil configuration using two-step transpose.
"""
function transpose_to_z_pencil!(dst::PencilArray, src::PencilArray, decomp::PencilDecomp)
    T = eltype(src)
    buffer_xz = _get_transpose_buffer(decomp, T)
    transpose!(buffer_xz, src)
    transpose!(dst, buffer_xz)
    return dst
end

"""
    transpose_to_xy_pencil!(dst, src, decomp::PencilDecomp)

Transpose data from z-pencil to xy-pencil configuration using two-step transpose.
"""
function transpose_to_xy_pencil!(dst::PencilArray, src::PencilArray, decomp::PencilDecomp)
    T = eltype(src)
    buffer_xz = _get_transpose_buffer(decomp, T)
    transpose!(buffer_xz, src)
    transpose!(dst, buffer_xz)
    return dst
end

# Note: _transpose_output_to_input! and _transpose_input_to_output! are defined after MPIPlans struct

# Grid-based versions
function transpose_to_z_pencil!(dst, src, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    transpose_to_z_pencil!(dst, src, decomp)
end

function transpose_to_xy_pencil!(dst, src, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    transpose_to_xy_pencil!(dst, src, decomp)
end

#=
================================================================================
                        GRID INITIALIZATION WITH MPI
================================================================================
=#

"""
    init_mpi_grid(params::QGParams, mpi_config::MPIConfig; decomp_dims=(2,3)) -> Grid

Initialize a Grid with MPI-distributed arrays using a 2D PencilArrays decomposition.
Note: the current PencilFFTs backend requires `decomp_dims=(2,3)` for FFT plans.
"""
function init_mpi_grid(params::QGParams, mpi_config::MPIConfig; decomp_dims=(2,3))
    T = Float64
    nx, ny, nz = params.nx, params.ny, params.nz

    # Create 2D pencil decomposition
    decomp = create_pencil_decomposition(nx, ny, nz, mpi_config; decomp_dims=decomp_dims)

    # Horizontal grid spacing
    dx = params.Lx / nx
    dy = params.Ly / ny

    # Vertical grid (same on all processes)
    z = if nz == 1
        T[params.Lz / 2]
    else
        T.(collect(range(0, params.Lz; length=nz)))
    end
    dz = nz > 1 ? diff(z) : T[params.Lz]

    # Wavenumbers (global arrays, same on all processes)
    kx = T.([i <= (nx+1)÷2 ? (2π/params.Lx)*(i-1) : (2π/params.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= (ny+1)÷2 ? (2π/params.Ly)*(j-1) : (2π/params.Ly)*(j-1-ny) for j in 1:ny])

    # Create distributed kh2 array in xy-pencil configuration (stored as z,x,y)
    kh2_pencil = PencilArray{T}(undef, decomp.pencil_xy)

    # Fill local portion of kh2
    local_range = decomp.local_range_xy
    parent_arr = parent(kh2_pencil)

    for k_local in axes(parent_arr, 1)
        for i_local in axes(parent_arr, 2)
            i_global = local_range[2][i_local]
            for j_local in axes(parent_arr, 3)
                j_global = local_range[3][j_local]
                parent_arr[k_local, i_local, j_local] = kx[i_global]^2 + ky[j_global]^2
            end
        end
    end

    return Grid{T, typeof(kh2_pencil)}(
        nx, ny, nz,
        params.Lx, params.Ly, params.Lz,
        dx, dy,
        z, dz,
        kx, ky,
        kh2_pencil,
        decomp
    )
end

# Alias for backward compatibility
const init_parallel_grid = init_mpi_grid

#=
================================================================================
                        STATE INITIALIZATION WITH MPI
================================================================================
=#

"""
    MPIWorkspace

Pre-allocated workspace arrays for transpose operations.
"""
struct MPIWorkspace{T, PA}
    q_z::PA
    psi_z::PA
    B_z::PA
    A_z::PA
    C_z::PA
    work_z::PA
end

"""
    MPIPlans

FFT plans for MPI-parallel execution using PencilFFTs.

This struct is defined here (before init_mpi_state) so it can be used as a type annotation.
The plan creation function `plan_mpi_transforms` is defined later in this file.
"""
struct MPIPlans{P, PI, PO}
    forward::P
    input_pencil::PI
    output_pencil::PO
    work_arrays::NamedTuple
    pencils_match::Bool
    decomp::Union{PencilDecomp, Nothing}
end

# Buffer cache for plan transpose operations (defined here since MPIPlans is now available)
function _get_plan_transpose_buffer(plans::MPIPlans, ::Type{T}) where T
    if PencilArrays.topology(plans.input_pencil) !== PencilArrays.topology(plans.output_pencil)
        error("PencilFFTs plan input/output topologies differ. " *
              "Ensure plans and arrays are created from the same MPI grid/topology.")
    end
    key = (objectid(plans), T)
    if !haskey(_plan_transpose_buffer_cache, key)
        pencil_xz = Pencil(plans.input_pencil; decomp_dims=(1, 3))
        _plan_transpose_buffer_cache[key] = PencilArray{T}(undef, pencil_xz)
    end
    return _plan_transpose_buffer_cache[key]::PencilArray{T}
end

# Plan-based transpose functions (defined after MPIPlans)
function _transpose_output_to_input!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    T = eltype(src)
    buffer_xz = _get_plan_transpose_buffer(plans, T)
    transpose!(buffer_xz, src)
    transpose!(dst, buffer_xz)
    return dst
end

function _transpose_input_to_output!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    T = eltype(src)
    buffer_xz = _get_plan_transpose_buffer(plans, T)
    transpose!(buffer_xz, src)
    transpose!(dst, buffer_xz)
    return dst
end

"""
    init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64) -> State
    init_mpi_state(grid::Grid, plans::MPIPlans, mpi_config::MPIConfig; T=Float64) -> State

Initialize a State with MPI-distributed PencilArrays in xy-pencil configuration.

Both methods allocate all arrays using the grid's pencil_xy for consistent memory layout.
The FFT functions handle any necessary permutation conversions internally.

# Example
```julia
G = init_mpi_grid(par, mpi_config)
plans = plan_mpi_transforms(G, mpi_config)
S = init_mpi_state(G, mpi_config)  # or init_mpi_state(G, plans, mpi_config)
```
"""
function init_mpi_state(grid::Grid, plans::MPIPlans, mpi_config::MPIConfig; T=Float64)
    # Delegate to the standard version - all arrays use pencil_xy
    return init_mpi_state(grid, mpi_config; T=T)
end

function init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition. Use init_mpi_grid() first.")
    end

    pencil_xy = decomp.pencil_xy

    # Allocate spectral (complex) fields
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

# Alias for backward compatibility
const init_parallel_state = init_mpi_state

"""
    init_mpi_workspace(grid::Grid, mpi_config::MPIConfig; T=Float64) -> MPIWorkspace

Initialize pre-allocated workspace arrays for transpose operations.
"""
function init_mpi_workspace(grid::Grid, mpi_config::MPIConfig; T=Float64)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end

    pencil_z = decomp.pencil_z

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

# Note: MPIPlans struct is defined earlier in this file (before init_mpi_state)

"""
    plan_mpi_transforms(grid::Grid, mpi_config::MPIConfig) -> MPIPlans

Create PencilFFTs plans for parallel 2D horizontal FFT execution.
"""
function plan_mpi_transforms(grid::Grid, mpi_config::MPIConfig)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end

    pencil_xy = decomp.pencil_xy
    if PencilArrays.decomposition(pencil_xy) != (2, 3)
        error("PencilFFTs requires decomp_dims=(2,3) for 3D FFT plans. " *
              "Use decomp_dims=(2,3) or switch FFT backend.")
    end

    # Storage order is (z, x, y): transform x and y, keep z untouched.
    transform = (
        PencilFFTs.Transforms.NoTransform(),
        PencilFFTs.Transforms.FFT(),
        PencilFFTs.Transforms.FFT()
    )

    # Use permute_dims=Val(false) to keep logical dimension order unchanged.
    # Note: output pencil may still differ in decomposition (handled in wrappers).
    plan = PencilFFTPlan(pencil_xy, transform; permute_dims=Val(false))

    # Allocate work arrays and get their pencil configurations
    work_in = allocate_input(plan)
    work_out = allocate_output(plan)

    input_pencil = PencilArrays.pencil(work_in)
    output_pencil = PencilArrays.pencil(work_out)

    pencils_match = _check_pencil_compatibility(input_pencil, output_pencil, pencil_xy)

    if !pencils_match && mpi_config.is_root
        @warn "PencilFFTs input/output pencils have different decompositions. " *
              "FFT wrappers will use transposes between pencils."
    end

    return MPIPlans(
        plan, input_pencil, output_pencil,
        (input=work_in, output=work_out),
        pencils_match,
        decomp
    )
end

function _check_pencil_compatibility(input_pencil, output_pencil, pencil_xy)
    in_range = range_local(input_pencil)
    xy_range = range_local(pencil_xy)
    if in_range != xy_range
        return false
    end
    out_range = range_local(output_pencil)
    if out_range != xy_range
        return false
    end
    return true
end

# FFT operations for MPIPlans
#
# With permute_dims=Val(false), input and output have the same logical dimension order.
# If pencils_match is true, all arrays use compatible pencil configurations.

function fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    if plans.pencils_match
        # Direct transform - pencils are compatible, zero-copy
        mul!(dst, plans.forward, src)
    else
        # Pencils have different MPI decompositions (FFT plan vs model pencils)
        # input_pencil matches model's pencil_xy, but output_pencil differs.
        # Use work arrays as intermediates and transpose between pencils.
        work_in = plans.work_arrays.input
        work_out = plans.work_arrays.output

        _copy_if_ranges_match!(work_in, src, "fft_forward! source -> plan input")

        # Execute FFT (output goes to work_out with output_pencil decomposition)
        mul!(work_out, plans.forward, work_in)

        # Transpose within the plan topology, then copy to model pencil
        _transpose_output_to_input!(work_in, work_out, plans)
        _copy_if_ranges_match!(dst, work_in, "fft_forward! plan input -> destination")
    end
    return dst
end

function fft_backward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    if plans.pencils_match
        # Direct transform - pencils are compatible, zero-copy
        ldiv!(dst, plans.forward, src)
    else
        # Pencils have different MPI decompositions (FFT plan vs model pencils)
        # input_pencil matches model's pencil_xy, but output_pencil differs.
        # Use work arrays as intermediates and transpose between pencils.
        work_in = plans.work_arrays.input
        work_out = plans.work_arrays.output

        _copy_if_ranges_match!(work_in, src, "fft_backward! source -> plan input")
        _transpose_input_to_output!(work_out, work_in, plans)

        # Execute inverse FFT
        ldiv!(work_in, plans.forward, work_out)

        _copy_if_ranges_match!(dst, work_in, "fft_backward! plan input -> destination")
    end
    return dst
end

"""
    allocate_physical(spectral_arr, plans::MPIPlans)

Allocate a physical-space array compatible with `fft_backward!` destination.

For MPI plans, the destination of `fft_backward!` must be on `input_pencil`,
not `output_pencil`. This helper allocates the correct array type.

# Arguments
- `spectral_arr`: A spectral-space array (on output_pencil) to use as template for element type
- `plans`: The MPIPlans containing pencil information

# Returns
A PencilArray allocated on `input_pencil` (physical space).

# Example
```julia
# CORRECT: destination on input_pencil
phys = allocate_physical(spectral_arr, plans)
fft_backward!(phys, spectral_arr, plans)

# WRONG: similar() creates array on same pencil as source
phys_wrong = similar(spectral_arr)  # on output_pencil!
fft_backward!(phys_wrong, spectral_arr, plans)  # ERROR!
```
"""
function allocate_physical(spectral_arr::PencilArray, plans::MPIPlans)
    return PencilArray{eltype(spectral_arr)}(undef, plans.input_pencil)
end

"""
    allocate_spectral(physical_arr, plans::MPIPlans)

Allocate a spectral-space array compatible with `fft_forward!` destination.

For MPI plans, the destination of `fft_forward!` must be on `output_pencil`,
not `input_pencil`. This helper allocates the correct array type.

# Arguments
- `physical_arr`: A physical-space array (on input_pencil) to use as template for element type
- `plans`: The MPIPlans containing pencil information

# Returns
A PencilArray allocated on `output_pencil` (spectral space).
"""
function allocate_spectral(physical_arr::PencilArray, plans::MPIPlans)
    return PencilArray{eltype(physical_arr)}(undef, plans.output_pencil)
end

"""
    allocate_fft_backward_dst(spectral_arr, plans)

Allocate a destination array for fft_backward! that is on the correct pencil.

For MPI plans with input_pencil, allocates on input_pencil (physical space).
For serial plans, uses similar() which works correctly.

This is the centralized helper function for all modules to use when allocating
arrays as destinations for fft_backward!.

# Arguments
- `spectral_arr`: A spectral-space array to use as template for element type
- `plans`: FFT plans (MPIPlans or serial)

# Returns
An array allocated on the correct pencil for fft_backward! destination.
"""
function allocate_fft_backward_dst(spectral_arr, plans)
    if hasfield(typeof(plans), :input_pencil) && plans.input_pencil !== nothing
        return PencilArray{eltype(spectral_arr)}(undef, plans.input_pencil)
    else
        return similar(spectral_arr)
    end
end


#=
================================================================================
                        LOCAL INDEX MAPPING
================================================================================
=#

function get_local_range_xy(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    return decomp.local_range_xy
end

"""
    get_local_range_physical(plans::MPIPlans)

Get local index ranges for physical space arrays (FFT input pencil).
Use this when indexing arrays allocated with the FFT input pencil.
"""
function get_local_range_physical(plans::MPIPlans)
    return range_local(plans.input_pencil)
end

"""
    get_local_range_spectral(plans::MPIPlans)

Get local index ranges for spectral space arrays (FFT output pencil).
Use this when indexing arrays allocated with the FFT output pencil.
"""
function get_local_range_spectral(plans::MPIPlans)
    return range_local(plans.output_pencil)
end

"""
    z_is_local(grid::Grid) -> Bool

Return true if the z-dimension is fully local on each rank.
"""
function z_is_local(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return true
    end
    return decomp.local_range_xy[1] == 1:grid.nz
end

function get_local_range_z(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    return decomp.local_range_z
end

function local_to_global_xy(local_idx::Int, dim::Int, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    return decomp.local_range_xy[dim][local_idx]
end

function local_to_global_z(local_idx::Int, dim::Int, grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    return decomp.local_range_z[dim][local_idx]
end

function local_indices(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    return decomp.local_range_xy
end

#=
================================================================================
                        UTILITY FUNCTIONS
================================================================================
=#

"""
    gather_to_root(arr::PencilArray, grid::Grid, mpi_config::MPIConfig)

Gather a distributed PencilArray to the root process.
"""
function gather_to_root(arr::PencilArray, grid::Grid, mpi_config::MPIConfig)
    gathered = gather(arr)
    return mpi_config.is_root ? gathered : nothing
end

# Also support regular arrays (for compatibility)
function gather_to_root(arr::AbstractArray, grid::Grid, mpi_config::MPIConfig)
    # For regular arrays, just return on root
    return mpi_config.is_root ? arr : nothing
end

"""
    scatter_from_root(arr, grid::Grid, mpi_config::MPIConfig)

Scatter an array from root to all processes as PencilArrays.
"""
function scatter_from_root(arr, grid::Grid, mpi_config::MPIConfig)
    decomp = grid.decomp
    pencil_xy = decomp.pencil_xy

    T = mpi_config.is_root ? eltype(arr) : ComplexF64
    T = MPI.bcast(T, 0, mpi_config.comm)

    distributed = PencilArray{T}(undef, pencil_xy)
    parent_arr = parent(distributed)

    if mpi_config.is_root
        cart_comm = PencilArrays.get_comm(pencil_xy)
        topo = PencilArrays.topology(pencil_xy)
        for coords in CartesianIndices(size(topo))
            coords_tuple = Tuple(coords)
            coords_zero = collect(coords_tuple .- 1)
            rank = MPI.Cart_rank(cart_comm, coords_zero)
            rank_range = range_remote(pencil_xy, coords_tuple)
            portion = arr[rank_range[1], rank_range[2], rank_range[3]]

            if rank == mpi_config.rank
                parent_arr .= portion
            else
                MPI.Send(Array(portion), rank, 0, mpi_config.comm)
            end
        end
    else
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
function mpi_barrier(mpi_config::MPIConfig)
    MPI.Barrier(mpi_config.comm)
end

"""
    mpi_reduce_sum(val, mpi_config::MPIConfig)

Sum a value across all processes.
"""
function mpi_reduce_sum(val, mpi_config::MPIConfig)
    return MPI.Allreduce(val, +, mpi_config.comm)
end

#=
================================================================================
                        ARRAY ALLOCATION HELPERS
================================================================================
=#

"""
    allocate_z_pencil(grid::Grid, ::Type{T}=ComplexF64) where T

Allocate an array in z-pencil configuration for vertical operations.
"""
function allocate_z_pencil(grid::Grid, ::Type{T}=ComplexF64) where T
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    arr = PencilArray{T}(undef, decomp.pencil_z)
    fill!(arr, zero(T))
    return arr
end

"""
    allocate_xy_pencil(grid::Grid, ::Type{T}=ComplexF64) where T

Allocate an array in xy-pencil configuration for horizontal operations (FFTs).
"""
function allocate_xy_pencil(grid::Grid, ::Type{T}=ComplexF64) where T
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    arr = PencilArray{T}(undef, decomp.pencil_xy)
    fill!(arr, zero(T))
    return arr
end

"""
    allocate_xz_pencil(grid::Grid, ::Type{T}=ComplexF64) where T

Allocate an array in xz-pencil (intermediate) configuration.
"""
function allocate_xz_pencil(grid::Grid, ::Type{T}=ComplexF64) where T
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition")
    end
    arr = PencilArray{T}(undef, decomp.pencil_xz)
    fill!(arr, zero(T))
    return arr
end

#=
================================================================================
                        INITIALIZATION HELPERS
================================================================================
=#

"""
    init_mpi_random_field!(arr::PencilArray, grid::Grid, amplitude, seed_offset=0)

Initialize a PencilArray with deterministic random values.
"""
function init_mpi_random_field!(arr::PencilArray, grid::Grid,
                                amplitude::Real, seed_offset::Int=0)
    decomp = grid.decomp
    local_range = decomp.local_range_xy
    parent_arr = parent(arr)

    for k_local in axes(parent_arr, 1)
        k_global = local_range[1][k_local]
        for i_local in axes(parent_arr, 2)
            i_global = local_range[2][i_local]
            for j_local in axes(parent_arr, 3)
                j_global = local_range[3][j_local]
                φ = 2π * ((hash((i_global, j_global, k_global, seed_offset)) % 1_000_000) / 1_000_000)
                parent_arr[k_local, i_local, j_local] = amplitude * cis(φ)
            end
        end
    end

    return arr
end

"""
    parallel_initialize_fields!(state, grid, plans, config, mpi_config; params=nothing)

Initialize fields with MPI support.
"""
function parallel_initialize_fields!(state, grid, plans, config, mpi_config; params=nothing, N2_profile=nothing)
    if config.initial_conditions.psi_type == :random
        init_mpi_random_field!(state.psi, grid, config.initial_conditions.psi_amplitude, 0)
    elseif config.initial_conditions.psi_type == :analytical
        init_analytical_psi!(state.psi, grid, config.initial_conditions.psi_amplitude, plans)
    end

    if config.initial_conditions.wave_type == :random
        init_mpi_random_field!(state.B, grid, config.initial_conditions.wave_amplitude, 1)
    elseif config.initial_conditions.wave_type == :analytical
        init_analytical_waves!(state.B, grid, config.initial_conditions.wave_amplitude, plans)
    end

    # Compute q from ψ
    if params !== nothing && hasfield(typeof(state), :q)
        add_balanced_component!(state, grid, params, plans; N2_profile=N2_profile)
    end
end

"""
    write_mpi_field(filename, varname, arr::PencilArray, grid, mpi_config)

Gather a distributed PencilArray to root for writing.
"""
function write_mpi_field(filename::String, varname::String,
                         arr::PencilArray, grid::Grid, mpi_config::MPIConfig)
    gathered = gather(arr)
    if mpi_config.is_root && gathered !== nothing
        return gathered
    end
    return nothing
end

#=
================================================================================
                        MPI REDUCTION HELPERS
================================================================================
=#

"""
    reduce_sum_if_mpi(val, mpi_config::MPIConfig)

Sum value across all MPI ranks.
"""
function reduce_sum_if_mpi(val::T, mpi_config::MPIConfig) where T
    return MPI.Allreduce(val, +, mpi_config.comm)
end

"""
    reduce_min_if_mpi(val, mpi_config::MPIConfig)

Get minimum across all MPI ranks.
"""
function reduce_min_if_mpi(val::T, mpi_config::MPIConfig) where T
    return MPI.Allreduce(val, min, mpi_config.comm)
end

"""
    reduce_max_if_mpi(val, mpi_config::MPIConfig)

Get maximum across all MPI ranks.
"""
function reduce_max_if_mpi(val::T, mpi_config::MPIConfig) where T
    return MPI.Allreduce(val, max, mpi_config.comm)
end

# Alias for backward compatibility
const gather_array_for_io = gather_to_root
