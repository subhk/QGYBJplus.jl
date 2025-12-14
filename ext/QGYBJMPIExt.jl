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

USAGE:
------
    using MPI
    using PencilArrays
    using PencilFFTs
    using QGYBJ

    # Now parallel functions are available
    MPI.Init()
    pconfig = QGYBJ.setup_mpi_environment()
    grid = QGYBJ.init_mpi_grid(params, pconfig)
    state = QGYBJ.init_mpi_state(grid, pconfig)
    plans = QGYBJ.plan_mpi_transforms(grid, pconfig)

================================================================================
=#

module QGYBJMPIExt

using QGYBJ
using MPI
using PencilArrays
using PencilFFTs

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

Configuration for MPI parallel execution with PencilArrays decomposition.
"""
struct MPIConfig
    comm::MPI.Comm
    rank::Int
    nprocs::Int
    topology::Tuple{Int,Int}  # Process topology for 2D decomposition
    is_root::Bool
end

"""
    setup_mpi_environment(; topology=(0,0)) -> MPIConfig

Initialize MPI and return configuration. Call this after MPI.Init().

# Arguments
- `topology::Tuple{Int,Int}`: Process grid (0,0 for auto-detection)

# Example
```julia
MPI.Init()
mpi_config = setup_mpi_environment()
```
"""
function QGYBJ.setup_mpi_environment(; topology::Tuple{Int,Int}=(0,0))
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Auto-determine topology if not specified
    if topology == (0, 0)
        # Try to make a roughly square decomposition for 2D
        p1 = isqrt(nprocs)
        while nprocs % p1 != 0
            p1 -= 1
        end
        p2 = nprocs ÷ p1
        topology = (p1, p2)
    end

    is_root = rank == 0

    if is_root
        @info "MPI initialized" nprocs topology
    end

    return MPIConfig(comm, rank, nprocs, topology, is_root)
end

#=
================================================================================
                        PENCIL DECOMPOSITION
================================================================================
=#

"""
    PencilDecomp

Wrapper for PencilArrays decomposition with metadata.
"""
struct PencilDecomp
    pencil::Pencil{3,2,MPI.Comm}  # 3D data, 2D decomposition
    local_range::NTuple{3, UnitRange{Int}}
    global_dims::NTuple{3, Int}
end

"""
    create_pencil_decomposition(nx, ny, nz, mpi_config) -> PencilDecomp

Create a 2D pencil decomposition for 3D data.

The decomposition is in the (y, z) dimensions, keeping x local for efficient
horizontal FFTs.
"""
function create_pencil_decomposition(nx::Int, ny::Int, nz::Int, mpi_config::MPIConfig)
    # Create topology for decomposition
    # Decompose in y and z, keep x local for efficient FFTs
    topo = MPITopology(mpi_config.comm, mpi_config.topology)

    # Create pencil: full domain size, decomposed in dims 2 and 3 (y and z)
    pencil = Pencil(topo, (nx, ny, nz), (2, 3))

    # Get local index ranges (LogicalOrder is default)
    local_range = range_local(pencil)

    return PencilDecomp(pencil, local_range, (nx, ny, nz))
end

#=
================================================================================
                        GRID INITIALIZATION WITH MPI
================================================================================
=#

"""
    init_mpi_grid(params::QGParams, mpi_config::MPIConfig) -> Grid

Initialize a Grid with MPI-distributed arrays using PencilArrays.

# Arguments
- `params::QGParams`: Model parameters
- `mpi_config::MPIConfig`: MPI configuration from setup_mpi_environment()

# Returns
Grid with:
- `decomp::PencilDecomp`: The pencil decomposition
- `kh2::PencilArray`: Distributed wavenumber squared array

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

    # Create pencil decomposition
    decomp = create_pencil_decomposition(nx, ny, nz, mpi_config)

    # Horizontal grid spacing
    dx = params.Lx / nx
    dy = params.Ly / ny

    # Vertical grid (same on all processes)
    z = T.(collect(range(0, 2π; length=nz)))
    dz = diff(z)

    # Wavenumbers (same on all processes for now)
    kx = T.([i <= nx÷2 ? (2π/params.Lx)*(i-1) : (2π/params.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/params.Ly)*(j-1) : (2π/params.Ly)*(j-1-ny) for j in 1:ny])

    # Create distributed kh2 array
    kh2_pencil = PencilArray{T}(undef, decomp.pencil)

    # Fill local portion of kh2
    local_range = decomp.local_range
    parent_arr = parent(kh2_pencil)

    for k_local in axes(parent_arr, 3)
        for j_local in axes(parent_arr, 2)
            for i_local in axes(parent_arr, 1)
                # Map local to global indices
                i_global = local_range[1][i_local]
                j_global = local_range[2][j_local]
                # k doesn't need mapping for wavenumbers (kh2 only depends on i,j)
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
    init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64) -> State

Initialize a State with MPI-distributed PencilArrays.

All fields are allocated as PencilArrays using the grid's decomposition.
"""
function QGYBJ.init_mpi_state(grid::Grid, mpi_config::MPIConfig; T=Float64)
    decomp = grid.decomp
    if decomp === nothing
        error("Grid does not have MPI decomposition. Use init_mpi_grid() first.")
    end

    pencil = decomp.pencil

    # Allocate spectral (complex) fields
    q   = PencilArray{Complex{T}}(undef, pencil); fill!(q, 0)
    psi = PencilArray{Complex{T}}(undef, pencil); fill!(psi, 0)
    A   = PencilArray{Complex{T}}(undef, pencil); fill!(A, 0)
    B   = PencilArray{Complex{T}}(undef, pencil); fill!(B, 0)
    C   = PencilArray{Complex{T}}(undef, pencil); fill!(C, 0)

    # Allocate real-space (real) fields
    u = PencilArray{T}(undef, pencil); fill!(u, 0)
    v = PencilArray{T}(undef, pencil); fill!(v, 0)
    w = PencilArray{T}(undef, pencil); fill!(w, 0)

    return State{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end

#=
================================================================================
                        FFT TRANSFORMS WITH PENCILFFTS
================================================================================
=#

"""
    MPIPlans

FFT plans for MPI-parallel execution using PencilFFTs.
"""
struct MPIPlans
    forward::PencilFFTs.PencilFFTPlan
    backward::PencilFFTs.PencilFFTPlan
    input_pencil::Pencil
    output_pencil::Pencil
    work_arrays::NamedTuple
end

"""
    plan_mpi_transforms(grid::Grid, mpi_config::MPIConfig) -> MPIPlans

Create PencilFFTs plans for parallel FFT execution.

Plans are created for 2D horizontal FFTs (dimensions 1 and 2) while
keeping data distributed in dimension 3 (vertical).

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

    pencil = decomp.pencil

    # Create transforms for each dimension:
    # - Dimensions 1 and 2 (x and y): FFT
    # - Dimension 3 (z): NoTransform (stays in physical space)
    # According to PencilFFTs API, pass a tuple of transforms for each dimension
    transform = (
        PencilFFTs.Transforms.FFT(),      # x dimension
        PencilFFTs.Transforms.FFT(),      # y dimension
        PencilFFTs.Transforms.NoTransform()  # z dimension (no transform)
    )

    # Create the PencilFFT plan
    plan = PencilFFTPlan(pencil, transform)

    # Get input and output pencil configurations
    input_pencil = first_pencil(plan)
    output_pencil = last_pencil(plan)

    # Allocate work arrays for transforms
    work_in = PencilArray{Complex{Float64}}(undef, input_pencil)
    work_out = PencilArray{Complex{Float64}}(undef, output_pencil)

    # Create inverse plan using inv()
    inv_plan = inv(plan)

    return MPIPlans(
        plan,
        inv_plan,
        input_pencil,
        output_pencil,
        (input=work_in, output=work_out)
    )
end

"""
    fft_forward!(dst, src, plans::MPIPlans)

Perform forward FFT using PencilFFTs.

Transforms dimensions 1 and 2 (horizontal) of the input array.
"""
function QGYBJ.fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    mul!(dst, plans.forward, src)
    return dst
end

"""
    fft_backward!(dst, src, plans::MPIPlans)

Perform inverse FFT using PencilFFTs.

Transforms dimensions 1 and 2 (horizontal) of the input array.

Note: Uses ldiv! which provides NORMALIZED inverse transform (consistent with FFTW.ifft).
This differs from mul!(dst, inv(plan), src) which would be unnormalized.
"""
function QGYBJ.fft_backward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)
    # Use ldiv! for normalized inverse transform
    # This is equivalent to: dst = plans.forward \ src
    ldiv!(dst, plans.forward, src)
    return dst
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

Note: PencilArrays doesn't have a built-in scatter function, so we implement
it manually by having root broadcast data and each process extract its local portion.
"""
function QGYBJ.scatter_from_root(arr, grid::Grid, mpi_config::MPIConfig)
    decomp = grid.decomp
    pencil = decomp.pencil
    local_range = decomp.local_range

    # Allocate distributed array
    distributed = PencilArray{eltype(arr)}(undef, pencil)
    parent_arr = parent(distributed)

    # Broadcast full array from root to all processes
    # (This is simple but not memory-efficient for very large arrays)
    if mpi_config.is_root
        global_arr = arr
    else
        global_arr = similar(arr)
    end
    MPI.Bcast!(global_arr, 0, mpi_config.comm)

    # Each process extracts its local portion
    for k_local in axes(parent_arr, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(parent_arr, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(parent_arr, 1)
                i_global = local_range[1][i_local]
                parent_arr[i_local, j_local, k_local] = global_arr[i_global, j_global, k_global]
            end
        end
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

Get the local index ranges for the current process.
"""
function QGYBJ.local_indices(grid::Grid)
    decomp = grid.decomp
    if decomp === nothing
        return (1:grid.nx, 1:grid.ny, 1:grid.nz)
    end
    return decomp.local_range
end

#=
================================================================================
                        PARALLEL I/O HELPERS
================================================================================
=#

"""
    write_mpi_field(filename, varname, arr::PencilArray, grid::Grid, mpi_config)

Write a distributed field to a NetCDF file using parallel I/O or gather.
"""
function QGYBJ.write_mpi_field(filename::String, varname::String,
                               arr::PencilArray, grid::Grid, mpi_config::MPIConfig)
    # For now, gather to root and write serially
    # Future: Use NCDatasets parallel I/O
    gathered = gather(arr)

    if mpi_config.is_root && gathered !== nothing
        # Write using standard NetCDF (handled by netcdf_io.jl)
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

Uses hash-based seeding to ensure reproducibility across different
process counts.
"""
function QGYBJ.init_mpi_random_field!(arr::PencilArray, grid::Grid,
                                       amplitude::Real, seed_offset::Int=0)
    decomp = grid.decomp
    local_range = decomp.local_range
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

# Export extension functions
export MPIConfig, MPIPlans, PencilDecomp

end # module QGYBJMPIExt
