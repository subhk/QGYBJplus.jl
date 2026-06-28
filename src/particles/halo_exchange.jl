"""
Halo exchange system for particle advection in parallel domains.

This module handles the communication of velocity field data between neighboring
MPI domains to enable accurate interpolation for particles near domain boundaries.

# Supported Decompositions
- 1D decomposition (x-direction only): Full y-domain on each rank
- 2D decomposition (x and y directions): Both dimensions distributed

# Halo Width Requirements
The halo width depends on the interpolation scheme used:
- TRILINEAR: 1 cell  (2×2×2 stencil)
- TRICUBIC:  2 cells (4×4×4 stencil)
- QUINTIC:   3 cells (6×6×6 stencil)
- ADAPTIVE:  3 cells (supports all schemes)

# 2D Decomposition Layout
For a 2D process grid (px × py), the neighbor layout is:

    NW(6) --- N(2) --- NE(7)
      |        |        |
    W(0) --- local --- E(1)
      |        |        |
    SW(4) --- S(3) --- SE(5)

where directions are: W=left(-x), E=right(+x), S=bottom(-y), N=top(+y)
"""

module HaloExchange

# Import interpolation types for halo width calculation
using ..InterpolationSchemes: InterpolationMethod, TRILINEAR, TRICUBIC, QUINTIC, ADAPTIVE, required_halo_width

export HaloInfo, setup_halo_exchange!, exchange_velocity_halos!,
       interpolate_velocity_with_halos,
       compute_process_grid, compute_local_size, compute_start_index, compute_neighbors_2d

# Direction indices for neighbors array
const DIR_W  = 1  # West  (left,  -x)
const DIR_E  = 2  # East  (right, +x)
const DIR_S  = 3  # South (bottom, -y)
const DIR_N  = 4  # North (top, +y)
const DIR_SW = 5  # Southwest corner
const DIR_SE = 6  # Southeast corner
const DIR_NW = 7  # Northwest corner
const DIR_NE = 8  # Northeast corner

"""
Information about halo regions for MPI communication in 2D decomposition.

# Fields
- `halo_width`: Number of ghost cells on each side (depends on interpolation scheme)
- `u_extended, v_extended, w_extended`: Extended velocity arrays (x halos always; y halos only for 2D)
- `local_start, local_end`: Indices of local domain within extended arrays
- `neighbors`: Array of 8 neighbor ranks [W, E, S, N, SW, SE, NW, NE], -1 if none
- `nx_global, ny_global, nz_global`: Global domain dimensions
- `nx_local, ny_local, nz_local`: Local domain dimensions (without halos)
- `px, py`: Process grid dimensions
- `rank_x, rank_y`: This rank's position in the process grid
- `periodic_x, periodic_y`: Boundary condition flags
- `is_2d_decomposition`: True if y is distributed across processes
- `comm, rank, nprocs`: MPI information
"""
mutable struct HaloInfo{T<:AbstractFloat}
    # Halo width (number of ghost cells on each side)
    halo_width::Int

    # Extended arrays including halos in x and (optionally) y
    # Layout: (nz_local, nx_local + 2*hw, ny_local + 2*hw_y)
    u_extended::Array{T,3}
    v_extended::Array{T,3}
    w_extended::Array{T,3}

    # Local domain indices in extended arrays
    local_start::NTuple{3,Int}  # (k_start, i_start, j_start) = (z, x, y)
    local_end::NTuple{3,Int}    # (k_end, i_end, j_end) = (z, x, y)

    # Neighbor information for 2D decomposition
    # [W, E, S, N, SW, SE, NW, NE] - value is rank or -1 if no neighbor
    neighbors::Vector{Int}

    # Communication buffers for x-direction (left/right)
    send_west::Vector{T}
    send_east::Vector{T}
    recv_west::Vector{T}
    recv_east::Vector{T}

    # Communication buffers for y-direction (bottom/top)
    send_south::Vector{T}
    send_north::Vector{T}
    recv_south::Vector{T}
    recv_north::Vector{T}

    # Communication buffers for corners (needed for 2D stencils)
    send_sw::Vector{T}
    send_se::Vector{T}
    send_nw::Vector{T}
    send_ne::Vector{T}
    recv_sw::Vector{T}
    recv_se::Vector{T}
    recv_nw::Vector{T}
    recv_ne::Vector{T}

    # MPI info
    comm::Any
    rank::Int
    nprocs::Int

    # Global domain dimensions
    nx_global::Int
    ny_global::Int
    nz_global::Int

    # Local domain dimensions (without halos)
    nx_local::Int
    ny_local::Int
    nz_local::Int

    # Process grid info
    px::Int  # Number of processes in x-direction
    py::Int  # Number of processes in y-direction
    rank_x::Int  # This rank's x-position in process grid (0-based)
    rank_y::Int  # This rank's y-position in process grid (0-based)

    # Boundary conditions
    periodic_x::Bool
    periodic_y::Bool

    # Decomposition type flag
    is_2d_decomposition::Bool

    function HaloInfo{T}(grid_info, rank::Int, nprocs::Int, comm;
                         halo_width::Union{Int,Nothing}=nothing,
                         interpolation_method::InterpolationMethod=TRILINEAR,
                         periodic_x::Bool=true,
                         periodic_y::Bool=true,
                         local_dims::Union{Nothing,Tuple{Int,Int,Int}}=nothing,
                         process_grid::Union{Nothing,Tuple{Int,Int}}=nothing) where T
        # grid_info can be a Grid object or a NamedTuple with (nx, ny, nz)

        # Extract global dimensions
        nx_global = grid_info.nx
        ny_global = grid_info.ny
        nz_global = grid_info.nz

        # Compute halo width from interpolation method if not explicitly provided
        if halo_width === nothing
            halo_width = required_halo_width(interpolation_method)
        end

        # Determine process grid (px × py)
        if process_grid !== nothing
            px, py = process_grid
        else
            # Auto-determine: try to make grid as square as possible
            px, py = compute_process_grid(nprocs)
        end

        @assert px * py == nprocs "Process grid ($px × $py) must equal nprocs ($nprocs)"

        # Compute this rank's position in the process grid
        # Layout: rank = rank_y * px + rank_x (row-major)
        rank_x = rank % px
        rank_y = rank ÷ px

        # Determine if this is 1D or 2D decomposition
        is_2d_decomposition = py > 1

        # Get LOCAL grid dimensions
        if local_dims !== nothing
            nz_local, nx_local, ny_local = local_dims
        else
            # Compute local dimensions from global and process grid
            nx_local = compute_local_size(nx_global, px, rank_x)
            ny_local = compute_local_size(ny_global, py, rank_y)
            nz_local = nz_global  # z is never decomposed for particles
        end

        max_halo = is_2d_decomposition ? min(nx_local, ny_local) : nx_local
        if max_halo < 1
            error("Local domain too small for halo exchange (nx_local=$nx_local, ny_local=$ny_local).")
        elseif halo_width < 1
            error("Halo width must be >= 1 (got $halo_width).")
        elseif halo_width > max_halo
            @warn "Halo width $halo_width exceeds local grid size (nx_local=$nx_local, ny_local=$ny_local); clamping to $max_halo."
            halo_width = max_halo
        end
        halo_width_y = is_2d_decomposition ? halo_width : 0

        # Extended grid dimensions (local + 2*halo_width in x, and y only for 2D)
        nx_ext = nx_local + 2 * halo_width
        ny_ext = ny_local + 2 * halo_width_y
        nz_ext = nz_local

        # Create extended arrays
        u_extended = zeros(T, nz_ext, nx_ext, ny_ext)
        v_extended = zeros(T, nz_ext, nx_ext, ny_ext)
        w_extended = zeros(T, nz_ext, nx_ext, ny_ext)

        # Local domain indices in extended array (1-based)
        local_start = (1, halo_width + 1, halo_width_y + 1)
        local_end = (nz_local, halo_width + nx_local, halo_width_y + ny_local)

        # Determine all 8 neighbors
        neighbors = compute_neighbors_2d(rank_x, rank_y, px, py, periodic_x, periodic_y)

        # Communication buffer sizes
        # X-direction buffers: halo_width × ny_local × nz_local × 3 components
        x_buffer_size = halo_width * ny_local * nz_local * 3
        send_west = Vector{T}(undef, x_buffer_size)
        send_east = Vector{T}(undef, x_buffer_size)
        recv_west = Vector{T}(undef, x_buffer_size)
        recv_east = Vector{T}(undef, x_buffer_size)

        # Y-direction buffers: nx_local × halo_width_y × nz_local × 3 components
        y_buffer_size = nx_local * halo_width_y * nz_local * 3
        send_south = Vector{T}(undef, y_buffer_size)
        send_north = Vector{T}(undef, y_buffer_size)
        recv_south = Vector{T}(undef, y_buffer_size)
        recv_north = Vector{T}(undef, y_buffer_size)

        # Corner buffers: halo_width × halo_width_y × nz_local × 3 components
        corner_buffer_size = halo_width * halo_width_y * nz_local * 3
        send_sw = Vector{T}(undef, corner_buffer_size)
        send_se = Vector{T}(undef, corner_buffer_size)
        send_nw = Vector{T}(undef, corner_buffer_size)
        send_ne = Vector{T}(undef, corner_buffer_size)
        recv_sw = Vector{T}(undef, corner_buffer_size)
        recv_se = Vector{T}(undef, corner_buffer_size)
        recv_nw = Vector{T}(undef, corner_buffer_size)
        recv_ne = Vector{T}(undef, corner_buffer_size)

        new{T}(
            halo_width,
            u_extended, v_extended, w_extended,
            local_start, local_end,
            neighbors,
            send_west, send_east, recv_west, recv_east,
            send_south, send_north, recv_south, recv_north,
            send_sw, send_se, send_nw, send_ne,
            recv_sw, recv_se, recv_nw, recv_ne,
            comm, rank, nprocs,
            nx_global, ny_global, nz_global,
            nx_local, ny_local, nz_local,
            px, py, rank_x, rank_y,
            periodic_x, periodic_y,
            is_2d_decomposition
        )
    end
end

"""
Compute an approximately square process grid for nprocs processes.
"""
function compute_process_grid(nprocs::Int)
    # Find factors closest to sqrt(nprocs)
    py = isqrt(nprocs)
    while nprocs % py != 0
        py -= 1
    end
    px = nprocs ÷ py
    return px, py
end

"""
Compute local size for a dimension given global size, number of processes, and rank.
"""
function compute_local_size(n_global::Int, nprocs::Int, rank::Int)
    base = n_global ÷ nprocs
    remainder = n_global % nprocs
    return rank < remainder ? base + 1 : base
end

"""
Compute start index (0-based) for a dimension given global size, number of processes, and rank.
"""
function compute_start_index(n_global::Int, nprocs::Int, rank::Int)
    base = n_global ÷ nprocs
    remainder = n_global % nprocs
    if rank < remainder
        return rank * (base + 1)
    else
        return remainder * (base + 1) + (rank - remainder) * base
    end
end

"""
Compute all 8 neighbors for 2D decomposition.
Returns vector of ranks: [W, E, S, N, SW, SE, NW, NE], -1 if no neighbor.
"""
function compute_neighbors_2d(rank_x::Int, rank_y::Int, px::Int, py::Int,
                              periodic_x::Bool, periodic_y::Bool)
    neighbors = fill(-1, 8)

    # Helper to convert (rx, ry) to rank
    function to_rank(rx, ry)
        return ry * px + rx
    end

    # West neighbor (left, -x)
    if rank_x > 0
        neighbors[DIR_W] = to_rank(rank_x - 1, rank_y)
    elseif periodic_x
        neighbors[DIR_W] = to_rank(px - 1, rank_y)
    end

    # East neighbor (right, +x)
    if rank_x < px - 1
        neighbors[DIR_E] = to_rank(rank_x + 1, rank_y)
    elseif periodic_x
        neighbors[DIR_E] = to_rank(0, rank_y)
    end

    # South neighbor (bottom, -y)
    if rank_y > 0
        neighbors[DIR_S] = to_rank(rank_x, rank_y - 1)
    elseif periodic_y
        neighbors[DIR_S] = to_rank(rank_x, py - 1)
    end

    # North neighbor (top, +y)
    if rank_y < py - 1
        neighbors[DIR_N] = to_rank(rank_x, rank_y + 1)
    elseif periodic_y
        neighbors[DIR_N] = to_rank(rank_x, 0)
    end

    # Southwest corner
    rx_sw = periodic_x && rank_x == 0 ? px - 1 : rank_x - 1
    ry_sw = periodic_y && rank_y == 0 ? py - 1 : rank_y - 1
    if rx_sw >= 0 && ry_sw >= 0
        neighbors[DIR_SW] = to_rank(rx_sw, ry_sw)
    end

    # Southeast corner
    rx_se = periodic_x && rank_x == px - 1 ? 0 : rank_x + 1
    ry_se = periodic_y && rank_y == 0 ? py - 1 : rank_y - 1
    if rx_se < px && ry_se >= 0
        neighbors[DIR_SE] = to_rank(rx_se, ry_se)
    end

    # Northwest corner
    rx_nw = periodic_x && rank_x == 0 ? px - 1 : rank_x - 1
    ry_nw = periodic_y && rank_y == py - 1 ? 0 : rank_y + 1
    if rx_nw >= 0 && ry_nw < py
        neighbors[DIR_NW] = to_rank(rx_nw, ry_nw)
    end

    # Northeast corner
    rx_ne = periodic_x && rank_x == px - 1 ? 0 : rank_x + 1
    ry_ne = periodic_y && rank_y == py - 1 ? 0 : rank_y + 1
    if rx_ne < px && ry_ne < py
        neighbors[DIR_NE] = to_rank(rx_ne, ry_ne)
    end

    return neighbors
end

"""
    setup_halo_exchange!(tracker)

Set up halo exchange system for particle tracker.
"""
function setup_halo_exchange!(tracker)
    if !tracker.is_parallel
        return nothing
    end

    T = eltype(tracker.u_field)

    # Create a minimal struct with just the required grid info
    grid_dims = (nx=tracker.nx, ny=tracker.ny, nz=tracker.nz)
    local_dims = size(tracker.u_field)
    process_grid = tracker.local_domain !== nothing && hasproperty(tracker.local_domain, :px) ?
                   (tracker.local_domain.px, tracker.local_domain.py) : nothing

    halo_info = HaloInfo{T}(grid_dims, tracker.rank, tracker.nprocs, tracker.comm;
                            local_dims=local_dims,
                            process_grid=process_grid,
                            periodic_x=tracker.config.periodic_x,
                            periodic_y=tracker.config.periodic_y,
                            interpolation_method=tracker.config.interpolation_method)

    return halo_info
end

"""
    exchange_velocity_halos!(halo_info, u_field, v_field, w_field)

Exchange velocity field halos between neighboring MPI domains in 2D.
Performs exchanges in order: x-direction, y-direction, then corners.
"""
function exchange_velocity_halos!(halo_info::HaloInfo{T},
                                 u_field::Array{T,3},
                                 v_field::Array{T,3},
                                 w_field::Array{T,3}) where T

    if halo_info.comm === nothing
        return halo_info
    end

    try
        if Base.find_package("MPI") === nothing
            @warn "MPI not available; skipping halo exchange"
            return halo_info
        end

        # Import MPI module
        M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))

        # Copy local data to extended arrays (center region)
        copy_local_to_extended_2d!(halo_info, u_field, v_field, w_field)

        # Exchange x-direction halos (West/East)
        exchange_x_halos!(halo_info, M)

        # Exchange y-direction halos (South/North)
        if halo_info.is_2d_decomposition
            exchange_y_halos!(halo_info, M)

            # Exchange corner halos (needed for 2D stencils)
            exchange_corner_halos!(halo_info, M)
        end

    catch e
        @warn "Halo exchange failed: $e"
        rethrow(e)
    end

    return halo_info
end

"""
Copy local velocity data to center of extended arrays.
"""
function copy_local_to_extended_2d!(halo_info::HaloInfo{T},
                                    u_field::Array{T,3},
                                    v_field::Array{T,3},
                                    w_field::Array{T,3}) where T

    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end

    # Copy local data to center of extended arrays
    halo_info.u_extended[k_start:k_end, i_start:i_end, j_start:j_end] .= u_field
    halo_info.v_extended[k_start:k_end, i_start:i_end, j_start:j_end] .= v_field
    halo_info.w_extended[k_start:k_end, i_start:i_end, j_start:j_end] .= w_field
end

"""
Exchange halos in x-direction (West/East neighbors).
"""
function exchange_x_halos!(halo_info::HaloInfo{T}, M) where T
    hw = halo_info.halo_width
    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end

    west = halo_info.neighbors[DIR_W]
    east = halo_info.neighbors[DIR_E]

    # Pack data for West neighbor (our left boundary)
    if west >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in i_start:(i_start+hw-1)
            halo_info.send_west[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_west[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_west[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack data for East neighbor (our right boundary)
    if east >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end-hw+1):i_end
            halo_info.send_east[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_east[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_east[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Non-blocking communication
    recv_reqs = M.Request[]
    send_reqs = M.Request[]

    # Post receives
    if west >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_west, west, 100, halo_info.comm))
    end
    if east >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_east, east, 101, halo_info.comm))
    end

    # Post sends
    if east >= 0
        push!(send_reqs, M.Isend(halo_info.send_east, east, 100, halo_info.comm))
    end
    if west >= 0
        push!(send_reqs, M.Isend(halo_info.send_west, west, 101, halo_info.comm))
    end

    # Wait for receives
    !isempty(recv_reqs) && M.Waitall(recv_reqs)

    # Unpack received data
    if west >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in 1:hw
            halo_info.u_extended[k, i, j] = halo_info.recv_west[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_west[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_west[idx+2]
            idx += 3
        end
    end

    if east >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end+1):(i_end+hw)
            halo_info.u_extended[k, i, j] = halo_info.recv_east[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_east[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_east[idx+2]
            idx += 3
        end
    end

    # Wait for sends to complete
    !isempty(send_reqs) && M.Waitall(send_reqs)
end

"""
Exchange halos in y-direction (South/North neighbors).
"""
function exchange_y_halos!(halo_info::HaloInfo{T}, M) where T
    hw = halo_info.halo_width
    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end

    south = halo_info.neighbors[DIR_S]
    north = halo_info.neighbors[DIR_N]

    # Pack data for South neighbor (our bottom boundary)
    if south >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:(j_start+hw-1), i in i_start:i_end
            halo_info.send_south[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_south[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_south[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack data for North neighbor (our top boundary)
    if north >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end-hw+1):j_end, i in i_start:i_end
            halo_info.send_north[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_north[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_north[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Non-blocking communication
    recv_reqs = M.Request[]
    send_reqs = M.Request[]

    # Post receives
    if south >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_south, south, 200, halo_info.comm))
    end
    if north >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_north, north, 201, halo_info.comm))
    end

    # Post sends
    if north >= 0
        push!(send_reqs, M.Isend(halo_info.send_north, north, 200, halo_info.comm))
    end
    if south >= 0
        push!(send_reqs, M.Isend(halo_info.send_south, south, 201, halo_info.comm))
    end

    # Wait for receives
    !isempty(recv_reqs) && M.Waitall(recv_reqs)

    # Unpack received data
    if south >= 0
        idx = 1
        for k in k_start:k_end, j in 1:hw, i in i_start:i_end
            halo_info.u_extended[k, i, j] = halo_info.recv_south[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_south[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_south[idx+2]
            idx += 3
        end
    end

    if north >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end+1):(j_end+hw), i in i_start:i_end
            halo_info.u_extended[k, i, j] = halo_info.recv_north[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_north[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_north[idx+2]
            idx += 3
        end
    end

    # Wait for sends to complete
    !isempty(send_reqs) && M.Waitall(send_reqs)
end

"""
Exchange corner halos (SW, SE, NW, NE).
Required for proper 2D interpolation stencils.
"""
function exchange_corner_halos!(halo_info::HaloInfo{T}, M) where T
    hw = halo_info.halo_width
    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end

    sw = halo_info.neighbors[DIR_SW]
    se = halo_info.neighbors[DIR_SE]
    nw = halo_info.neighbors[DIR_NW]
    ne = halo_info.neighbors[DIR_NE]

    # Pack SW corner (our bottom-left)
    if sw >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:(j_start+hw-1), i in i_start:(i_start+hw-1)
            halo_info.send_sw[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_sw[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_sw[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack SE corner (our bottom-right)
    if se >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:(j_start+hw-1), i in (i_end-hw+1):i_end
            halo_info.send_se[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_se[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_se[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack NW corner (our top-left)
    if nw >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end-hw+1):j_end, i in i_start:(i_start+hw-1)
            halo_info.send_nw[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_nw[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_nw[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack NE corner (our top-right)
    if ne >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end-hw+1):j_end, i in (i_end-hw+1):i_end
            halo_info.send_ne[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_ne[idx+1] = halo_info.v_extended[k, i, j]
            halo_info.send_ne[idx+2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Non-blocking communication
    recv_reqs = M.Request[]
    send_reqs = M.Request[]

    # Post receives
    if sw >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_sw, sw, 300, halo_info.comm))
    end
    if se >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_se, se, 301, halo_info.comm))
    end
    if nw >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_nw, nw, 302, halo_info.comm))
    end
    if ne >= 0
        push!(recv_reqs, M.Irecv!(halo_info.recv_ne, ne, 303, halo_info.comm))
    end

    # Post sends (to opposite corners)
    if ne >= 0
        push!(send_reqs, M.Isend(halo_info.send_ne, ne, 300, halo_info.comm))
    end
    if nw >= 0
        push!(send_reqs, M.Isend(halo_info.send_nw, nw, 301, halo_info.comm))
    end
    if se >= 0
        push!(send_reqs, M.Isend(halo_info.send_se, se, 302, halo_info.comm))
    end
    if sw >= 0
        push!(send_reqs, M.Isend(halo_info.send_sw, sw, 303, halo_info.comm))
    end

    # Wait for receives
    !isempty(recv_reqs) && M.Waitall(recv_reqs)

    # Unpack corners
    if sw >= 0
        idx = 1
        for k in k_start:k_end, j in 1:hw, i in 1:hw
            halo_info.u_extended[k, i, j] = halo_info.recv_sw[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_sw[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_sw[idx+2]
            idx += 3
        end
    end

    if se >= 0
        idx = 1
        for k in k_start:k_end, j in 1:hw, i in (i_end+1):(i_end+hw)
            halo_info.u_extended[k, i, j] = halo_info.recv_se[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_se[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_se[idx+2]
            idx += 3
        end
    end

    if nw >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end+1):(j_end+hw), i in 1:hw
            halo_info.u_extended[k, i, j] = halo_info.recv_nw[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_nw[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_nw[idx+2]
            idx += 3
        end
    end

    if ne >= 0
        idx = 1
        for k in k_start:k_end, j in (j_end+1):(j_end+hw), i in (i_end+1):(i_end+hw)
            halo_info.u_extended[k, i, j] = halo_info.recv_ne[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_ne[idx+1]
            halo_info.w_extended[k, i, j] = halo_info.recv_ne[idx+2]
            idx += 3
        end
    end

    # Wait for sends
    !isempty(send_reqs) && M.Waitall(send_reqs)
end

"""
    interpolate_velocity_with_halos(x, y, z, tracker, halo_info)

Interpolate velocity using extended arrays with halo data.
Supports both 1D and 2D MPI decomposition.

Uses trilinear interpolation with:
- X-direction: halo data handles cross-boundary interpolation
- Y-direction: halo data handles cross-boundary interpolation for 2D; periodic wrapping for 1D
- Z-direction: clamping to [1, nz] (vertical boundaries are not periodic)
"""
function interpolate_velocity_with_halos(x::T, y::T, z::T,
                                       tracker, halo_info::HaloInfo{T}) where T

    # Handle periodic boundaries using GLOBAL domain lengths
    x_periodic = halo_info.periodic_x ? tracker.x0 + mod(x - tracker.x0, tracker.Lx) : x
    y_periodic = halo_info.periodic_y ? tracker.y0 + mod(y - tracker.y0, tracker.Ly) : y
    z_min = -tracker.Lz
    z0 = z_min + tracker.dz / 2
    z_max = zero(T)
    z_clamped = clamp(z, z0, z_max)

    # Compute local domain start positions
    x_start = tracker.x0 + compute_start_index(halo_info.nx_global, halo_info.px, halo_info.rank_x) * tracker.dx
    y_start = tracker.y0 + compute_start_index(halo_info.ny_global, halo_info.py, halo_info.rank_y) * tracker.dy

    # Convert to local domain coordinates
    x_local = x_periodic - x_start
    y_local = y_periodic - y_start

    hw = halo_info.halo_width
    hy = halo_info.is_2d_decomposition ? halo_info.halo_width : 0

    # Convert to extended grid indices (accounting for halo offset)
    fx = x_local / tracker.dx + hw + 1  # +1 for 1-based indexing
    fy = y_local / tracker.dy + hy + 1
    fz = (z_clamped - z0) / tracker.dz + 1

    # Get integer and fractional parts
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)

    rx = fx - ix
    ry = fy - iy
    rz = fz - iz

    # Bounds check for extended arrays
    nz_ext, nx_ext, ny_ext = size(halo_info.u_extended)

    # Clamp indices to extended array bounds (halos handle boundary data)
    ix1 = max(1, min(nx_ext, ix))
    ix2 = max(1, min(nx_ext, ix + 1))
    if !halo_info.is_2d_decomposition && halo_info.periodic_y
        iy1 = mod(iy - 1, ny_ext) + 1
        iy2 = mod(iy, ny_ext) + 1
    else
        iy1 = max(1, min(ny_ext, iy))
        iy2 = max(1, min(ny_ext, iy + 1))
    end
    iz1 = max(1, min(nz_ext, iz))
    iz2 = max(1, min(nz_ext, iz + 1))

    # Trilinear interpolation using extended arrays
    # Bottom face (z1)
    u_z1_y1 = (1-rx) * halo_info.u_extended[iz1, ix1, iy1] + rx * halo_info.u_extended[iz1, ix2, iy1]
    u_z1_y2 = (1-rx) * halo_info.u_extended[iz1, ix1, iy2] + rx * halo_info.u_extended[iz1, ix2, iy2]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2

    v_z1_y1 = (1-rx) * halo_info.v_extended[iz1, ix1, iy1] + rx * halo_info.v_extended[iz1, ix2, iy1]
    v_z1_y2 = (1-rx) * halo_info.v_extended[iz1, ix1, iy2] + rx * halo_info.v_extended[iz1, ix2, iy2]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2

    w_z1_y1 = (1-rx) * halo_info.w_extended[iz1, ix1, iy1] + rx * halo_info.w_extended[iz1, ix2, iy1]
    w_z1_y2 = (1-rx) * halo_info.w_extended[iz1, ix1, iy2] + rx * halo_info.w_extended[iz1, ix2, iy2]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2

    # Top face (z2)
    u_z2_y1 = (1-rx) * halo_info.u_extended[iz2, ix1, iy1] + rx * halo_info.u_extended[iz2, ix2, iy1]
    u_z2_y2 = (1-rx) * halo_info.u_extended[iz2, ix1, iy2] + rx * halo_info.u_extended[iz2, ix2, iy2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2

    v_z2_y1 = (1-rx) * halo_info.v_extended[iz2, ix1, iy1] + rx * halo_info.v_extended[iz2, ix2, iy1]
    v_z2_y2 = (1-rx) * halo_info.v_extended[iz2, ix1, iy2] + rx * halo_info.v_extended[iz2, ix2, iy2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2

    w_z2_y1 = (1-rx) * halo_info.w_extended[iz2, ix1, iy1] + rx * halo_info.w_extended[iz2, ix2, iy1]
    w_z2_y2 = (1-rx) * halo_info.w_extended[iz2, ix1, iy2] + rx * halo_info.w_extended[iz2, ix2, iy2]
    w_z2 = (1-ry) * w_z2_y1 + ry * w_z2_y2

    # Final interpolation in z
    u_interp = (1-rz) * u_z1 + rz * u_z2
    v_interp = (1-rz) * v_z1 + rz * v_z2
    w_interp = (1-rz) * w_z1 + rz * w_z2

    # For 2D advection, set w to zero
    if !tracker.config.use_3d_advection
        w_interp = zero(T)
    end

    return u_interp, v_interp, w_interp
end

# Legacy compatibility: alias for old function names
const copy_local_to_extended! = copy_local_to_extended_2d!
const pack_halo_data! = nothing  # Removed, now integrated into exchange functions
const unpack_halo_data! = nothing  # Removed, now integrated into exchange functions

end # module HaloExchange

using .HaloExchange
