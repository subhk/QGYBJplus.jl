"""
Halo exchange system for particle advection in parallel domains.

This module handles the communication of velocity field data between neighboring
MPI domains to enable accurate interpolation for particles near domain boundaries.

# Halo Width Requirements
The halo width depends on the interpolation scheme used:
- TRILINEAR: 1 cell  (2×2×2 stencil)
- TRICUBIC:  2 cells (4×4×4 stencil)
- QUINTIC:   3 cells (6×6×6 stencil)
- ADAPTIVE:  3 cells (supports all schemes)
"""

module HaloExchange

# Access Grid from grandparent module (QGYBJplus)
const _PARENT = Base.parentmodule(@__MODULE__)           # UnifiedParticleAdvection
const _GRANDPARENT = Base.parentmodule(_PARENT)          # QGYBJplus
const Grid = _GRANDPARENT.Grid

# Import interpolation types for halo width calculation
using ..InterpolationSchemes: InterpolationMethod, TRILINEAR, TRICUBIC, QUINTIC, ADAPTIVE, required_halo_width

export HaloInfo, setup_halo_exchange!, exchange_velocity_halos!,
       interpolate_velocity_with_halos

"""
Information about halo regions for MPI communication.
"""
mutable struct HaloInfo{T<:AbstractFloat}
    # Halo width (number of ghost cells on each side)
    halo_width::Int
    
    # Extended arrays including halos
    u_extended::Array{T,3}
    v_extended::Array{T,3}
    w_extended::Array{T,3}
    
    # Local domain indices in extended arrays
    local_start::NTuple{3,Int}  # (k_start, i_start, j_start) = (z, x, y)
    local_end::NTuple{3,Int}    # (k_end, i_end, j_end) = (z, x, y)
    
    # Neighbor information
    left_neighbor::Int     # Rank of left neighbor (-1 if none)
    right_neighbor::Int    # Rank of right neighbor (-1 if none)
    
    # Communication buffers
    send_left::Vector{T}
    send_right::Vector{T}
    recv_left::Vector{T}
    recv_right::Vector{T}
    
    # MPI info
    comm::Any
    rank::Int
    nprocs::Int
    
    function HaloInfo{T}(grid::Grid, rank::Int, nprocs::Int, comm;
                         halo_width::Union{Int,Nothing}=nothing,
                         interpolation_method::InterpolationMethod=TRILINEAR,
                         periodic_x::Bool=true,
                         local_dims::Union{Nothing,Tuple{Int,Int,Int}}=nothing) where T
        # Compute halo width from interpolation method if not explicitly provided
        if halo_width === nothing
            halo_width = required_halo_width(interpolation_method)
        end
        # Get LOCAL grid dimensions
        # For 2D pencil decomposition, all three dimensions may be < global size
        # If local_dims provided, use them (order: nz, nx, ny); otherwise compute from 1D decomposition in x
        if local_dims !== nothing
            nz_local, nx_local, ny_local = local_dims
        else
            # Fallback: assume 1D decomposition in x only
            nx_global = grid.nx
            nx_local = nx_global ÷ nprocs
            remainder = nx_global % nprocs
            if rank < remainder
                nx_local += 1
            end
            ny_local = grid.ny
            nz_local = grid.nz
        end

        # Extended grid dimensions (local + 2*halo_width in x-direction)
        # Note: For 2D decomposition, we only add halos in x (slab decomposition for particles)
        nx_ext = nx_local + 2*halo_width
        ny_ext = ny_local  # Use LOCAL ny (may be < grid.ny in 2D decomposition)
        nz_ext = nz_local  # Use LOCAL nz (may be < grid.nz in 2D decomposition)

        # Create extended arrays (sized for LOCAL domain + halos)
        u_extended = zeros(T, nz_ext, nx_ext, ny_ext)
        v_extended = zeros(T, nz_ext, nx_ext, ny_ext)
        w_extended = zeros(T, nz_ext, nx_ext, ny_ext)

        # Local domain indices in extended array (1-based)
        local_start = (1, halo_width + 1, 1)
        local_end = (nz_local, halo_width + nx_local, ny_local)

        # Determine neighbors (1D decomposition in x with periodic boundaries)
        if periodic_x
            # Periodic: rank 0 connects to last rank, last rank connects to rank 0
            left_neighbor = rank > 0 ? rank - 1 : nprocs - 1
            right_neighbor = rank < nprocs - 1 ? rank + 1 : 0
        else
            # Non-periodic: boundary ranks have no neighbor on that side
            left_neighbor = rank > 0 ? rank - 1 : -1
            right_neighbor = rank < nprocs - 1 ? rank + 1 : -1
        end

        # Communication buffer sizes (halo_width * ny_local * nz_local * 3_components)
        buffer_size = halo_width * ny_local * nz_local * 3
        send_left = Vector{T}(undef, buffer_size)
        send_right = Vector{T}(undef, buffer_size)
        recv_left = Vector{T}(undef, buffer_size)
        recv_right = Vector{T}(undef, buffer_size)

        new{T}(
            halo_width,
            u_extended, v_extended, w_extended,
            local_start, local_end,
            left_neighbor, right_neighbor,
            send_left, send_right, recv_left, recv_right,
            comm, rank, nprocs
        )
    end
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
    grid_info = (nx=tracker.nx, ny=tracker.ny, nz=tracker.nz, 
                Lx=tracker.Lx, Ly=tracker.Ly, Lz=tracker.Lz)
    grid = Grid(grid_info)  # Create minimal grid info
    
    halo_info = HaloInfo{T}(grid, tracker.rank, tracker.nprocs, tracker.comm)
    
    return halo_info
end

"""
    exchange_velocity_halos!(halo_info, u_field, v_field, w_field)

Exchange velocity field halos between neighboring MPI domains.
"""
function exchange_velocity_halos!(halo_info::HaloInfo{T}, 
                                 u_field::Array{T,3}, 
                                 v_field::Array{T,3}, 
                                 w_field::Array{T,3}) where T
    
    if halo_info.comm === nothing
        return
    end
    
    try
        if Base.find_package("MPI") === nothing
            @warn "MPI not available; skipping halo exchange"; return halo_info
        end

        # Import MPI module
        M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))

        # Copy local data to extended arrays
        copy_local_to_extended!(halo_info, u_field, v_field, w_field)

        # Prepare send buffers
        pack_halo_data!(halo_info)

        # Post non-blocking receives
        recv_reqs = M.Request[]

        if halo_info.left_neighbor >= 0
            req = M.Irecv!(halo_info.recv_left, halo_info.left_neighbor, 0, halo_info.comm)
            push!(recv_reqs, req)
        end

        if halo_info.right_neighbor >= 0
            req = M.Irecv!(halo_info.recv_right, halo_info.right_neighbor, 1, halo_info.comm)
            push!(recv_reqs, req)
        end

        # Send data
        send_reqs = M.Request[]

        if halo_info.right_neighbor >= 0
            req = M.Isend(halo_info.send_right, halo_info.right_neighbor, 0, halo_info.comm)
            push!(send_reqs, req)
        end

        if halo_info.left_neighbor >= 0
            req = M.Isend(halo_info.send_left, halo_info.left_neighbor, 1, halo_info.comm)
            push!(send_reqs, req)
        end

        # Wait for receives to complete
        if !isempty(recv_reqs)
            M.Waitall(recv_reqs)
        end

        # Unpack received data
        unpack_halo_data!(halo_info)

        # Wait for sends to complete
        if !isempty(send_reqs)
            M.Waitall(send_reqs)
        end

    catch e
        @warn "Halo exchange failed: $e"
    end
    
    return halo_info
end

"""
Copy local velocity data to extended arrays.
"""
function copy_local_to_extended!(halo_info::HaloInfo{T}, 
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
Pack halo data into send buffers.

The left neighbor needs our LEFT edge data (to fill their RIGHT halo).
The right neighbor needs our RIGHT edge data (to fill their LEFT halo).
"""
function pack_halo_data!(halo_info::HaloInfo{T}) where T
    hw = halo_info.halo_width
    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end

    # Pack data to send to left neighbor (our LEFT boundary, for their RIGHT halo)
    if halo_info.left_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in i_start:(i_start+hw-1)
            halo_info.send_left[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_left[idx + 1] = halo_info.v_extended[k, i, j]
            halo_info.send_left[idx + 2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end

    # Pack data to send to right neighbor (our RIGHT boundary, for their LEFT halo)
    if halo_info.right_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end-hw+1):i_end
            halo_info.send_right[idx] = halo_info.u_extended[k, i, j]
            halo_info.send_right[idx + 1] = halo_info.v_extended[k, i, j]
            halo_info.send_right[idx + 2] = halo_info.w_extended[k, i, j]
            idx += 3
        end
    end
end

"""
Unpack received halo data.
"""
function unpack_halo_data!(halo_info::HaloInfo{T}) where T
    hw = halo_info.halo_width
    k_start, i_start, j_start = halo_info.local_start
    k_end, i_end, j_end = halo_info.local_end
    
    # Unpack data from left neighbor (fills left halo region)
    if halo_info.left_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in 1:hw
            halo_info.u_extended[k, i, j] = halo_info.recv_left[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_left[idx + 1]
            halo_info.w_extended[k, i, j] = halo_info.recv_left[idx + 2]
            idx += 3
        end
    end
    
    # Unpack data from right neighbor (fills right halo region)
    if halo_info.right_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end+1):(i_end+hw)
            halo_info.u_extended[k, i, j] = halo_info.recv_right[idx]
            halo_info.v_extended[k, i, j] = halo_info.recv_right[idx + 1]
            halo_info.w_extended[k, i, j] = halo_info.recv_right[idx + 2]
            idx += 3
        end
    end
end

"""
    interpolate_velocity_with_halos(x, y, z, tracker, halo_info)

Interpolate velocity using extended arrays with halo data.
"""
function interpolate_velocity_with_halos(x::T, y::T, z::T, 
                                       tracker, halo_info::HaloInfo{T}) where T
    
    # Handle periodic boundaries
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x
    y_periodic = tracker.config.periodic_y ? mod(y, tracker.Ly) : y
    z_clamped = clamp(z, 0, tracker.Lz)
    
    # Convert to local domain coordinates
    local_domain = tracker.local_domain
    x_local = x_periodic - local_domain.x_start
    
    # Convert to extended grid indices (accounting for halo offset)
    fx = x_local / tracker.dx + halo_info.halo_width + 1  # +1 for 1-based indexing
    fy = y_periodic / tracker.dy + 1
    fz = z_clamped / tracker.dz + 1
    
    # Get integer and fractional parts
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    
    rx = fx - ix
    ry = fy - iy
    rz = fz - iz
    
    # Bounds check for extended arrays
    nz_ext, nx_ext, ny_ext = size(halo_info.u_extended)
    
    # Handle boundary indices (now we have halo data!)
    ix1 = max(1, min(nx_ext, ix))
    ix2 = max(1, min(nx_ext, ix + 1))
    
    if tracker.config.periodic_y
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
        w_interp = 0.0
    end
    
    return u_interp, v_interp, w_interp
end

end # module HaloExchange

using .HaloExchange
