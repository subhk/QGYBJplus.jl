"""
Halo exchange system for particle advection in parallel domains.

This module handles the communication of velocity field data between neighboring
MPI domains to enable accurate interpolation for particles near domain boundaries.
"""

module HaloExchange

using ..QGYBJ: Grid

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
    local_start::NTuple{3,Int}  # (i_start, j_start, k_start)
    local_end::NTuple{3,Int}    # (i_end, j_end, k_end)
    
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
    
    function HaloInfo{T}(grid::Grid, rank::Int, nprocs::Int, comm, halo_width::Int=2) where T
        # Extended grid dimensions (original + 2*halo_width in x-direction)
        nx_ext = grid.nx + 2*halo_width
        ny_ext = grid.ny  # No halo in y for 1D decomposition
        nz_ext = grid.nz  # No halo in z for 1D decomposition
        
        # Create extended arrays
        u_extended = zeros(T, nx_ext, ny_ext, nz_ext)
        v_extended = zeros(T, nx_ext, ny_ext, nz_ext)
        w_extended = zeros(T, nx_ext, ny_ext, nz_ext)
        
        # Local domain indices in extended array (1-based)
        local_start = (halo_width + 1, 1, 1)
        local_end = (halo_width + grid.nx, grid.ny, grid.nz)
        
        # Determine neighbors (1D decomposition in x)
        left_neighbor = rank > 0 ? rank - 1 : -1
        right_neighbor = rank < nprocs - 1 ? rank + 1 : -1
        
        # Communication buffer sizes (halo_width * ny * nz * 3_components)
        buffer_size = halo_width * grid.ny * grid.nz * 3
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
        import MPI
        
        # Copy local data to extended arrays
        copy_local_to_extended!(halo_info, u_field, v_field, w_field)
        
        # Prepare send buffers
        pack_halo_data!(halo_info)
        
        # Post non-blocking receives
        recv_reqs = MPI.Request[]
        
        if halo_info.left_neighbor >= 0
            req = MPI.Irecv!(halo_info.recv_left, halo_info.left_neighbor, 0, halo_info.comm)
            push!(recv_reqs, req)
        end
        
        if halo_info.right_neighbor >= 0
            req = MPI.Irecv!(halo_info.recv_right, halo_info.right_neighbor, 1, halo_info.comm)
            push!(recv_reqs, req)
        end
        
        # Send data
        send_reqs = MPI.Request[]
        
        if halo_info.right_neighbor >= 0
            req = MPI.Isend(halo_info.send_right, halo_info.right_neighbor, 0, halo_info.comm)
            push!(send_reqs, req)
        end
        
        if halo_info.left_neighbor >= 0
            req = MPI.Isend(halo_info.send_left, halo_info.left_neighbor, 1, halo_info.comm)
            push!(send_reqs, req)
        end
        
        # Wait for receives to complete
        if !isempty(recv_reqs)
            MPI.Waitall(recv_reqs)
        end
        
        # Unpack received data
        unpack_halo_data!(halo_info)
        
        # Wait for sends to complete
        if !isempty(send_reqs)
            MPI.Waitall(send_reqs)
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
    
    i_start, j_start, k_start = halo_info.local_start
    i_end, j_end, k_end = halo_info.local_end
    
    # Copy local data to center of extended arrays
    halo_info.u_extended[i_start:i_end, j_start:j_end, k_start:k_end] .= u_field
    halo_info.v_extended[i_start:i_end, j_start:j_end, k_start:k_end] .= v_field
    halo_info.w_extended[i_start:i_end, j_start:j_end, k_start:k_end] .= w_field
end

"""
Pack halo data into send buffers.
"""
function pack_halo_data!(halo_info::HaloInfo{T}) where T
    hw = halo_info.halo_width
    i_start, j_start, k_start = halo_info.local_start
    i_end, j_end, k_end = halo_info.local_end
    
    # Pack data to send to left neighbor (right boundary of local domain)
    if halo_info.left_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end-hw+1):i_end
            halo_info.send_left[idx] = halo_info.u_extended[i, j, k]
            halo_info.send_left[idx + 1] = halo_info.v_extended[i, j, k]
            halo_info.send_left[idx + 2] = halo_info.w_extended[i, j, k]
            idx += 3
        end
    end
    
    # Pack data to send to right neighbor (left boundary of local domain)
    if halo_info.right_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in i_start:(i_start+hw-1)
            halo_info.send_right[idx] = halo_info.u_extended[i, j, k]
            halo_info.send_right[idx + 1] = halo_info.v_extended[i, j, k]
            halo_info.send_right[idx + 2] = halo_info.w_extended[i, j, k]
            idx += 3
        end
    end
end

"""
Unpack received halo data.
"""
function unpack_halo_data!(halo_info::HaloInfo{T}) where T
    hw = halo_info.halo_width
    i_start, j_start, k_start = halo_info.local_start
    i_end, j_end, k_end = halo_info.local_end
    
    # Unpack data from left neighbor (fills left halo region)
    if halo_info.left_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in 1:hw
            halo_info.u_extended[i, j, k] = halo_info.recv_left[idx]
            halo_info.v_extended[i, j, k] = halo_info.recv_left[idx + 1]
            halo_info.w_extended[i, j, k] = halo_info.recv_left[idx + 2]
            idx += 3
        end
    end
    
    # Unpack data from right neighbor (fills right halo region)
    if halo_info.right_neighbor >= 0
        idx = 1
        for k in k_start:k_end, j in j_start:j_end, i in (i_end+1):(i_end+hw)
            halo_info.u_extended[i, j, k] = halo_info.recv_right[idx]
            halo_info.v_extended[i, j, k] = halo_info.recv_right[idx + 1]
            halo_info.w_extended[i, j, k] = halo_info.recv_right[idx + 2]
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
    nx_ext, ny_ext, nz_ext = size(halo_info.u_extended)
    
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
    u_z1_y1 = (1-rx) * halo_info.u_extended[ix1,iy1,iz1] + rx * halo_info.u_extended[ix2,iy1,iz1]
    u_z1_y2 = (1-rx) * halo_info.u_extended[ix1,iy2,iz1] + rx * halo_info.u_extended[ix2,iy2,iz1]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * halo_info.v_extended[ix1,iy1,iz1] + rx * halo_info.v_extended[ix2,iy1,iz1]
    v_z1_y2 = (1-rx) * halo_info.v_extended[ix1,iy2,iz1] + rx * halo_info.v_extended[ix2,iy2,iz1]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * halo_info.w_extended[ix1,iy1,iz1] + rx * halo_info.w_extended[ix2,iy1,iz1]
    w_z1_y2 = (1-rx) * halo_info.w_extended[ix1,iy2,iz1] + rx * halo_info.w_extended[ix2,iy2,iz1]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)
    u_z2_y1 = (1-rx) * halo_info.u_extended[ix1,iy1,iz2] + rx * halo_info.u_extended[ix2,iy1,iz2]
    u_z2_y2 = (1-rx) * halo_info.u_extended[ix1,iy2,iz2] + rx * halo_info.u_extended[ix2,iy2,iz2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * halo_info.v_extended[ix1,iy1,iz2] + rx * halo_info.v_extended[ix2,iy1,iz2]
    v_z2_y2 = (1-rx) * halo_info.v_extended[ix1,iy2,iz2] + rx * halo_info.v_extended[ix2,iy2,iz2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * halo_info.w_extended[ix1,iy1,iz2] + rx * halo_info.w_extended[ix2,iy1,iz2]
    w_z2_y2 = (1-rx) * halo_info.w_extended[ix1,iy2,iz2] + rx * halo_info.w_extended[ix2,iy2,iz2]
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