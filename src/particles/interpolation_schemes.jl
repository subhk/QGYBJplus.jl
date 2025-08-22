"""
Advanced interpolation schemes for particle advection.

This module provides higher-order interpolation methods beyond trilinear
interpolation, including tricubic and adaptive schemes for improved accuracy.
"""

module InterpolationSchemes

export InterpolationMethod, interpolate_velocity_advanced,
       CubicSpline1D, tricubic_weights, bicubic_interpolation

"""
Available interpolation methods.
"""
@enum InterpolationMethod begin
    TRILINEAR = 1    # O(h²) - Fast, standard trilinear 
    TRICUBIC = 2     # O(h⁴) - High accuracy, C¹ smooth
    ADAPTIVE = 3     # Adaptive based on local gradient
    QUINTIC = 4      # O(h⁶) - Highest accuracy (experimental)
end

"""
1D cubic spline interpolation weights and derivatives.
"""
struct CubicSpline1D{T<:AbstractFloat}
    # Cubic spline basis functions for uniform grid
    # Using Catmull-Rom spline (C¹ continuous)
    function CubicSpline1D{T}() where T
        new{T}()
    end
end

"""
    cubic_basis_functions(t)

Catmull-Rom cubic spline basis functions.
For parameter t ∈ [0,1], returns weights for points at [-1, 0, 1, 2].
"""
function cubic_basis_functions(t::T) where T
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom basis functions
    w0 = -0.5*t3 + t2 - 0.5*t           # Weight for point at -1
    w1 = 1.5*t3 - 2.5*t2 + 1.0          # Weight for point at 0  
    w2 = -1.5*t3 + 2.0*t2 + 0.5*t       # Weight for point at 1
    w3 = 0.5*t3 - 0.5*t2                # Weight for point at 2
    
    return w0, w1, w2, w3
end

"""
    cubic_derivative_weights(t)

Derivatives of Catmull-Rom basis functions.
"""
function cubic_derivative_weights(t::T) where T
    t2 = t * t
    
    # Derivatives of basis functions
    dw0 = -1.5*t2 + 2.0*t - 0.5
    dw1 = 4.5*t2 - 5.0*t
    dw2 = -4.5*t2 + 4.0*t + 0.5
    dw3 = 1.5*t2 - t
    
    return dw0, dw1, dw2, dw3
end

"""
    tricubic_interpolation(x, y, z, field, grid_info, boundary_conditions)

High-accuracy tricubic interpolation using 4×4×4 = 64 grid points.
Provides O(h⁴) accuracy and C¹ continuity.
"""
function tricubic_interpolation(x::T, y::T, z::T,
                               u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                               grid_info, boundary_conditions) where T
    
    nx, ny, nz = size(u_field)
    dx, dy, dz = grid_info.dx, grid_info.dy, grid_info.dz
    Lx, Ly, Lz = grid_info.Lx, grid_info.Ly, grid_info.Lz
    
    # Handle periodic boundaries
    x_periodic = boundary_conditions.periodic_x ? mod(x, Lx) : x
    y_periodic = boundary_conditions.periodic_y ? mod(y, Ly) : y
    z_clamped = clamp(z, 0, Lz)
    
    # Convert to grid coordinates
    fx = x_periodic / dx
    fy = y_periodic / dy
    fz = z_clamped / dz
    
    # Get integer parts and fractional coordinates
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    
    tx = fx - ix  # Parameter in [0,1]
    ty = fy - iy
    tz = fz - iz
    
    # Get cubic spline weights
    wx0, wx1, wx2, wx3 = cubic_basis_functions(tx)
    wy0, wy1, wy2, wy3 = cubic_basis_functions(ty)
    wz0, wz1, wz2, wz3 = cubic_basis_functions(tz)
    
    # Initialize interpolated values
    u_interp = zero(T)
    v_interp = zero(T)
    w_interp = zero(T)
    
    # 4×4×4 interpolation stencil
    for k in 0:3, j in 0:3, i in 0:3
        # Grid indices with boundary handling
        gx = get_grid_index(ix + i - 1, nx, boundary_conditions.periodic_x)
        gy = get_grid_index(iy + j - 1, ny, boundary_conditions.periodic_y)
        gz = get_grid_index(iz + k - 1, nz, false)  # Z never periodic
        
        if gx > 0 && gy > 0 && gz > 0  # Valid indices
            # Combined weight
            weight = (i == 0 ? wx0 : i == 1 ? wx1 : i == 2 ? wx2 : wx3) *
                    (j == 0 ? wy0 : j == 1 ? wy1 : j == 2 ? wy2 : wy3) *
                    (k == 0 ? wz0 : k == 1 ? wz1 : k == 2 ? wz2 : wz3)
            
            u_interp += weight * u_field[gx, gy, gz]
            v_interp += weight * v_field[gx, gy, gz]
            w_interp += weight * w_field[gx, gy, gz]
        end
    end
    
    return u_interp, v_interp, w_interp
end

"""
    get_grid_index(idx, n, periodic)

Get valid grid index with boundary conditions.
"""
function get_grid_index(idx::Int, n::Int, periodic::Bool)
    if periodic
        return mod(idx - 1, n) + 1  # Periodic wrapping (1-based)
    else
        return max(1, min(n, idx))  # Clamping for non-periodic
    end
end

"""
    adaptive_interpolation(x, y, z, fields, grid_info, boundary_conditions)

Adaptive interpolation that chooses method based on local field smoothness.
"""
function adaptive_interpolation(x::T, y::T, z::T,
                               u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                               grid_info, boundary_conditions) where T
    
    # Estimate local smoothness using finite differences
    smoothness = estimate_field_smoothness(x, y, z, u_field, v_field, w_field, grid_info)
    
    # Choose interpolation method based on smoothness
    if smoothness > 0.1  # Rough field - use trilinear for stability
        return trilinear_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    else  # Smooth field - use tricubic for accuracy
        return tricubic_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    end
end

"""
Estimate local field smoothness using second derivatives.
"""
function estimate_field_smoothness(x::T, y::T, z::T,
                                  u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                                  grid_info) where T
    
    nx, ny, nz = size(u_field)
    dx, dy, dz = grid_info.dx, grid_info.dy, grid_info.dz
    
    # Convert to grid indices
    ix = clamp(round(Int, x / dx), 2, nx-1)
    iy = clamp(round(Int, y / dy), 2, ny-1)
    iz = clamp(round(Int, z / dz), 2, nz-1)
    
    # Compute second derivatives (finite differences)
    d2u_dx2 = (u_field[ix+1,iy,iz] - 2*u_field[ix,iy,iz] + u_field[ix-1,iy,iz]) / dx^2
    d2u_dy2 = (u_field[ix,iy+1,iz] - 2*u_field[ix,iy,iz] + u_field[ix,iy-1,iz]) / dy^2
    d2u_dz2 = (u_field[ix,iy,iz+1] - 2*u_field[ix,iy,iz] + u_field[ix,iy,iz-1]) / dz^2
    
    # RMS curvature as smoothness indicator
    curvature = sqrt(d2u_dx2^2 + d2u_dy2^2 + d2u_dz2^2)
    
    return curvature
end

"""
    interpolate_velocity_advanced(x, y, z, fields, grid_info, boundary_conditions, method)

Advanced interpolation dispatcher with multiple methods.
"""
function interpolate_velocity_advanced(x::T, y::T, z::T,
                                     u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                                     grid_info, boundary_conditions, 
                                     method::InterpolationMethod) where T
    
    if method == TRILINEAR
        return trilinear_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    elseif method == TRICUBIC
        return tricubic_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    elseif method == ADAPTIVE
        return adaptive_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    elseif method == QUINTIC
        return quintic_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
    else
        error("Unknown interpolation method: $method")
    end
end

"""
Original trilinear interpolation for comparison.
"""
function trilinear_interpolation(x::T, y::T, z::T,
                                u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                                grid_info, boundary_conditions) where T
    
    nx, ny, nz = size(u_field)
    dx, dy, dz = grid_info.dx, grid_info.dy, grid_info.dz
    Lx, Ly, Lz = grid_info.Lx, grid_info.Ly, grid_info.Lz
    
    # Handle periodic boundaries
    x_periodic = boundary_conditions.periodic_x ? mod(x, Lx) : x
    y_periodic = boundary_conditions.periodic_y ? mod(y, Ly) : y
    z_clamped = clamp(z, 0, Lz)
    
    # Convert to grid indices
    fx = x_periodic / dx
    fy = y_periodic / dy
    fz = z_clamped / dz
    
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    
    rx = fx - ix
    ry = fy - iy
    rz = fz - iz
    
    # Get grid indices with boundary handling
    ix1 = get_grid_index(ix + 1, nx, boundary_conditions.periodic_x)
    ix2 = get_grid_index(ix + 2, nx, boundary_conditions.periodic_x)
    iy1 = get_grid_index(iy + 1, ny, boundary_conditions.periodic_y)
    iy2 = get_grid_index(iy + 2, ny, boundary_conditions.periodic_y)
    iz1 = get_grid_index(iz + 1, nz, false)
    iz2 = get_grid_index(iz + 2, nz, false)
    
    # Trilinear interpolation
    # Bottom face (z1)
    u_z1_y1 = (1-rx) * u_field[ix1,iy1,iz1] + rx * u_field[ix2,iy1,iz1]
    u_z1_y2 = (1-rx) * u_field[ix1,iy2,iz1] + rx * u_field[ix2,iy2,iz1]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * v_field[ix1,iy1,iz1] + rx * v_field[ix2,iy1,iz1]
    v_z1_y2 = (1-rx) * v_field[ix1,iy2,iz1] + rx * v_field[ix2,iy2,iz1]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * w_field[ix1,iy1,iz1] + rx * w_field[ix2,iy1,iz1]
    w_z1_y2 = (1-rx) * w_field[ix1,iy2,iz1] + rx * w_field[ix2,iy2,iz1]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)
    u_z2_y1 = (1-rx) * u_field[ix1,iy1,iz2] + rx * u_field[ix2,iy1,iz2]
    u_z2_y2 = (1-rx) * u_field[ix1,iy2,iz2] + rx * u_field[ix2,iy2,iz2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * v_field[ix1,iy1,iz2] + rx * v_field[ix2,iy1,iz2]
    v_z2_y2 = (1-rx) * v_field[ix1,iy2,iz2] + rx * v_field[ix2,iy2,iz2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * w_field[ix1,iy1,iz2] + rx * w_field[ix2,iy1,iz2]
    w_z2_y2 = (1-rx) * w_field[ix1,iy2,iz2] + rx * w_field[ix2,iy2,iz2]
    w_z2 = (1-ry) * w_z2_y1 + ry * w_z2_y2
    
    # Final interpolation in z
    u_interp = (1-rz) * u_z1 + rz * u_z2
    v_interp = (1-rz) * v_z1 + rz * v_z2
    w_interp = (1-rz) * w_z1 + rz * w_z2
    
    return u_interp, v_interp, w_interp
end

"""
Experimental quintic interpolation (O(h⁶) accuracy).
"""
function quintic_interpolation(x::T, y::T, z::T,
                              u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                              grid_info, boundary_conditions) where T
    
    # For now, fall back to tricubic (quintic requires 6×6×6 = 216 points!)
    @warn "Quintic interpolation not yet implemented, using tricubic"
    return tricubic_interpolation(x, y, z, u_field, v_field, w_field, grid_info, boundary_conditions)
end

"""
    interpolation_error_estimate(method, grid_spacing)

Theoretical error estimate for different interpolation methods.
"""
function interpolation_error_estimate(method::InterpolationMethod, h::T) where T
    if method == TRILINEAR
        return h^2  # O(h²)
    elseif method == TRICUBIC
        return h^4  # O(h⁴)
    elseif method == QUINTIC
        return h^6  # O(h⁶)
    else
        return h^2  # Conservative estimate
    end
end

end # module InterpolationSchemes

using .InterpolationSchemes