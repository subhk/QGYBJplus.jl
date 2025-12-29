"""
Advanced interpolation schemes for particle advection.

This module provides higher-order interpolation methods beyond trilinear
interpolation, including tricubic and adaptive schemes for improved accuracy.
"""

module InterpolationSchemes

export InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
       interpolate_velocity_advanced,
       CubicSpline1D, tricubic_weights, bicubic_interpolation,
       quintic_basis_functions, quintic_interpolation,
       required_halo_width

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
    required_halo_width(method::InterpolationMethod)

Return the minimum halo width required for a given interpolation method.

The halo width is the number of ghost cells needed on each side of the local
domain to perform interpolation for particles near domain boundaries.

# Stencil Requirements
- TRILINEAR: 2×2×2 stencil, needs 1 halo cell (accesses [0, +1])
- TRICUBIC:  4×4×4 stencil, needs 2 halo cells (accesses [-1, 0, +1, +2])
- QUINTIC:   6×6×6 stencil, needs 3 halo cells (accesses [-2, -1, 0, +1, +2, +3])
- ADAPTIVE:  May use any scheme, defaults to max (3)
"""
function required_halo_width(method::InterpolationMethod)
    if method == TRILINEAR
        return 1
    elseif method == TRICUBIC
        return 2
    elseif method == QUINTIC
        return 3
    elseif method == ADAPTIVE
        return 3  # Conservative: support all schemes
    else
        return 2  # Default fallback
    end
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
    
    nz, nx, ny = size(u_field)
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
            
            u_interp += weight * u_field[gz, gx, gy]
            v_interp += weight * v_field[gz, gx, gy]
            w_interp += weight * w_field[gz, gx, gy]
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
    
    nz, nx, ny = size(u_field)
    dx, dy, dz = grid_info.dx, grid_info.dy, grid_info.dz
    
    # Convert to grid indices
    ix = clamp(round(Int, x / dx), 2, nx-1)
    iy = clamp(round(Int, y / dy), 2, ny-1)
    iz = clamp(round(Int, z / dz), 2, nz-1)
    
    # Compute second derivatives (finite differences)
    d2u_dx2 = (u_field[iz, ix+1, iy] - 2*u_field[iz, ix, iy] + u_field[iz, ix-1, iy]) / dx^2
    d2u_dy2 = (u_field[iz, ix, iy+1] - 2*u_field[iz, ix, iy] + u_field[iz, ix, iy-1]) / dy^2
    d2u_dz2 = (u_field[iz+1, ix, iy] - 2*u_field[iz, ix, iy] + u_field[iz-1, ix, iy]) / dz^2
    
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
    
    nz, nx, ny = size(u_field)
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
    u_z1_y1 = (1-rx) * u_field[iz1, ix1, iy1] + rx * u_field[iz1, ix2, iy1]
    u_z1_y2 = (1-rx) * u_field[iz1, ix1, iy2] + rx * u_field[iz1, ix2, iy2]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * v_field[iz1, ix1, iy1] + rx * v_field[iz1, ix2, iy1]
    v_z1_y2 = (1-rx) * v_field[iz1, ix1, iy2] + rx * v_field[iz1, ix2, iy2]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * w_field[iz1, ix1, iy1] + rx * w_field[iz1, ix2, iy1]
    w_z1_y2 = (1-rx) * w_field[iz1, ix1, iy2] + rx * w_field[iz1, ix2, iy2]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)
    u_z2_y1 = (1-rx) * u_field[iz2, ix1, iy1] + rx * u_field[iz2, ix2, iy1]
    u_z2_y2 = (1-rx) * u_field[iz2, ix1, iy2] + rx * u_field[iz2, ix2, iy2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * v_field[iz2, ix1, iy1] + rx * v_field[iz2, ix2, iy1]
    v_z2_y2 = (1-rx) * v_field[iz2, ix1, iy2] + rx * v_field[iz2, ix2, iy2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * w_field[iz2, ix1, iy1] + rx * w_field[iz2, ix2, iy1]
    w_z2_y2 = (1-rx) * w_field[iz2, ix1, iy2] + rx * w_field[iz2, ix2, iy2]
    w_z2 = (1-ry) * w_z2_y1 + ry * w_z2_y2
    
    # Final interpolation in z
    u_interp = (1-rz) * u_z1 + rz * u_z2
    v_interp = (1-rz) * v_z1 + rz * v_z2
    w_interp = (1-rz) * w_z1 + rz * w_z2
    
    return u_interp, v_interp, w_interp
end

"""
    quintic_basis_functions(t)

Quintic B-spline basis functions for uniform grid interpolation.
For parameter t ∈ [0,1], returns weights for points at [-2, -1, 0, 1, 2, 3].
Provides O(h⁶) accuracy with C⁴ continuity.
"""
function quintic_basis_functions(t::T) where T
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Quintic B-spline basis functions (cardinal spline form)
    # These ensure smooth interpolation with C⁴ continuity
    w0 = (-t5 + 5*t4 - 10*t3 + 10*t2 - 5*t + 1) / 120.0
    w1 = (5*t5 - 20*t4 + 20*t3 + 20*t2 - 50*t + 26) / 120.0
    w2 = (-10*t5 + 30*t4 - 60*t2 + 66) / 120.0
    w3 = (10*t5 - 20*t4 - 20*t3 + 20*t2 + 50*t + 26) / 120.0
    w4 = (-5*t5 + 5*t4 + 10*t3 + 10*t2 + 5*t + 1) / 120.0
    w5 = t5 / 120.0

    return w0, w1, w2, w3, w4, w5
end

"""
Quintic interpolation (O(h⁶) accuracy).
Uses a 6×6×6 = 216 point stencil for high-accuracy interpolation.
Provides C⁴ continuous interpolation with excellent smoothness properties.
"""
function quintic_interpolation(x::T, y::T, z::T,
                              u_field::Array{T,3}, v_field::Array{T,3}, w_field::Array{T,3},
                              grid_info, boundary_conditions) where T

    nz, nx, ny = size(u_field)
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

    # Get quintic spline weights (6 points: -2, -1, 0, 1, 2, 3)
    wx = quintic_basis_functions(tx)
    wy = quintic_basis_functions(ty)
    wz = quintic_basis_functions(tz)

    # Initialize interpolated values
    u_interp = zero(T)
    v_interp = zero(T)
    w_interp = zero(T)

    # 6×6×6 interpolation stencil
    for k in 0:5, j in 0:5, i in 0:5
        # Grid indices with boundary handling
        # Stencil points: ix + i - 2 covers [-2, -1, 0, 1, 2, 3] relative to ix
        gx = get_grid_index(ix + i - 2, nx, boundary_conditions.periodic_x)
        gy = get_grid_index(iy + j - 2, ny, boundary_conditions.periodic_y)
        gz = get_grid_index(iz + k - 2, nz, false)  # Z never periodic

        if gx > 0 && gy > 0 && gz > 0  # Valid indices
            # Combined weight from tensor product
            weight = wx[i+1] * wy[j+1] * wz[k+1]

            u_interp += weight * u_field[gz, gx, gy]
            v_interp += weight * v_field[gz, gx, gy]
            w_interp += weight * w_field[gz, gx, gy]
        end
    end

    return u_interp, v_interp, w_interp
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
