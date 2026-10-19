"""
Interpolation of cell-centred velocity fields at particle positions.

Every scheme is a tensor product of one-dimensional interpolating stencils
that reproduce the grid values exactly. Horizontally the fields are periodic.
Vertically each column is extended past the lid and the bottom by its mirror
image, even for the horizontal velocity and odd for the vertical velocity, so
that `w` vanishes at both rigid boundaries.
"""
module InterpolationSchemes

export InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
       interpolate_velocity_advanced, interpolate_fields

"""
Available interpolation methods. The errors are for smooth fields on a grid
of spacing h.

- `TRILINEAR`: linear in each direction (2-point stencil), O(h²), continuous.
- `TRICUBIC`: Catmull-Rom cubic (4-point stencil), O(h³), continuously
  differentiable. The default.
- `QUINTIC`: degree-5 Lagrange polynomial (6-point stencil), O(h⁶),
  continuous.
- `ADAPTIVE`: deprecated, now the same as `TRICUBIC`.
"""
@enum InterpolationMethod begin
    TRILINEAR = 1
    TRICUBIC = 2
    ADAPTIVE = 3
    QUINTIC = 4
end

"""Linear weights for the points at offsets 0 and 1 from the cell's lower node."""
@inline _stencil_weights(::Val{2}, t) = (1 - t, t)

"""Catmull-Rom weights for the points at offsets -1, 0, 1, and 2."""
@inline function _stencil_weights(::Val{4}, t)
    t2 = t * t
    t3 = t2 * t
    return ((-t3 + 2t2 - t) / 2,
            (3t3 - 5t2 + 2) / 2,
            (-3t3 + 4t2 + t) / 2,
            (t3 - t2) / 2)
end

"""Degree-5 Lagrange weights for the points at offsets -2, -1, 0, 1, 2, and 3."""
@inline function _stencil_weights(::Val{6}, t)
    a, b, c, d, e, g = t + 2, t + 1, t, t - 1, t - 2, t - 3
    return (-b * c * d * e * g / 120,
            a * c * d * e * g / 24,
            -a * b * d * e * g / 12,
            a * b * c * e * g / 12,
            -a * b * c * d * g / 24,
            a * b * c * d * e / 120)
end

"""Horizontal index `i` wrapped (periodic) or clamped into `1:n`."""
@inline get_grid_index(i::Int, n::Int, periodic::Bool) =
    periodic ? mod(i - 1, n) + 1 : clamp(i, 1, n)

"""
Level `k` of a column of `nz` cell centres extended by mirror images about the
bottom and the lid: the stored level it reads, and whether it is reflected an
odd number of times.
"""
@inline function _mirrored_level(k::Int, nz::Int)
    m = mod(k - 1, 2nz)
    return m < nz ? (m + 1, false) : (2nz - m, true)
end

"""
    interpolate_fields(fields, parities, x, y, z, grid_info, boundary_conditions, method)

Interpolate each array of the tuple `fields` (cell centred, `(nz, nx, ny)`
layout) at `(x, y, z)`, with `x` and `y` measured from the domain origin.
Returns a tuple with one value per field.

Above the lid `z = 0` and below the bottom `z = -Lz`, field `n` continues as
the mirror image of its column: evenly for `parities[n] = 1` (zero normal
derivative, as for the horizontal velocity, since ψ_z = 0 there) and oddly
for `parities[n] = -1` (zero value, as for the vertical velocity). A position
outside `[-Lz, 0]` takes the value at the nearer boundary.
"""
function interpolate_fields(fields::NTuple{N, AbstractArray{T, 3}},
                            parities::NTuple{N, Int}, x::T, y::T, z::T,
                            grid_info, boundary_conditions,
                            method::InterpolationMethod) where {N, T}
    if method == TRILINEAR
        return _interpolate_stencil(Val(2), fields, parities, x, y, z,
                                    grid_info, boundary_conditions)
    elseif method == TRICUBIC || method == ADAPTIVE
        return _interpolate_stencil(Val(4), fields, parities, x, y, z,
                                    grid_info, boundary_conditions)
    elseif method == QUINTIC
        return _interpolate_stencil(Val(6), fields, parities, x, y, z,
                                    grid_info, boundary_conditions)
    end
    throw(ArgumentError("unknown interpolation method $method"))
end

function _interpolate_stencil(::Val{S}, fields::NTuple{N, AbstractArray{T, 3}},
                              parities::NTuple{N, Int}, x::T, y::T, z::T,
                              grid_info, boundary_conditions) where {S, N, T}
    nz, nx, ny = size(first(fields))
    dx, dy, dz = grid_info.dx, grid_info.dy, grid_info.dz
    z_min = hasproperty(grid_info, :z_min) ? grid_info.z_min : -grid_info.Lz
    z_max = hasproperty(grid_info, :z_max) ? grid_info.z_max : zero(T)
    z0 = hasproperty(grid_info, :z0) ? grid_info.z0 : z_min + dz / 2
    periodic_x = boundary_conditions.periodic_x
    periodic_y = boundary_conditions.periodic_y

    # Positions in grid units: node i (1-based) sits at (i - 1) dx, level k at
    # z0 + (k - 1) dz.
    fx = (periodic_x ? mod(x, grid_info.Lx) : x) / dx
    fy = (periodic_y ? mod(y, grid_info.Ly) : y) / dy
    fz = (clamp(z, z_min, z_max) - z0) / dz
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    wx = _stencil_weights(Val(S), fx - ix)
    wy = _stencil_weights(Val(S), fy - iy)
    wz = _stencil_weights(Val(S), fz - iz)

    # The cell's lower node has index ix + 1; the stencil starts S/2 - 1
    # nodes before it.
    first_offset = 1 - S ÷ 2
    levels = ntuple(k -> _mirrored_level(iz + first_offset + k, nz), Val(S))

    values = ntuple(_ -> zero(T), Val(N))
    for j in 1:S
        gy = get_grid_index(iy + first_offset + j, ny, periodic_y)
        for i in 1:S
            gx = get_grid_index(ix + first_offset + i, nx, periodic_x)
            wxy = wx[i] * wy[j]
            for k in 1:S
                gz, mirrored = levels[k]
                values = _accumulate(values, fields, parities, mirrored,
                                     wxy * wz[k], gz, gx, gy)
            end
        end
    end
    return values
end

# A separate function, so that the closure does not capture the reassigned
# accumulator (which would box it).
@inline function _accumulate(values::NTuple{N, T}, fields, parities, mirrored,
                             weight, gz, gx, gy) where {N, T}
    return ntuple(Val(N)) do n
        sign = mirrored ? parities[n] : 1
        @inbounds values[n] + sign * weight * fields[n][gz, gx, gy]
    end
end

"""
    interpolate_velocity_advanced(x, y, z, u, v, w, grid_info, boundary_conditions, method)

Interpolate the cell-centred velocity components `u`, `v`, and `w` at
`(x, y, z)` with `method`. `w` vanishes at the lid and the bottom, and `u`
and `v` have zero vertical derivative there. See [`interpolate_fields`](@ref).
The fields are read point by point, so they must be host arrays: for a model
on a GPU, pass `Array(model.fields.u)` and so on.
"""
function interpolate_velocity_advanced(x::T, y::T, z::T,
                                       u_field::AbstractArray{T, 3},
                                       v_field::AbstractArray{T, 3},
                                       w_field::AbstractArray{T, 3},
                                       grid_info, boundary_conditions,
                                       method::InterpolationMethod) where T
    return interpolate_fields((u_field, v_field, w_field), (1, 1, -1),
                              x, y, z, grid_info, boundary_conditions, method)
end

end # module InterpolationSchemes

using .InterpolationSchemes
