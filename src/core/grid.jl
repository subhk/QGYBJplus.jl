"""
    RectilinearGrid(; size, extent=nothing, x=nothing, y=nothing, z=nothing,
                      centered=false)

Immutable owner of the model's global rectilinear geometry. Horizontal
coordinates are periodic Fourier collocation nodes on half-open intervals;
vertical coordinates are cell centers between `z_faces`.

Provide `size=(nx, ny, nz)` together with either `extent=(Lx, Ly, Lz)` or
explicit bounds `x=(x₁, x₂)`, `y=(y₁, y₂)`, and `z=(z₁, z₂)`.
When only `extent` is supplied, the horizontal origin is zero (or centered
when `centered=true`) and the vertical interval is `(-Lz, 0)`.
The `kx` and `ky` properties use FFTW's spectral ordering.
"""
struct RectilinearGrid{T}
    size::NTuple{3, Int}
    extent::NTuple{3, T}
    origin::NTuple{2, T}
    z_bounds::NTuple{2, T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    x_faces::Vector{T}
    y_faces::Vector{T}
    z_faces::Vector{T}
    dx::T
    dy::T
    dz::T
    kx::Vector{T}
    ky::Vector{T}
    kh2::Matrix{T}
end

"""
    RuntimeGeometry(base, kh2, decomposition)

Runtime-local spectral metadata derived from a [`RectilinearGrid`](@ref).
Global geometry remains owned exactly once by `base`; only the distributed
horizontal wavenumber field and decomposition live here.
"""
struct RuntimeGeometry{G, K, D}
    base::G
    kh2::K
    decomposition::D
end

@inline function Base.getproperty(geometry::RuntimeGeometry, name::Symbol)
    name === :base && return getfield(geometry, :base)
    name === :kh2 && return getfield(geometry, :kh2)
    name === :decomposition && return getfield(geometry, :decomposition)

    base = getfield(geometry, :base)
    name === :nx && return base.size[1]
    name === :ny && return base.size[2]
    name === :nz && return base.size[3]
    name === :Lx && return base.extent[1]
    name === :Ly && return base.extent[2]
    name === :Lz && return base.extent[3]
    name === :x0 && return base.origin[1]
    name === :y0 && return base.origin[2]
    name === :dx && return base.dx
    name === :dy && return base.dy
    name === :dz && return base.dz
    name === :x && return base.x
    name === :y && return base.y
    name === :z && return base.z
    name === :kx && return base.kx
    name === :ky && return base.ky
    return getfield(geometry, name)
end

Base.propertynames(::RuntimeGeometry, private::Bool=false) =
    (:base, :kh2, :decomposition, :nx, :ny, :nz, :Lx, :Ly, :Lz,
     :x0, :y0, :dx, :dy, :dz, :x, :y, :z, :kx, :ky)

get_local_range(geometry::RectilinearGrid) =
    (1:geometry.size[3], 1:geometry.size[1], 1:geometry.size[2])
get_local_range(geometry::RuntimeGeometry) =
    geometry.decomposition === nothing ? get_local_range(geometry.base) :
                                         geometry.decomposition.local_range_xy

@inline local_to_global(index::Int, dimension::Int,
                        ::AbstractArray) = index
@inline function local_to_global(index::Int, dimension::Int,
                                 array::PencilArray)
    ranges = PencilArrays.range_local(PencilArrays.pencil(array))
    return ranges[dimension][index]
end
@inline local_to_global(index::Int, dimension::Int,
                        ::RectilinearGrid) = index
@inline function local_to_global(index::Int, dimension::Int,
                                 geometry::RuntimeGeometry)
    return get_local_range(geometry)[dimension][index]
end

get_local_dims(array) = size(parent(array))
is_parallel_array(array) = array isa PencilArray

@inline get_kx(index::Int, geometry::RuntimeGeometry) =
    geometry.kx[local_to_global(index, 2, geometry)]
@inline get_ky(index::Int, geometry::RuntimeGeometry) =
    geometry.ky[local_to_global(index, 3, geometry)]
@inline function get_kh2(i::Int, j::Int, k::Int, array,
                         geometry::RuntimeGeometry)
    geometry.decomposition === nothing && return geometry.kh2[i, j]
    return real(parent(geometry.kh2)[k, i, j])
end

function _grid_bounds(bounds, name::String)
    bounds isa Tuple && length(bounds) == 2 ||
        throw(ArgumentError("$name must be a two-element tuple"))
    all(value -> value isa Real, bounds) ||
        throw(ArgumentError("$name bounds must be real numbers"))
    first_bound, last_bound = float.(bounds)
    all(isfinite, (first_bound, last_bound)) ||
        throw(ArgumentError("$name bounds must be finite"))
    last_bound > first_bound ||
        throw(ArgumentError("$name upper bound must exceed its lower bound"))
    return first_bound, last_bound
end

function _spectral_wavenumbers(::Type{T}, n::Int, length::T) where T
    scale = T(2π) / length
    return T[scale * (i <= (n + 1) ÷ 2 ? i - 1 : i - 1 - n) for i in 1:n]
end

function RectilinearGrid(; size,
    extent=nothing,
    x=nothing,
    y=nothing,
    z=nothing,
    centered::Bool=false)

    size isa Tuple && length(size) == 3 ||
        throw(ArgumentError("size must be a three-element tuple"))
    all(value -> value isa Integer, size) ||
        throw(ArgumentError("grid dimensions must be integers"))
    dimensions = Int.(size)
    all(>(0), dimensions) || throw(ArgumentError("all grid dimensions must be positive"))
    nx, ny, nz = dimensions

    centered && (x !== nothing || y !== nothing) &&
        throw(ArgumentError("centered=true cannot be combined with explicit x or y ranges"))

    x_input = x === nothing ? nothing : _grid_bounds(x, "x")
    y_input = y === nothing ? nothing : _grid_bounds(y, "y")
    z_input = z === nothing ? nothing : _grid_bounds(z, "z")

    if extent === nothing
        x_input === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or x=(x₁, x₂)"))
        y_input === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or y=(y₁, y₂)"))
        z_input === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or z=(z₁, z₂)"))
        raw_extent = (x_input[2] - x_input[1],
                      y_input[2] - y_input[1],
                      z_input[2] - z_input[1])
    else
        extent isa Tuple && length(extent) == 3 ||
            throw(ArgumentError("extent must be a three-element tuple"))
        all(value -> value isa Real, extent) ||
            throw(ArgumentError("grid extents must be real numbers"))
        raw_extent = float.(extent)
        all(isfinite, raw_extent) || throw(ArgumentError("all grid extents must be finite"))
        all(>(0), raw_extent) || throw(ArgumentError("all grid extents must be positive"))
    end

    numeric_values = Any[raw_extent...]
    for bounds in (x_input, y_input, z_input)
        bounds === nothing || append!(numeric_values, bounds)
    end
    T = promote_type(map(typeof, numeric_values)...)
    Lx, Ly, Lz = T.(raw_extent)

    function check_extent(bounds, length, name)
        bounds === nothing && return
        actual = T(bounds[2] - bounds[1])
        tolerance = T(10) * eps(T) * max(abs(length), one(T))
        isapprox(actual, length; rtol=T(10) * eps(T), atol=tolerance) ||
            throw(ArgumentError("$name range length $actual does not match extent $length"))
    end
    check_extent(x_input, Lx, "x")
    check_extent(y_input, Ly, "y")
    check_extent(z_input, Lz, "z")

    x0 = x_input === nothing ? (centered ? -Lx / 2 : zero(T)) : T(x_input[1])
    y0 = y_input === nothing ? (centered ? -Ly / 2 : zero(T)) : T(y_input[1])
    z_bottom, z_top = z_input === nothing ? (-Lz, zero(T)) : T.(z_input)

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    x_nodes = T[x0 + (i - 1) * dx for i in 1:nx]
    y_nodes = T[y0 + (j - 1) * dy for j in 1:ny]
    z_faces = T[z_bottom + (k - 1) * dz for k in 1:(nz + 1)]
    z_nodes = T[(z_faces[k] + z_faces[k + 1]) / 2 for k in 1:nz]
    x_faces = T[x0 + (i - 1) * dx for i in 1:(nx + 1)]
    y_faces = T[y0 + (j - 1) * dy for j in 1:(ny + 1)]

    kx = _spectral_wavenumbers(T, nx, Lx)
    ky = _spectral_wavenumbers(T, ny, Ly)
    kh2 = T[kx_i^2 + ky_j^2 for kx_i in kx, ky_j in ky]

    return RectilinearGrid{T}(
        dimensions, (Lx, Ly, Lz), (x0, y0), (z_bottom, z_top),
        x_nodes, y_nodes, z_nodes, x_faces, y_faces, z_faces,
        dx, dy, dz, kx, ky, kh2)
end

function Base.show(io::IO, grid::RectilinearGrid)
    print(io, "RectilinearGrid(size=$(grid.size), extent=$(grid.extent), " *
              "origin=$(grid.origin))")
end
