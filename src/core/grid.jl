"""
    RectilinearGrid([architecture=CPU()], [FT=Float64]; size, extent=nothing,
                    x=nothing, y=nothing, z=nothing, centered=false)

Immutable owner of the model's global rectilinear geometry. Horizontal
coordinates are periodic Fourier collocation nodes on half-open intervals;
vertical coordinates are cell centers between `z_faces`.

Provide `size=(nx, ny, nz)` together with either `extent=(Lx, Ly, Lz)` or
explicit bounds `x=(x₁, x₂)`, `y=(y₁, y₂)`, and `z=(z₁, z₂)`.
When only `extent` is supplied, the horizontal origin is zero (or centered
when `centered=true`) and the vertical interval is `(-Lz, 0)`.
The `kx` and `ky` properties use FFTW's spectral ordering.

`architecture` selects where a model built on this grid stores and computes
its fields: `CPU()` (default) or `GPU()`. The coordinate vectors held by the
grid itself always live on the host. A GPU backend is stored as its arrays
report it (see `array_architecture`).

`FT` is the floating-point type of the grid and of every array of a model
built on it. `Float32` halves the model's memory and is much faster on most
GPUs; see the performance page for its accuracy.

```julia
grid = RectilinearGrid(GPU(); size=(128, 128, 64), extent=(1e5, 1e5, 4e3))
grid32 = RectilinearGrid(GPU(), Float32; size=(256, 256, 128), extent=(1e5, 1e5, 4e3))
```
"""
struct RectilinearGrid{T, A<:AbstractArchitecture}
    architecture::A
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

architecture(grid::RectilinearGrid) = grid.architecture
Base.eltype(::RectilinearGrid{T}) where {T} = T

"""
    RuntimeGeometry(grid)

Kernel-facing geometry derived from a [`RectilinearGrid`](@ref). Global
geometry remains owned exactly once by `base`; this adds copies of the
horizontal wavenumbers stored on the grid's architecture, which is what the
spectral kernels index.
"""
struct RuntimeGeometry{G, A, V, K}
    base::G
    architecture::A
    kx::V
    ky::V
    kh2::K
end

function RuntimeGeometry(grid::RectilinearGrid)
    architecture = grid.architecture
    return RuntimeGeometry(grid, architecture,
                           on_architecture(architecture, grid.kx),
                           on_architecture(architecture, grid.ky),
                           on_architecture(architecture, grid.kh2))
end

@inline function Base.getproperty(geometry::RuntimeGeometry, name::Symbol)
    name === :base && return getfield(geometry, :base)
    name === :architecture && return getfield(geometry, :architecture)
    name === :kx && return getfield(geometry, :kx)
    name === :ky && return getfield(geometry, :ky)
    name === :kh2 && return getfield(geometry, :kh2)

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
    name === :z_faces && return base.z_faces
    return getfield(geometry, name)
end

Base.propertynames(::RuntimeGeometry, private::Bool=false) =
    (:base, :architecture, :kx, :ky, :kh2, :nx, :ny, :nz, :Lx, :Ly, :Lz,
     :x0, :y0, :dx, :dy, :dz, :x, :y, :z, :z_faces)

architecture(geometry::RuntimeGeometry) = geometry.architecture

"""Uniform vertical spacing used by the vertical stencils (1 for one level)."""
vertical_spacing(geometry::RuntimeGeometry) =
    geometry.nz > 1 ? geometry.z[2] - geometry.z[1] : one(geometry.dz)

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

RectilinearGrid(FT::Type{<:AbstractFloat}; kwargs...) =
    RectilinearGrid(CPU(), FT; kwargs...)

function RectilinearGrid(architecture::AbstractArchitecture=CPU(),
    FT::Type{<:AbstractFloat}=Float64;
    size,
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

    T = FT
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

    canonical = array_architecture(architecture)
    return RectilinearGrid{T, typeof(canonical)}(
        canonical, dimensions, (Lx, Ly, Lz), (x0, y0), (z_bottom, z_top),
        x_nodes, y_nodes, z_nodes, x_faces, y_faces, z_faces,
        dx, dy, dz, kx, ky, kh2)
end

function Base.show(io::IO, grid::RectilinearGrid)
    print(io, "RectilinearGrid($(grid.architecture), size=$(grid.size), " *
              "extent=$(grid.extent), origin=$(grid.origin))")
end
