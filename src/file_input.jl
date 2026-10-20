#=
Reading initial fields and stratification profiles from NetCDF or JLD2 files,
and interpolating fields given on another rectilinear grid onto the model's.
=#

using NCDatasets
using JLD2: JLD2, jldopen

_is_jld2(path) = lowercase(splitext(path)[2]) == ".jld2"

_source_description(filename, ::Nothing) = repr(filename)
_source_description(filename, group) = "$(repr(group)) in $(repr(filename))"

_group_path(::Nothing) = ()
_group_path(group) = split(group, '/'; keepempty=false)

_missing_group(filename, group) =
    throw(ArgumentError("$(repr(filename)) has no group $(repr(group))"))

"""The NetCDF group at the `/`-separated path `group` of an open `dataset`."""
function _netcdf_container(dataset, filename, group)
    container = dataset
    for part in _group_path(group)
        haskey(container.group, part) || _missing_group(filename, group)
        container = container.group[part]
    end
    return container
end

"""The group, NamedTuple, or Dict at the path `group` of an open JLD2 `file`."""
function _jld2_container(file, filename, group)
    container = file
    for part in _group_path(group)
        part in _entry_names(container) || _missing_group(filename, group)
        container = _entry(container, part)
        container isa Union{JLD2.Group, NamedTuple, AbstractDict} ||
            throw(ArgumentError("$(_source_description(filename, group)) is a " *
                                "$(typeof(container)), not a group, " *
                                "NamedTuple, or Dict"))
    end
    return container
end

# JLD2 files and groups are indexed by name; a stored NamedTuple or Dict by
# its keys.
_entry_names(container::Union{NamedTuple, AbstractDict}) =
    String[string(key) for key in keys(container)]
_entry_names(container) = keys(container)
_entry(container::NamedTuple, name) = container[Symbol(name)]
_entry(container::AbstractDict, name) =
    haskey(container, name) ? container[name] : container[Symbol(name)]
_entry(container, name) = container[name]

function _real_values(data, name, source)
    data isa AbstractArray && all(value -> value isa Real, data) ||
        throw(ArgumentError("$name in $source must be an array of real numbers " *
                            "without missing values; got $(typeof(data))"))
    return collect(Float64, vec(data))
end

"""
Depths, positive downward, from the vertical coordinate named `name` of a
stratification file: `depth` is positive downward and `z` positive upward,
both from the surface, except that a `z` without negative values is read as
depth, as in earlier versions.
"""
_file_depths(name, values) =
    name == "z" && any(<(zero(eltype(values))), values) ? -values : values

"""
Heights, in the grid's coordinates, of the vertical coordinate named `name`
of a field file on a grid whose top (its lid) is at `top`: `depth` is
measured down from the lid; `z` is a height, except that a `z` with no point
below the lid, and some above it, is read as depth, as in earlier versions.
"""
function _field_heights(name, values, top)
    as_depth = name == "depth" || (all(>=(top), values) && any(>(top), values))
    return as_depth ? top .- values : values
end

"""
    _read_field(source, grid)

The values of `source` in z-x-y layout, checked to be finite, and the
coordinate vectors `(; x, y, z)` of its points (z in the grid's
coordinates), or `nothing` in place of the coordinates when they are not
all given.
"""
function _read_field(source::FieldArray, grid::RectilinearGrid)
    values = _checked_field_values(Array(source.values), "the FieldArray")
    return _to_zxy(values, source.layout === :zxy ? (1, 2, 3) : (3, 1, 2)),
           source.coordinates
end

function _read_field(source::FieldFile, grid::RectilinearGrid)
    isfile(source.path) ||
        throw(ArgumentError("field file does not exist: $(source.path)"))
    location = _source_description(source.path, source.group)
    description = "$(repr(source.variable)) in $location"
    missing_variable() = throw(ArgumentError(
        "$location has no variable $(repr(source.variable))"))
    top = Float64(last(grid.z_faces))
    # Everything is read and checked while the file is open: an error must
    # not hold on to a variable of a closed file.
    if _is_jld2(source.path)
        return jldopen(source.path, "r") do file
            container = _jld2_container(file, source.path, source.group)
            names = _entry_names(container)
            source.variable in names || missing_variable()
            values = _entry(container, source.variable)
            values isa AbstractArray{<:Number, 3} || throw(ArgumentError(
                "$description must be a three-dimensional numeric array; " *
                "got $(typeof(values))"))
            values = _checked_field_values(Array(values), description)
            (_to_zxy(values, source.layout === :zxy ? (1, 2, 3) : (3, 1, 2)),
             _stored_coordinates(in(names), name -> _entry(container, name),
                                 location, top))
        end
    else
        return NCDataset(source.path, "r") do dataset
            container = _netcdf_container(dataset, source.path, source.group)
            haskey(container, source.variable) || missing_variable()
            variable = container[source.variable]
            dimensions = String[string(name) for name in NCDatasets.dimnames(variable)]
            values = Array(variable)
            values, dimensions = _drop_extra_dimensions(values, dimensions, description)
            permutation = _zxy_permutation(dimensions, source.layout, description)
            values = _checked_field_values(values, description)
            (_to_zxy(values, permutation),
             _stored_coordinates(name -> haskey(container, name),
                                 name -> container[name][:], location, top))
        end
    end
end

_to_zxy(values, permutation) =
    permutation == (1, 2, 3) ? values : permutedims(values, permutation)

"""
`values` without the missing (fill) values a NetCDF variable reads as
`missing`, in a numeric element type; an error when any is missing or not
finite.
"""
function _checked_field_values(values::AbstractArray, description)
    missing_count = count(ismissing, values)
    missing_count == 0 || throw(ArgumentError(
        "$description has $missing_count missing or fill values; fill them " *
        "before setting the field"))
    T = nonmissingtype(eltype(values))
    T <: Number || throw(ArgumentError(
        "$description must hold numbers; got $(eltype(values))"))
    numbers = convert(Array{T}, values)
    all(isfinite, numbers) || throw(ArgumentError(
        "$description has non-finite values (NaN or Inf)"))
    return numbers
end

"""Dimension names that identify each axis of a NetCDF field."""
const _AXIS_DIMENSION_NAMES = (
    x=("x", "xc", "x_c", "xh", "lon", "longitude", "nx"),
    y=("y", "yc", "y_c", "yh", "lat", "latitude", "ny"),
    z=("z", "zc", "z_c", "depth", "level", "lev", "nz"),
)

function _axis_of(dimension)
    name = lowercase(dimension)
    for axis in (:x, :y, :z)
        name in _AXIS_DIMENSION_NAMES[axis] && return axis
    end
    return nothing
end

"""
`values` without dimensions of length one beyond the three of a field, such
as a single time; an error unless three dimensions remain.
"""
function _drop_extra_dimensions(values, dimensions, description)
    while ndims(values) > 3
        index = findfirst(d -> size(values, d) == 1 && _axis_of(dimensions[d]) === nothing,
                          1:ndims(values))
        index === nothing && break
        values = dropdims(values; dims=index)
        dimensions = deleteat!(copy(dimensions), index)
    end
    ndims(values) == 3 || throw(ArgumentError(
        "$description has dimensions ($(join(dimensions, ", "))) of sizes " *
        "$(size(values)); a field needs three, x, y, and z"))
    return values, dimensions
end

"""
The permutation that puts a NetCDF field in z-x-y order: from its dimension
names when they name the axes (x, y, z or depth, and common alternatives;
with two named, the third is the remaining axis), or else from `layout`.
"""
function _zxy_permutation(dimensions, layout, description)
    axes = map(_axis_of, dimensions)
    known = filter(!isnothing, collect(axes))
    allunique(known) || throw(ArgumentError(
        "the dimensions ($(join(dimensions, ", "))) of $description name an axis twice"))
    if length(known) == 2
        remaining = only(setdiff((:x, :y, :z), known))
        axes = map(axis -> axis === nothing ? remaining : axis, axes)
    elseif length(known) == 1
        throw(ArgumentError(
            "only one dimension of $description, of ($(join(dimensions, ", "))), " *
            "names an axis; name all three (x, y, and z or depth), or none and " *
            "give the layout"))
    end
    isempty(known) && return layout === :zxy ? (1, 2, 3) : (3, 1, 2)
    return (findfirst(==(:z), axes), findfirst(==(:x), axes), findfirst(==(:y), axes))
end

function _stored_coordinates(has, read, location, top)
    vertical = findfirst(has, ("z", "depth"))
    has("x") && has("y") && vertical !== nothing || return nothing
    name = ("z", "depth")[vertical]
    heights = _field_heights(name, _real_values(read(name), name, location), top)
    return (x=_real_values(read("x"), "x", location),
            y=_real_values(read("y"), "y", location),
            z=heights)
end

"""Whether `coordinates` are the model's own points, to within roundoff."""
function _on_model_grid(coordinates, grid::RectilinearGrid)
    matches(points, nodes, spacing) = length(points) == length(nodes) &&
        all(abs(point - node) <= 1e-6 * spacing for (point, node) in zip(points, nodes))
    return matches(coordinates.x, grid.x, grid.dx) &&
           matches(coordinates.y, grid.y, grid.dy) &&
           matches(coordinates.z, grid.z, grid.dz)
end

"""
    _interpolate_to_grid(values, coordinates, grid, source)

Interpolate `values[k, i, j]` (z-x-y layout), given at the points of the
rectilinear `coordinates` (z positive upward), onto the cell centers of
`grid`: linearly along each axis, periodically in x and y with the domain's
period, and holding the end values above and below the data in z.
"""
function _interpolate_to_grid(values, coordinates, grid::RectilinearGrid, source)
    lengths = (length(coordinates.z), length(coordinates.x), length(coordinates.y))
    size(values) == lengths || throw(DimensionMismatch(
        "$source has size $(size(values)) in z-x-y layout, but its z, x, and y " *
        "coordinates have lengths $lengths"))
    wx = _periodic_weights(coordinates.x, grid.x, grid.origin[1], grid.extent[1],
                           grid.dx, "x", source)
    wy = _periodic_weights(coordinates.y, grid.y, grid.origin[2], grid.extent[2],
                           grid.dy, "y", source)
    wz = _vertical_weights(coordinates.z, grid, source)

    nx, ny, nz = grid.size
    result = Array{ComplexF64}(undef, nz, nx, ny)
    for j in 1:ny, i in 1:nx
        i₀, i₁, α = wx[i]
        j₀, j₁, β = wy[j]
        for k in 1:nz
            k₀, k₁, γ = wz[k]
            result[k, i, j] =
                (1 - β) * ((1 - α) * _lerp(values, k₀, k₁, γ, i₀, j₀) +
                           α * _lerp(values, k₀, k₁, γ, i₁, j₀)) +
                β * ((1 - α) * _lerp(values, k₀, k₁, γ, i₀, j₁) +
                     α * _lerp(values, k₀, k₁, γ, i₁, j₁))
        end
    end
    return result
end

@inline _lerp(values, k₀, k₁, γ, i, j) =
    @inbounds (1 - γ) * values[k₀, i, j] + γ * values[k₁, i, j]

_phase(x, origin, period) = (phase = mod(x - origin, period);
                             phase < period ? phase : zero(phase))

"""
Weights `(lower, upper, fraction)` interpolating data at `coordinate` linearly
to each of `nodes`, periodically with `period` from `origin`. The data must
lie within one period; a last point one period after the first repeats it.
"""
function _periodic_weights(coordinate, nodes, origin, period, spacing, axis, source)
    isempty(coordinate) && throw(ArgumentError("$source has no $axis coordinates"))
    all(isfinite, coordinate) ||
        throw(ArgumentError("the $axis coordinates of $source must be finite"))
    order = sortperm(coordinate)
    sorted = coordinate[order]
    span = sorted[end] - sorted[1]
    tolerance = 1e-6 * period
    span <= period + tolerance || throw(ArgumentError(
        "the $axis coordinates of $source span $span, more than the domain's " *
        "period $period; give one period of the periodic field"))
    span >= period - tolerance && pop!(order)

    phases = [_phase(coordinate[index], origin, period) for index in order]
    wrap = sortperm(phases)
    phase, points = phases[wrap], order[wrap]
    n = length(points)
    if n > 1
        steps = diff(phase)
        all(>(0), steps) ||
            throw(ArgumentError("the $axis coordinates of $source must be unique"))
        gap = phase[1] + period - phase[n]
        gap - maximum(steps) > spacing &&
            @warn "The $axis coordinates of $source cover $(sorted[1]) to " *
                  "$(sorted[end]), leaving a gap of $(round(gap; sigdigits=3)) " *
                  "at the domain's periodic boundary; it is filled by linear " *
                  "interpolation across the boundary."
    end

    weights = Vector{Tuple{Int, Int, Float64}}(undef, length(nodes))
    for (index, node) in enumerate(nodes)
        if n == 1
            weights[index] = (points[1], points[1], 0.0)
            continue
        end
        p = _phase(Float64(node), origin, period)
        j = searchsortedlast(phase, p)
        lower, upper = 0 < j < n ? (j, j + 1) : (n, 1)
        left = j == 0 ? phase[n] - period : phase[lower]
        right = j == n ? phase[1] + period : phase[upper]
        weights[index] = (points[lower], points[upper], (p - left) / (right - left))
    end
    return weights
end

"""
Weights `(lower, upper, fraction)` interpolating data at heights `coordinate`
linearly to the cell centers of `grid`, holding the end values beyond the
data, with a warning when that leaves more than one grid spacing uncovered.
"""
function _vertical_weights(coordinate, grid::RectilinearGrid, source)
    n = length(coordinate)
    n > 0 || throw(ArgumentError("$source has no z coordinates"))
    all(isfinite, coordinate) ||
        throw(ArgumentError("the z coordinates of $source must be finite"))
    order = sortperm(coordinate)
    sorted = coordinate[order]
    all(>(0), diff(sorted)) ||
        throw(ArgumentError("the z coordinates of $source must be unique"))

    bottom, top = grid.z_faces[1], grid.z_faces[end]
    if sorted[1] - bottom > grid.dz || top - sorted[end] > grid.dz
        @warn "$(uppercasefirst(source)) covers z = $(sorted[1]) to " *
              "$(sorted[end]), but the domain spans z = $bottom to $top; " *
              "outside that range it is held at the nearest data value."
    end

    weights = Vector{Tuple{Int, Int, Float64}}(undef, length(grid.z))
    for (index, node) in enumerate(grid.z)
        weights[index] = if node <= sorted[1]
            (order[1], order[1], 0.0)
        elseif node >= sorted[n]
            (order[n], order[n], 0.0)
        else
            j = searchsortedlast(sorted, node)
            (order[j], order[j + 1], (node - sorted[j]) / (sorted[j + 1] - sorted[j]))
        end
    end
    return weights
end
