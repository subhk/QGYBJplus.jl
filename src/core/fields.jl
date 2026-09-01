"""
    ModelFields{T, RT, CT}

Prognostic and diagnostic arrays owned by a QG-YBJ model. All fields use
`(z, x, y)` dimension order. The prognostic fields `q` and `B`, together with
the diagnostic fields `psi`, `A`, and `C`, are spectral and complex-valued;
the velocity fields `u`, `v`, and `w` are real-valued.

`RT` and `CT` may be ordinary arrays or distributed pencil arrays.
"""
Base.@kwdef mutable struct ModelFields{
    T,
    RT<:AbstractArray{T, 3},
    CT<:AbstractArray{Complex{T}, 3},
}
    q::CT
    B::CT
    psi::CT
    A::CT
    C::CT
    u::RT
    v::RT
    w::RT
end

function _field_dimensions(dimensions)
    dimensions isa Tuple && length(dimensions) == 3 ||
        throw(ArgumentError("field dimensions must be a three-element tuple"))
    all(value -> value isa Integer, dimensions) ||
        throw(ArgumentError("field dimensions must be integers"))
    result = Int.(dimensions)
    all(>(0), result) || throw(ArgumentError("field dimensions must be positive"))
    return result
end

"""
    ModelFields(T, dimensions)

Allocate zero-filled serial model fields with `(nz, nx, ny) == dimensions`.
"""
function ModelFields(::Type{T}, dimensions) where {T<:AbstractFloat}
    dims = _field_dimensions(dimensions)
    q = zeros(Complex{T}, dims)
    B = zeros(Complex{T}, dims)
    psi = zeros(Complex{T}, dims)
    A = zeros(Complex{T}, dims)
    C = zeros(Complex{T}, dims)
    u = zeros(T, dims)
    v = zeros(T, dims)
    w = zeros(T, dims)
    return ModelFields{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end

"""
    allocate_field(T, grid; complex=false)

Allocate an uninitialized field from an internal `RuntimeGeometry`.
Distributed allocation is handled by `ModelRuntime`.
"""
function allocate_field(::Type{T}, grid::RuntimeGeometry; complex::Bool=false) where {T}
    dimensions = (grid.nz, grid.nx, grid.ny)
    if grid.decomposition === nothing
        return complex ? Array{Complex{T}}(undef, dimensions) :
                         Array{T}(undef, dimensions)
    end

    error("Parallel field allocation requires the MPI runtime")
end

"""Allocate zero-filled serial fields for a public `RectilinearGrid`."""
function allocate_fields(grid::RectilinearGrid; T::Type{<:AbstractFloat}=Float64)
    nx, ny, nz = grid.size
    return ModelFields(T, (nz, nx, ny))
end

"""Allocate zero-filled fields for the current computational `RuntimeGeometry`."""
function allocate_fields(grid::RuntimeGeometry; T::Type{<:AbstractFloat}=Float64)
    q = allocate_field(T, grid; complex=true)
    B = allocate_field(T, grid; complex=true)
    psi = allocate_field(T, grid; complex=true)
    A = allocate_field(T, grid; complex=true)
    C = allocate_field(T, grid; complex=true)
    u = allocate_field(T, grid)
    v = allocate_field(T, grid)
    w = allocate_field(T, grid)

    for field in (q, B, psi, A, C, u, v, w)
        fill!(field, zero(eltype(field)))
    end

    return ModelFields{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end

"""
    copy_fields!(destination::ModelFields, source::ModelFields)

Copy every array of `source` into the matching array of `destination`, which
must already have the same layout. Used by scheduled diagnostics, which observe
a snapshot every write and must not allocate a field set each time.
"""
function copy_fields!(destination::ModelFields, source::ModelFields)
    copyto!(destination.q, source.q)
    copyto!(destination.B, source.B)
    copyto!(destination.psi, source.psi)
    copyto!(destination.A, source.A)
    copyto!(destination.C, source.C)
    copyto!(destination.u, source.u)
    copyto!(destination.v, source.v)
    copyto!(destination.w, source.w)
    return destination
end

"""
    copy_fields(fields::ModelFields)

Copy model fields with `similar`, preserving pencil decompositions for
distributed arrays while keeping every copied array independent.
"""
function copy_fields(fields::ModelFields{T, RT, CT}) where {T, RT, CT}
    q = copyto!(similar(fields.q), fields.q)
    B = copyto!(similar(fields.B), fields.B)
    psi = copyto!(similar(fields.psi), fields.psi)
    A = copyto!(similar(fields.A), fields.A)
    C = copyto!(similar(fields.C), fields.C)
    u = copyto!(similar(fields.u), fields.u)
    v = copyto!(similar(fields.v), fields.v)
    w = copyto!(similar(fields.w), fields.w)
    return ModelFields{T, RT, CT}(q, B, psi, A, C, u, v, w)
end
