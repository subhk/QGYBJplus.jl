"""
    ModelFields{T, RT, CT}

Prognostic and diagnostic arrays owned by a QG-YBJ model. All fields use
`(z, x, y)` dimension order. The prognostic fields `q` and `B`, together with
the diagnostic fields `psi`, `A`, and `C`, are spectral and complex-valued;
the velocity fields `u`, `v`, and `w` are real-valued.

`RT` and `CT` are the array types of the model's architecture: `Array` on the
CPU, or the GPU package's device array (e.g. `CuArray`) on a GPU.

`step!` updates these arrays in place, so references to the container and to
its individual arrays both stay live across steps.
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
    ModelFields([architecture=CPU()], T, dimensions)

Allocate zero-filled model fields with `(nz, nx, ny) == dimensions` on
`architecture`.
"""
ModelFields(::Type{T}, dimensions) where {T<:AbstractFloat} =
    ModelFields(CPU(), T, dimensions)

function ModelFields(architecture::AbstractArchitecture, ::Type{T},
                     dimensions) where {T<:AbstractFloat}
    dims = _field_dimensions(dimensions)
    q = zeros_on(architecture, Complex{T}, dims...)
    B = zeros_on(architecture, Complex{T}, dims...)
    psi = zeros_on(architecture, Complex{T}, dims...)
    A = zeros_on(architecture, Complex{T}, dims...)
    C = zeros_on(architecture, Complex{T}, dims...)
    u = zeros_on(architecture, T, dims...)
    v = zeros_on(architecture, T, dims...)
    w = zeros_on(architecture, T, dims...)
    return ModelFields{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end

"""Allocate zero-filled fields on the architecture, and in the precision, of a `RectilinearGrid`."""
function allocate_fields(grid::RectilinearGrid; T::Type{<:AbstractFloat}=eltype(grid))
    nx, ny, nz = grid.size
    return ModelFields(grid.architecture, T, (nz, nx, ny))
end

"""Allocate zero-filled fields on the architecture of a `RuntimeGeometry`."""
allocate_fields(grid::RuntimeGeometry; T::Type{<:AbstractFloat}=eltype(grid.base)) =
    allocate_fields(grid.base; T)

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

Copy model fields with `similar`, keeping every copied array independent and
on the same architecture as `fields`.
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

architecture(fields::ModelFields) = architecture(fields.q)
