#=
================================================================================
            kernel_utils.jl - Small kernels shared by the operators
================================================================================

Elementwise maps and dealiasing projections used throughout the pseudo-spectral
operators. They are KernelAbstractions kernels rather than broadcasts so that
host runs use KernelAbstractions' threaded CPU backend just like device runs
use the GPU.
================================================================================
=#

# `@kernel` defines fixed constructor methods per name, so each arity needs its
# own kernel name.
@kernel function _map1_kernel!(f, destination, a)
    index = @index(Global, Linear)
    @inbounds destination[index] = f(a[index])
end

@kernel function _map2_kernel!(f, destination, a, b)
    index = @index(Global, Linear)
    @inbounds destination[index] = f(a[index], b[index])
end

@kernel function _map3_kernel!(f, destination, a, b, c)
    index = @index(Global, Linear)
    @inbounds destination[index] = f(a[index], b[index], c[index])
end

"""
    in_precision(array, T)

`array` with the element type of precision `T`: `T` for a real array and
`Complex{T}` for a complex one; `nothing` stays `nothing`. A coefficient
profile in another precision would make a kernel compute in that precision.
"""
in_precision(::Nothing, ::Type{T}) where {T<:AbstractFloat} = nothing
function in_precision(array::AbstractArray, ::Type{T}) where {T<:AbstractFloat}
    S = eltype(array) <: Complex ? Complex{T} : T
    return eltype(array) === S ? array : S.(array)
end

_map_kernel(::Val{1}) = _map1_kernel!
_map_kernel(::Val{2}) = _map2_kernel!
_map_kernel(::Val{3}) = _map3_kernel!

"""
    kernel_map!(f, destination, arrays...)

Set `destination[I] = f(arrays[1][I], ...)` for every index with a kernel on
the destination's backend, for one to three source arrays. `destination` may
be one of `arrays`.
"""
function kernel_map!(f::F, destination, arrays::Vararg{AbstractArray, N}) where {F, N}
    for array in arrays
        size(array) == size(destination) || throw(DimensionMismatch(
            "kernel_map! needs equally sized arrays; got $(size(array)) and $(size(destination))"))
    end
    launch!(_map_kernel(Val(N)), destination, (length(destination),),
            f, destination, arrays...)
    return destination
end

@kernel function _dealias_kernel!(destination, source, mask)
    k, i, j = @index(Global, NTuple)
    @inbounds destination[k, i, j] =
        keep_mode(mask, i, j) ? source[k, i, j] : zero(eltype(destination))
end

"""
    dealias!(destination, source, mask)

Copy the horizontal modes of `source` that `mask` keeps into `destination` and
zero the rest. `mask` is a kernel dealiasing rule or Bool matrix (see
[`kernel_mask`](@ref)); `destination` may be `source`.
"""
function dealias!(destination, source, mask)
    size(source) == size(destination) || throw(DimensionMismatch(
        "dealias! needs equally sized arrays; got $(size(source)) and $(size(destination))"))
    launch!(_dealias_kernel!, destination, size(destination),
            destination, source, mask)
    return destination
end
