#=
================================================================================
                    transforms.jl - FFT Planning and Execution
================================================================================

Horizontal FFTs for `(nz, nx, ny)` fields. A single in-place, batched plan
transforms dimensions 2 and 3 (x and y) of every vertical level at once, so a
transform is one library call with no per-level copies.

The plans come from AbstractFFTs, which dispatches on the array type:
FFTW for host `Array`s and cuFFT for `CuArray`s. A backend whose arrays have
no AbstractFFTs implementation can add a `horizontal_fft_plans` method.

TRANSFORM CONVENTION:
- Horizontal 2D complex-to-complex FFTs for each vertical level
- `fft_backward!` is NORMALIZED (divides by N = nx*ny internally)
- No manual normalization is needed after `fft_backward!`
- Wavenumber layout follows FFTW convention (see grid.jl)
================================================================================
=#

module Transforms

using ..QGYBJplus: RuntimeGeometry, AbstractArchitecture, device
import FFTW
import AbstractFFTs
import KernelAbstractions

"""
    Plans

Cached horizontal FFT plans for `(nz, nx, ny)` complex fields on one
architecture. `forward` and `backward` are in-place plans over dimensions
`(2, 3)`; `backward` is normalized, so no manual scaling is needed.
"""
struct Plans{F, B}
    forward::F
    backward::B
    size::NTuple{3, Int}
end

"""
    horizontal_fft_plans(buffer) -> (forward, backward)

In-place forward and normalized inverse plans over dimensions `(2, 3)` of the
complex 3D `buffer`. The plans are applied to other arrays of the same type
and size with `plan * array`. Add a method for array types that AbstractFFTs
does not cover.
"""
function horizontal_fft_plans(buffer::AbstractArray{<:Complex, 3})
    _check_fft_support(buffer)
    forward = AbstractFFTs.plan_fft!(buffer, (2, 3))
    backward = AbstractFFTs.plan_ifft!(buffer, (2, 3))
    return forward, backward
end

# FFTW accepts every dense array type, including device arrays it cannot read,
# so a device array that only FFTW would plan for has no FFT library.
function _check_fft_support(buffer)
    KernelAbstractions.get_backend(buffer) isa KernelAbstractions.CPU && return nothing
    signature = Tuple{typeof(buffer), Tuple{Int, Int}}
    supported = hasmethod(AbstractFFTs.plan_fft!, signature) &&
                parentmodule(which(AbstractFFTs.plan_fft!, signature)) !== FFTW
    supported || throw(ArgumentError(
        "no FFT library plans transforms of $(typeof(buffer)). Load one for this " *
        "backend, or add a method of QGYBJplus.Transforms.horizontal_fft_plans " *
        "for this array type (see the CPU and GPU execution page of the docs)."))
    return nothing
end

"""Build horizontal FFT plans for `(nz, nx, ny)` fields on `architecture`."""
function Plans(architecture::AbstractArchitecture, nz::Integer, nx::Integer,
               ny::Integer, ::Type{T}=ComplexF64) where {T<:Complex}
    buffer = KernelAbstractions.zeros(device(architecture), T, nz, nx, ny)
    forward, backward = horizontal_fft_plans(buffer)
    return Plans(forward, backward, (Int(nz), Int(nx), Int(ny)))
end

"""
    plan_transforms!(G::RuntimeGeometry) -> Plans

Create horizontal FFT plans for the fields of `G` on its architecture.
"""
plan_transforms!(G::RuntimeGeometry) = Plans(G.architecture, G.nz, G.nx, G.ny)

function _check_transform_size(P::Plans, array)
    size(array) == P.size || throw(DimensionMismatch(
        "transform plans are for size $(P.size); got an array of size $(size(array))"))
    return nothing
end

"""
    fft_forward!(dst, src, P::Plans)

Compute the horizontal forward FFT (complex-to-complex) of `src` for every
z-level and store it in `dst`. `dst` may be `src`.
"""
function fft_forward!(dst, src, P::Plans)
    _check_transform_size(P, dst)
    dst === src || copyto!(dst, src)
    P.forward * dst
    return dst
end

function fft_forward!(dst, src, runtime)
    hasproperty(runtime, :plans) ||
        throw(ArgumentError("third argument must be transform plans or a model runtime"))
    return fft_forward!(dst, src, runtime.plans)
end

"""
    fft_backward!(dst, src, P::Plans)

Compute the normalized horizontal inverse FFT of `src` for every z-level and
store it in `dst`. `dst` may be `src`.
"""
function fft_backward!(dst, src, P::Plans)
    _check_transform_size(P, dst)
    dst === src || copyto!(dst, src)
    P.backward * dst
    return dst
end

function fft_backward!(dst, src, runtime)
    hasproperty(runtime, :plans) ||
        throw(ArgumentError("third argument must be transform plans or a model runtime"))
    return fft_backward!(dst, src, runtime.plans)
end

"""
    allocate_fft_backward_dst(spectral_array, plans)

Allocate a destination for `fft_backward!` of `spectral_array`: physical and
spectral fields share shape, element type, and architecture.
"""
allocate_fft_backward_dst(spectral_array, plans) = similar(spectral_array)

end # module Transforms

using .Transforms: Plans, plan_transforms!, horizontal_fft_plans,
                   fft_forward!, fft_backward!, allocate_fft_backward_dst
