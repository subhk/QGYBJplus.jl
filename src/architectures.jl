#=
================================================================================
                architectures.jl - Where model arrays live and run
================================================================================

Every array kernel in QGYBJ+.jl is a KernelAbstractions.jl kernel, launched on
the backend that owns the arrays it writes. An architecture names that backend:

    CPU()          host `Array` storage, KernelAbstractions.CPU(), FFTW
    GPU()          device storage from the loaded GPU package (CUDA.jl), cuFFT
    GPU(backend)   any KernelAbstractions backend, e.g. GPU(CUDABackend())

The architecture is chosen once, on the grid,

    grid = RectilinearGrid(GPU(); size=(256, 256, 64), extent=(L, L, H))

and every resource the model builds from that grid (fields, transform plans,
scratch buffers, coefficient profiles, the dealiasing mask) lives on it.

Kernels never look up an architecture themselves: they run on the backend of
the arrays they are handed (`KernelAbstractions.get_backend`). Small host-side
inputs a caller may pass instead — a stratification profile, a mask — are moved
to that backend with `on_backend` first, which is a no-op when they are already
there.
================================================================================
=#

abstract type AbstractArchitecture end

"""
    CPU()

Run on the host with `Array` storage. Kernels use KernelAbstractions' CPU
backend, which is multithreaded when Julia is started with several threads, and
horizontal transforms use FFTW.
"""
struct CPU <: AbstractArchitecture end

"""
    GPU(backend)
    GPU()

Run on a single GPU described by a KernelAbstractions `backend`. `GPU()` uses
the default backend of the loaded GPU package: after `using CUDA` this is
`CUDA.CUDABackend()`, with cuFFT for the horizontal transforms.
"""
struct GPU{B} <: AbstractArchitecture
    backend::B
end

"""Backend used by `GPU()`. GPU package extensions add the method."""
function default_gpu_backend end

function GPU()
    applicable(default_gpu_backend) || throw(ArgumentError(
        "GPU() needs a GPU package. Load one first (`using CUDA`), or pass " *
        "a KernelAbstractions backend explicitly: GPU(backend)."))
    return GPU(default_gpu_backend())
end

Base.show(io::IO, ::CPU) = print(io, "CPU()")
Base.show(io::IO, architecture::GPU) =
    print(io, "GPU(", nameof(typeof(architecture.backend)), ")")

"""KernelAbstractions backend of an architecture."""
device(::CPU) = KernelAbstractions.CPU()
device(architecture::GPU) = architecture.backend

"""
    architecture(x)

Architecture that owns `x`: an array, a KernelAbstractions backend, a grid, a
model, or a simulation.
"""
function architecture end

architecture(::KernelAbstractions.CPU) = CPU()
architecture(backend::KernelAbstractions.Backend) = GPU(backend)
architecture(array::AbstractArray) =
    architecture(KernelAbstractions.get_backend(array))

"""
    on_architecture(architecture, x)

Return `x` with every array it contains stored on `architecture`. Arrays that
are already there are returned as they are, without a copy.
"""
on_architecture(architecture::AbstractArchitecture, x) =
    Adapt.adapt(device(architecture), x)

"""Move `x` to a KernelAbstractions backend; a no-op when it is already there."""
on_backend(backend::KernelAbstractions.Backend, x) = Adapt.adapt(backend, x)

"""Host copy of `x`, or `x` itself when it already lives on the host."""
on_host(x) = Adapt.adapt(KernelAbstractions.CPU(), x)

"""
    first_value(array)

First element of `array`, read with a one-element copy so device arrays are
never indexed from the host.
"""
first_value(array::Array) = first(array)
function first_value(array::AbstractArray)
    value = Vector{eltype(array)}(undef, 1)
    copyto!(value, 1, array, 1, 1)
    return only(value)
end

"""
    array_architecture(architecture)

`architecture` with the backend that its arrays report. Kernels run on
`get_backend(array)`, so backend options that arrays do not record, such as
`CUDABackend(; prefer_blocks=true)`, cannot reach a kernel launch. Grids store
this form, which keeps `architecture(model) == architecture(model.fields)`.
"""
array_architecture(architecture::CPU) = architecture
function array_architecture(architecture::GPU)
    probe = KernelAbstractions.allocate(architecture.backend, Float32, 0)
    backend = KernelAbstractions.get_backend(probe)
    if backend != architecture.backend
        @warn "Kernels run on the backend their arrays report, $backend; the " *
              "options of $(architecture.backend) are not used." maxlog=1
    end
    return GPU(backend)
end

"""Zero-filled array of element type `T` on `architecture`."""
zeros_on(architecture::AbstractArchitecture, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.zeros(device(architecture), T, dims...)

"""
    launch!(kernel!, backend_or_array, ndrange, args...)

Run the KernelAbstractions `kernel!` over `ndrange` on the given backend, or on
the backend of the given array. Empty ranges launch nothing.
"""
@inline function launch!(kernel!::F, backend::KernelAbstractions.Backend,
                         ndrange::Tuple, args::Vararg{Any, N}) where {F, N}
    any(iszero, ndrange) && return nothing
    kernel!(backend)(args...; ndrange=ndrange)
    return nothing
end

@inline launch!(kernel!::F, array::AbstractArray, ndrange::Tuple,
                args::Vararg{Any, N}) where {F, N} =
    launch!(kernel!, KernelAbstractions.get_backend(array), ndrange, args...)

#=
Kernel-side dealiasing rules. Public functions accept a `Lmask` keyword that is
either a Bool matrix indexed by (i, j) or `nothing`; what `nothing` means is
documented per function. Kernels instead receive one of these, so the rule is
resolved at compile time and a "keep everything" mask costs no memory.
=#

"""Kernel dealiasing rule that keeps every horizontal mode."""
struct KeepAllModes end

"""Kernel dealiasing rule implementing the two-thirds cutoff of [`is_dealiased`](@ref)."""
struct RadialDealiasing
    nx::Int
    ny::Int
end

@inline keep_mode(::KeepAllModes, i, j) = true
@inline keep_mode(rule::RadialDealiasing, i, j) =
    is_dealiased(i, j, rule.nx, rule.ny)
@inline keep_mode(mask::AbstractMatrix{Bool}, i, j) = @inbounds mask[i, j]

_dense_host_mask(mask::BitArray) = Array{Bool}(mask)
_dense_host_mask(mask) = mask

"""
    kernel_mask(Lmask, array, default)

Turn a user-facing `Lmask` into a kernel argument for the backend of the
`(nz, nx, ny)` `array`: `nothing` selects `default`, and a Bool matrix, which
must be `(nx, ny)` because kernels read it unchecked, is copied to a device
backend.
"""
kernel_mask(::Nothing, array, default) = default
kernel_mask(rule::Union{KeepAllModes, RadialDealiasing}, array, default) = rule
function kernel_mask(mask::AbstractMatrix{Bool}, array, default)
    _check_mask_size(mask, size(array, 2), size(array, 3))
    backend = KernelAbstractions.get_backend(array)
    backend isa KernelAbstractions.CPU && return on_host(mask)
    return on_backend(backend, _dense_host_mask(mask))
end

_check_mask_size(mask, nx, ny) = size(mask) == (nx, ny) || throw(DimensionMismatch(
    "the dealiasing mask has size $(size(mask)); expected (nx, ny) = $((nx, ny))"))
