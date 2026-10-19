#=
================================================================================
                scratch.jl - Reusable grid-sized temporaries
================================================================================

The pseudo-spectral kernels need a handful of grid-sized complex temporaries
each. Allocating those with `similar` on every call put tens of megabytes of
garbage per timestep on the heap (or, on a GPU, through the device allocator).

A `ModelWorkspace` hands the same buffers back on every subsequent call.
Borrowing is scoped, so nested kernels each get their own buffers and give
them back:

```julia
with_scratch(workspace) do
    uk = scratch_like(workspace, S.psi)
    tmpu = scratch_physical(workspace, uk, plans)
    ...
end                       # buffers returned to the pool here
```

Spectral and physical buffers have the same shape and element type, so one
pool serves both; `scratch_physical` and `scratch_phys_like` remain separate
names because they say which space a buffer holds at the call site.

Rules for call sites:
- A borrowed buffer must not outlive its `with_scratch` block. Never return one,
  never store one in a struct.
- Open a scope in the kernel that borrows, not in its caller, so nested kernels
  nest their scopes too.
- `workspace === nothing` still works: every helper falls back to a fresh
  allocation.
- The pool keeps its high-water mark, the deepest nesting of borrows over the
  whole run, for good, and that sets much of the model's memory. Form a
  spectral quantity directly in its buffer and transform it in place
  (`fft_backward!(buffer, buffer, plans)`), write pointwise products over an
  input when it is no longer needed, and return buffers before calling into
  another borrowing kernel.

The workspace also owns one real `(nz, nx, ny)` array, `tridiagonal`, which the
column kernels use for the modified superdiagonal of their Thomas sweeps. Each
solve finishes before the next kernel on the same backend starts, so a single
array serves every solve.
================================================================================
=#

"""Grow-on-demand bank of identically shaped arrays, borrowed LIFO."""
mutable struct ScratchPool{A}
    buffers::Vector{A}
    taken::Int
end

ScratchPool{A}() where {A} = ScratchPool{A}(Vector{A}(), 0)

"""Runtime-owned scratch storage on the model's architecture."""
struct ModelWorkspace{C, R}
    pool::ScratchPool{C}
    tridiagonal::R
end

"""
    ModelWorkspace(architecture, dims; T=Float64)

Allocate an empty scratch workspace for `Complex{T}` fields of size `dims` on
`architecture`.
"""
function ModelWorkspace(architecture::AbstractArchitecture, dims::NTuple{3, Int};
                        T::Type{<:AbstractFloat}=Float64)
    tridiagonal = zeros_on(architecture, T, dims...)
    # A zero-size array fixes the pool's buffer type without a full allocation.
    buffer_type = typeof(similar(tridiagonal, Complex{T}, (0, 0, 0)))
    return ModelWorkspace(ScratchPool{buffer_type}(), tridiagonal)
end

_scratch_pool(workspace::ModelWorkspace) = workspace.pool
_scratch_pool(workspace) = nothing

"""
    with_scratch(kernel, workspace)

Run `kernel`, returning every buffer it borrows to `workspace` afterwards.
"""
function with_scratch(kernel, workspace)
    pool = _scratch_pool(workspace)
    pool === nothing && return kernel()
    mark = pool.taken
    try
        return kernel()
    finally
        pool.taken = mark
    end
end

@inline function _borrow(pool::ScratchPool{A}, template) where {A}
    template isa A || return similar(template)
    index = pool.taken + 1
    pool.taken = index
    if index > length(pool.buffers)
        push!(pool.buffers, similar(template))
    elseif size(pool.buffers[index]) != size(template)
        pool.buffers[index] = similar(template)
    end
    return pool.buffers[index]
end

"""Borrow a spectral buffer shaped like `template`, or allocate a fresh one."""
@inline function scratch_like(workspace, template)
    pool = _scratch_pool(workspace)
    pool === nothing && return similar(template)
    return _borrow(pool, template)
end

"""
    scratch_physical(workspace, template, plans)

Borrow a buffer suitable as an `fft_backward!` destination for `template`, or
allocate a fresh one.
"""
@inline scratch_physical(workspace, template, plans) =
    scratch_like(workspace, template)

"""Borrow a buffer shaped like a physical-space `template`, or allocate one."""
@inline scratch_phys_like(workspace, template) = scratch_like(workspace, template)

"""Real `(nz, nx, ny)` scratch for Thomas sweeps, borrowed or freshly allocated."""
@inline function tridiagonal_scratch(workspace::ModelWorkspace, template)
    scratch = workspace.tridiagonal
    compatible = size(scratch) == size(template) &&
                 eltype(scratch) === real(eltype(template)) &&
                 typeof(KernelAbstractions.get_backend(scratch)) ===
                 typeof(KernelAbstractions.get_backend(template))
    return compatible ? scratch : similar(template, real(eltype(template)))
end
@inline tridiagonal_scratch(workspace, template) =
    similar(template, real(eltype(template)))
