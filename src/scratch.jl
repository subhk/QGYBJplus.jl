#=
================================================================================
                scratch.jl - Reusable grid-sized temporaries
================================================================================

The pseudo-spectral kernels need a handful of grid-sized temporaries each:
spectral buffers on the FFT output pencil and physical buffers on the FFT input
pencil. Allocating those with `similar` on every call put tens of megabytes of
garbage per timestep on the heap, which the GC then had to chase.

A `ScratchPool` hands the same buffers back on every subsequent call. Borrowing
is scoped, so nested kernels each get their own buffers and give them back:

```julia
with_scratch(workspace) do
    uk = scratch_like(workspace, S.psi)
    tmpu = scratch_physical(workspace, uk, plans)
    ...
end                       # buffers returned to the pool here
```

Rules for call sites:
- A borrowed buffer must not outlive its `with_scratch` block. Never return one,
  never store one in a struct.
- Open a scope in the kernel that borrows, not in its caller, so nested kernels
  nest their scopes too.
- `workspace === nothing` still works: every helper falls back to a fresh
  allocation, which is what the code did before pools existed.

A pool belongs to one runtime, so its physical buffers always share that
runtime's FFT input pencil; only the spectral pool re-checks shapes, because
callers can legitimately hand it arrays from different spectral pencils.
================================================================================
=#

"""Grow-on-demand bank of identically shaped arrays, borrowed LIFO."""
mutable struct ScratchPool{A}
    buffers::Vector{A}
    taken::Int
end

ScratchPool{A}() where {A} = ScratchPool{A}(Vector{A}(), 0)

_scratch_pools(workspace) =
    workspace !== nothing && hasproperty(workspace, :spectral) &&
    hasproperty(workspace, :physical)

"""
    with_scratch(kernel, workspace)

Run `kernel`, returning every buffer it borrows to `workspace` afterwards.
"""
function with_scratch(kernel, workspace)
    _scratch_pools(workspace) || return kernel()
    spectral_mark = workspace.spectral.taken
    physical_mark = workspace.physical.taken
    try
        return kernel()
    finally
        workspace.spectral.taken = spectral_mark
        workspace.physical.taken = physical_mark
    end
end

"""Borrow a spectral buffer shaped like `template`, or allocate a fresh one."""
@inline function scratch_like(workspace, template)
    _scratch_pools(workspace) || return similar(template)
    pool = workspace.spectral
    index = pool.taken + 1
    pool.taken = index
    if index > length(pool.buffers)
        push!(pool.buffers, similar(template))
    elseif size(parent(pool.buffers[index])) != size(parent(template))
        pool.buffers[index] = similar(template)
    end
    return pool.buffers[index]
end

"""
    scratch_physical(workspace, template, plans)

Borrow a buffer suitable as an `fft_backward!` destination for `template`, or
allocate a fresh one. Only the first borrow at a given depth allocates.
"""
@inline function scratch_physical(workspace, template, plans)
    _scratch_pools(workspace) ||
        return allocate_fft_backward_dst(template, plans)
    pool = workspace.physical
    index = pool.taken + 1
    pool.taken = index
    if index > length(pool.buffers)
        push!(pool.buffers, allocate_fft_backward_dst(template, plans))
    end
    return pool.buffers[index]
end

"""
Borrow a buffer shaped like a *physical-pencil* `template`, or allocate one.
Use this for temporaries derived from an existing physical array, where
`scratch_physical` has no spectral template to transform from.
"""
@inline function scratch_phys_like(workspace, template)
    _scratch_pools(workspace) || return similar(template)
    pool = workspace.physical
    index = pool.taken + 1
    pool.taken = index
    if index > length(pool.buffers)
        push!(pool.buffers, similar(template))
    elseif size(parent(pool.buffers[index])) != size(parent(template))
        pool.buffers[index] = similar(template)
    end
    return pool.buffers[index]
end
