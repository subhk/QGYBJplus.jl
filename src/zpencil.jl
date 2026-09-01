#=
================================================================================
                zpencil.jl - Running vertical kernels on distributed z
================================================================================

Spectral model fields live on the PencilFFTs *output* pencil, which decomposes
the vertical dimension whenever the process topology has more than one rank in
its first direction (`-n 4` -> a (2, 2) grid). Any operator that couples
neighbouring vertical levels — a tridiagonal solve, a finite difference in z,
a running sum over z — therefore cannot index its arrays directly.

`with_z_local` is the single place that knows how to fix that. It hands the
kernel arrays whose vertical dimension is fully local, transposing to and from
the z-pencil only when the arrays are actually distributed in z:

```julia
with_z_local(geometry, (destination, source), (:out, :in);
             scratch=z_scratch(workspace, :psi_z, :q_z)) do psi_z, q_z
    my_vertical_kernel!(psi_z, q_z, geometry)
end
```

`z_scratch` always returns a tuple, so a one-buffer call site cannot
accidentally pass a bare array.

The kernel receives whole arrays, not their parents, so
`local_to_global(i, 2, array)` resolves horizontal wavenumbers correctly in
both cases — pass-through and transposed. That is what lets one kernel serve
the serial, 1D-decomposed and 2D-decomposed paths instead of the near-identical
`_direct!` / `_2d!` pairs this helper replaces.

Directions:
- `:in`    read by the kernel; transposed in, not written back
- `:out`   written by the kernel; not transposed in, transposed back out
- `:inout` both
================================================================================
=#

"""Vertical dimension of every array in `arrays` is already fully local."""
function _all_z_local(geometry::RuntimeGeometry, arrays::Tuple)
    geometry.decomposition === nothing && return true
    hasfield(typeof(geometry.decomposition), :pencil_z) || return true
    return all(array -> z_is_local(array, geometry), arrays)
end

function _z_scratch(scratch::Tuple, index::Int, template, geometry::RuntimeGeometry)
    if index <= length(scratch)
        candidate = scratch[index]
        candidate === nothing ||
            return candidate::AbstractArray{eltype(template), 3}
    end
    return allocate_z_pencil(geometry, eltype(template))
end

"""
    with_z_local(kernel, geometry, arrays, directions; scratch=())

Run `kernel` on versions of `arrays` whose vertical dimension is fully local.

`directions` marks each array `:in`, `:out`, or `:inout`. Arrays that are
already z-local are passed straight through with no communication. Otherwise
each is mirrored onto a z-pencil buffer, taken from `scratch` when one is
supplied at the matching position and allocated when it is not.

Returns whatever `kernel` returns.
"""
function with_z_local(kernel, geometry::RuntimeGeometry, arrays::Tuple,
                      directions::Tuple; scratch::Tuple=())

    length(arrays) == length(directions) || throw(ArgumentError(
        "with_z_local needs one direction per array " *
        "(got $(length(arrays)) arrays, $(length(directions)) directions)"))
    all(direction -> direction in (:in, :out, :inout), directions) ||
        throw(ArgumentError("directions must each be :in, :out, or :inout"))

    _all_z_local(geometry, arrays) && return kernel(arrays...)

    views = ntuple(index -> _z_scratch(scratch, index, arrays[index], geometry),
                   length(arrays))
    for outer_index in eachindex(views), inner_index in 1:(outer_index - 1)
        views[outer_index] === views[inner_index] && throw(ArgumentError(
            "with_z_local requires distinct z-pencil buffers " *
            "(arguments $inner_index and $outer_index share one)"))
    end

    for index in eachindex(arrays)
        directions[index] === :out ||
            transpose_to_z_pencil!(views[index], arrays[index], geometry)
    end
    result = kernel(views...)
    for index in eachindex(arrays)
        directions[index] === :in ||
            transpose_to_xy_pencil!(arrays[index], views[index], geometry)
    end
    return result
end

"""One named z-pencil scratch array from `workspace`, or `nothing`."""
@inline function _named_z_scratch(workspace, name::Symbol)
    workspace === nothing && return nothing
    hasproperty(workspace, name) || return nothing
    return getproperty(workspace, name)
end

"""
    z_scratch(workspace, names...) -> Tuple

Named z-pencil scratch arrays from `workspace`, in the order given, as a tuple
ready for `with_z_local`'s `scratch` keyword. Always a tuple, including for a
single name, so a one-buffer call site cannot accidentally pass a bare array.
Missing names come back as `nothing`, which makes `with_z_local` allocate.
"""
@inline z_scratch(workspace, names::Symbol...) =
    map(name -> _named_z_scratch(workspace, name), names)
