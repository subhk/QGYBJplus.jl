#=
================================================================================
                    transforms.jl - FFT Planning and Execution
================================================================================

This module provides FFT transforms for the QG-YBJ+ model. It uses FFTW.jl
for serial execution and supports MPI-parallel execution via the QGYBJMPIExt
extension with PencilFFTs.jl.

SERIAL MODE (default):
- Uses FFTW.jl for efficient FFT computation
- FFTW.jl is a required dependency

PARALLEL MODE (with extension):
- Uses PencilFFTs.jl for distributed FFTs
- Automatically enabled when MPI, PencilArrays, PencilFFTs are loaded
- See ext/QGYBJMPIExt.jl for parallel implementation

TRANSFORM CONVENTION:
- Horizontal 2D FFTs (x,y dimensions) for each vertical level
- FFTW.ifft is NORMALIZED (divides by N = nx*ny internally)
- No manual normalization needed after fft_backward!
- Wavenumber layout follows FFTW convention (see grid.jl)

================================================================================
=#

module Transforms

using ..QGYBJ: Grid
using LinearAlgebra
import FFTW

#=
================================================================================
                        NAIVE FFT (for reference/testing only)
================================================================================
These naive DFT implementations are provided for reference and tiny-size testing.
They are NOT used in production - FFTW.jl is always used. O(n^4) per plane.
================================================================================
=#

"""
    _naive_fft2!(dst, src)

Compute a simple 2D DFT of `src` into `dst` (complex arrays) without FFTW.
Intended only as a tiny-size fallback for testing; O(n^4) per plane.
"""
function _naive_fft2!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T<:Complex}
    nx, ny = size(src)
    for ky in 0:ny-1, kx in 0:nx-1
        s = zero(T)
        for y in 0:ny-1, x in 0:nx-1
            angle = -2π * ( (kx*x)/nx + (ky*y)/ny )
            s += src[x+1, y+1] * cis(angle)
        end
        dst[kx+1, ky+1] = s
    end
    return dst
end

"""
    _naive_ifft2!(dst, src)

Inverse 2D DFT fallback; NORMALIZED to match FFTW.ifft behavior.
FFTW.ifft divides by N (nx*ny for 2D), so this implementation does the same.
"""
function _naive_ifft2!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T<:Complex}
    nx, ny = size(src)
    norm_factor = one(real(T)) / (nx * ny)  # Normalization to match FFTW.ifft
    for y in 0:ny-1, x in 0:nx-1
        s = zero(T)
        for ky in 0:ny-1, kx in 0:nx-1
            angle = 2π * ( (kx*x)/nx + (ky*y)/ny )
            s += src[kx+1, ky+1] * cis(angle)
        end
        dst[x+1, y+1] = s * norm_factor
    end
    return dst
end

#=
================================================================================
                        FFT PLAN STRUCTURE
================================================================================
=#

"""
    Plans

Container for FFT plans. Used for serial FFTW execution.

For parallel execution with PencilFFTs, the extension module (QGYBJMPIExt)
provides `MPIPlans` which wraps `PencilFFTPlan` and uses `ldiv!` for the
normalized inverse transform.

# Fields
- `backend::Symbol`: Always `:fftw` for this struct
- `p_forward`: Reserved for future FFTW plan caching
- `p_backward`: Reserved for future FFTW plan caching
- `f_forward`: FFTW-specific forward plan (for pre-planned transforms)
- `f_backward`: FFTW-specific inverse plan (for pre-planned transforms)

# Note
When MPI/PencilArrays/PencilFFTs are loaded, use `plan_mpi_transforms()` instead,
which returns `MPIPlans` from the extension module.
"""
Base.@kwdef mutable struct Plans
    backend::Symbol = :fftw          # :fftw for serial mode
    # Reserved for future FFTW plan caching
    p_forward::Any = nothing
    p_backward::Any = nothing
    # FFTW pre-planned transforms (optional optimization)
    f_forward::Any = nothing
    f_backward::Any = nothing
end

#=
================================================================================
                        FFT PLANNING
================================================================================
=#

"""
    plan_transforms!(G::Grid, parallel_config=nothing) -> Plans

Create forward/backward FFT plans appropriate to the environment.

# Serial Mode (default)
Returns Plans with `:fftw` backend for per-slice FFT execution.

# Parallel Mode
If `parallel_config` indicates MPI is active and the grid has decomposition,
attempts to use PencilFFTs via the extension module.

# Arguments
- `G::Grid`: Grid structure (determines array sizes)
- `parallel_config`: Optional parallel configuration

# Returns
Plans struct with appropriate backend and plans.

# Example
```julia
G = init_grid(par)
plans = plan_transforms!(G)  # Serial FFTW
```
"""
function plan_transforms!(G::Grid, parallel_config=nothing)
    # If parallel_config indicates MPI is active, try parallel setup
    if parallel_config !== nothing
        if hasproperty(parallel_config, :use_mpi) && parallel_config.use_mpi && G.decomp !== nothing
            # Parallel mode requested - try extension
            return setup_parallel_transforms(G, parallel_config)
        end
    end

    # Default: serial FFTW mode
    # Note: We don't pre-plan here for simplicity. FFTW caches plans internally.
    return Plans(backend=:fftw)
end

"""
    setup_parallel_transforms(grid::Grid, pconfig) -> Plans

Set up FFT plans for parallel execution.

This is a stub that returns FFTW plans. The actual parallel implementation
is provided by the QGYBJMPIExt extension when PencilFFTs is loaded.

For true parallel FFTs, ensure MPI, PencilArrays, and PencilFFTs are loaded
before using QGYBJ, then use `plan_mpi_transforms()` from the extension.
"""
function setup_parallel_transforms(grid::Grid, pconfig)
    # This stub returns FFTW plans as fallback
    # The extension overrides this with PencilFFTs
    @warn "Parallel transforms requested but PencilFFTs extension not loaded. Falling back to FFTW."
    return Plans(backend=:fftw)
end

#=
================================================================================
                        FFT EXECUTION
================================================================================
=#

"""
    fft_forward!(dst, src, P::Plans)

Compute horizontal forward FFT (complex-to-complex) for each z-plane.

# Algorithm
Serial FFTW backend: Loops over z-slices and applies 2D FFT to each (x,y) plane.

# Arguments
- `dst`: Destination array (spectral space)
- `src`: Source array (physical space)
- `P::Plans`: FFT plans

# Returns
Modified dst array.

# Note
For parallel execution with PencilArrays, the extension module (QGYBJMPIExt)
provides a separate `fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)`
method that handles distributed transforms automatically.
"""
function fft_forward!(dst, src, P::Plans)
    # Serial FFTW path: transform each z-slice independently
    @inbounds for k in axes(src, 3)
        dst[:,:,k] .= FFTW.fft(src[:,:,k])
    end
    return dst
end

"""
    fft_backward!(dst, src, P::Plans)

Compute horizontal inverse FFT (complex-to-complex) for each z-plane.

# Algorithm
Serial FFTW backend: Loops over z-slices and applies 2D inverse FFT to each (x,y) plane.
FFTW.ifft is NORMALIZED (divides by N automatically).

# Arguments
- `dst`: Destination array (physical space, normalized)
- `src`: Source array (spectral space)
- `P::Plans`: FFT plans

# Returns
Modified dst array.

# Note
For parallel execution with PencilArrays, the extension module (QGYBJMPIExt)
provides a separate `fft_backward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)`
method that uses `ldiv!` for normalized inverse transforms.
"""
function fft_backward!(dst, src, P::Plans)
    # Serial FFTW path: transform each z-slice independently
    # FFTW.ifft is normalized (divides by nx*ny)
    @inbounds for k in axes(src, 3)
        dst[:,:,k] .= FFTW.ifft(src[:,:,k])
    end
    return dst
end

end # module Transforms

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
