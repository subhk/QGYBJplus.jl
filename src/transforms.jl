#=
================================================================================
                    transforms.jl - FFT Planning and Execution
================================================================================

This module provides FFT transforms for the QG-YBJ+ model. It uses FFTW.jl
for serial execution and supports MPI-parallel execution via the built-in
`mpi.jl` support with PencilFFTs.jl.

SERIAL MODE (default):
- Uses FFTW.jl for efficient FFT computation
- FFTW.jl is a required dependency

PARALLEL MODE:
- Uses PencilFFTs.jl for distributed FFTs
- Enabled when MPI, PencilArrays, PencilFFTs are loaded and a parallel config is passed
- See mpi.jl for MPI plan setup

TRANSFORM CONVENTION:
- Horizontal 2D FFTs (x,y dimensions) for each vertical level
- FFTW.ifft is NORMALIZED (divides by N = nx*ny internally)
- No manual normalization needed after fft_backward!
- Wavenumber layout follows FFTW convention (see grid.jl)

================================================================================
=#

module Transforms

using ..QGYBJplus: Grid
using LinearAlgebra
import FFTW

#=
================================================================================
                        FFT PLAN STRUCTURE
================================================================================
=#

"""
    Plans

Container for FFT plans. Used for serial FFTW execution.

For parallel execution with PencilFFTs, the MPI path
provides `MPIPlans` which wraps `PencilFFTPlan` and uses `ldiv!` for the
normalized inverse transform.

# Fields
- `backend::Symbol`: Always `:fftw` for this struct
- `fft_plan`: Cached in-place FFTW forward plan for a single (nx, ny) plane
- `ifft_plan`: Cached in-place FFTW normalized inverse plan for a single (nx, ny) plane
- `buf`: Contiguous (nx, ny) scratch buffer reused for every z-slice

# Current Implementation
`fft_forward!`/`fft_backward!` copy each strided `A[k,:,:]` plane into `buf`,
apply the cached in-place plan, and copy back. Planning once (in `Plans(G)`) and
reusing `buf` avoids allocating a fresh FFTW plan on every call and avoids running
the transform over non-unit-stride views.

# Note
When MPI/PencilArrays/PencilFFTs are loaded, use `plan_mpi_transforms()` instead,
which returns `MPIPlans`.
"""
mutable struct Plans{FP, IP, B}
    backend::Symbol                  # :fftw for serial mode
    fft_plan::FP                     # cached in-place 2D forward plan (per x-y plane)
    ifft_plan::IP                    # cached in-place 2D normalized inverse plan
    buf::B                           # contiguous (nx, ny) scratch reused for every z-slice
end

# Build cached FFTW plans for per-(x,y)-plane transforms. Planning once here and
# reusing one contiguous buffer avoids (1) allocating a fresh FFTW plan on every
# fft_forward!/fft_backward! call (~336 B/slice) and (2) transforming strided
# `A[k,:,:]` views (non-unit stride is cache-hostile and FFTW-unfriendly).
function Plans(G::Grid)
    buf = Array{ComplexF64}(undef, G.nx, G.ny)
    fft_plan = FFTW.plan_fft!(buf)
    ifft_plan = FFTW.plan_ifft!(buf)   # normalized inverse (matches FFTW.ifft!)
    return Plans(:fftw, fft_plan, ifft_plan, buf)
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
attempts to use PencilFFTs via the MPI support.

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
            # Parallel mode requested
            return setup_parallel_transforms(G, parallel_config)
        end
    end

    # Default: serial FFTW mode
    # Note: FFTW threading is NOT enabled by default because for small grids
    # thread overhead often exceeds the benefit.
    # Users can enable FFTW threading manually if needed for large grids
    return Plans(G)
end

"""
    setup_parallel_transforms(grid::Grid, pconfig) -> Plans

Set up FFT plans for parallel execution.

This delegates to `plan_mpi_transforms` from the MPI support when available,
and falls back to FFTW plans otherwise.
"""
function setup_parallel_transforms(grid::Grid, pconfig)
    PARENT = Base.parentmodule(@__MODULE__)
    if isdefined(PARENT, :plan_mpi_transforms)
        return PARENT.plan_mpi_transforms(grid, pconfig)
    end
    @warn "Parallel transforms requested but MPI plan setup not available. Falling back to FFTW."
    return Plans(grid)
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
For parallel execution with PencilArrays, the MPI support
provides a separate `fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)`
method that handles distributed transforms automatically.
"""
function fft_forward!(dst, src, P::Plans)
    # Serial FFTW path: transform each (x,y) plane independently for each z.
    # Copy each strided z-slice into the cached contiguous buffer, apply the
    # cached in-place plan, then copy back — no per-call plan allocation.
    if eltype(dst) <: Complex && eltype(src) <: Complex
        buf = P.buf
        @inbounds for k in axes(src, 1)
            @views copyto!(buf, src[k, :, :])
            P.fft_plan * buf
            @views copyto!(dst[k, :, :], buf)
        end
    else
        @inbounds for k in axes(src, 1)
            @views dst[k, :, :] .= FFTW.fft(src[k, :, :])
        end
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
For parallel execution with PencilArrays, the MPI support
provides a separate `fft_backward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)`
method that uses `ldiv!` for normalized inverse transforms.
"""
function fft_backward!(dst, src, P::Plans)
    # Serial FFTW path: transform each (x,y) plane independently for each z
    # FFTW.ifft is normalized (divides by nx*ny)
    if eltype(dst) <: Complex && eltype(src) <: Complex
        buf = P.buf
        @inbounds for k in axes(src, 1)
            @views copyto!(buf, src[k, :, :])
            P.ifft_plan * buf
            @views copyto!(dst[k, :, :], buf)
        end
    else
        @inbounds for k in axes(src, 1)
            @views dst[k, :, :] .= FFTW.ifft(src[k, :, :])
        end
    end
    return dst
end

end # module Transforms

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
