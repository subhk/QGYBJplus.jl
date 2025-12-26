#=
================================================================================
                    transforms.jl - FFT Planning and Execution
================================================================================

This module provides FFT transforms for the QG-YBJ+ model. It uses FFTW.jl
for serial execution and supports MPI-parallel execution via the built-in
`parallel_mpi.jl` support with PencilFFTs.jl.

SERIAL MODE (default):
- Uses FFTW.jl for efficient FFT computation
- FFTW.jl is a required dependency

PARALLEL MODE:
- Uses PencilFFTs.jl for distributed FFTs
- Enabled when MPI, PencilArrays, PencilFFTs are loaded and a parallel config is passed
- See parallel_mpi.jl for MPI plan setup

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
- `p_forward`: Reserved (not currently used)
- `p_backward`: Reserved (not currently used)
- `f_forward`: Reserved (not currently used)
- `f_backward`: Reserved (not currently used)

# Current Implementation
The current implementation uses FFTW.fft/ifft directly without pre-planning.
FFTW internally caches plans for repeated transforms of the same size/type,
so explicit pre-planning is not critical for performance. The reserved fields
are kept for potential future optimization with explicit FFTW plan objects.

# Note
When MPI/PencilArrays/PencilFFTs are loaded, use `plan_mpi_transforms()` instead,
which returns `MPIPlans`.
"""
Base.@kwdef mutable struct Plans
    backend::Symbol = :fftw          # :fftw for serial mode
    # Reserved fields for potential future FFTW plan caching
    # Currently unused - FFTW.fft/ifft are called directly
    p_forward::Any = nothing
    p_backward::Any = nothing
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
    # Note: We don't pre-plan here for simplicity. FFTW caches plans internally.
    return Plans(backend=:fftw)
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
For parallel execution with PencilArrays, the MPI support
provides a separate `fft_forward!(dst::PencilArray, src::PencilArray, plans::MPIPlans)`
method that handles distributed transforms automatically.
"""
function fft_forward!(dst, src, P::Plans)
    # Serial FFTW path: transform each (x,y) plane independently for each z
    @inbounds for k in axes(src, 1)
        dst[k, :, :] .= FFTW.fft(src[k, :, :])
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
    @inbounds for k in axes(src, 1)
        dst[k, :, :] .= FFTW.ifft(src[k, :, :])
    end
    return dst
end

end # module Transforms

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
