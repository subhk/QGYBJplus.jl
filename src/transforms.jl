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

using ..QGYBJplus: RuntimeGeometry
using LinearAlgebra
import FFTW

#=
================================================================================
                        FFT PLAN STRUCTURE
================================================================================
=#

"""
    Plans

Cached FFTW plans for serial (non-distributed) execution.

Each horizontal plane is transformed in place through a single contiguous
`(nx, ny)` buffer, so a transform allocates nothing after construction.
Transforming a strided `A[k, :, :]` view directly, as an earlier version did,
allocated a fresh copy per plane per call.

`fft_backward!` uses FFTW's normalized inverse, so no manual scaling is needed.

Distributed runtimes use `plan_distributed_transforms()` instead, which returns
`MPIPlans`.
"""
struct Plans{F, I, B<:AbstractMatrix}
    backend::Symbol
    forward::F
    inverse::I
    buffer::B
end

"""Build serial FFTW plans for an `nx × ny` horizontal plane."""
function Plans(nx::Integer, ny::Integer, ::Type{T}=ComplexF64) where {T}
    buffer = Matrix{T}(undef, nx, ny)
    fill!(buffer, zero(T))
    # Note: FFTW threading is deliberately not enabled here. It conflicts with
    # Julia threading in the nonlinear kernels, and for the grids this serial
    # path serves the thread overhead exceeds the benefit.
    forward = FFTW.plan_fft!(buffer)
    inverse = FFTW.plan_ifft!(buffer)
    return Plans(:fftw, forward, inverse, buffer)
end

#=
================================================================================
                        FFT PLANNING
================================================================================
=#

"""
    plan_transforms!(G::RuntimeGeometry, parallel_config=nothing) -> Plans

Create forward/backward FFT plans appropriate to the environment.

# Serial Mode (default)
Returns Plans with `:fftw` backend for per-slice FFT execution.

# Parallel Mode
If `parallel_config` indicates MPI is active and the grid has decomposition,
attempts to use PencilFFTs via the MPI support.

# Arguments
- `G::RuntimeGeometry`: RuntimeGeometry structure (determines array sizes)
- `parallel_config`: Optional parallel configuration

# Returns
Plans struct with appropriate backend and plans.

# Example
```julia
G = init_grid(par)
plans = plan_transforms!(G)  # Serial FFTW
```
"""
function plan_transforms!(G::RuntimeGeometry, parallel_config=nothing)
    # If parallel_config indicates MPI is active, try parallel setup
    if parallel_config !== nothing
        if hasproperty(parallel_config, :use_mpi) && parallel_config.use_mpi && G.decomposition !== nothing
            # Parallel mode requested
            return setup_parallel_transforms(G, parallel_config)
        end
    end

    # Default: serial FFTW mode
    # Note: FFTW threading is NOT enabled by default because:
    # 1. It can conflict with Julia threading in nonlinear kernels
    # 2. For small grids, thread overhead exceeds benefit
    # Users can enable FFTW threading manually if needed for large grids
    return Plans(G.nx, G.ny)
end

"""
    setup_parallel_transforms(grid::RuntimeGeometry, pconfig) -> Plans

Set up FFT plans for parallel execution.

This delegates to the distributed planner from the MPI support when available,
and falls back to FFTW plans otherwise.
"""
function setup_parallel_transforms(grid::RuntimeGeometry, pconfig)
    PARENT = Base.parentmodule(@__MODULE__)
    if isdefined(PARENT, :plan_distributed_transforms)
        return PARENT.plan_distributed_transforms(grid, pconfig)
    end
    @warn "Parallel transforms requested but MPI plan setup not available. Falling back to FFTW."
    return Plans(grid.nx, grid.ny)
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
    # Serial FFTW path: transform each (x,y) plane independently for each z,
    # through the plan's contiguous buffer so nothing is allocated per plane.
    buffer = P.buffer
    @inbounds for k in axes(src, 1)
        copyto!(buffer, view(src, k, :, :))
        P.forward * buffer
        copyto!(view(dst, k, :, :), buffer)
    end
    return dst
end

function fft_forward!(dst, src, runtime)
    hasproperty(runtime, :plans) ||
        throw(ArgumentError("third argument must be transform plans or a model runtime"))
    return fft_forward!(dst, src, runtime.plans)
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
    # Serial FFTW path, normalized inverse (FFTW's ifft divides by nx*ny).
    buffer = P.buffer
    @inbounds for k in axes(src, 1)
        copyto!(buffer, view(src, k, :, :))
        P.inverse * buffer
        copyto!(view(dst, k, :, :), buffer)
    end
    return dst
end

function fft_backward!(dst, src, runtime)
    hasproperty(runtime, :plans) ||
        throw(ArgumentError("third argument must be transform plans or a model runtime"))
    return fft_backward!(dst, src, runtime.plans)
end

end # module Transforms

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
