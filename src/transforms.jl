"""
FFT planning and wrappers. Uses PencilFFTs when available, otherwise FFTW in serial.
Currently plans 2D FFTs along x,y for each z-slab, which matches the Fortran
pseudo-spectral horizontal approach.
"""
module Transforms

using ..QGYBJ: Grid
using LinearAlgebra

const HAS_PENCILFFTS = Base.find_package("PencilFFTs") !== nothing
function ensure_pencil()
    try
        @eval import PencilFFTs
        return PencilFFTs
    catch
        return nothing
    end
end
function ensure_fftw()
    try
        @eval import FFTW
        return FFTW
    catch
        return nothing
    end
end

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

Inverse 2D DFT fallback; unnormalized (match FFTW.ifft convention we assume
and divide by nx*ny outside).
"""
function _naive_ifft2!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T<:Complex}
    nx, ny = size(src)
    for y in 0:ny-1, x in 0:nx-1
        s = zero(T)
        for ky in 0:ny-1, kx in 0:nx-1
            angle = 2π * ( (kx*x)/nx + (ky*y)/ny )
            s += src[kx+1, ky+1] * cis(angle)
        end
        dst[x+1, y+1] = s
    end
    return dst
end

Base.@kwdef mutable struct Plans
    backend::Symbol                  # :pencil or :fftw
    # PencilFFTs plans
    p_forward::Any = nothing
    p_backward::Any = nothing
    # FFTW fallback plans
    f_forward::Any = nothing
    f_backward::Any = nothing
end

"""
    plan_transforms!(G::Grid, parallel_config=nothing)

Create forward/backward FFT plans appropriate to the environment.
Unified interface for both serial and parallel execution.
"""
function plan_transforms!(G::Grid, parallel_config=nothing)
    # If parallel_config is provided and MPI is active, use parallel planning
    if parallel_config !== nothing && parallel_config.use_mpi && G.decomp !== nothing
        return setup_parallel_transforms(G, parallel_config)
    end
    
    # Try PencilFFTs for serial case if decomp exists
    if HAS_PENCILFFTS && G.decomp !== nothing
        try
            PF = ensure_pencil()
            p = PF.plan_fft((G.nx, G.ny); dims=(1,2))
            ip = PF.plan_ifft((G.nx, G.ny); dims=(1,2))
            return Plans(backend=:pencil, p_forward=p, p_backward=ip)
        catch err
            @info "PencilFFTs planning fallback", err
        end
    end
    # Fallback to FFTW
    return Plans(backend=:fftw)
end

"""
    setup_parallel_transforms(grid::Grid, pconfig)

Set up FFT plans for parallel execution (called from unified interface).
"""
function setup_parallel_transforms(grid::Grid, pconfig)
    if grid.decomp !== nothing && HAS_PENCILFFTS
        try
            PF = ensure_pencil()
            # Create plans for the pencil decomposition
            # Transform in x and y dimensions (dims 1 and 2)
            forward_plan = PF.PencilFFTPlan(grid.decomp, Complex{Float64}; 
                                                   transform=(PF.Transforms.FFT(),
                                                            PF.Transforms.FFT()),
                                                   transform_dims=(1, 2))
            
            backward_plan = PF.PencilIFFTPlan(grid.decomp, Complex{Float64};
                                                     transform=(PF.Transforms.IFFT(),
                                                              PF.Transforms.IFFT()),
                                                     transform_dims=(1, 2))
            
            return Plans(backend=:pencil, p_forward=forward_plan, p_backward=backward_plan)
            
        catch e
            @warn "Failed to create PencilFFTs plans: $e"
        end
    end
    
    # Fallback to FFTW
    return Plans(backend=:fftw)
end

"""
    fft_forward!(dst, src, P)

Compute horizontal forward FFT (complex-to-complex) for each z-plane.
"""
function fft_forward!(dst, src, P::Plans)
    if P.backend === :pencil
        PF = ensure_pencil()
        PF.fft!(dst, src; plan=P.p_forward)
    else
        # src, dst: Array{Complex,3}; transform x,y per z index
        @inbounds for k in axes(src,3)
            FF = ensure_fftw()
            if FF === nothing
                _naive_fft2!(dst[:,:,k], src[:,:,k])
            else
                dst[:,:,k] .= Base.invokelatest(FF.fft, src[:,:,k])
            end
        end
    end
    return dst
end

"""
    fft_backward!(dst, src, P)

Compute horizontal inverse FFT (complex-to-complex) for each z-plane.
"""
function fft_backward!(dst, src, P::Plans)
    if P.backend === :pencil
        PF = ensure_pencil()
        PF.ifft!(dst, src; plan=P.p_backward)
    else
        @inbounds for k in axes(src,3)
            FF = ensure_fftw()
            if FF === nothing
                _naive_ifft2!(dst[:,:,k], src[:,:,k])
            else
                dst[:,:,k] .= Base.invokelatest(FF.ifft, src[:,:,k])
            end
        end
    end
    return dst
end

end # module

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
