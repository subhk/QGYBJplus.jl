"""
FFT planning and wrappers. Uses PencilFFTs when available, otherwise FFTW in serial.
Currently plans 2D FFTs along x,y for each z-slab, which matches the Fortran
pseudo-spectral horizontal approach.
"""
module Transforms

using ..QGYBJ: Grid
using LinearAlgebra

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
    try
        import PencilFFTs
        if G.decomp !== nothing
            # Plan along (x,y). Different PencilFFTs versions may vary; try common API.
            try
                p = PencilFFTs.plan_fft((G.nx, G.ny); dims=(1,2))
                ip = PencilFFTs.plan_ifft((G.nx, G.ny); dims=(1,2))
                return Plans(backend=:pencil, p_forward=p, p_backward=ip)
            catch err
                @info "PencilFFTs planning fallback", err
            end
        end
    catch err
        @info "PencilFFTs not available; using FFTW", err
    end
    
    # Fallback to FFTW
    using FFTW
    return Plans(backend=:fftw)
end

"""
    setup_parallel_transforms(grid::Grid, pconfig)

Set up FFT plans for parallel execution (called from unified interface).
"""
function setup_parallel_transforms(grid::Grid, pconfig)
    if grid.decomp !== nothing
        try
            import PencilFFTs
            
            # Create plans for the pencil decomposition
            # Transform in x and y dimensions (dims 1 and 2)
            forward_plan = PencilFFTs.PencilFFTPlan(grid.decomp, Complex{Float64}; 
                                                   transform=(PencilFFTs.Transforms.FFT(),
                                                            PencilFFTs.Transforms.FFT()),
                                                   transform_dims=(1, 2))
            
            backward_plan = PencilFFTs.PencilIFFTPlan(grid.decomp, Complex{Float64};
                                                     transform=(PencilFFTs.Transforms.IFFT(),
                                                              PencilFFTs.Transforms.IFFT()),
                                                     transform_dims=(1, 2))
            
            return Plans(backend=:pencil, p_forward=forward_plan, p_backward=backward_plan)
            
        catch e
            @warn "Failed to create PencilFFTs plans: $e"
        end
    end
    
    # Fallback to FFTW
    using FFTW
    return Plans(backend=:fftw)
end

"""
    fft_forward!(dst, src, P)

Compute horizontal forward FFT (complex-to-complex) for each z-plane.
"""
function fft_forward!(dst, src, P::Plans)
    if P.backend === :pencil
        import PencilFFTs
        PencilFFTs.fft!(dst, src; plan=P.p_forward)
    else
        using FFTW
        # src, dst: Array{Complex,3}; transform x,y per z index
        @inbounds for k in axes(src,3)
            dst[:,:,k] .= fft(src[:,:,k])
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
        import PencilFFTs
        PencilFFTs.ifft!(dst, src; plan=P.p_backward)
    else
        using FFTW
        @inbounds for k in axes(src,3)
            dst[:,:,k] .= ifft(src[:,:,k])
        end
    end
    return dst
end

end # module

using .Transforms: Plans, plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!
