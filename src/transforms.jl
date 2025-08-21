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
    plan_transforms!(G)

Create forward/backward FFT plans appropriate to the environment.
"""
function plan_transforms!(G::Grid)
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

using .Transforms: Plans, plan_transforms!, fft_forward!, fft_backward!
