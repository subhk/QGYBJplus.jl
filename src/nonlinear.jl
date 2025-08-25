"""
Nonlinear and linear tendency computations: Jacobians, refraction placeholder,
and hyperdiffusion factors with 2/3 dealiasing support.
"""

module Nonlinear

using ..QGYBJ: Grid
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!

"""
    jacobian_spectral!(dstk, phik, chik, G, plans)

Compute J(phi, chi) = phi_x chi_y - phi_y chi_x using spectral derivatives and
2D transforms per z-slab. Writes result in spectral space dstk.
"""
function jacobian_spectral!(dstk, phik, chik, G::Grid, plans)
    nx, ny, nz = G.nx, G.ny, G.nz
    # spectral derivatives
    phixk = similar(phik); phiyk = similar(phik)
    chixk = similar(chik); chiyk = similar(chik)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        phixk[i,j,k] = im*G.kx[i]*phik[i,j,k]
        phiyk[i,j,k] = im*G.ky[j]*phik[i,j,k]
        chixk[i,j,k] = im*G.kx[i]*chik[i,j,k]
        chiyk[i,j,k] = im*G.ky[j]*chik[i,j,k]
    end
    # inverse to real space
    phix = similar(phik); phiy = similar(phik)
    chix = similar(chik); chiy = similar(chik)
    fft_backward!(phix, phixk, plans)
    fft_backward!(phiy, phiyk, plans)
    fft_backward!(chix, chixk, plans)
    fft_backward!(chiy, chiyk, plans)
    # form J in real space
    J = similar(phik)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        J[i,j,k] = (real(phix[i,j,k])*real(chiy[i,j,k]) - real(phiy[i,j,k])*real(chix[i,j,k]))
    end
    # forward to spectral
    fft_forward!(dstk, J, plans)
    # normalization of FFTs (inverse wasn’t normalized)
    norm = nx*ny
    @inbounds dstk .= dstk ./ norm
    return dstk
end

"""
    convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G, plans; Lmask=true)

Mirror of Fortran convol_waqg using divergence form: FFT of u*q and v*q.
Also computes J(ψ,BR) and J(ψ,BI) similarly using BRr/BIr.
"""
function convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    # Real-space fields
    qr  = similar(qk)
    BRr = similar(BRk)
    BIr = similar(BIk)
    fft_backward!(qr,  qk,  plans)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)
    # Products u*q and v*q
    uterm = similar(qk); vterm = similar(qk)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        uterm[i,j,k] = u[i,j,k]*real(qr[i,j,k])
        vterm[i,j,k] = v[i,j,k]*real(qr[i,j,k])
    end
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            nqk[i,j,k] = im*G.kx[i]*uterm[i,j,k] + im*G.ky[j]*vterm[i,j,k]
        else
            nqk[i,j,k] = 0
        end
    end
    # J(ψ,BR)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        uterm[i,j,k] = u[i,j,k]*real(BRr[i,j,k])
        vterm[i,j,k] = v[i,j,k]*real(BRr[i,j,k])
    end
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            nBRk[i,j,k] = im*G.kx[i]*uterm[i,j,k] + im*G.ky[j]*vterm[i,j,k]
        else
            nBRk[i,j,k] = 0
        end
    end
    # J(ψ,BI)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        uterm[i,j,k] = u[i,j,k]*real(BIr[i,j,k])
        vterm[i,j,k] = v[i,j,k]*real(BIr[i,j,k])
    end
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            nBIk[i,j,k] = im*G.kx[i]*uterm[i,j,k] + im*G.ky[j]*vterm[i,j,k]
        else
            nBIk[i,j,k] = 0
        end
    end
    # Normalize due to IFFT lack of scaling
    norm = nx*ny
    nqk  ./= norm
    nBRk ./= norm
    nBIk ./= norm
    return nqk, nBRk, nBIk
end

"""
    refraction_waqg!(rBRk, rBIk, BRk, BIk, psik, G, plans; Lmask=true)

Compute rB = B * ζ where ζ = -kh² ψ, in spectral space with dealiasing.
"""
function refraction_waqg!(rBRk, rBIk, BRk, BIk, psik, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    zetak = similar(psik)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        zetak[i,j,k] = -G.kh2[i,j]*psik[i,j,k]
    end
    zetar = similar(zetak)
    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(zetar, zetak, plans)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)
    rBRr = similar(BRr); rBIr = similar(BIr)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        rBRr[i,j,k] = real(zetar[i,j,k])*real(BRr[i,j,k])
        rBIr[i,j,k] = real(zetar[i,j,k])*real(BIr[i,j,k])
    end
    fft_forward!(rBRk, rBRr, plans)
    fft_forward!(rBIk, rBIr, plans)
    norm = nx*ny
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            rBRk[i,j,k] /= norm
            rBIk[i,j,k] /= norm
        else
            rBRk[i,j,k] = 0
            rBIk[i,j,k] = 0
        end
    end
    return rBRk, rBIk
end

"""
    compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=true)

Compute wave feedback q^w combining (i/2)J(B*,B) and −(1/4)∇²|B|², then scale by W2F.
"""
function compute_qw!(qwk, BRk, BIk, par, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    BRxk = similar(BRk); BRyk = similar(BRk)
    BIxk = similar(BIk); BIyk = similar(BIk)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        BRxk[i,j,k] = im*G.kx[i]*BRk[i,j,k]
        BRyk[i,j,k] = im*G.ky[j]*BRk[i,j,k]
        BIxk[i,j,k] = im*G.kx[i]*BIk[i,j,k]
        BIyk[i,j,k] = im*G.ky[j]*BIk[i,j,k]
    end
    BRxr = similar(BRk); BRyr = similar(BRk)
    BIxr = similar(BIk); BIyr = similar(BIk)
    fft_backward!(BRxr, BRxk, plans)
    fft_backward!(BRyr, BRyk, plans)
    fft_backward!(BIxr, BIxk, plans)
    fft_backward!(BIyr, BIyk, plans)
    qwr = similar(qwk)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        qwr[i,j,k] = real(BRyr[i,j,k])*real(BIxr[i,j,k]) - real(BRxr[i,j,k])*real(BIyr[i,j,k])
    end
    # |B|^2 term
    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)
    mag2 = similar(BRk)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        mag2[i,j,k] = real(BRr[i,j,k])^2 + real(BIr[i,j,k])^2
    end
    tempk = similar(BRk)
    fft_forward!(tempk, mag2, plans)
    # Assemble qwk in spectral space
    fft_forward!(qwk, qwr, plans)
    norm = nx*ny
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        if L[i,j]
            qwk[i,j,k] = (qwk[i,j,k] - 0.25*kh2*tempk[i,j,k]) / norm
        else
            qwk[i,j,k] = 0
        end
        qwk[i,j,k] *= par.W2F  # Ro normalization removed
    end
    return qwk
end

"""
    dissipation_q_nv!(dqk, qok, par, G)

Vertical diffusion of q at time n−1 with Neumann boundaries.
"""
function dissipation_q_nv!(dqk, qok, par, G::Grid)
    nx, ny, nz = G.nx, G.ny, G.nz
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    invdz2 = 1/(dz*dz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if k == 1
            dqk[i,j,k] = par.nuz * ( qok[i,j,k+1] - qok[i,j,k] ) * invdz2
        elseif k == nz
            dqk[i,j,k] = par.nuz * ( qok[i,j,k-1] - qok[i,j,k] ) * invdz2
        else
            dqk[i,j,k] = par.nuz * ( qok[i,j,k+1] - 2qok[i,j,k] + qok[i,j,k-1] ) * invdz2
        end
    end
    return dqk
end

"""
    int_factor(kx, ky, par; waves=false)

Compute horizontal hyperdiffusion integrating factor matching test1.
"""
function int_factor(kx::Real, ky::Real, par; waves::Bool=false)
    if waves
        return par.dt * ( par.nuh1w*(abs(kx)^(2par.ilap1w) + abs(ky)^(2par.ilap1w)) +
                          par.nuh2w*(abs(kx)^(2par.ilap2w) + abs(ky)^(2par.ilap2w)) )
    else
        return par.dt * ( par.nuh1 *(abs(kx)^(2par.ilap1 ) + abs(ky)^(2par.ilap1 )) +
                          par.nuh2 *(abs(kx)^(2par.ilap2 ) + abs(ky)^(2par.ilap2 )) )
    end
end

end # module

using .Nonlinear: jacobian_spectral!, convol_waqg!, refraction_waqg!, compute_qw!, dissipation_q_nv!, int_factor
