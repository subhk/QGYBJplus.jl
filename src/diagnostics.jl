"""
Diagnostics: omega_eqn_rhs! and simple wave energy measures.
"""

module Diagnostics

using ..QGYBJ: Grid
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!

"""
    omega_eqn_rhs!(rhs, psi, G, plans; Lmask)

Compute RHS of omega equation: 2 J(psi_z, ∇² psi) in spectral space.
Approximates stag/unstag averaging from Fortran by using centered psi_z and
averaged psi across adjacent levels.
"""
function omega_eqn_rhs!(rhs, psi, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    # psi_z in spectral space (simple finite difference)
    psizk = similar(psi)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if k == nz
            psizk[i,j,k] = 0  # Neumann top
        else
            psizk[i,j,k] = (psi[i,j,k+1] - psi[i,j,k]) / dz
        end
    end
    # Build needed spectral derivatives
    bxk = similar(psi); byk = similar(psi)
    xxk = similar(psi); xyk = similar(psi)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        bxk[i,j,k] = im*G.kx[i]*psizk[i,j,k]
        byk[i,j,k] = im*G.ky[j]*psizk[i,j,k]
        # average psi between k and k+1; at top use psi at top
        ψavg = k < nz ? 0.5*(psi[i,j,k+1] + psi[i,j,k]) : psi[i,j,k]
        xxk[i,j,k] = -im*G.kx[i]*kh2*ψavg
        xyk[i,j,k] = -im*G.ky[j]*kh2*ψavg
    end
    # To real space
    bxr = similar(psi); byr = similar(psi)
    xxr = similar(psi); xyr = similar(psi)
    fft_backward!(bxr, bxk, plans)
    fft_backward!(byr, byk, plans)
    fft_backward!(xxr, xxk, plans)
    fft_backward!(xyr, xyk, plans)
    # Real-space RHS
    rhsr = similar(psi)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        rhsr[i,j,k] = 2.0 * ( real(bxr[i,j,k])*real(xyr[i,j,k]) - real(byr[i,j,k])*real(xxr[i,j,k]) )
    end
    # Back to spectral
    fft_forward!(rhs, rhsr, plans)
    # Normalize and dealias
    norm = nx*ny
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            rhs[i,j,k] /= norm
        else
            rhs[i,j,k] = 0
        end
    end
    return rhs
end

"""
    wave_energy(B, A) -> (E_B, E_A)

Simple domain-sum energy-like diagnostics for |B|^2 and |A|^2.
"""
function wave_energy(B, A)
    EB = 0.0; EA = 0.0
    @inbounds for x in B; EB += abs2(x); end
    @inbounds for x in A; EA += abs2(x); end
    return EB, EA
end

end # module

using .Diagnostics: omega_eqn_rhs!, wave_energy

