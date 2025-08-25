"""
Normal YBJ operators (non-plus branch): sigma, A via vertical integration, and sumB.
These mirror the Fortran routines compute_sigma, compute_A, and sumB.
"""

module YBJNormal

using ..QGYBJ: Grid, QGParams
using ..QGYBJ: N2_ut

"""
    sumB!(B, G; Lmask)

Subtract vertical mean of B at each horizontal wavenumber for kh>0 and
kept by dealias mask L.
"""
function sumB!(B::AbstractArray{Complex,3}, G::Grid; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    ave = zeros(ComplexF64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        if L[i,j] && G.kh2[i,j] > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += B[i,j,k]
            end
            aveij = s / nz
            ave[i,j] = aveij
            for k in 1:nz
                B[i,j,k] -= aveij
            end
        else
            for k in 1:nz; B[i,j,k] = 0; end
        end
    end
    return B
end

"""
    compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask) -> sigma

Compute vertical integral sigma(kx,ky) used by the normal YBJ method,
following Fortran: sigma = sum_z [ (rBRk + 2 nBIk)/kh^2 + i (rBIk - 2 nBRk)/kh^2 ].
"""
function compute_sigma(par::QGParams, G::Grid,
                       nBRk, nBIk, rBRk, rBIk; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    sigma = zeros(ComplexF64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        if L[i,j] && kh2 > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += ( rBRk[i,j,k] + 2*nBIk[i,j,k] + im*( rBIk[i,j,k] - 2*nBRk[i,j,k] ) )/kh2
            end
            sigma[i,j] = s  # Normalized (Bu*Ro = 1.0)
        else
            sigma[i,j] = 0
        end
    end
    return sigma
end

"""
    compute_A!(A, C, BRk, BIk, sigma, par, G; Lmask)

Normal YBJ recovery of A from B by vertical integration with r_2ut = N^2(z),
then enforcing vertical-mean constraint via sigma. Also computes C=A_z, with
Neumann C(top)=0.
"""
function compute_A!(A::AbstractArray{Complex,3}, C::AbstractArray{Complex,3},
                    BRk::AbstractArray{Complex,3}, BIk::AbstractArray{Complex,3},
                    sigma::AbstractArray{Complex,2}, par::QGParams, G::Grid; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    N2 = N2_ut(par, G)
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    # Temporary arrays per (i,j): AR(z), AI(z)
    @inbounds for j in 1:ny, i in 1:nx
        if L[i,j] && G.kh2[i,j] > 0
            # Stage 1: build \tilde{A} by cumulative vertical integration
            sBR = 0.0 + 0.0im
            sBI = 0.0 + 0.0im
            A[i,j,1] = 0
            for k in 2:nz
                sBR += BRk[i,j,k-1]
                sBI += BIk[i,j,k-1]
                A[i,j,k] = A[i,j,k-1] + ( sBR + im*sBI ) * N2[k-1]*dz*dz
            end
            # Stage 2: compute vertical sums sumAR/sumAI
            sumA = 0.0 + 0.0im
            for k in 1:nz
                sumA += A[i,j,k]
            end
            # Adjust to enforce mean(A) = sigma(i,j)/n3
            adj = (sigma[i,j] - sumA)/nz
            for k in 1:nz
                A[i,j,k] += adj
            end
            # C = A_z, forward diff; top C=0
            for k in 1:nz-1
                C[i,j,k] = (A[i,j,k+1] - A[i,j,k])/dz
            end
            C[i,j,nz] = 0
        else
            for k in 1:nz
                A[i,j,k] = 0; C[i,j,k] = 0
            end
        end
    end
    return A, C
end

end # module

using .YBJNormal: sumB!, compute_sigma, compute_A!

