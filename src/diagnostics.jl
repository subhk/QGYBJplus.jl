#=
================================================================================
                    diagnostics.jl - Energy and Field Diagnostics
================================================================================

This file provides diagnostic routines for analyzing QG-YBJ+ simulations,
including energy computations, the omega equation RHS, and field slicing.

ENERGY DIAGNOSTICS:
-------------------
Energy is a key diagnostic for verifying model behavior:

1. FLOW KINETIC ENERGY:
   KE = (1/2) ∫∫∫ (u² + v²) dx dy dz

   In QG, KE is related to enstrophy and streamfunction.
   Conservation/decay of KE indicates model stability.

2. WAVE ENERGY:
   WE_B = (1/2) ∫∫∫ |B|² dx dy dz   (envelope-based)
   WE_A = (1/2) ∫∫∫ |A|² dx dy dz   (amplitude-based)

   Wave energy transfer between scales indicates cascade direction.
   Energy exchange with mean flow shows wave-mean interaction.

OMEGA EQUATION RHS:
-------------------
The omega equation RHS drives ageostrophic vertical motion:

    ∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)

The RHS 2J(ψ_z, ∇²ψ) represents:
- Jacobian of vertical shear (thermal wind) and vorticity
- Physically: differential advection creating divergence
- Strong near fronts and eddy boundaries

FIELD SLICING:
--------------
Utility functions for extracting 2D slices from 3D spectral fields:
- slice_horizontal: x-y plane at fixed z (good for surface fields)
- slice_vertical_xz: x-z plane at fixed y (good for vertical structure)

FORTRAN CORRESPONDENCE:
-----------------------
- omega_eqn_rhs! → omega_eqn_rhs in diagnostics.f90
- wave_energy → energy diagnostics in diagnostics.f90
- flow_kinetic_energy → ke_flow in diagnostics.f90

================================================================================
=#

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
    flow_kinetic_energy(u, v) -> KE

Domain-sum kinetic energy 0.5(u^2+v^2) in nondimensional units.
"""
function flow_kinetic_energy(u, v)
    KE = 0.0
    @inbounds for i in eachindex(u)
        KE += 0.5 * (u[i]^2 + v[i]^2)
    end
    return KE
end

"""
    wave_energy_vavg(B, G, plans) -> WE_ave::Array{Float64,2}

Vertically averaged wave energy density 0.5|B|^2 in real space.
Returns an (nx,ny) array normalized by nz (matches we_vave).
"""
function wave_energy_vavg(B, G::Grid, plans)
    nx, ny, nz = G.nx, G.ny, G.nz
    # Build BRk, BIk and invert to real
    BRk = similar(B); BIk = similar(B)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        BRk[i,j,k] = Complex(real(B[i,j,k]), 0)
        BIk[i,j,k] = Complex(imag(B[i,j,k]), 0)
    end
    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)
    # Accumulate 0.5|B|^2 and normalize by nx*ny (IFFT unnormalized) and nz
    WE = zeros(Float64, nx, ny)
    norm = nx*ny
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        WE[i,j] += 0.5*((real(BRr[i,j,k])/norm)^2 + (real(BIr[i,j,k])/norm)^2)
    end
    WE ./= nz
    return WE
end

"""
    slice_horizontal(field, G, plans; k::Int) -> Array{Float64,2}

Return a horizontal x–y slice at vertical index k from a complex spectral field
by inverse FFT to real space.
"""
function slice_horizontal(field, G::Grid, plans; k::Int)
    nx, ny, nz = G.nx, G.ny, G.nz
    @assert 1 <= k <= nz
    # Inverse FFT entire field to get real slice
    Xr = similar(field)
    fft_backward!(Xr, field, plans)
    norm = nx*ny
    sl = Array{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        sl[i,j] = real(Xr[i,j,k]) / norm
    end
    return sl
end

"""
    slice_vertical_xz(field, G, plans; j::Int) -> Array{Float64,2}

Return an x–z slice at fixed y-index j from a complex spectral field by inverse
FFT to real space.
"""
function slice_vertical_xz(field, G::Grid, plans; j::Int)
    nx, ny, nz = G.nx, G.ny, G.nz
    @assert 1 <= j <= ny
    Xr = similar(field)
    fft_backward!(Xr, field, plans)
    norm = nx*ny
    sl = Array{Float64}(undef, nx, nz)
    @inbounds for k in 1:nz, i in 1:nx
        sl[i,k] = real(Xr[i,j,k]) / norm
    end
    return sl
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
