#=
================================================================================
                    nonlinear.jl - Nonlinear Tendency Terms
================================================================================

This file computes the nonlinear advection and interaction terms in the
QG-YBJ+ equations. These are the heart of the model's physics.

KEY PHYSICS:
------------
The nonlinear terms represent:

1. JACOBIAN ADVECTION: J(ψ, q) = ∂ψ/∂x ∂q/∂y - ∂ψ/∂y ∂q/∂x
   - Mean flow advects potential vorticity
   - Mean flow advects wave envelope B

2. REFRACTION: B × ζ
   - Waves are refracted by gradients in relative vorticity ζ = ∇²ψ
   - This causes wave focusing in anticyclones, defocusing in cyclones

3. WAVE FEEDBACK: qʷ = (i/2)J(B*, B) - (1/4)∇²|B|²
   - Waves can modify the mean flow through nonlinear wave-wave interactions
   - This is the Xie & Vanneste (2015) wave feedback term

4. HYPERDIFFUSION: -ν₁(-∇²)^n₁ - ν₂(-∇²)^n₂
   - Numerical dissipation for stability
   - Two operators allow selective damping at different scales

NUMERICAL METHOD:
-----------------
All nonlinear products are computed using the pseudo-spectral method:
1. Transform fields to real space (inverse FFT)
2. Compute products in real space (pointwise multiplication)
3. Transform result back to spectral space (forward FFT)
4. Apply 2/3 dealiasing mask to remove aliased modes

This is more efficient than computing convolutions directly in spectral space.

DEALIASING:
-----------
The 2/3 rule removes wavenumbers with |k| > 2/3 kmax to prevent aliasing
from quadratic nonlinearities. The Lmask array encodes which modes to keep.

FORTRAN CORRESPONDENCE:
----------------------
- convol_waqg!      ↔ convol_waqg (derivatives.f90)
- refraction_waqg!  ↔ refraction_waqg (derivatives.f90)
- compute_qw!       ↔ compute_qw (derivatives.f90)
- dissipation_q_nv! ↔ dissipation_q_nv (derivatives.f90)
- int_factor        ↔ integrating factor computation in main_waqg.f90

================================================================================
=#

module Nonlinear

using ..QGYBJ: Grid, local_to_global, get_local_dims
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!
using ..QGYBJ: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJ: local_to_global_z, allocate_z_pencil

# Reference to parent module for accessing is_dealiased
const PARENT = Base.parentmodule(@__MODULE__)

#=
================================================================================
                        JACOBIAN OPERATOR
================================================================================
The Jacobian J(φ, χ) = φₓχᵧ - φᵧχₓ represents advection of χ by the flow
derived from φ. In QG, φ = ψ (streamfunction) gives the geostrophic flow.

The Jacobian conserves both φ and χ integrals (energy and enstrophy).
================================================================================
=#

"""
    jacobian_spectral!(dstk, phik, chik, G, plans)

Compute the Jacobian J(φ, χ) = ∂φ/∂x ∂χ/∂y - ∂φ/∂y ∂χ/∂x using pseudo-spectral method.

# Mathematical Definition
The Jacobian (also called Poisson bracket) is:

    J(φ, χ) = ∂φ/∂x ∂χ/∂y - ∂φ/∂y ∂χ/∂x

In vector form: J(φ, χ) = ẑ · (∇φ × ∇χ)

# Physical Interpretation
- J(ψ, q): Advection of PV by geostrophic flow
- J(ψ, B): Advection of wave envelope by mean flow
- The Jacobian conserves both integrals ∫φ and ∫χ

# Algorithm
1. Compute spectral derivatives: φ̂ₓ = ikₓφ̂, φ̂ᵧ = ikᵧφ̂
2. Transform derivatives to real space
3. Compute product: J = φₓχᵧ - φᵧχₓ (pointwise)
4. Transform result back to spectral space

# Arguments
- `dstk`: Output array for Ĵ(φ, χ) in spectral space
- `phik`: φ̂ in spectral space
- `chik`: χ̂ in spectral space
- `G::Grid`: Grid with wavenumber arrays
- `plans`: FFT plans from plan_transforms!

# Note
Result is normalized by (nx × ny) to account for unnormalized inverse FFT.

# Example
```julia
# Compute J(ψ, q)
jacobian_spectral!(Jpsi_q, psi_k, q_k, grid, plans)
```
"""
function jacobian_spectral!(dstk, phik, chik, G::Grid, plans)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    phi_arr = parent(phik)
    chi_arr = parent(chik)
    dst_arr = parent(dstk)
    nx_local, ny_local, nz_local = size(phi_arr)

    #= Step 1: Compute spectral derivatives
    In spectral space: ∂/∂x → ikₓ, ∂/∂y → ikᵧ =#
    phixk = similar(phik); phiyk = similar(phik)
    chixk = similar(chik); chiyk = similar(chik)

    phix_arr = parent(phixk); phiy_arr = parent(phiyk)
    chix_arr = parent(chixk); chiy_arr = parent(chiyk)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]

        phix_arr[i_local, j_local, k] = im*kx_val*phi_arr[i_local, j_local, k]   # φ̂ₓ = ikₓ φ̂
        phiy_arr[i_local, j_local, k] = im*ky_val*phi_arr[i_local, j_local, k]   # φ̂ᵧ = ikᵧ φ̂
        chix_arr[i_local, j_local, k] = im*kx_val*chi_arr[i_local, j_local, k]   # χ̂ₓ = ikₓ χ̂
        chiy_arr[i_local, j_local, k] = im*ky_val*chi_arr[i_local, j_local, k]   # χ̂ᵧ = ikᵧ χ̂
    end

    #= Step 2: Transform derivatives to real space =#
    phix = similar(phik); phiy = similar(phik)
    chix = similar(chik); chiy = similar(chik)
    fft_backward!(phix, phixk, plans)
    fft_backward!(phiy, phiyk, plans)
    fft_backward!(chix, chixk, plans)
    fft_backward!(chiy, chiyk, plans)

    phix_r = parent(phix); phiy_r = parent(phiy)
    chix_r = parent(chix); chiy_r = parent(chiy)

    #= Step 3: Compute Jacobian in real space (pointwise multiplication)
    J = φₓχᵧ - φᵧχₓ =#
    J = similar(phik)
    J_arr = parent(J)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        J_arr[i_local, j_local, k] = (real(phix_r[i_local, j_local, k])*real(chiy_r[i_local, j_local, k]) -
                                      real(phiy_r[i_local, j_local, k])*real(chix_r[i_local, j_local, k]))
    end

    #= Step 4: Transform back to spectral space =#
    fft_forward!(dstk, J, plans)

    #= Normalization note:
    The pseudo-spectral convolution involves:
    - 4 normalized IFFTs (each divides by N internally)
    - Pointwise product in physical space
    - 1 unnormalized FFT (gives N × amplitude)

    For the correct spectral convolution amplitude, we divide by N once.
    This accounts for the FFT's unnormalized output and yields proper scaling
    for the tendency terms used in time-stepping. =#
    norm = nx*ny
    @inbounds dst_arr .= dst_arr ./ norm

    return dstk
end

#=
================================================================================
                    CONVOLUTION ADVECTION (convol_waqg)
================================================================================
This computes the advection terms J(ψ, q), J(ψ, BR), J(ψ, BI) using the
divergence form:

    J(ψ, q) = ∂(uq)/∂x + ∂(vq)/∂y = ikₓ(ûq) + ikᵧ(v̂q)

where u = -∂ψ/∂y, v = ∂ψ/∂x are the geostrophic velocities.

This form is used in the Fortran code for better conservation properties.
================================================================================
=#

"""
    convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G, plans; Lmask=nothing)

Compute advection terms in divergence form, matching Fortran `convol_waqg`.

# Mathematical Form
Uses the divergence form of the Jacobian:

    J(ψ, q) = ∂(uq)/∂x + ∂(vq)/∂y

where u, v are the geostrophic velocities (in real space).

# Output
- `nqk`:  Ĵ(ψ, q) - advection of QGPV
- `nBRk`: Ĵ(ψ, BR) - advection of wave real part
- `nBIk`: Ĵ(ψ, BI) - advection of wave imaginary part

# Arguments
- `nqk, nBRk, nBIk`: Output arrays (spectral)
- `u, v`: Real-space velocity arrays (precomputed)
- `qk, BRk, BIk`: Input fields (spectral)
- `G::Grid`: Grid struct
- `plans`: FFT plans
- `Lmask`: Dealiasing mask (true = keep mode, false = zero)

# Algorithm
For each field χ ∈ {q, BR, BI}:
1. Transform χ̂ → χ (inverse FFT)
2. Compute uχ and vχ (pointwise in real space)
3. Transform back: (ûχ), (v̂χ)
4. Compute divergence: ikₓ(ûχ) + ikᵧ(v̂χ)
5. Apply dealiasing mask

# Fortran Correspondence
This matches `convol_waqg` in derivatives.f90.

# Note
The velocities u, v should be precomputed and passed in real space.
"""
function convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    u_arr = parent(u); v_arr = parent(v)
    nqk_arr = parent(nqk); nBRk_arr = parent(nBRk); nBIk_arr = parent(nBIk)
    nx_local, ny_local, nz_local = size(u_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    # This avoids allocating a full (nx, ny) mask on each process
    use_inline_dealias = isnothing(Lmask)
    # Helper function: check if mode should be kept
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Transform input fields to real space =#
    qr  = similar(qk)
    BRr = similar(BRk)
    BIr = similar(BIk)
    fft_backward!(qr,  qk,  plans)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)

    qr_arr = parent(qr); BRr_arr = parent(BRr); BIr_arr = parent(BIr)

    #= ---- J(ψ, q): Advection of QGPV ---- =#
    # Compute products u*q and v*q in real space
    uterm = similar(qk); vterm = similar(qk)
    uterm_arr = parent(uterm); vterm_arr = parent(vterm)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        uterm_arr[i_local, j_local, k] = u_arr[i_local, j_local, k]*real(qr_arr[i_local, j_local, k])
        vterm_arr[i_local, j_local, k] = v_arr[i_local, j_local, k]*real(qr_arr[i_local, j_local, k])
    end

    # Transform to spectral and compute divergence
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    uterm_arr = parent(uterm); vterm_arr = parent(vterm)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        if should_keep(i_global, j_global)
            # J(ψ,q) = ∂(uq)/∂x + ∂(vq)/∂y = ikₓ(ûq) + ikᵧ(v̂q)
            nqk_arr[i_local, j_local, k] = im*kx_val*uterm_arr[i_local, j_local, k] + im*ky_val*vterm_arr[i_local, j_local, k]
        else
            nqk_arr[i_local, j_local, k] = 0  # Dealiased
        end
    end

    #= ---- J(ψ, BR): Advection of wave real part ---- =#
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        uterm_arr[i_local, j_local, k] = u_arr[i_local, j_local, k]*real(BRr_arr[i_local, j_local, k])
        vterm_arr[i_local, j_local, k] = v_arr[i_local, j_local, k]*real(BRr_arr[i_local, j_local, k])
    end
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    uterm_arr = parent(uterm); vterm_arr = parent(vterm)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        if should_keep(i_global, j_global)
            nBRk_arr[i_local, j_local, k] = im*kx_val*uterm_arr[i_local, j_local, k] + im*ky_val*vterm_arr[i_local, j_local, k]
        else
            nBRk_arr[i_local, j_local, k] = 0
        end
    end

    #= ---- J(ψ, BI): Advection of wave imaginary part ---- =#
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        uterm_arr[i_local, j_local, k] = u_arr[i_local, j_local, k]*real(BIr_arr[i_local, j_local, k])
        vterm_arr[i_local, j_local, k] = v_arr[i_local, j_local, k]*real(BIr_arr[i_local, j_local, k])
    end
    fft_forward!(uterm, uterm, plans)
    fft_forward!(vterm, vterm, plans)
    uterm_arr = parent(uterm); vterm_arr = parent(vterm)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        if should_keep(i_global, j_global)
            nBIk_arr[i_local, j_local, k] = im*kx_val*uterm_arr[i_local, j_local, k] + im*ky_val*vterm_arr[i_local, j_local, k]
        else
            nBIk_arr[i_local, j_local, k] = 0
        end
    end

    #= Normalize for unnormalized inverse FFT =#
    norm = nx*ny
    nqk_arr  ./= norm
    nBRk_arr ./= norm
    nBIk_arr ./= norm

    return nqk, nBRk, nBIk
end

#=
================================================================================
                        WAVE REFRACTION
================================================================================
Near-inertial waves are refracted by gradients in relative vorticity ζ = ∇²ψ.
This causes:
- Focusing of waves in anticyclones (ζ < 0)
- Defocusing in cyclones (ζ > 0)

The refraction term is: B × ζ (complex multiplication by real ζ)

In terms of real/imaginary parts:
- rBR = BR × ζ
- rBI = BI × ζ
================================================================================
=#

"""
    refraction_waqg!(rBRk, rBIk, BRk, BIk, psik, G, plans; Lmask=nothing)

Compute wave refraction term: B × ζ where ζ = ∇²ψ is relative vorticity.

# Physical Interpretation
Near-inertial waves are refracted by vorticity gradients:
- Anticyclones (ζ < 0): Wave focusing, amplitude increase
- Cyclones (ζ > 0): Wave defocusing, amplitude decrease

This is the "wave capture" mechanism that traps NIWs in anticyclonic eddies.

# Mathematical Form
    refraction = B × ζ

where ζ = ∇²ψ = -kₕ²ψ̂ in spectral space.

# Output
- `rBRk`: Real part of refraction term (spectral)
- `rBIk`: Imaginary part of refraction term (spectral)

# Algorithm
1. Compute ζ̂ = -kₕ²ψ̂ (spectral)
2. Transform ζ̂, B̂R, B̂I to real space
3. Compute products: rBR = ζ × BR, rBI = ζ × BI
4. Transform back and apply dealiasing

# Fortran Correspondence
This matches `refraction_waqg` in derivatives.f90.

# Example
```julia
refraction_waqg!(rBR, rBI, BR, BI, psi, grid, plans; Lmask=L)
# rBR, rBI now contain the refraction tendencies
```
"""
function refraction_waqg!(rBRk, rBIk, BRk, BIk, psik, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    psi_arr = parent(psik)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)
    nx_local, ny_local, nz_local = size(psi_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Compute relative vorticity ζ = ∇²ψ = -kₕ²ψ̂ =#
    zetak = similar(psik)
    zetak_arr = parent(zetak)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2
        zetak_arr[i_local, j_local, k] = -kh2*psi_arr[i_local, j_local, k]
    end

    #= Transform to real space =#
    zetar = similar(zetak)
    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(zetar, zetak, plans)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)

    zetar_arr = parent(zetar)
    BRr_arr = parent(BRr); BIr_arr = parent(BIr)

    #= Compute products in real space: rB = ζ × B =#
    rBRr = similar(BRr); rBIr = similar(BIr)
    rBRr_arr = parent(rBRr); rBIr_arr = parent(rBIr)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        rBRr_arr[i_local, j_local, k] = real(zetar_arr[i_local, j_local, k])*real(BRr_arr[i_local, j_local, k])
        rBIr_arr[i_local, j_local, k] = real(zetar_arr[i_local, j_local, k])*real(BIr_arr[i_local, j_local, k])
    end

    #= Transform back to spectral and apply dealiasing =#
    fft_forward!(rBRk, rBRr, plans)
    fft_forward!(rBIk, rBIr, plans)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)

    norm = nx*ny
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        if should_keep(i_global, j_global)
            rBRk_arr[i_local, j_local, k] /= norm
            rBIk_arr[i_local, j_local, k] /= norm
        else
            rBRk_arr[i_local, j_local, k] = 0  # Dealiased
            rBIk_arr[i_local, j_local, k] = 0
        end
    end

    return rBRk, rBIk
end

#=
================================================================================
                        WAVE FEEDBACK ON MEAN FLOW
================================================================================
Waves can modify the mean flow through the wave feedback term qʷ.
This represents the averaged effect of nonlinear wave-wave interactions
on the balanced flow (Xie & Vanneste 2015).

qʷ = W2F × [(i/2)J(B*, B) - (1/4)∇²|B|²]

where W2F = (Uw/U)² is the wave-to-flow velocity ratio squared.
================================================================================
=#

"""
    compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=nothing)

Compute wave feedback on mean flow: qʷ from wave field B.

# Physical Interpretation
The wave feedback qʷ represents how near-inertial waves modify the
quasi-geostrophic flow. This is a key component of wave-mean flow
interaction in the QG-YBJ+ model.

# Mathematical Form (Xie & Vanneste 2015)
    qʷ = W2F × [(i/2)J(B*, B) - (1/4)∇²|B|²]

where:
- B* is the complex conjugate of B
- J(B*, B) = B*ₓBᵧ - B*ᵧBₓ is the Jacobian
- |B|² = BR² + BI² is the wave energy density
- W2F = (Uw/U)² scales by wave-to-flow velocity ratio

# Decomposition
Let B = BR + i×BI. Then:
- J(B*, B) = 2(BRᵧBIₓ - BRₓBIᵧ) [imaginary-valued]
- ∇²|B|² = ∇²(BR² + BI²)

The final qʷ is real-valued after combining terms.

# Arguments
- `qwk`: Output array for q̂ʷ (spectral)
- `BRk, BIk`: Wave field components (spectral)
- `par`: QGParams (for W2F scaling)
- `G::Grid`: Grid struct
- `plans`: FFT plans
- `Lmask`: Dealiasing mask

# Fortran Correspondence
This matches `compute_qw` in derivatives.f90.

# Example
```julia
compute_qw!(qw, BR, BI, params, grid, plans; Lmask=L)
# qw now contains wave feedback term
```
"""
function compute_qw!(qwk, BRk, BIk, par, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    BRk_arr = parent(BRk); BIk_arr = parent(BIk)
    qwk_arr = parent(qwk)
    nx_local, ny_local, nz_local = size(BRk_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Compute derivatives of BR and BI =#
    BRxk = similar(BRk); BRyk = similar(BRk)
    BIxk = similar(BIk); BIyk = similar(BIk)
    BRxk_arr = parent(BRxk); BRyk_arr = parent(BRyk)
    BIxk_arr = parent(BIxk); BIyk_arr = parent(BIyk)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        BRxk_arr[i_local, j_local, k] = im*kx_val*BRk_arr[i_local, j_local, k]  # ∂BR/∂x
        BRyk_arr[i_local, j_local, k] = im*ky_val*BRk_arr[i_local, j_local, k]  # ∂BR/∂y
        BIxk_arr[i_local, j_local, k] = im*kx_val*BIk_arr[i_local, j_local, k]  # ∂BI/∂x
        BIyk_arr[i_local, j_local, k] = im*ky_val*BIk_arr[i_local, j_local, k]  # ∂BI/∂y
    end

    #= Transform derivatives to real space =#
    BRxr = similar(BRk); BRyr = similar(BRk)
    BIxr = similar(BIk); BIyr = similar(BIk)
    fft_backward!(BRxr, BRxk, plans)
    fft_backward!(BRyr, BRyk, plans)
    fft_backward!(BIxr, BIxk, plans)
    fft_backward!(BIyr, BIyk, plans)

    BRxr_arr = parent(BRxr); BRyr_arr = parent(BRyr)
    BIxr_arr = parent(BIxr); BIyr_arr = parent(BIyr)

    #= Compute (i/2)J(B*, B) term
    J(B*, B) = 2(BRᵧBIₓ - BRₓBIᵧ)
    So (i/2)J(B*, B) contributes: BRᵧBIₓ - BRₓBIᵧ =#
    qwr = similar(qwk)
    qwr_arr = parent(qwr)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        qwr_arr[i_local, j_local, k] = real(BRyr_arr[i_local, j_local, k])*real(BIxr_arr[i_local, j_local, k]) -
                                        real(BRxr_arr[i_local, j_local, k])*real(BIyr_arr[i_local, j_local, k])
    end

    #= Compute |B|² = BR² + BI² for the ∇²|B|² term =#
    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)

    BRr_arr = parent(BRr); BIr_arr = parent(BIr)
    mag2 = similar(BRk)
    mag2_arr = parent(mag2)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        mag2_arr[i_local, j_local, k] = real(BRr_arr[i_local, j_local, k])^2 + real(BIr_arr[i_local, j_local, k])^2
    end

    #= Transform |B|² to spectral for ∇² operation =#
    tempk = similar(BRk)
    fft_forward!(tempk, mag2, plans)
    tempk_arr = parent(tempk)

    #= Assemble qʷ in spectral space
    qʷ = J_term - (1/4)∇²|B|²
    where ∇² → -kₕ² in spectral space =#
    fft_forward!(qwk, qwr, plans)
    qwk_arr = parent(qwk)

    norm = nx*ny
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2
        if should_keep(i_global, j_global)
            qwk_arr[i_local, j_local, k] = (qwk_arr[i_local, j_local, k] - 0.25*kh2*tempk_arr[i_local, j_local, k]) / norm
        else
            qwk_arr[i_local, j_local, k] = 0
        end
        # Scale by Ro * W2F to match Fortran (derivatives.f90 line 1027)
        # W2F = (Uw/U)² (wave-to-flow velocity ratio)
        # Ro = U/(f*L) (Rossby number)
        qwk_arr[i_local, j_local, k] *= (par.Ro * par.W2F)
    end

    return qwk
end

#=
================================================================================
                        VERTICAL DIFFUSION
================================================================================
Vertical diffusion of q provides small-scale dissipation in the vertical.
This is usually small or zero in nondimensional units.

The operator is: νz ∂²q/∂z²

with Neumann boundary conditions (∂q/∂z = 0 at top/bottom).
================================================================================
=#

"""
    dissipation_q_nv!(dqk, qok, par, G)

Compute vertical diffusion of q with Neumann boundary conditions.

# Mathematical Form
    D = νz ∂²q/∂z²

with ∂q/∂z = 0 at z = 0 and z = H.

# Discretization
Interior points (1 < k < nz):
    D[k] = νz (q[k+1] - 2q[k] + q[k-1]) / dz²

Boundary points (Neumann):
    D[1]  = νz (q[2] - q[1]) / dz²
    D[nz] = νz (q[nz-1] - q[nz]) / dz²

# Arguments
- `dqk`: Output array for diffusion term
- `qok`: Input q field at time n-1 (for leapfrog)
- `par`: QGParams (for nuz coefficient)
- `G::Grid`: Grid struct

# Note
This operates on spectral q but the vertical derivative is in physical space,
so the operation is the same for each (kx, ky) mode.

# Fortran Correspondence
This matches `dissipation_q_nv` in derivatives.f90.
"""
function dissipation_q_nv!(dqk, qok, par, G::Grid; workspace=nothing)
    nz = G.nz

    # Check if we need 2D decomposition transpose
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

    if need_transpose
        _dissipation_q_nv_2d!(dqk, qok, par, G, workspace)
    else
        _dissipation_q_nv_direct!(dqk, qok, par, G)
    end

    return dqk
end

"""
Direct vertical diffusion for serial or 1D decomposition (z fully local).
"""
function _dissipation_q_nv_direct!(dqk, qok, par, G::Grid)
    nz = G.nz

    # Get underlying arrays
    dqk_arr = parent(dqk)
    qok_arr = parent(qok)
    nx_local, ny_local, nz_local = size(dqk_arr)

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local"

    # Vertical grid spacing
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    invdz2 = 1/(dz*dz)

    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        if k == 1
            # Bottom boundary: Neumann (q_z = 0)
            dqk_arr[i_local, j_local, k] = par.nuz * ( qok_arr[i_local, j_local, k+1] - qok_arr[i_local, j_local, k] ) * invdz2
        elseif k == nz
            # Top boundary: Neumann (q_z = 0)
            dqk_arr[i_local, j_local, k] = par.nuz * ( qok_arr[i_local, j_local, k-1] - qok_arr[i_local, j_local, k] ) * invdz2
        else
            # Interior: standard central difference
            dqk_arr[i_local, j_local, k] = par.nuz * ( qok_arr[i_local, j_local, k+1] - 2qok_arr[i_local, j_local, k] + qok_arr[i_local, j_local, k-1] ) * invdz2
        end
    end
end

"""
2D decomposition vertical diffusion with transposes.
"""
function _dissipation_q_nv_2d!(dqk, qok, par, G::Grid, workspace)
    nz = G.nz

    # Allocate z-pencil workspace
    qok_z = workspace !== nothing ? workspace.q_z : allocate_z_pencil(G, ComplexF64)
    dqk_z = workspace !== nothing ? workspace.work_z : allocate_z_pencil(G, ComplexF64)

    # Transpose input to z-pencil
    transpose_to_z_pencil!(qok_z, qok, G)

    # Get underlying arrays in z-pencil format
    qok_z_arr = parent(qok_z)
    dqk_z_arr = parent(dqk_z)
    nx_local, ny_local, nz_local = size(qok_z_arr)

    @assert nz_local == nz "After transpose, z must be fully local"

    # Vertical grid spacing
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    invdz2 = 1/(dz*dz)

    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        if k == 1
            dqk_z_arr[i_local, j_local, k] = par.nuz * ( qok_z_arr[i_local, j_local, k+1] - qok_z_arr[i_local, j_local, k] ) * invdz2
        elseif k == nz
            dqk_z_arr[i_local, j_local, k] = par.nuz * ( qok_z_arr[i_local, j_local, k-1] - qok_z_arr[i_local, j_local, k] ) * invdz2
        else
            dqk_z_arr[i_local, j_local, k] = par.nuz * ( qok_z_arr[i_local, j_local, k+1] - 2qok_z_arr[i_local, j_local, k] + qok_z_arr[i_local, j_local, k-1] ) * invdz2
        end
    end

    # Transpose output back to xy-pencil
    transpose_to_xy_pencil!(dqk, dqk_z, G)
end

#=
================================================================================
                        HYPERDIFFUSION (Integrating Factor)
================================================================================
Hyperdiffusion provides numerical stability by damping small-scale noise.
It uses higher powers of the Laplacian to be scale-selective.

The model uses TWO hyperdiffusion operators:
    Dissipation = -ν₁(-∇²)^n₁ - ν₂(-∇²)^n₂

Typical choices:
- n₁ = 2 (biharmonic): Damps intermediate scales
- n₂ = 6 (hyper-6): Sharp cutoff at grid scale

The integrating factor method incorporates hyperdiffusion exactly:
    q(n+1) = exp(-λ×dt) × [time-stepped q without diffusion]

where λ = ν₁kₕ^(2n₁) + ν₂kₕ^(2n₂)
================================================================================
=#

"""
    int_factor(kx, ky, par; waves=false)

Compute hyperdiffusion integrating factor for given wavenumber.

# Mathematical Background
The hyperdiffusion operator is:

    D = -ν₁(-∇²)^n₁ - ν₂(-∇²)^n₂

In spectral space, this becomes multiplication by:

    λ = ν₁|k|^(2n₁) + ν₂|k|^(2n₂)

The integrating factor for one time step is: exp(-λ×dt)

For efficiency, we return just λ×dt (the exponent).

# Arguments
- `kx, ky`: Horizontal wavenumber components
- `par`: QGParams (contains ν₁, ν₂, n₁, n₂)
- `waves::Bool`: If true, use wave hyperdiffusion (nuh1w, ilap1w, etc.)

# Returns
    λ×dt = dt × [ν₁(|kx|^(2n₁) + |ky|^(2n₁)) + ν₂(|kx|^(2n₂) + |ky|^(2n₂))]

# Usage in Time Stepping
```julia
# After computing tendency
factor = exp(-int_factor(kx, ky, par))
q_new = factor * q_tendency
```

# Fortran Correspondence
This matches the integrating factor computation in the main loop of main_waqg.f90.

# Example
```julia
# Get integrating factor for wavenumber (3, 4)
lambda_dt = int_factor(3.0, 4.0, params)
factor = exp(-lambda_dt)  # Multiply solution by this
```
"""
function int_factor(kx::Real, ky::Real, par; waves::Bool=false)
    if waves
        # Wave field hyperdiffusion (often smaller or zero)
        return par.dt * ( par.nuh1w*(abs(kx)^(2par.ilap1w) + abs(ky)^(2par.ilap1w)) +
                          par.nuh2w*(abs(kx)^(2par.ilap2w) + abs(ky)^(2par.ilap2w)) )
    else
        # Mean flow hyperdiffusion
        return par.dt * ( par.nuh1 *(abs(kx)^(2par.ilap1 ) + abs(ky)^(2par.ilap1 )) +
                          par.nuh2 *(abs(kx)^(2par.ilap2 ) + abs(ky)^(2par.ilap2 )) )
    end
end

end # module

# Export nonlinear operators to main QGYBJ module
using .Nonlinear: jacobian_spectral!, convol_waqg!, refraction_waqg!, compute_qw!, dissipation_q_nv!, int_factor
