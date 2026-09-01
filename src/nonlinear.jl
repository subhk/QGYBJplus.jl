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

3. WAVE FEEDBACK: qʷ = (i/2f₀)J(B*, B) + (1/4f₀)∇²|B|²
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
================================================================================
=#

module Nonlinear

using ..QGYBJplus: RuntimeGeometry, HorizontalHyperdiffusivity,
                   FlowHyperdiffusivity, WaveHyperdiffusivity,
                   local_to_global, z_is_local
using ..QGYBJplus: fft_forward!, fft_backward!
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: allocate_z_pencil
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
using ..QGYBJplus: with_scratch, scratch_like, scratch_physical, scratch_phys_like
using ..QGYBJplus: with_z_local, z_scratch
import PencilArrays: PencilArray

# Reference to parent module for accessing is_dealiased
const PARENT = Base.parentmodule(@__MODULE__)

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

# Prefilter spectral inputs to the 2/3 mask before nonlinear products.
function _prefilter_spectral!(dst, src, G::RuntimeGeometry, Lmask)
    nx, ny = G.nx, G.ny
    src_arr = parent(src)
    dst_arr = parent(dst)
    nz_local, nx_local, ny_local = size(src_arr)

    use_inline_dealias = isnothing(Lmask)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, src)
        j_global = local_to_global(j_local, 3, src)
        keep = use_inline_dealias ? PARENT.is_dealiased(i_global, j_global, nx, ny) : Lmask[i_global, j_global]
        dst_arr[k, i_local, j_local] = keep ? src_arr[k, i_local, j_local] : zero(eltype(dst_arr))
    end
    return dst
end

#=
================================================================================
                        JACOBIAN OPERATOR
================================================================================
The Jacobian J(φ, χ) = φₓχᵧ - φᵧχₓ represents advection of χ by the flow
derived from φ. In QG, φ = ψ (streamfunction) gives the geostrophic flow.

The Jacobian conserves both φ and χ integrals (energy and enstrophy).
================================================================================
=#

function _component_int_factor(kₓ::Real, kᵧ::Real, Δt::Real, closure)
    kₕ² = kₓ^2 + kᵧ^2
    return Δt * sum(
        coefficient * kₕ²^(order ÷ 2)
        for (coefficient, order) in zip(closure.coefficients, closure.orders)
    )
end

"""
    jacobian_spectral!(dstk, phik, chik, G, plans; Lmask=nothing)

Compute the Jacobian J(φ, χ) = ∂φ/∂x ∂χ/∂y - ∂φ/∂y ∂χ/∂x using pseudo-spectral method.

!!! note "Usage Note"
    This function is exported for user convenience but is **not used** in the main
    time-stepping code. The main code uses `convol_waqg!` instead, which computes
    advection terms using the divergence form with precomputed velocities.

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
2. Transform derivatives to physical space
3. Compute product: J = φₓχᵧ - φᵧχₓ (pointwise)
4. Transform result back to spectral space

# Arguments
- `dstk`: Output array for Ĵ(φ, χ) in spectral space
- `phik`: φ̂ in spectral space (must be real field, i.e., Hermitian symmetric)
- `chik`: χ̂ in spectral space (must be real field, i.e., Hermitian symmetric)
- `G::RuntimeGeometry`: RuntimeGeometry with wavenumber arrays
- `plans`: FFT plans from plan_transforms!
- `Lmask`: Optional 2/3 dealiasing mask (true = keep mode, false = zero)

# Important
This function assumes φ and χ are **real-valued fields** in physical space. For real
fields, IFFT of spectral derivatives (im*k*φ̂) yields real results (up to roundoff),
so the physical derivatives are extracted via `real()`.

# Example
```julia
# Compute J(ψ, q) for real fields ψ and q
jacobian_spectral!(Jpsi_q, psi_k, q_k, grid, plans)
```
"""
function jacobian_spectral!(dstk, φₖ, χₖ, G::RuntimeGeometry, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    φ_arr = parent(φₖ)
    χ_arr = parent(χₖ)
    dst_arr = parent(dstk)
    nz_local, nx_local, ny_local = size(φ_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Step 1: Compute spectral derivatives
    In spectral space: ∂/∂x → ikₓ, ∂/∂y → ikᵧ =#
    φₓₖ = similar(φₖ); φᵧₖ = similar(φₖ)
    χₓₖ = similar(χₖ); χᵧₖ = similar(χₖ)

    φₓ_arr = parent(φₓₖ); φᵧ_arr = parent(φᵧₖ)
    χₓ_arr = parent(χₓₖ); χᵧ_arr = parent(χᵧₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, φₖ)
        j_global = local_to_global(j_local, 3, φₖ)
    
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]

        if should_keep(i_global, j_global)
            φₓ_arr[k, i_local, j_local] = im*kₓ*φ_arr[k, i_local, j_local]   # φ̂ₓ = ikₓ φ̂
            φᵧ_arr[k, i_local, j_local] = im*kᵧ*φ_arr[k, i_local, j_local]   # φ̂ᵧ = ikᵧ φ̂
            χₓ_arr[k, i_local, j_local] = im*kₓ*χ_arr[k, i_local, j_local]   # χ̂ₓ = ikₓ χ̂
            χᵧ_arr[k, i_local, j_local] = im*kᵧ*χ_arr[k, i_local, j_local]   # χ̂ᵧ = ikᵧ χ̂
        else
            φₓ_arr[k, i_local, j_local] = 0
            φᵧ_arr[k, i_local, j_local] = 0
            χₓ_arr[k, i_local, j_local] = 0
            χᵧ_arr[k, i_local, j_local] = 0
        end
    end

    #= Step 2: Transform derivatives to real space =#
    φₓ = _allocate_fft_dst(φₓₖ, plans); φᵧ = _allocate_fft_dst(φᵧₖ, plans)
    χₓ = _allocate_fft_dst(χₓₖ, plans); χᵧ = _allocate_fft_dst(χᵧₖ, plans)

    fft_backward!(φₓ, φₓₖ, plans)
    fft_backward!(φᵧ, φᵧₖ, plans)
    fft_backward!(χₓ, χₓₖ, plans)
    fft_backward!(χᵧ, χᵧₖ, plans)

    φₓᵣ = parent(φₓ); φᵧᵣ = parent(φᵧ)
    χₓᵣ = parent(χₓ); χᵧᵣ = parent(χᵧ)

    #= Step 3: Compute Jacobian in physical space (pointwise multiplication)
    J = φₓχᵧ - φᵧχₓ

    For real fields: IFFT(im*k*φ̂) is real (up to roundoff), so we use real()
    to extract the physical derivative. =#
    Jᵣ = _allocate_fft_dst(φₖ, plans)
    J_arr = parent(Jᵣ)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(φₓᵣ)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        J_arr[k, i_local, j_local] = (real(φₓᵣ[k, i_local, j_local])*real(χᵧᵣ[k, i_local, j_local]) -
                                      real(φᵧᵣ[k, i_local, j_local])*real(χₓᵣ[k, i_local, j_local]))
    end

    #= Step 4: Transform back to spectral space and apply dealiasing =#
    fft_forward!(dstk, Jᵣ, plans)

    # Apply 2/3 dealiasing mask to remove aliased modes from quadratic nonlinearity
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, dstk)
        j_global = local_to_global(j_local, 3, dstk)
        if !should_keep(i_global, j_global)
            dst_arr[k, i_local, j_local] = 0  # Zero aliased modes
        end
    end

    #= Normalization note:
    The pseudo-spectral convolution involves:
    - 4 normalized IFFTs (each divides by N internally via FFTW.ifft)
    - Pointwise product in physical space
    - 1 FFT (FFTW.fft, which is already properly normalized in spectral convention)

    Since fft_backward! uses normalized IFFT (divides by N), the pseudo-spectral
    product is already correctly scaled. No additional normalization is needed.
    Previous code incorrectly divided by nx*ny, weakening nonlinear dynamics. =#

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
- `G::RuntimeGeometry`: RuntimeGeometry struct
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
function convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G::RuntimeGeometry, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    u_arr = parent(u); v_arr = parent(v)
    nqk_arr = parent(nqk); nBRk_arr = parent(nBRk); nBIk_arr = parent(nBIk)
    # Physical array dimensions (u, v are in physical space)
    nz_phys, nx_phys, ny_phys = size(u_arr)
    # Spectral array dimensions (may differ in 2D decomposition)
    nz_spec, nx_spec, ny_spec = size(nqk_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    # This avoids allocating a full (nx, ny) mask on each process
    use_inline_dealias = isnothing(Lmask)
    # Helper function: check if mode should be kept
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Transform input fields to real space =#
    qᵣ  = _allocate_fft_dst(qk, plans)
    BRᵣ = _allocate_fft_dst(BRk, plans)
    BIᵣ = _allocate_fft_dst(BIk, plans)

    qk_f  = similar(qk)
    BRk_f = similar(BRk)
    BIk_f = similar(BIk)
    _prefilter_spectral!(qk_f,  qk,  G, Lmask)
    _prefilter_spectral!(BRk_f, BRk, G, Lmask)
    _prefilter_spectral!(BIk_f, BIk, G, Lmask)

    fft_backward!(qᵣ,  qk_f,  plans)
    fft_backward!(BRᵣ, BRk_f, plans)
    fft_backward!(BIᵣ, BIk_f, plans)

    qᵣ_arr = parent(qᵣ); BRᵣ_arr = parent(BRᵣ); BIᵣ_arr = parent(BIᵣ)

    #= ---- J(ψ, q): Advection of QGPV ---- =#
    # Compute products u*q and v*q in real space (input pencil)
    uterm_r = _allocate_fft_dst(qk, plans)
    vterm_r = _allocate_fft_dst(qk, plans)
    uterm_r_arr = parent(uterm_r); vterm_r_arr = parent(vterm_r)
    uterm_k = similar(qk); vterm_k = similar(qk)

    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local]*real(qᵣ_arr[k, i_local, j_local])
        vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local]*real(qᵣ_arr[k, i_local, j_local])
    end

    # Transform to spectral and compute divergence
    fft_forward!(uterm_k, uterm_r, plans)
    fft_forward!(vterm_k, vterm_r, plans)

    uterm_arr = parent(uterm_k); vterm_arr = parent(vterm_k)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, uterm_k)
        j_global = local_to_global(j_local, 3, uterm_k)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        if should_keep(i_global, j_global)
            # J(ψ,q) = ∂(uq)/∂x + ∂(vq)/∂y = ikₓ(ûq) + ikᵧ(v̂q)
            nqk_arr[k, i_local, j_local] = im*kₓ*uterm_arr[k, i_local, j_local] + im*kᵧ*vterm_arr[k, i_local, j_local]
        else
            nqk_arr[k, i_local, j_local] = 0  # Dealiased
        end
    end

    #= ---- J(ψ, BR): Advection of wave real part ---- =#
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local]*real(BRᵣ_arr[k, i_local, j_local])
        vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local]*real(BRᵣ_arr[k, i_local, j_local])
    end
    fft_forward!(uterm_k, uterm_r, plans)
    fft_forward!(vterm_k, vterm_r, plans)

    uterm_arr = parent(uterm_k); vterm_arr = parent(vterm_k)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, uterm_k)
        j_global = local_to_global(j_local, 3, uterm_k)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        if should_keep(i_global, j_global)
            nBRk_arr[k, i_local, j_local] = im*kₓ*uterm_arr[k, i_local, j_local] + im*kᵧ*vterm_arr[k, i_local, j_local]
        else
            nBRk_arr[k, i_local, j_local] = 0
        end
    end

    #= ---- J(ψ, BI): Advection of wave imaginary part ---- =#
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local]*real(BIᵣ_arr[k, i_local, j_local])
        vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local]*real(BIᵣ_arr[k, i_local, j_local])
    end
    fft_forward!(uterm_k, uterm_r, plans)
    fft_forward!(vterm_k, vterm_r, plans)

    uterm_arr = parent(uterm_k); vterm_arr = parent(vterm_k)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, uterm_k)
        j_global = local_to_global(j_local, 3, uterm_k)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
    
        if should_keep(i_global, j_global)
            nBIk_arr[k, i_local, j_local] = im*kₓ*uterm_arr[k, i_local, j_local] + im*kᵧ*vterm_arr[k, i_local, j_local]
        else
            nBIk_arr[k, i_local, j_local] = 0
        end
    end

    #= No additional normalization needed:
    fft_backward! uses normalized IFFT (divides by N internally).
    Previous code incorrectly divided by nx*ny, weakening advection terms. =#

    return nqk, nBRk, nBIk
end

# Advection helper for complex fields (q or B) without splitting into BR/BI.
function _convol_advect!(nχk, u, v, χk, G::RuntimeGeometry, plans;
                         Lmask=nothing, use_real::Bool=false, workspace=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    u_arr = parent(u); v_arr = parent(v)
    nχk_arr = parent(nχk)
    # Physical array dimensions (u, v are in physical space)
    nz_phys, nx_phys, ny_phys = size(u_arr)
    # Spectral array dimensions (may differ in 2D decomposition)
    nz_spec, nx_spec, ny_spec = size(nχk_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    χᵣ = scratch_physical(workspace, χk, plans)
    χk_f = scratch_like(workspace, χk)
    _prefilter_spectral!(χk_f, χk, G, Lmask)
    fft_backward!(χᵣ, χk_f, plans)
    χᵣ_arr = parent(χᵣ)

    uterm_r = scratch_physical(workspace, χk, plans)
    vterm_r = scratch_physical(workspace, χk, plans)
    uterm_r_arr = parent(uterm_r); vterm_r_arr = parent(vterm_r)
    uterm_k = scratch_like(workspace, χk); vterm_k = scratch_like(workspace, χk)

    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        χval = use_real ? real(χᵣ_arr[k, i_local, j_local]) : χᵣ_arr[k, i_local, j_local]
        uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local] * χval
        vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local] * χval
    end

    fft_forward!(uterm_k, uterm_r, plans)
    fft_forward!(vterm_k, vterm_r, plans)

    uterm_arr = parent(uterm_k); vterm_arr = parent(vterm_k)
    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, uterm_k)
        j_global = local_to_global(j_local, 3, uterm_k)
        if should_keep(i_global, j_global)
            kₓ = G.kx[i_global]
            kᵧ = G.ky[j_global]
            nχk_arr[k, i_local, j_local] = im*kₓ*uterm_arr[k, i_local, j_local] + im*kᵧ*vterm_arr[k, i_local, j_local]
        else
            nχk_arr[k, i_local, j_local] = 0
        end
    end

    return nχk
end

"""
    convol_waqg_q!(nqk, u, v, qk, G, plans; Lmask=nothing)

Compute advection of q using divergence form without splitting wave fields.
"""
function convol_waqg_q!(nqk, u, v, qk, G::RuntimeGeometry, plans;
                        Lmask=nothing, workspace=nothing)
    return with_scratch(workspace) do
        _convol_advect!(nqk, u, v, qk, G, plans;
                        Lmask=Lmask, use_real=true, workspace)
    end
end

"""
    convol_waqg_B!(nBk, u, v, Bk, G, plans; Lmask=nothing)

Compute advection of complex B directly (YBJ+ path).
"""
function convol_waqg_B!(nBk, u, v, Bk, G::RuntimeGeometry, plans;
                        Lmask=nothing, workspace=nothing)
    return with_scratch(workspace) do
        _convol_advect!(nBk, u, v, Bk, G, plans;
                        Lmask=Lmask, use_real=false, workspace)
    end
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
function refraction_waqg!(rBRk, rBIk, BRk, BIk, ψₖ, G::RuntimeGeometry, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    ψ_arr = parent(ψₖ)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)
    # Spectral array dimensions
    nz_spec, nx_spec, ny_spec = size(ψ_arr)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    #= Compute relative vorticity ζ = ∇²ψ = -kₕ²ψ̂ =#
    ζₖ = scratch_like(workspace, ψₖ)
    ζₖ_arr = parent(ζₖ)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, ψₖ)
        j_global = local_to_global(j_local, 3, ψₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        if should_keep(i_global, j_global)
            ζₖ_arr[k, i_local, j_local] = -kₕ²*ψ_arr[k, i_local, j_local]
        else
            ζₖ_arr[k, i_local, j_local] = 0
        end
    end

    #= Transform to real space =#
    ζᵣ = _allocate_fft_dst(ζₖ, plans)
    BRᵣ = _allocate_fft_dst(BRk, plans); BIᵣ = _allocate_fft_dst(BIk, plans)

    BRk_f = similar(BRk)
    BIk_f = similar(BIk)
    _prefilter_spectral!(BRk_f, BRk, G, Lmask)
    _prefilter_spectral!(BIk_f, BIk, G, Lmask)

    fft_backward!(ζᵣ, ζₖ, plans)
    fft_backward!(BRᵣ, BRk_f, plans)
    fft_backward!(BIᵣ, BIk_f, plans)

    ζᵣ_arr = parent(ζᵣ)
    BRᵣ_arr = parent(BRᵣ); BIᵣ_arr = parent(BIᵣ)

    #= Compute products in real space: rB = ζ × B =#
    rBRᵣ = similar(BRᵣ); rBIᵣ = similar(BIᵣ)
    rBRᵣ_arr = parent(rBRᵣ); rBIᵣ_arr = parent(rBIᵣ)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(ζᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        rBRᵣ_arr[k, i_local, j_local] = real(ζᵣ_arr[k, i_local, j_local])*real(BRᵣ_arr[k, i_local, j_local])
        rBIᵣ_arr[k, i_local, j_local] = real(ζᵣ_arr[k, i_local, j_local])*real(BIᵣ_arr[k, i_local, j_local])
    end

    #= Transform back to spectral and apply dealiasing =#
    fft_forward!(rBRk, rBRᵣ, plans)
    fft_forward!(rBIk, rBIᵣ, plans)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)

    #= No additional normalization needed:
    fft_backward! uses normalized IFFT (divides by N internally).
    Previous code incorrectly divided by nx*ny, weakening refraction terms.
    Just apply dealiasing mask. =#

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, rBRk)
        j_global = local_to_global(j_local, 3, rBRk)
        if !should_keep(i_global, j_global)
            rBRk_arr[k, i_local, j_local] = 0  # Dealiased
            rBIk_arr[k, i_local, j_local] = 0
        end
    end

    return rBRk, rBIk
end

"""
    refraction_waqg_B!(rBk, Bk, ψₖ, G, plans; Lmask=nothing)

Compute wave refraction term ζ*B directly for complex B (YBJ+ path).
"""
function refraction_waqg_B!(rBk, Bk, ψₖ, G::RuntimeGeometry, plans;
                            Lmask=nothing, workspace=nothing)
    return with_scratch(workspace) do
        _refraction_waqg_B!(rBk, Bk, ψₖ, G, plans, Lmask, workspace)
    end
end

function _refraction_waqg_B!(rBk, Bk, ψₖ, G::RuntimeGeometry, plans, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz

    ψ_arr = parent(ψₖ)
    rBk_arr = parent(rBk)
    # Spectral array dimensions
    nz_spec, nx_spec, ny_spec = size(ψ_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    ζₖ = scratch_like(workspace, ψₖ)
    ζₖ_arr = parent(ζₖ)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, ψₖ)
        j_global = local_to_global(j_local, 3, ψₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        if should_keep(i_global, j_global)
            ζₖ_arr[k, i_local, j_local] = -kₕ²*ψ_arr[k, i_local, j_local]
        else
            ζₖ_arr[k, i_local, j_local] = 0
        end
    end

    ζᵣ = scratch_physical(workspace, ζₖ, plans)
    Bᵣ = scratch_physical(workspace, Bk, plans)
    Bk_f = scratch_like(workspace, Bk)
    _prefilter_spectral!(Bk_f, Bk, G, Lmask)
    fft_backward!(ζᵣ, ζₖ, plans)
    fft_backward!(Bᵣ, Bk_f, plans)

    ζᵣ_arr = parent(ζᵣ)
    Bᵣ_arr = parent(Bᵣ)

    rBᵣ = scratch_phys_like(workspace, Bᵣ)
    rBᵣ_arr = parent(rBᵣ)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(ζᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        rBᵣ_arr[k, i_local, j_local] = real(ζᵣ_arr[k, i_local, j_local]) * Bᵣ_arr[k, i_local, j_local]
    end

    fft_forward!(rBk, rBᵣ, plans)
    rBk_arr = parent(rBk)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, rBk)
        j_global = local_to_global(j_local, 3, rBk)
        if !should_keep(i_global, j_global)
            rBk_arr[k, i_local, j_local] = 0
        end
    end

    return rBk
end

#=
================================================================================
                        WAVE FEEDBACK ON MEAN FLOW
================================================================================
Waves can modify the mean flow through the wave feedback term qʷ.
This represents the averaged effect of nonlinear wave-wave interactions
on the balanced flow (Xie & Vanneste 2015).

For dimensional equations where B has actual velocity units:
    qʷ = (i/2f₀)J(B*, B) + (1/4f₀)∇²|B|²

The factor 1/f₀ converts the quadratic velocity-gradient terms to PV units.
================================================================================
=#

"""
    compute_qw!(qwk, BRk, BIk, G, plans; f, Lmask=nothing)

Compute wave feedback on mean flow: qʷ from wave field B.

# Physical Interpretation
The wave feedback qʷ represents how near-inertial waves modify the
quasi-geostrophic flow. This is a key component of wave-mean flow
interaction in the QG-YBJ+ model.

# Mathematical Form (Xie & Vanneste 2015)
For dimensional equations where B has velocity units [m/s]:

    qʷ = (i/2f₀)J(B*, B) + (1/4f₀)∇²|B|²

where:
- B* is the complex conjugate of B
- J(B*, B) = B*ₓBᵧ - B*ᵧBₓ is the Jacobian
- |B|² = BR² + BI² is the wave energy density

No separate wave-amplitude scaling is applied since B already has its actual
dimensional amplitude. The Coriolis normalization 1/f₀ remains required.

# Decomposition
Let B = BR + i×BI. Then:
- J(B*, B) = 2i(BRₓBIᵧ - BRᵧBIₓ) [purely imaginary]
- ∇²|B|² = ∇²(BR² + BI²)

The final qʷ is real-valued after combining terms.

# Arguments
- `qwk`: Output array for q̂ʷ (spectral)
- `BRk, BIk`: Wave field components (spectral)
- `G::RuntimeGeometry`: RuntimeGeometry struct
- `plans`: FFT plans
- `f`: Coriolis frequency f₀
- `Lmask`: Dealiasing mask

# Fortran Correspondence
This is the dimensional counterpart of `compute_qw` in derivatives.f90. The
legacy nondimensional amplitude factors are absent, while 1/f₀ is retained.

# Example
```julia
compute_qw!(qw, BR, BI, grid, plans; f=f₀, Lmask=L)
# qw now contains wave feedback term
```
"""
function compute_qw!(qʷₖ, BRk, BIk, G::RuntimeGeometry, plans;
                     f::Real, Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("wave feedback requires a finite, nonzero Coriolis frequency"))
    inv_f = inv(float(f))

    # Get underlying arrays
    qʷₖ_arr = parent(qʷₖ)

    # Dealiasing: use inline check for efficiency when Lmask not provided
    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    # Prefilter inputs to avoid aliasing when upstream fields are not masked
    BRk_f = similar(BRk)
    BIk_f = similar(BIk)
    _prefilter_spectral!(BRk_f, BRk, G, Lmask)
    _prefilter_spectral!(BIk_f, BIk, G, Lmask)

    BRk_arr = parent(BRk_f); BIk_arr = parent(BIk_f)
    nz_local, nx_local, ny_local = size(BRk_arr)

    #= Compute derivatives of BR and BI =#
    BRₓₖ = similar(BRk); BRᵧₖ = similar(BRk)
    BIₓₖ = similar(BIk); BIᵧₖ = similar(BIk)
    BRₓₖ_arr = parent(BRₓₖ); BRᵧₖ_arr = parent(BRᵧₖ)
    BIₓₖ_arr = parent(BIₓₖ); BIᵧₖ_arr = parent(BIᵧₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, BRk_f)
        j_global = local_to_global(j_local, 3, BRk_f)
      
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
      
        BRₓₖ_arr[k, i_local, j_local] = im*kₓ*BRk_arr[k, i_local, j_local]  # ∂BR/∂x
        BRᵧₖ_arr[k, i_local, j_local] = im*kᵧ*BRk_arr[k, i_local, j_local]  # ∂BR/∂y
        BIₓₖ_arr[k, i_local, j_local] = im*kₓ*BIk_arr[k, i_local, j_local]  # ∂BI/∂x
        BIᵧₖ_arr[k, i_local, j_local] = im*kᵧ*BIk_arr[k, i_local, j_local]  # ∂BI/∂y
    end

    #= Transform derivatives to real space =#
    BRₓᵣ = _allocate_fft_dst(BRₓₖ, plans); BRᵧᵣ = _allocate_fft_dst(BRᵧₖ, plans)
    BIₓᵣ = _allocate_fft_dst(BIₓₖ, plans); BIᵧᵣ = _allocate_fft_dst(BIᵧₖ, plans)
    fft_backward!(BRₓᵣ, BRₓₖ, plans)
    fft_backward!(BRᵧᵣ, BRᵧₖ, plans)
    fft_backward!(BIₓᵣ, BIₓₖ, plans)
    fft_backward!(BIᵧᵣ, BIᵧₖ, plans)

    BRₓᵣ_arr = parent(BRₓᵣ); BRᵧᵣ_arr = parent(BRᵧᵣ)
    BIₓᵣ_arr = parent(BIₓᵣ); BIᵧᵣ_arr = parent(BIᵧᵣ)

    #= Compute the unscaled (i/2)J(B*, B) term
    J(B*, B) = 2i(BRₓBIᵧ - BRᵧBIₓ)  [purely imaginary]
    So (i/2)J(B*, B) = i² × (BRₓBIᵧ - BRᵧBIₓ) = -(BRₓBIᵧ - BRᵧBIₓ) = BRᵧBIₓ - BRₓBIᵧ =#
    qʷᵣ = _allocate_fft_dst(qʷₖ, plans)
    qʷᵣ_arr = parent(qʷᵣ)
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(qʷᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        qʷᵣ_arr[k, i_local, j_local] = real(BRᵧᵣ_arr[k, i_local, j_local])*real(BIₓᵣ_arr[k, i_local, j_local]) -
                                        real(BRₓᵣ_arr[k, i_local, j_local])*real(BIᵧᵣ_arr[k, i_local, j_local])
    end

    #= Compute |B|² = BR² + BI² for the ∇²|B|² term =#
    BRᵣ = _allocate_fft_dst(BRk, plans); BIᵣ = _allocate_fft_dst(BIk, plans)
    fft_backward!(BRᵣ, BRk_f, plans)
    fft_backward!(BIᵣ, BIk_f, plans)

    BRᵣ_arr = parent(BRᵣ); BIᵣ_arr = parent(BIᵣ)
    mag² = _allocate_fft_dst(BRk, plans)
    mag²_arr = parent(mag²)

    # Physical array dimensions (already defined above as nz_phys, nx_phys, ny_phys)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        mag²_arr[k, i_local, j_local] = real(BRᵣ_arr[k, i_local, j_local])^2 + real(BIᵣ_arr[k, i_local, j_local])^2
    end

    #= Transform |B|² to spectral for ∇² operation =#
    tempₖ = similar(BRk)
    fft_forward!(tempₖ, mag², plans)
    tempₖ_arr = parent(tempₖ)

    #= Assemble qʷ in spectral space
    qʷ = (1/f₀) [J_term + (1/4)∇²|B|²]
    where ∇² → -kₕ² in spectral space =#
    fft_forward!(qʷₖ, qʷᵣ, plans)
    qʷₖ_arr = parent(qʷₖ)

    #= No additional normalization needed:
    fft_backward! uses normalized IFFT (divides by N internally).
    Previous code incorrectly divided by nx*ny, weakening wave feedback.
    Just combine terms and apply dealiasing. =#
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, qʷₖ)
        j_global = local_to_global(j_local, 3, qʷₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
      
        if should_keep(i_global, j_global)
            qʷₖ_arr[k, i_local, j_local] = inv_f *
                (qʷₖ_arr[k, i_local, j_local] -
                 0.25*kₕ²*tempₖ_arr[k, i_local, j_local])
        else
            qʷₖ_arr[k, i_local, j_local] = 0
        end
    end

    return qʷₖ
end

"""
    compute_qw_complex!(qʷₖ, Bk, G, plans; f, Lmask=nothing)

Compute wave feedback directly from complex B without spectral BR/BI splitting.

Implements the wave part of the "XV⁺" potential vorticity, Asselin & Young
(2019) equation (3.5):

    q = ΔΨ + LΨ + (i/2f) J(L⁺A*, L⁺A) + (1/4f) Δ|L⁺A|²

The argument is `B = L⁺A`, **not** the backrotated velocity `LA`. That is
deliberate, not an oversight: A&Y obtain (3.5) from Xie & Vanneste's PV by the
substitution L ↦ L⁺, and it is precisely that substitution which gives the
coupled system (1.4), (3.4), (3.5) its nonlinear "coupled energy" conservation
law, A&Y equations (3.6)–(3.7). Passing `LA` here would reproduce the original
XV potential vorticity and break that conservation law.

Note the deliberate asymmetry with the wave kinetic energy, which *does* use
`LA` — A&Y equation (4.7) and the remark following it: "to define WKE for YBJ⁺
we use L, not L⁺". See `_local_energy_components` in core/io.jl.
"""
function compute_qw_complex!(qʷₖ, Bk, G::RuntimeGeometry, plans;
                             f::Real, Lmask=nothing, workspace=nothing)
    return with_scratch(workspace) do
        _compute_qw_complex!(qʷₖ, Bk, G, plans, f, Lmask, workspace)
    end
end

function _compute_qw_complex!(qʷₖ, Bk, G::RuntimeGeometry, plans, f, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("wave feedback requires a finite, nonzero Coriolis frequency"))
    inv_f = inv(float(f))

    qʷₖ_arr = parent(qʷₖ)

    # Prefilter inputs to avoid aliasing when upstream fields are not masked
    Bk_f = scratch_like(workspace, Bk)
    _prefilter_spectral!(Bk_f, Bk, G, Lmask)

    Bk_arr = parent(Bk_f)
    nz_local, nx_local, ny_local = size(Bk_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    # Spectral derivatives of B
    Bₓₖ = scratch_like(workspace, Bk); Bᵧₖ = scratch_like(workspace, Bk)
    Bₓₖ_arr = parent(Bₓₖ); Bᵧₖ_arr = parent(Bᵧₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, Bk_f)
        j_global = local_to_global(j_local, 3, Bk_f)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        Bₓₖ_arr[k, i_local, j_local] = im*kₓ*Bk_arr[k, i_local, j_local]
        Bᵧₖ_arr[k, i_local, j_local] = im*kᵧ*Bk_arr[k, i_local, j_local]
    end

    # Transform to physical space
    Bᵣ = scratch_physical(workspace, Bk, plans)
    Bₓᵣ = scratch_physical(workspace, Bₓₖ, plans)
    Bᵧᵣ = scratch_physical(workspace, Bᵧₖ, plans)
    fft_backward!(Bᵣ, Bk_f, plans)
    fft_backward!(Bₓᵣ, Bₓₖ, plans)
    fft_backward!(Bᵧᵣ, Bᵧₖ, plans)

    Bᵣ_arr = parent(Bᵣ)
    Bₓᵣ_arr = parent(Bₓᵣ)
    Bᵧᵣ_arr = parent(Bᵧᵣ)

    # (i/2)J(B*, B) term in physical space
    qʷᵣ = scratch_phys_like(workspace, Bᵣ)
    qʷᵣ_arr = parent(qʷᵣ)
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(qʷᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        Jval = conj(Bₓᵣ_arr[k, i_local, j_local]) * Bᵧᵣ_arr[k, i_local, j_local] -
               conj(Bᵧᵣ_arr[k, i_local, j_local]) * Bₓᵣ_arr[k, i_local, j_local]
        qʷᵣ_arr[k, i_local, j_local] = real(0.5im * Jval)
    end

    # |B|^2 term
    mag² = scratch_physical(workspace, Bk, plans)
    mag²_arr = parent(mag²)
    # Physical array dimensions (already defined above as nz_phys, nx_phys, ny_phys)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        mag²_arr[k, i_local, j_local] = real(conj(Bᵣ_arr[k, i_local, j_local]) * Bᵣ_arr[k, i_local, j_local])
    end

    # Transform to spectral
    tempₖ = scratch_like(workspace, Bk)
    fft_forward!(tempₖ, mag², plans)
    fft_forward!(qʷₖ, qʷᵣ, plans)
    tempₖ_arr = parent(tempₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, qʷₖ)
        j_global = local_to_global(j_local, 3, qʷₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        if should_keep(i_global, j_global)
            qʷₖ_arr[k, i_local, j_local] = inv_f *
                (qʷₖ_arr[k, i_local, j_local] -
                 0.25*kₕ²*tempₖ_arr[k, i_local, j_local])
        else
            qʷₖ_arr[k, i_local, j_local] = 0
        end
    end

    return qʷₖ
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
    dissipation_q_nv!(dqk, qok, vertical_diffusivity, G; workspace=nothing)

Compute vertical diffusion of q with Neumann boundary conditions.

# Mathematical Form
    D = νz ∂²q/∂z²

with ∂q/∂z = 0 at z = -Lz and z = 0.

# Discretization
Interior points (1 < k < nz):
    D[k] = νz (q[k+1] - 2q[k] + q[k-1]) / dz²

Boundary points (Neumann):
    D[1]  = νz (q[2] - q[1]) / dz²
    D[nz] = νz (q[nz-1] - q[nz]) / dz²

# Arguments
- `dqk`: Output array for diffusion term
- `qok`: Input q field at the current Runge-Kutta stage
- `vertical_diffusivity`: Scalar vertical diffusivity coefficient
- `G::RuntimeGeometry`: RuntimeGeometry struct
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Note
This operates on spectral q but the vertical derivative is in physical space,
so the operation is the same for each (kx, ky) mode.

# Fortran Correspondence
This matches `dissipation_q_nv` in derivatives.f90.
"""
function dissipation_q_nv!(dqk, qok, vertical_diffusivity::Real, G::RuntimeGeometry;
    workspace=nothing)

    # A single layer has no vertical diffusion, and no transpose to arrange.
    if G.nz <= 1
        fill!(parent(dqk), zero(eltype(parent(dqk))))
        return dqk
    end

    with_z_local(G, (dqk, qok), (:out, :in);
                 scratch=z_scratch(workspace, :work_z, :q_z)) do dq_z, q_z
        _dissipation_q_nv!(dq_z, q_z, vertical_diffusivity, G)
    end
    return dqk
end

"""
Second-order vertical diffusion with Neumann (q_z = 0) top and bottom.
Requires a fully local vertical dimension, which `dissipation_q_nv!` arranges.
"""
function _dissipation_q_nv!(dqk, qok, vertical_diffusivity, G::RuntimeGeometry)
    nz = G.nz

    dqk_arr = parent(dqk)
    qok_arr = parent(qok)
    nz_local, nx_local, ny_local = size(dqk_arr)
    @assert nz_local == nz "Vertical dimension must be fully local"

    Δz = G.z[2] - G.z[1]
    Δz⁻² = 1/(Δz*Δz)
    νz = vertical_diffusivity

    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        if k == 1
            # Bottom boundary: Neumann (q_z = 0)
            dqk_arr[k, i_local, j_local] = νz * ( qok_arr[k+1, i_local, j_local] - qok_arr[k, i_local, j_local] ) * Δz⁻²
        elseif k == nz
            # Top boundary: Neumann (q_z = 0)
            dqk_arr[k, i_local, j_local] = νz * ( qok_arr[k-1, i_local, j_local] - qok_arr[k, i_local, j_local] ) * Δz⁻²
        else
            # Interior: standard central difference
            dqk_arr[k, i_local, j_local] = νz * ( qok_arr[k+1, i_local, j_local] - 2qok_arr[k, i_local, j_local] + qok_arr[k-1, i_local, j_local] ) * Δz⁻²
        end
    end
    return dqk
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
    int_factor(kx, ky, Δt, closure; waves=false, inviscid=false)

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
- `Δt`: Time-step length
- `closure`: a horizontal hyperdiffusivity component
- `inviscid`: Disable the integrating factor when true
- `waves::Bool`: If true, use wave hyperdiffusion (nuh1w, ilap1w, etc.)

# Returns
    λ×dt = dt × [ν₁(kx² + ky²)^n₁ + ν₂(kx² + ky²)^n₂] = dt × [ν₁ kₕ^(2n₁) + ν₂ kₕ^(2n₂)]

Note: Uses isotropic form `(kx² + ky²)^n` for proper damping of diagonal modes.

# Usage in Time Stepping
```julia
# After computing tendency
factor = exp(-int_factor(kx, ky, Δt, closure))
q_new = factor * q_tendency
```

# Fortran Correspondence
This matches the integrating factor computation in the main loop of main_waqg.f90.

# Example
```julia
# Get integrating factor for wavenumber (3, 4)
lambda_dt = int_factor(3.0, 4.0, Δt, closure)
factor = exp(-lambda_dt)  # Multiply solution by this
```
"""
function int_factor(kₓ::Real, kᵧ::Real, Δt::Real,
    closure::HorizontalHyperdiffusivity; waves::Bool=false,
    inviscid::Bool=false)

    inviscid && return 0.0
    component = waves ? closure.wave : closure.flow
    return _component_int_factor(kₓ, kᵧ, Δt, component)
end

function int_factor(kₓ::Real, kᵧ::Real, Δt::Real,
    closure::FlowHyperdiffusivity; waves::Bool=false,
    inviscid::Bool=false)

    (inviscid || waves) && return 0.0
    return _component_int_factor(kₓ, kᵧ, Δt, closure)
end

function int_factor(kₓ::Real, kᵧ::Real, Δt::Real,
    closure::WaveHyperdiffusivity; waves::Bool=false,
    inviscid::Bool=false)

    (inviscid || !waves) && return 0.0
    return _component_int_factor(kₓ, kᵧ, Δt, closure)
end

end # module

# Export nonlinear operators to main QGYBJplus module
using .Nonlinear: jacobian_spectral!, convol_waqg!, convol_waqg_q!, convol_waqg_B!,
                  refraction_waqg!, refraction_waqg_B!, compute_qw!, compute_qw_complex!,
                  dissipation_q_nv!, int_factor
