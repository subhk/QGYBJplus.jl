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

3. WAVE FEEDBACK: qʷ = (i/2f)J(B*, B) + (1/4f)∇²|B|²  (PDF Eq. 8, 29, 47)
   - Waves can modify the mean flow through nonlinear wave-wave interactions
   - This is the Xie & Vanneste (2015) wave feedback term
   - Note: f is the Coriolis parameter in the denominator

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

using ..QGYBJplus: Grid, local_to_global, z_is_local
using ..QGYBJplus: fft_forward!, fft_backward!
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: allocate_z_pencil
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
import PencilArrays: PencilArray

# Reference to parent module for accessing is_dealiased
const PARENT = Base.parentmodule(@__MODULE__)

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

"""
    NonlinearWorkspace(spectral_template, plans)

Reusable scratch arrays for pseudo-spectral nonlinear products.

The spectral arrays use the same pencil as `spectral_template`; physical arrays
use the FFT backward destination pencil. The workspace is intentionally generic
because nonlinear kernels run sequentially and can reuse the same buffers for
advection, refraction, Jacobians, and wave-feedback terms.
"""
struct NonlinearWorkspace{A, P, R, C}
    spectral1::A
    spectral2::A
    spectral3::A
    spectral4::A
    spectral5::A
    spectral6::A
    physical1::P
    physical2::P
    physical3::P
    physical4::P
    physical5::P
    physical6::P
    vertical_d::Vector{R}
    vertical_dₗ::Vector{R}
    vertical_dᵤ::Vector{R}
    vertical_rhs::Vector{C}
    vertical_dₗ_work::Vector{R}
    vertical_d_work::Vector{R}
    vertical_dᵤ_work::Vector{R}
    vertical_rhsᵣ::Vector{R}
    vertical_rhsᵢ::Vector{R}
    vertical_solᵣ::Vector{R}
    vertical_solᵢ::Vector{R}
    N2_profile::Vector{R}
end

function NonlinearWorkspace(spectral_template, plans)
    spectral1 = similar(spectral_template)
    spectral2 = similar(spectral_template)
    spectral3 = similar(spectral_template)
    spectral4 = similar(spectral_template)
    spectral5 = similar(spectral_template)
    spectral6 = similar(spectral_template)

    physical1 = _allocate_fft_dst(spectral_template, plans)
    physical2 = _allocate_fft_dst(spectral_template, plans)
    physical3 = _allocate_fft_dst(spectral_template, plans)
    physical4 = _allocate_fft_dst(spectral_template, plans)
    physical5 = _allocate_fft_dst(spectral_template, plans)
    physical6 = _allocate_fft_dst(spectral_template, plans)

    C = eltype(spectral_template)
    R = typeof(real(zero(C)))
    nz = size(parent(spectral_template), 1)
    n_interior = max(nz - 2, 0)
    n_offdiag = max(n_interior - 1, 0)

    return NonlinearWorkspace(spectral1, spectral2, spectral3, spectral4, spectral5, spectral6,
                              physical1, physical2, physical3, physical4, physical5, physical6,
                              zeros(R, n_interior),
                              zeros(R, n_offdiag),
                              zeros(R, n_offdiag),
                              zeros(C, n_interior),
                              zeros(R, n_offdiag),
                              zeros(R, n_interior),
                              zeros(R, n_offdiag),
                              zeros(R, n_interior),
                              zeros(R, n_interior),
                              zeros(R, n_interior),
                              zeros(R, n_interior),
                              zeros(R, nz))
end

# Prefilter spectral inputs to the 2/3 mask before nonlinear products.
function _prefilter_spectral!(dst, src, G::Grid, Lmask)
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
- `G::Grid`: Grid with wavenumber arrays
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
function jacobian_spectral!(dstk, φₖ, χₖ, G::Grid, plans; Lmask=nothing, workspace=nothing)
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
    φₓₖ = workspace === nothing ? similar(φₖ) : workspace.spectral1
    φᵧₖ = workspace === nothing ? similar(φₖ) : workspace.spectral2
    χₓₖ = workspace === nothing ? similar(χₖ) : workspace.spectral3
    χᵧₖ = workspace === nothing ? similar(χₖ) : workspace.spectral4

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
    φₓ = workspace === nothing ? _allocate_fft_dst(φₓₖ, plans) : workspace.physical1
    φᵧ = workspace === nothing ? _allocate_fft_dst(φᵧₖ, plans) : workspace.physical2
    χₓ = workspace === nothing ? _allocate_fft_dst(χₓₖ, plans) : workspace.physical3
    χᵧ = workspace === nothing ? _allocate_fft_dst(χᵧₖ, plans) : workspace.physical4

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
    Jᵣ = workspace === nothing ? _allocate_fft_dst(φₖ, plans) : workspace.physical5
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

    uterm_r_arr = parent(uterm_r); 
    vterm_r_arr = parent(vterm_r)

    uterm_k = similar(qk); 
    vterm_k = similar(qk)

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
function _convol_advect!(nχk, u, v, χk, G::Grid, plans; Lmask=nothing,
                         use_real::Bool=false, workspace=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    u_arr = parent(u); v_arr = parent(v)
    nχk_arr = parent(nχk)

    # Physical array dimensions (u, v are in physical space)
    nz_phys, nx_phys, ny_phys = size(u_arr)

    # Spectral array dimensions (may differ in 2D decomposition)
    nz_spec, nx_spec, ny_spec = size(nχk_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    χᵣ = workspace === nothing ? _allocate_fft_dst(χk, plans) : workspace.physical1
    χk_f = workspace === nothing ? similar(χk) : workspace.spectral1

    _prefilter_spectral!(χk_f, χk, G, Lmask)
    fft_backward!(χᵣ, χk_f, plans)
    χᵣ_arr = parent(χᵣ)

    uterm_r = workspace === nothing ? _allocate_fft_dst(χk, plans) : workspace.physical2
    vterm_r = workspace === nothing ? _allocate_fft_dst(χk, plans) : workspace.physical3

    uterm_r_arr = parent(uterm_r) 
    vterm_r_arr = parent(vterm_r)

    uterm_k = workspace === nothing ? similar(χk) : workspace.spectral2
    vterm_k = workspace === nothing ? similar(χk) : workspace.spectral3

    # Hoist the `use_real` branch out of the loop so χval has a single concrete
    # type per loop body (Float64 for q advection, Complex for L⁺A). Inside the
    # loop the ternary made χval a Union{Float64,ComplexF64}, forcing a per-cell
    # type check on every grid point.
    if use_real
        @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
            χval = real(χᵣ_arr[k, i_local, j_local])
            uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local] * χval
            vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local] * χval
        end
    else
        @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
            χval = χᵣ_arr[k, i_local, j_local]
            uterm_r_arr[k, i_local, j_local] = u_arr[k, i_local, j_local] * χval
            vterm_r_arr[k, i_local, j_local] = v_arr[k, i_local, j_local] * χval
        end
    end

    fft_forward!(uterm_k, uterm_r, plans)
    fft_forward!(vterm_k, vterm_r, plans)

    uterm_arr = parent(uterm_k)
    vterm_arr = parent(vterm_k)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, uterm_k)
        j_global = local_to_global(j_local, 3, uterm_k)
        if should_keep(i_global, j_global)
            kₓ = G.kx[i_global]
            kᵧ = G.ky[j_global]
            nχk_arr[k, i_local, j_local] = im*kₓ*uterm_arr[k, i_local, j_local] + 
                                        im*kᵧ*vterm_arr[k, i_local, j_local]
        else
            nχk_arr[k, i_local, j_local] = 0.0
        end
    end

    return nχk
end

"""
    convol_waqg_q!(nqk, u, v, qk, G, plans; Lmask=nothing)

Compute advection of q using divergence form without splitting wave fields.
"""
function convol_waqg_q!(nqk, u, v, qk, G::Grid, plans; Lmask=nothing, workspace=nothing)
    return _convol_advect!(nqk, u, v, qk, G, plans; Lmask=Lmask,
                           use_real=true, workspace=workspace)
end

"""
    convol_waqg_L⁺A!(nL⁺Ak, u, v, L⁺Ak, G, plans; Lmask=nothing)

Compute advection of complex L⁺A directly (YBJ+ path).
"""
function convol_waqg_L⁺A!(nL⁺Ak, u, v, L⁺Ak, G::Grid, plans; Lmask=nothing, workspace=nothing)
    return _convol_advect!(nL⁺Ak, u, v, L⁺Ak, G, plans; Lmask=Lmask,
                           use_real=false, workspace=workspace)
end

#=
================================================================================
                        WAVE REFRACTION
================================================================================
Near-inertial waves are refracted by gradients in relative vorticity ζ = ∇²ψ.
This causes:
- Focusing of waves in anticyclones (ζ < 0)
- Defocusing in cyclones (ζ > 0)

The refraction term is: L⁺A × ζ (complex multiplication by real ζ)

In terms of real/imaginary parts:
- rL⁺AR = L⁺AR × ζ
- rL⁺AI = L⁺AI × ζ
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
function refraction_waqg!(rBRk, rBIk, BRk, BIk, ψₖ, G::Grid, plans; Lmask=nothing)
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
    ζₖ = similar(ψₖ)
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
            ζₖ_arr[k, i_local, j_local] = 0.0
        end
    end

    #= Transform to real space =#
    ζᵣ  = _allocate_fft_dst(ζₖ, plans)

    BRᵣ = _allocate_fft_dst(BRk, plans); 
    BIᵣ = _allocate_fft_dst(BIk, plans)

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
    rBRᵣ = similar(BRᵣ); 
    rBIᵣ = similar(BIᵣ)
    
    rBRᵣ_arr = parent(rBRᵣ); 
    rBIᵣ_arr = parent(rBIᵣ)

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
            rBRk_arr[k, i_local, j_local] = 0.0  # Dealiased
            rBIk_arr[k, i_local, j_local] = 0.0
        end
    end

    return rBRk, rBIk
end

"""
    refraction_waqg_L⁺A!(rL⁺Ak, L⁺Ak, ψₖ, G, plans; Lmask=nothing)

Compute wave refraction term ζ*L⁺A directly for complex L⁺A (YBJ+ path).
"""
function refraction_waqg_L⁺A!(rL⁺Ak, L⁺Ak, ψₖ, G::Grid, plans; Lmask=nothing, workspace=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    ψ_arr = parent(ψₖ)
    rL⁺Ak_arr = parent(rL⁺Ak)
    # Spectral array dimensions
    nz_spec, nx_spec, ny_spec = size(ψ_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    ζₖ = workspace === nothing ? similar(ψₖ) : workspace.spectral1
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
            ζₖ_arr[k, i_local, j_local] = 0.0
        end
    end

    ζᵣ = workspace === nothing ? _allocate_fft_dst(ζₖ, plans) : workspace.physical1
    L⁺Aᵣ = workspace === nothing ? _allocate_fft_dst(L⁺Ak, plans) : workspace.physical2
    L⁺Ak_f = workspace === nothing ? similar(L⁺Ak) : workspace.spectral2
    _prefilter_spectral!(L⁺Ak_f, L⁺Ak, G, Lmask)
    fft_backward!(ζᵣ, ζₖ, plans)
    fft_backward!(L⁺Aᵣ, L⁺Ak_f, plans)

    ζᵣ_arr = parent(ζᵣ)
    L⁺Aᵣ_arr = parent(L⁺Aᵣ)

    rL⁺Aᵣ = workspace === nothing ? similar(L⁺Aᵣ) : workspace.physical3
    rL⁺Aᵣ_arr = parent(rL⁺Aᵣ)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(ζᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        rL⁺Aᵣ_arr[k, i_local, j_local] = real(ζᵣ_arr[k, i_local, j_local]) * L⁺Aᵣ_arr[k, i_local, j_local]
    end

    fft_forward!(rL⁺Ak, rL⁺Aᵣ, plans)
    rL⁺Ak_arr = parent(rL⁺Ak)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, rL⁺Ak)
        j_global = local_to_global(j_local, 3, rL⁺Ak)
        if !should_keep(i_global, j_global)
            rL⁺Ak_arr[k, i_local, j_local] = 0.0  # Dealiased
        end
    end

    return rL⁺Ak
end

#=
================================================================================
                        WAVE FEEDBACK ON MEAN FLOW
================================================================================
Waves can modify the mean flow through the wave feedback term qʷ.
This represents the averaged effect of nonlinear wave-wave interactions
on the balanced flow (Xie & Vanneste 2015).

For dimensional equations where L⁺A has actual velocity units:
    qʷ = (i/2f)J(L⁺A*, L⁺A) + (1/4f)∇²|L⁺A|²

No additional scaling is needed since L⁺A already contains the wave amplitude.
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
For dimensional equations where B has velocity units [m/s]:

    qʷ = (i/2f)J(B*, B) + (1/4f)∇²|B|²

where:
- B* is the complex conjugate of B
- J(B*, B) = B*ₓBᵧ - B*ᵧBₓ is the Jacobian
- |B|² = BR² + BI² is the wave energy density

No W2F scaling is applied since B already has its actual dimensional amplitude.

# Decomposition
Let B = BR + i×BI. Then:
- J(B*, B) = 2i(BRₓBIᵧ - BRᵧBIₓ) [purely imaginary]
- ∇²|B|² = ∇²(BR² + BI²)

The final qʷ is real-valued after combining terms.

# Arguments
- `qwk`: Output array for q̂ʷ (spectral)
- `BRk, BIk`: Wave field components (spectral)
- `par`: QGParams
- `G::Grid`: Grid struct
- `plans`: FFT plans
- `Lmask`: Dealiasing mask (true = keep mode, false = zero)

# Example
```julia
compute_qw!(qw, BR, BI, params, grid, plans; Lmask=L)
# qw now contains wave feedback term
```
"""
function compute_qw!(qʷₖ, BRk, BIk, par, G::Grid, plans; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

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
    BRₓᵣ = _allocate_fft_dst(BRₓₖ, plans); 
    BRᵧᵣ = _allocate_fft_dst(BRᵧₖ, plans)

    BIₓᵣ = _allocate_fft_dst(BIₓₖ, plans); 
    BIᵧᵣ = _allocate_fft_dst(BIᵧₖ, plans)

    fft_backward!(BRₓᵣ, BRₓₖ, plans)
    fft_backward!(BRᵧᵣ, BRᵧₖ, plans)
    fft_backward!(BIₓᵣ, BIₓₖ, plans)
    fft_backward!(BIᵧᵣ, BIᵧₖ, plans)

    BRₓᵣ_arr = parent(BRₓᵣ); BRᵧᵣ_arr = parent(BRᵧᵣ)
    BIₓᵣ_arr = parent(BIₓᵣ); BIᵧᵣ_arr = parent(BIᵧᵣ)

    #= Compute (i/2f)J(B*, B) term (PDF Eq. 47)
    J(B*, B) = 2i(BRₓBIᵧ - BRᵧBIₓ)  [purely imaginary]
    So (i/2f)J(B*, B) = (i²/f)(BRₓBIᵧ - BRᵧBIₓ) = (1/f)(BRᵧBIₓ - BRₓBIᵧ) =#
    f₀ = par.f₀
    qʷᵣ = _allocate_fft_dst(qʷₖ, plans)
    qʷᵣ_arr = parent(qʷᵣ)
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(qʷᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        qʷᵣ_arr[k, i_local, j_local] = (real(BRᵧᵣ_arr[k, i_local, j_local])*real(BIₓᵣ_arr[k, i_local, j_local]) -
                                        real(BRₓᵣ_arr[k, i_local, j_local])*real(BIᵧᵣ_arr[k, i_local, j_local])) / f₀
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
    qʷ = J_term + (1/4f)∇²|B|²
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
            # qʷ = (i/2f)J(B*, B) - (k²/4f)|B|² (PDF Eq. 29, 47)
            # ∇² → -kₕ² in spectral space, so (1/4f)∇²|B|² → -(kₕ²/4f)|B|²
            qʷₖ_arr[k, i_local, j_local] = qʷₖ_arr[k, i_local, j_local] - (0.25/f₀)*kₕ²*tempₖ_arr[k, i_local, j_local]
        else
            qʷₖ_arr[k, i_local, j_local] = 0
        end
    end

    return qʷₖ
end

"""
    compute_qw_complex!(qʷₖ, Bk, par, G, plans; Lmask=nothing)

Compute wave feedback directly from complex B without spectral BR/BI splitting.
"""
function compute_qw_complex!(qʷₖ, Bk, par, G::Grid, plans; Lmask=nothing, workspace=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    qʷₖ_arr = parent(qʷₖ)

    # Prefilter inputs to avoid aliasing when upstream fields are not masked
    Bk_f = workspace === nothing ? similar(Bk) : workspace.spectral1
    _prefilter_spectral!(Bk_f, Bk, G, Lmask)

    Bk_arr = parent(Bk_f)
    nz_local, nx_local, ny_local = size(Bk_arr)

    use_inline_dealias = isnothing(Lmask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? PARENT.is_dealiased(i_g, j_g, nx, ny) : Lmask[i_g, j_g]

    # Spectral derivatives of B
    Bₓₖ = workspace === nothing ? similar(Bk) : workspace.spectral2
    Bᵧₖ = workspace === nothing ? similar(Bk) : workspace.spectral3
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
    Bᵣ = workspace === nothing ? _allocate_fft_dst(Bk, plans) : workspace.physical1
    Bₓᵣ = workspace === nothing ? _allocate_fft_dst(Bₓₖ, plans) : workspace.physical2
    Bᵧᵣ = workspace === nothing ? _allocate_fft_dst(Bᵧₖ, plans) : workspace.physical3
    fft_backward!(Bᵣ, Bk_f, plans)
    fft_backward!(Bₓᵣ, Bₓₖ, plans)
    fft_backward!(Bᵧᵣ, Bᵧₖ, plans)

    Bᵣ_arr = parent(Bᵣ)
    Bₓᵣ_arr = parent(Bₓᵣ)
    Bᵧᵣ_arr = parent(Bᵧᵣ)

    # (i/2f)J(B*, B) term in physical space (PDF Eq. 29, 47)
    f₀ = par.f₀
    qʷᵣ = workspace === nothing ? similar(Bᵣ) : workspace.physical4
    qʷᵣ_arr = parent(qʷᵣ)
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(qʷᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        Jval = conj(Bₓᵣ_arr[k, i_local, j_local]) * Bᵧᵣ_arr[k, i_local, j_local] -
               conj(Bᵧᵣ_arr[k, i_local, j_local]) * Bₓᵣ_arr[k, i_local, j_local]
        qʷᵣ_arr[k, i_local, j_local] = real(0.5im * Jval) / f₀
    end

    # |B|^2 term
    mag² = workspace === nothing ? _allocate_fft_dst(Bk, plans) : workspace.physical5
    mag²_arr = parent(mag²)
    # Physical array dimensions (already defined above as nz_phys, nx_phys, ny_phys)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        mag²_arr[k, i_local, j_local] = real(conj(Bᵣ_arr[k, i_local, j_local]) * Bᵣ_arr[k, i_local, j_local])
    end

    # Transform to spectral
    tempₖ = workspace === nothing ? similar(Bk) : workspace.spectral4
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
            # -(k²/4f)|B|² term (PDF Eq. 29, 47)
            qʷₖ_arr[k, i_local, j_local] = qʷₖ_arr[k, i_local, j_local] - (0.25/f₀)*kₕ²*tempₖ_arr[k, i_local, j_local]
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
    dissipation_q_nv!(dqk, qok, par, G; workspace=nothing)

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
- `qok`: Input q field
- `par`: QGParams (for nuz coefficient)
- `G::Grid`: Grid struct
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Note
This operates on spectral q but the vertical derivative is in physical space,
so the operation is the same for each (kx, ky) mode.

# Fortran Correspondence
This matches `dissipation_q_nv` in derivatives.f90.
"""
function dissipation_q_nv!(dqk, qok, par, G::Grid; workspace=nothing)
    nz = G.nz

    # Check if we need 2D decomposition transpose
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(qok, G)

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
    nz_local, nx_local, ny_local = size(dqk_arr)

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local"

    # Handle nz=1 case: no vertical diffusion possible with single layer
    if nz <= 1
        fill!(dqk_arr, zero(eltype(dqk_arr)))
        return
    end

    # Vertical grid spacing (safe now since nz >= 2)
    Δz = G.z[2] - G.z[1]
    Δz⁻² = 1/(Δz*Δz)
    νz = par.νz

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
end

"""
2D decomposition vertical diffusion with transposes.
"""
function _dissipation_q_nv_2d!(dqk, qok, par, G::Grid, workspace)
    nz = G.nz

    # Handle nz=1 case: no vertical diffusion possible with single layer
    if nz <= 1
        dqk_arr = parent(dqk)
        fill!(dqk_arr, zero(eltype(dqk_arr)))
        return
    end

    # Allocate z-pencil workspace
    qok_z = workspace !== nothing ? workspace.q_z : allocate_z_pencil(G, ComplexF64)
    dqk_z = workspace !== nothing ? workspace.work_z : allocate_z_pencil(G, ComplexF64)

    # Transpose input to z-pencil
    transpose_to_z_pencil!(qok_z, qok, G)

    # Get underlying arrays in z-pencil format
    qok_z_arr = parent(qok_z)
    dqk_z_arr = parent(dqk_z)
    nz_local, nx_local, ny_local = size(qok_z_arr)

    @assert nz_local == nz "After transpose, z must be fully local"

    # Vertical grid spacing (safe now since nz >= 2)
    Δz = G.z[2] - G.z[1]
    Δz⁻² = 1/(Δz*Δz)
    νz = par.νz

    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        if k == 1
            dqk_z_arr[k, i_local, j_local] = νz * ( qok_z_arr[k+1, i_local, j_local] - qok_z_arr[k, i_local, j_local] ) * Δz⁻²
        elseif k == nz
            dqk_z_arr[k, i_local, j_local] = νz * ( qok_z_arr[k-1, i_local, j_local] - qok_z_arr[k, i_local, j_local] ) * Δz⁻²
        else
            dqk_z_arr[k, i_local, j_local] = νz * ( qok_z_arr[k+1, i_local, j_local] - 2qok_z_arr[k, i_local, j_local] + qok_z_arr[k-1, i_local, j_local] ) * Δz⁻²
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

    λ = ν₁(|kx|^(2n₁) + |ky|^(2n₁)) + ν₂(|kx|^(2n₂) + |ky|^(2n₂))

The integrating factor for one time step is: exp(-λ×dt)

For efficiency, we return just λ×dt (the exponent).

# Arguments
- `kx, ky`: Horizontal wavenumber components
- `par`: QGParams (contains ν₁, ν₂, n₁, n₂)
- `waves::Bool`: If true, use wave hyperdiffusion (nuh1w, ilap1w, etc.)

# Returns
    λ×dt = dt × [ν₁(|kx|^(2n₁) + |ky|^(2n₁)) +
                 ν₂(|kx|^(2n₂) + |ky|^(2n₂))]

Note: This follows QG-YBJp's separable hyperdiffusion operator.

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
function int_factor(kₓ::Real, kᵧ::Real, par; waves::Bool=false)
    # When inviscid=true, disable ALL dissipation including hyperdiffusion
    # Return 0 so that exp(-0) = 1 (no damping)
    if hasfield(typeof(par), :inviscid) && par.inviscid
        return 0.0
    end

    Δt = par.dt
    if waves
        # Wave field hyperdiffusion (often smaller or zero)
        ν₁ʷ = par.νₕ₁ʷ; n₁ʷ = par.ilap1w
        ν₂ʷ = par.νₕ₂ʷ; n₂ʷ = par.ilap2w
        return Δt * ( ν₁ʷ * (abs(kₓ)^(2n₁ʷ) + abs(kᵧ)^(2n₁ʷ)) +
                      ν₂ʷ * (abs(kₓ)^(2n₂ʷ) + abs(kᵧ)^(2n₂ʷ)) )
    else
        # Mean flow hyperdiffusion
        ν₁ = par.νₕ₁; n₁ = par.ilap1
        ν₂ = par.νₕ₂; n₂ = par.ilap2
        return Δt * ( ν₁ * (abs(kₓ)^(2n₁) + abs(kᵧ)^(2n₁)) +
                      ν₂ * (abs(kₓ)^(2n₂) + abs(kᵧ)^(2n₂)) )
    end
end

end # module

# Export nonlinear operators to main QGYBJplus module
using .Nonlinear: jacobian_spectral!, convol_waqg!, convol_waqg_q!, convol_waqg_L⁺A!,
                  refraction_waqg!, refraction_waqg_L⁺A!, compute_qw!, compute_qw_complex!,
                  dissipation_q_nv!, int_factor, NonlinearWorkspace
