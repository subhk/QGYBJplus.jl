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
Every pointwise and spectral loop is a KernelAbstractions kernel.

DEALIASING:
-----------
The 2/3 rule removes wavenumbers with |k| > 2/3 kmax to prevent aliasing
from quadratic nonlinearities. The Lmask array encodes which modes to keep;
`Lmask=nothing` applies the radial two-thirds rule directly.
================================================================================
=#

module Nonlinear

using KernelAbstractions: @kernel, @index, @Const, get_backend
using ..QGYBJplus: RuntimeGeometry, HorizontalHyperdiffusivity,
                   FlowHyperdiffusivity, WaveHyperdiffusivity
using ..QGYBJplus: fft_forward!, fft_backward!
using ..QGYBJplus: allocate_fft_backward_dst
using ..QGYBJplus: with_scratch, scratch_like, scratch_physical, scratch_phys_like
using ..QGYBJplus: launch!, kernel_map!, kernel_mask, keep_mode, dealias!,
                   KeepAllModes, RadialDealiasing, vertical_spacing

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

"""Kernel rule for a nonlinear-term `Lmask`: `nothing` is the radial 2/3 rule."""
_nonlinear_mask(Lmask, G::RuntimeGeometry, array) =
    kernel_mask(Lmask, array, RadialDealiasing(G.nx, G.ny))

# Prefilter spectral inputs to the 2/3 mask before nonlinear products.
function _prefilter_spectral!(dst, src, G::RuntimeGeometry, Lmask)
    return dealias!(dst, src, _nonlinear_mask(Lmask, G, dst))
end

#=
================================================================================
                        SHARED SPECTRAL KERNELS
================================================================================
=#

"""`∂χ/∂x` and `∂χ/∂y` in spectral space, optionally projected by `mask`."""
@kernel function _gradient_kernel!(χx, χy, @Const(χ), @Const(kx), @Const(ky), mask)
    k, i, j = @index(Global, NTuple)
    @inbounds if keep_mode(mask, i, j)
        χx[k, i, j] = im * kx[i] * χ[k, i, j]
        χy[k, i, j] = im * ky[j] * χ[k, i, j]
    else
        χx[k, i, j] = zero(eltype(χx))
        χy[k, i, j] = zero(eltype(χy))
    end
end

"""Divergence-form advection `i kₓ (uχ)^ + i kᵧ (vχ)^`, projected by `mask`."""
@kernel function _divergence_kernel!(nχ, @Const(uχ), @Const(vχ), @Const(kx),
                                     @Const(ky), mask)
    k, i, j = @index(Global, NTuple)
    @inbounds if keep_mode(mask, i, j)
        nχ[k, i, j] = im * kx[i] * uχ[k, i, j] + im * ky[j] * vχ[k, i, j]
    else
        nχ[k, i, j] = zero(eltype(nχ))
    end
end

"""Relative vorticity `ζ̂ = -kₕ² ψ̂`, projected by `mask`."""
@kernel function _vorticity_kernel!(ζ, @Const(ψ), @Const(kx), @Const(ky), mask)
    k, i, j = @index(Global, NTuple)
    @inbounds if keep_mode(mask, i, j)
        kₕ² = kx[i]^2 + ky[j]^2
        ζ[k, i, j] = -kₕ² * ψ[k, i, j]
    else
        ζ[k, i, j] = zero(eltype(ζ))
    end
end

"""`qʷ = (1/f₀)[J + (1/4)∇²|B|²]` from the transformed pieces, projected by `mask`."""
@kernel function _wave_feedback_assembly_kernel!(qʷ, @Const(magnitude), @Const(kx),
                                                 @Const(ky), mask, inv_f)
    k, i, j = @index(Global, NTuple)
    @inbounds if keep_mode(mask, i, j)
        kₕ² = kx[i]^2 + ky[j]^2
        qʷ[k, i, j] = inv_f * (qʷ[k, i, j] - kₕ² * magnitude[k, i, j] / 4)
    else
        qʷ[k, i, j] = zero(eltype(qʷ))
    end
end

_real_product(a, b) = real(a) * real(b)
_jacobian_product(φx, φy, χx, χy) = real(φx) * real(χy) - real(φy) * real(χx)

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
    return _hyperdiffusion_exponent(closure, kₕ², Δt)
end

"""
`Δt Σ νₙ kₕ^orderₙ` for a hyperdiffusivity component. Pure scalar code, so the
ETD coefficient kernel calls it on every backend.

The powers are taken in at least double precision. A single-precision model
passes a `Float32` kₕ², and dimensional wavenumbers (kₕ ~ 10⁻⁴ m⁻¹) raised to
the twelfth or sixteenth power underflow `Float32`, which would switch the
damping off.
"""
@inline function _hyperdiffusion_exponent(closure, kₕ², Δt)
    T = promote_type(Float64, typeof(kₕ²), eltype(closure.coefficients))
    k² = convert(T, kₕ²)
    rate = zero(T)
    for (coefficient, order) in zip(closure.coefficients, closure.orders)
        # Besides saving work for disabled closure terms, skipping before the
        # power prevents 0 * Inf from turning a legitimate zero rate into NaN.
        iszero(coefficient) && continue
        rate += coefficient * k²^(order ÷ 2)
    end
    return Δt * rate
end

"""
    jacobian_spectral!(dstk, phik, chik, G, plans; Lmask=nothing)

Compute the Jacobian J(φ, χ) = ∂φ/∂x ∂χ/∂y - ∂φ/∂y ∂χ/∂x using pseudo-spectral method.

!!! note "Usage Note"
    This function is provided for user convenience but is **not used** in the
    main time-stepping code. The main code uses `convol_waqg_q!` instead,
    which computes advection terms using the divergence form with
    precomputed velocities.

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
    backend = get_backend(dstk)
    mask = _nonlinear_mask(Lmask, G, dstk)

    # Step 1: spectral derivatives, ∂/∂x → ikₓ and ∂/∂y → ikᵧ
    φₓₖ = similar(φₖ); φᵧₖ = similar(φₖ)
    χₓₖ = similar(χₖ); χᵧₖ = similar(χₖ)
    launch!(_gradient_kernel!, backend, size(φₖ), φₓₖ, φᵧₖ, φₖ, G.kx, G.ky, mask)
    launch!(_gradient_kernel!, backend, size(χₖ), χₓₖ, χᵧₖ, χₖ, G.kx, G.ky, mask)

    # Step 2: transform derivatives to real space
    φₓ = _allocate_fft_dst(φₓₖ, plans); φᵧ = _allocate_fft_dst(φᵧₖ, plans)
    χₓ = _allocate_fft_dst(χₓₖ, plans); χᵧ = _allocate_fft_dst(χᵧₖ, plans)
    fft_backward!(φₓ, φₓₖ, plans)
    fft_backward!(φᵧ, φᵧₖ, plans)
    fft_backward!(χₓ, χₓₖ, plans)
    fft_backward!(χᵧ, χᵧₖ, plans)

    # Step 3: J = φₓχᵧ - φᵧχₓ in physical space. For real fields the inverse
    # transforms of ik φ̂ are real up to roundoff, so take their real parts.
    Jᵣ = _allocate_fft_dst(φₖ, plans)
    launch!(_jacobian_product_kernel!, backend, size(Jᵣ), Jᵣ, φₓ, φᵧ, χₓ, χᵧ)

    # Step 4: back to spectral space; dealias the quadratic product. The
    # normalized inverse transforms leave the product correctly scaled.
    fft_forward!(dstk, Jᵣ, plans)
    dealias!(dstk, dstk, mask)
    return dstk
end

@kernel function _jacobian_product_kernel!(J, @Const(φx), @Const(φy), @Const(χx), @Const(χy))
    index = @index(Global, Linear)
    @inbounds J[index] = _jacobian_product(φx[index], φy[index], χx[index], χy[index])
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
    # Every input is read before any output is written, so an output may alias
    # any of the inputs.
    results = map((qk, BRk, BIk)) do field
        _convol_advect!(similar(field), u, v, field, G, plans;
                        Lmask=Lmask, use_real=true, workspace=nothing)
    end
    copyto!(nqk, results[1])
    copyto!(nBRk, results[2])
    copyto!(nBIk, results[3])
    return nqk, nBRk, nBIk
end

# Advection helper for complex fields (q or B) without splitting into BR/BI.
function _convol_advect!(nχk, u, v, χk, G::RuntimeGeometry, plans;
                         Lmask=nothing, use_real::Bool=false, workspace=nothing)
    backend = get_backend(nχk)
    mask = _nonlinear_mask(Lmask, G, nχk)
    product = use_real ? _real_product : *

    # Two buffers, transformed in place: χ (then vχ) and uχ.
    χ = scratch_like(workspace, χk)
    dealias!(χ, χk, mask)
    fft_backward!(χ, χ, plans)
    uχ = scratch_like(workspace, χk)
    kernel_map!(product, uχ, u, χ)
    kernel_map!(product, χ, v, χ)
    fft_forward!(uχ, uχ, plans)
    fft_forward!(χ, χ, plans)

    # The normalized inverse transform leaves the products correctly scaled.
    launch!(_divergence_kernel!, backend, size(nχk),
            nχk, uχ, χ, G.kx, G.ky, mask)
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
    backend = get_backend(rBRk)
    mask = _nonlinear_mask(Lmask, G, rBRk)

    # Relative vorticity ζ = ∇²ψ = -kₕ²ψ̂
    ζₖ = similar(ψₖ)
    launch!(_vorticity_kernel!, backend, size(ζₖ), ζₖ, ψₖ, G.kx, G.ky, mask)

    # Transform to real space
    ζᵣ = _allocate_fft_dst(ζₖ, plans)
    BRᵣ = _allocate_fft_dst(BRk, plans); BIᵣ = _allocate_fft_dst(BIk, plans)
    BRk_f = similar(BRk)
    BIk_f = similar(BIk)
    dealias!(BRk_f, BRk, mask)
    dealias!(BIk_f, BIk, mask)
    fft_backward!(ζᵣ, ζₖ, plans)
    fft_backward!(BRᵣ, BRk_f, plans)
    fft_backward!(BIᵣ, BIk_f, plans)

    # Products in real space: rB = ζ × B
    rBRᵣ = similar(BRᵣ); rBIᵣ = similar(BIᵣ)
    kernel_map!(_real_product, rBRᵣ, ζᵣ, BRᵣ)
    kernel_map!(_real_product, rBIᵣ, ζᵣ, BIᵣ)

    # Back to spectral; the normalized inverse transform leaves the products
    # correctly scaled, so only dealias.
    fft_forward!(rBRk, rBRᵣ, plans)
    fft_forward!(rBIk, rBIᵣ, plans)
    dealias!(rBRk, rBRk, mask)
    dealias!(rBIk, rBIk, mask)
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

_refraction_product(ζ, B) = real(ζ) * B

function _refraction_waqg_B!(rBk, Bk, ψₖ, G::RuntimeGeometry, plans, Lmask, workspace)
    backend = get_backend(rBk)
    mask = _nonlinear_mask(Lmask, G, rBk)

    # Two buffers, transformed in place: ζ and B (then ζB).
    ζ = scratch_like(workspace, ψₖ)
    launch!(_vorticity_kernel!, backend, size(ζ), ζ, ψₖ, G.kx, G.ky, mask)
    fft_backward!(ζ, ζ, plans)
    B = scratch_like(workspace, Bk)
    dealias!(B, Bk, mask)
    fft_backward!(B, B, plans)

    kernel_map!(_refraction_product, B, ζ, B)
    fft_forward!(rBk, B, plans)
    dealias!(rBk, rBk, mask)
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
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("wave feedback requires a finite, nonzero Coriolis frequency"))
    inv_f = convert(real(eltype(qʷₖ)), inv(float(f)))
    backend = get_backend(qʷₖ)
    mask = _nonlinear_mask(Lmask, G, qʷₖ)

    # Prefilter inputs to avoid aliasing when upstream fields are not masked
    BRk_f = similar(BRk)
    BIk_f = similar(BIk)
    dealias!(BRk_f, BRk, mask)
    dealias!(BIk_f, BIk, mask)

    # Derivatives of the prefiltered BR and BI
    BRₓₖ = similar(BRk); BRᵧₖ = similar(BRk)
    BIₓₖ = similar(BIk); BIᵧₖ = similar(BIk)
    launch!(_gradient_kernel!, backend, size(BRk), BRₓₖ, BRᵧₖ, BRk_f, G.kx, G.ky,
            KeepAllModes())
    launch!(_gradient_kernel!, backend, size(BIk), BIₓₖ, BIᵧₖ, BIk_f, G.kx, G.ky,
            KeepAllModes())

    # Transform derivatives to real space
    BRₓᵣ = _allocate_fft_dst(BRₓₖ, plans); BRᵧᵣ = _allocate_fft_dst(BRᵧₖ, plans)
    BIₓᵣ = _allocate_fft_dst(BIₓₖ, plans); BIᵧᵣ = _allocate_fft_dst(BIᵧₖ, plans)
    fft_backward!(BRₓᵣ, BRₓₖ, plans)
    fft_backward!(BRᵧᵣ, BRᵧₖ, plans)
    fft_backward!(BIₓᵣ, BIₓₖ, plans)
    fft_backward!(BIᵧᵣ, BIᵧₖ, plans)

    # J(B*, B) = 2i(BRₓBIᵧ - BRᵧBIₓ) is purely imaginary, so
    # (i/2)J(B*, B) = BRᵧBIₓ - BRₓBIᵧ.
    qʷᵣ = _allocate_fft_dst(qʷₖ, plans)
    launch!(_jacobian_product_kernel!, backend, size(qʷᵣ),
            qʷᵣ, BRᵧᵣ, BRₓᵣ, BIᵧᵣ, BIₓᵣ)

    # |B|² = BR² + BI² for the ∇²|B|² term
    BRᵣ = _allocate_fft_dst(BRk, plans); BIᵣ = _allocate_fft_dst(BIk, plans)
    fft_backward!(BRᵣ, BRk_f, plans)
    fft_backward!(BIᵣ, BIk_f, plans)
    mag² = _allocate_fft_dst(BRk, plans)
    kernel_map!(_split_magnitude², mag², BRᵣ, BIᵣ)

    # Assemble qʷ = (1/f₀)[J_term + (1/4)∇²|B|²] with ∇² → -kₕ²
    tempₖ = similar(BRk)
    fft_forward!(tempₖ, mag², plans)
    fft_forward!(qʷₖ, qʷᵣ, plans)
    launch!(_wave_feedback_assembly_kernel!, backend, size(qʷₖ),
            qʷₖ, tempₖ, G.kx, G.ky, mask, inv_f)
    return qʷₖ
end

_split_magnitude²(BR, BI) = real(BR)^2 + real(BI)^2

"""
    compute_qw_complex!(qʷₖ, Bk, G, plans; f, Lmask=nothing)

Compute wave feedback directly from complex B without spectral BR/BI splitting.

For `YBJPlus`, this implements the wave part of the "XV⁺" potential
vorticity, Asselin & Young (2019) equation (3.5):

    q = ΔΨ + LΨ + (i/2f) J(L⁺A*, L⁺A) + (1/4f) Δ|L⁺A|²

The caller supplies the formulation's prognostic envelope: `B = L⁺A` for
`YBJPlus`, or `B = LA` for the original `YBJ` formulation. A&Y obtain (3.5)
from Xie & Vanneste's PV by the substitution L ↦ L⁺, and it is precisely
that substitution which gives the YBJ⁺ coupled system (1.4), (3.4), (3.5)
its nonlinear "coupled energy" conservation law, A&Y equations (3.6)–(3.7).
Substituting `LA` for the YBJ⁺ state would instead recover the original XV
potential vorticity and lose that YBJ⁺ conservation law.

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

# (i/2) J(B*, B) = -Im(J(B*, B)) / 2, as J(B*, B) is purely imaginary.
_complex_jacobian(Bx, By) = -imag(conj(Bx) * By - conj(By) * Bx) / 2
_complex_magnitude²(B) = real(conj(B) * B)

function _compute_qw_complex!(qʷₖ, Bk, G::RuntimeGeometry, plans, f, Lmask, workspace)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("wave feedback requires a finite, nonzero Coriolis frequency"))
    inv_f = convert(real(eltype(qʷₖ)), inv(float(f)))
    backend = get_backend(qʷₖ)
    mask = _nonlinear_mask(Lmask, G, qʷₖ)

    # Three buffers, transformed in place: B (prefiltered, to avoid aliasing
    # when upstream fields are not masked, then |B|²), ∂ₓB (then the
    # Jacobian term), and ∂ᵧB.
    B = scratch_like(workspace, Bk)
    dealias!(B, Bk, mask)
    Bₓ = scratch_like(workspace, Bk); Bᵧ = scratch_like(workspace, Bk)
    launch!(_gradient_kernel!, backend, size(Bk), Bₓ, Bᵧ, B, G.kx, G.ky,
            KeepAllModes())
    fft_backward!(B, B, plans)
    fft_backward!(Bₓ, Bₓ, plans)
    fft_backward!(Bᵧ, Bᵧ, plans)

    # (i/2)J(B*, B) and |B|² in physical space
    kernel_map!(_complex_jacobian, Bₓ, Bₓ, Bᵧ)
    kernel_map!(_complex_magnitude², B, B)

    # Transform to spectral and assemble
    fft_forward!(B, B, plans)
    fft_forward!(qʷₖ, Bₓ, plans)
    launch!(_wave_feedback_assembly_kernel!, backend, size(qʷₖ),
            qʷₖ, B, G.kx, G.ky, mask, inv_f)
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
- `workspace`: Unused; accepted for call-site compatibility

# Note
This operates on spectral q but the vertical derivative is in physical space,
so the operation is the same for each (kx, ky) mode.

# Fortran Correspondence
This matches `dissipation_q_nv` in derivatives.f90.
"""
function dissipation_q_nv!(dqk, qok, vertical_diffusivity::Real, G::RuntimeGeometry;
    workspace=nothing)

    # A single layer has no vertical diffusion.
    if G.nz <= 1
        fill!(dqk, zero(eltype(dqk)))
        return dqk
    end

    nz = size(dqk, 1)
    nz == G.nz || throw(DimensionMismatch("dqk has $nz levels; the grid has $(G.nz)"))
    size(qok) == size(dqk) || throw(DimensionMismatch(
        "qok has size $(size(qok)); dqk has $(size(dqk))"))
    # The stencil reads q at k ± 1 after dq[k] is written, so they must differ.
    dqk === qok && throw(ArgumentError("dqk and qok must be distinct arrays"))
    Δz = vertical_spacing(G)
    launch!(_vertical_diffusion_kernel!, dqk, size(dqk),
            dqk, qok, convert(typeof(Δz), vertical_diffusivity), 1 / (Δz * Δz), nz)
    return dqk
end

@kernel function _vertical_diffusion_kernel!(dq, @Const(q), νz, Δz⁻², nz)
    k, i, j = @index(Global, NTuple)
    @inbounds if k == 1
        # Bottom boundary: Neumann (q_z = 0)
        dq[k, i, j] = νz * (q[k + 1, i, j] - q[k, i, j]) * Δz⁻²
    elseif k == nz
        # Top boundary: Neumann (q_z = 0)
        dq[k, i, j] = νz * (q[k - 1, i, j] - q[k, i, j]) * Δz⁻²
    else
        # Interior: standard central difference
        dq[k, i, j] = νz * (q[k + 1, i, j] - 2q[k, i, j] + q[k - 1, i, j]) * Δz⁻²
    end
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
# Build the scalar damping exponent used by the ETD-RK2 coefficient table.
lambda_dt = int_factor(kx, ky, Δt, closure)
E, hφ1, hφ2 = _etd_coefficients(lambda_dt, Δt)
```

# Fortran Correspondence
This matches the integrating factor computation in the main loop of main_waqg.f90.

# Example
```julia
# Get integrating factor for wavenumber (3, 4)
lambda_dt = int_factor(3.0, 4.0, Δt, closure)
factor = exp(-lambda_dt)
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
