#=
================================================================================
                    ybj_normal.jl - Normal YBJ Operators (Non-Plus)
================================================================================

This file implements the "normal" YBJ (Young-Ben Jelloul) wave operators, as
opposed to the YBJ+ formulation. The key difference is how A is recovered from B.

YBJ vs YBJ+ COMPARISON:
-----------------------
1. YBJ+ (Plus formulation):
   - B = L⁺A where L⁺ is an elliptic operator
   - Recover A from B via tridiagonal solve (invert_B_to_A!)
   - More accurate for high vertical wavenumber modes

2. Normal YBJ (this file):
   - B = L A, where L = ∂z[(f₀²/N²)∂z]
   - Recover A from B via vertical INTEGRATION
   - Computationally simpler but less accurate for high-kz

PHYSICAL CONTEXT:
-----------------
The normal YBJ equation describes near-inertial wave evolution:

    ∂B/∂t + J(ψ, B) = i·αdisp·kh²·A - (i/2)ζ·B

where:
- αdisp = f₀/2 is the dispersion coefficient
- ζ = ∇²ψ is the relative vorticity
- B is related to the wave amplitude A by `B = ∂z[(f₀²/N²)A_z]`

To recover A from B, we integrate twice in z with appropriate constraints.

VERTICAL INTEGRATION METHOD:
----------------------------
Given B(z), recover A(z) by:

1. First integration: ∫B dz gives ∂A/∂z (up to constant)
2. Second integration: ∫∫B dz² gives A (up to linear function)
3. Apply constraints:
   - Zero mean constraint: ∫A dz = σ (sigma)
   - Boundary conditions: A_z = 0 at top/bottom

SIGMA CONSTRAINT:
-----------------
The sigma parameter σ(kx,ky) is computed from the nonlinear terms to ensure
proper vertical mean of A. This is the "solvability condition" for the
integration problem.

Each operator is a column kernel: one work item per horizontal mode, looping
over z.

FORTRAN CORRESPONDENCE:
-----------------------
- sumB! → sumB in derivatives.f90
- compute_sigma → compute_sigma in derivatives.f90
- compute_A! → compute_A in derivatives.f90

================================================================================
=#

module YBJNormal

using KernelAbstractions: @kernel, @index, @Const, get_backend
using ..QGYBJplus: RuntimeGeometry, launch!, kernel_mask, keep_mode, on_backend,
                   KeepAllModes, vertical_spacing, in_precision

"""Kernel rule for a normal-YBJ `Lmask`: `nothing` keeps every mode."""
_column_mask(Lmask, array) = kernel_mask(Lmask, array, KeepAllModes())

#=
================================================================================
                    VERTICAL MEAN SUBTRACTION
================================================================================
For normal YBJ, we need to remove the vertical mean of B before integration.
================================================================================
=#

"""
    sumB!(B, G; Lmask=nothing, workspace=nothing)

Remove the vertical mean from the wave envelope B at each horizontal wavenumber.

# Physical Background
In the normal YBJ formulation, the wave envelope B is related to amplitude A by:

    B = ∂z[(f₀²/N²)∂z A]

With the Neumann conditions A_z = 0 at the bottom and the lid, B has zero
vertical mean in every column, including the horizontally uniform one
(kₕ = 0). This function enforces that constraint.

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ) within the dealiasing mask:
1. Compute vertical mean: B̄(kₓ,kᵧ) = (1/nz) Σₖ B(kₓ,kᵧ,k)
2. Subtract mean: B(kₓ,kᵧ,k) ← B(kₓ,kᵧ,k) - B̄

Wavenumbers outside the mask are set to zero.

# Arguments
- `B::Array{Complex,3}`: Wave envelope (modified in-place)
- `G::RuntimeGeometry`: RuntimeGeometry structure with wavenumbers
- `Lmask`: Optional dealiasing mask (default: all modes kept)
- `workspace`: Unused; accepted for call-site compatibility

# Returns
Modified B array with zero vertical mean at each (kₓ, kᵧ).

# Fortran Correspondence
Follows `sumB` in derivatives.f90, but keeps the depth-varying part of the
kₕ = 0 column, the horizontally uniform waves.
"""
function sumB!(B::AbstractArray{<:Complex,3}, G::RuntimeGeometry; Lmask=nothing, workspace=nothing)
    nz, nx, ny = size(B)
    @assert nz == G.nz "Vertical dimension must match the grid"
    launch!(_sumB_kernel!, B, (nx, ny), B, G.kx, G.ky, _column_mask(Lmask, B), nz)
    return B
end

@kernel function _sumB_kernel!(B, @Const(kx), @Const(ky), mask, nz)
    i, j = @index(Global, NTuple)
    if keep_mode(mask, i, j)
        s = zero(eltype(B))
        for k in 1:nz
            @inbounds s += B[k, i, j]
        end
        aveij = s / nz
        for k in 1:nz
            @inbounds B[k, i, j] -= aveij
        end
    else
        for k in 1:nz
            @inbounds B[k, i, j] = zero(eltype(B))
        end
    end
end

#=
================================================================================
                    SIGMA CONSTRAINT COMPUTATION
================================================================================
Sigma is the solvability condition for the vertical integration.
================================================================================
=#

"""
    compute_sigma(f, G, nBk, rBk; Lmask=nothing,
                  workspace=nothing) -> sigma
    compute_sigma(f, G, nBRk, nBIk, rBRk, rBIk; kwargs...) -> sigma

Compute the sigma constraint for normal YBJ A recovery.

# Physical Background
When recovering A from B via vertical integration, we need to determine the
vertical mean of A. The sigma parameter provides this constraint from the
nonlinear forcing terms.

# Mathematical Formula
For each horizontal wavenumber (kₓ, kᵧ):

```
σ(kₓ,kᵧ) = (1/f₀) Σₖ (rBk - 2i·nBk)/kh²
```

where:
- nBk: Complex nonlinear-advection tendency
- rBk: Complex refraction product
- kh² = kₓ² + kᵧ²

The component overload is retained for direct use and satisfies
`nBk = nBRk + i*nBIk` and `rBk = rBRk + i*rBIk`.

# Arguments
- `f`: Coriolis frequency
- `G::RuntimeGeometry`: RuntimeGeometry with wavenumbers
- `nBk`, `rBk`: Complex advection and refraction fields
- `nBRk, nBIk, rBRk, rBIk`: Separate component spectra accepted by the
  compatibility overload
- `Lmask`: Optional dealiasing mask
- `workspace`: Unused; accepted for call-site compatibility

# Returns
2D complex array sigma(nx, ny) with the constraint values, on the same
architecture as the inputs.

# Fortran Correspondence
Matches `compute_sigma` in derivatives.f90.
"""
function compute_sigma(f::Real, G::RuntimeGeometry, nBk, rBk;
                       Lmask=nothing, workspace=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ sigma requires a finite, nonzero Coriolis frequency"))
    nz, nx, ny = size(nBk)
    nz == G.nz || throw(DimensionMismatch("nBk has $nz levels; the grid has $(G.nz)"))
    size(rBk) == size(nBk) || throw(DimensionMismatch(
        "rBk has size $(size(rBk)); nBk has $(size(nBk))"))
    σ = similar(nBk, eltype(nBk), nx, ny)
    launch!(_sigma_kernel!, σ, (nx, ny), σ, nBk, rBk, G.kx, G.ky,
            _column_mask(Lmask, nBk), convert(real(eltype(σ)), f), nz)
    return σ
end

@kernel function _sigma_kernel!(σ, @Const(nB), @Const(rB), @Const(kx), @Const(ky),
                                mask, f, nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    value = zero(eltype(σ))
    if keep_mode(mask, i, j) && kₕ² > 0
        for k in 1:nz
            @inbounds value += (rB[k, i, j] - 2im * nB[k, i, j]) / kₕ²
        end
    end
    @inbounds σ[i, j] = value / f
end

function compute_sigma(f::Real, G::RuntimeGeometry,
                       nBRk, nBIk, rBRk, rBIk;
                       Lmask=nothing, workspace=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ sigma requires a finite, nonzero Coriolis frequency"))
    nz, nx, ny = size(nBRk)
    nz == G.nz || throw(DimensionMismatch("nBRk has $nz levels; the grid has $(G.nz)"))
    all(array -> size(array) == size(nBRk), (nBIk, rBRk, rBIk)) ||
        throw(DimensionMismatch("the split spectra must have the same size"))
    σ = similar(nBRk, eltype(nBRk), nx, ny)
    launch!(_split_sigma_kernel!, σ, (nx, ny), σ, nBRk, nBIk, rBRk, rBIk,
            G.kx, G.ky, _column_mask(Lmask, nBRk), convert(real(eltype(σ)), f), nz)
    return σ
end

@kernel function _split_sigma_kernel!(σ, @Const(nBR), @Const(nBI), @Const(rBR),
                                      @Const(rBI), @Const(kx), @Const(ky), mask, f, nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    s = zero(eltype(σ))
    if keep_mode(mask, i, j) && kₕ² > 0
        for k in 1:nz
            @inbounds s += (rBR[k, i, j] + 2 * nBI[k, i, j] +
                            im * (rBI[k, i, j] - 2 * nBR[k, i, j])) / kₕ²
        end
    end
    # Enforce zero vertical mean of the dimensional YBJ tendency, whose
    # dispersion coefficient is f₀/2.
    @inbounds σ[i, j] = s / f
end

#=
================================================================================
                    WAVE AMPLITUDE RECOVERY (NORMAL YBJ)
================================================================================
Recover the true wave amplitude A from the evolved envelope B via vertical
integration (as opposed to YBJ+ which uses tridiagonal inversion).
================================================================================
=#

"""
    compute_A!(A, C, B, sigma, G; f, Lmask=nothing, workspace=nothing,
               N2_profile=nothing)
    compute_A!(A, C, BRk, BIk, sigma, G; kwargs...)

Recover wave amplitude A from envelope B using normal YBJ vertical integration.

# Physical Background
In normal YBJ, B and A are related by:

    B = ∂z[(f₀²/N²)∂z A]

To recover A from B, we integrate twice using the inverse elliptic coefficient:
1. First integral: ``(f₀²/N²)∂A/∂z = ∫ B dz + c₁``
2. Second integral: ``A = ∫ (N²/f₀²)∫B dz² + c₂``

The constants are determined by:
- Boundary condition: ∂A/∂z = 0 at top (Neumann)
- Mean constraint: ∫A dz = σ (from sigma)

At kₕ = 0 the dispersion term vanishes, so the depth mean of A is never
needed; σ is zero there and A has zero mean.

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ):

**Stage 1: Cumulative Integration**
```julia
Ã[1] = 0
Ã[k] = Ã[k-1] + (Σⱼ₌₁ᵏ⁻¹ B[j]) × N²_face[k-1]/f₀² × dz²
```

**Stage 2: Apply Sigma Constraint**
```julia
sumA = Σₖ Ã[k]
adj = (σ - sumA) / nz
A[k] = Ã[k] + adj   # Enforce ∫A = σ
```

**Stage 3: Compute Vertical Derivative**
```julia
C[k] = (A[k+1] - A[k]) / dz   # Forward difference
C[nz] = 0                      # Neumann BC at top
```

# Arguments
- `A::Array{Complex,3}`: Output wave amplitude (modified in-place)
- `C::Array{Complex,3}`: Output vertical derivative A_z (modified in-place)
- `B`: Complex wave-envelope spectrum
- `BRk, BIk`: Separate component spectra accepted by the compatibility
  overload, with `B = BRk + i*BIk`
- `sigma::Array{Complex,2}`: Sigma constraint from compute_sigma
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `f`: Coriolis frequency ``f₀``
- `Lmask`: Optional dealiasing mask
- `workspace`: Unused; accepted for call-site compatibility
- `N2_profile`: Optional N²(z) sampled at the upper faces
  `G.z_faces[2:end]` for variable stratification. If not provided, uses a unit
  profile.

# Returns
Tuple (A, C) with recovered amplitude and its vertical derivative.

# Fortran Correspondence
Matches `compute_A` in derivatives.f90.

# Note
This is the NORMAL YBJ recovery method. For YBJ+, use `invert_B_to_A!` instead,
which solves the full L⁺A = B elliptic problem via tridiagonal solve.
"""
function compute_A!(A::AbstractArray{<:Complex,3}, C::AbstractArray{<:Complex,3},
                    B::AbstractArray{<:Complex,3},
                    sigma::AbstractArray{<:Complex,2}, G::RuntimeGeometry;
                    f::Real, Lmask=nothing, workspace=nothing,
                    N2_profile=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ recovery requires a finite, nonzero Coriolis frequency"))
    nz, nx, ny = size(A)
    _check_recovery_sizes(A, C, sigma, G, B)
    backend = get_backend(A)
    launch!(_compute_A_kernel!, backend, (nx, ny), A, C, B, nothing,
            on_backend(backend, sigma), G.kx, G.ky, _column_mask(Lmask, A),
            on_backend(backend, _recovery_N²(N2_profile, nz, real(eltype(A)))),
            convert(real(eltype(A)), inv(float(f))^2),
            real(eltype(A))(vertical_spacing(G)), nz)
    return A, C
end

function compute_A!(A::AbstractArray{<:Complex,3}, C::AbstractArray{<:Complex,3},
                    BRk::AbstractArray{<:Complex,3}, BIk::AbstractArray{<:Complex,3},
                    sigma::AbstractArray{<:Complex,2}, G::RuntimeGeometry;
                    f::Real, Lmask=nothing, workspace=nothing,
                    N2_profile=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ recovery requires a finite, nonzero Coriolis frequency"))
    nz, nx, ny = size(A)
    _check_recovery_sizes(A, C, sigma, G, BRk, BIk)
    backend = get_backend(A)
    launch!(_compute_A_kernel!, backend, (nx, ny), A, C, BRk, BIk,
            on_backend(backend, sigma), G.kx, G.ky, _column_mask(Lmask, A),
            on_backend(backend, _recovery_N²(N2_profile, nz, real(eltype(A)))),
            convert(real(eltype(A)), inv(float(f))^2),
            real(eltype(A))(vertical_spacing(G)), nz)
    return A, C
end

"""
Error unless `A`, `C`, and the envelope spectra share one size on the grid and
`sigma` is one horizontal plane of it; the envelope must not be `A` or `C`.
"""
function _check_recovery_sizes(A, C, sigma, G::RuntimeGeometry, envelopes...)
    nz, nx, ny = size(A)
    nz == G.nz || throw(DimensionMismatch("A has $nz levels; the grid has $(G.nz)"))
    size(C) == size(A) && all(B -> size(B) == size(A), envelopes) ||
        throw(DimensionMismatch("A, C, and B must have the same size"))
    size(sigma) == (nx, ny) || throw(DimensionMismatch(
        "sigma has size $(size(sigma)); expected $((nx, ny))"))
    any(B -> B === A || B === C, envelopes) &&
        throw(ArgumentError("B must be distinct from A and C"))
    return nothing
end

"""The N² profile of the recovery in precision `T`: unit N² when none fits."""
_recovery_N²(N2_profile, nz, ::Type{T}) where T =
    (N2_profile !== nothing && length(N2_profile) == nz) ?
    in_precision(N2_profile, T) : ones(T, nz)

"""Envelope value used by the recovery: `B`, or `BR + i BI` for split spectra."""
@inline _envelope_increment(B, ::Nothing, k, i, j) = @inbounds B[k, i, j]
@inline _envelope_increment(BR, BI, k, i, j) =
    @inbounds BR[k, i, j], BI[k, i, j]

@inline _accumulate(total, increment) = total + increment
@inline _accumulate(total::Tuple, increment::Tuple) =
    (total[1] + increment[1], total[2] + increment[2])

@inline _combine(total) = total
@inline _combine(total::Tuple) = total[1] + im * total[2]

@inline _zero_accumulator(B, ::Nothing) = zero(eltype(B))
@inline _zero_accumulator(BR, BI) = (zero(eltype(BR)), zero(eltype(BI)))

@kernel function _compute_A_kernel!(A, C, @Const(B), BI, @Const(σ), @Const(kx),
                                    @Const(ky), mask, @Const(N²), inv_f², Δz, nz)
    i, j = @index(Global, NTuple)
    if keep_mode(mask, i, j)
        # Stage 1: build Ã by cumulative vertical integration
        cumulative_B = _zero_accumulator(B, BI)
        @inbounds A[1, i, j] = zero(eltype(A))
        for k in 2:nz
            @inbounds begin
                cumulative_B = _accumulate(cumulative_B,
                                           _envelope_increment(B, BI, k - 1, i, j))
                A[k, i, j] = A[k - 1, i, j] +
                    _combine(cumulative_B) * N²[k - 1] * inv_f² * Δz^2
            end
        end
        # Stage 2: enforce mean(A) = σ(i,j)/nz
        sum_A = zero(eltype(A))
        for k in 1:nz
            @inbounds sum_A += A[k, i, j]
        end
        @inbounds adjustment = (σ[i, j] - sum_A) / nz
        for k in 1:nz
            @inbounds A[k, i, j] += adjustment
        end
        # Stage 3: C = A_z by forward differences; top C = 0
        for k in 1:(nz - 1)
            @inbounds C[k, i, j] = (A[k + 1, i, j] - A[k, i, j]) / Δz
        end
        @inbounds C[nz, i, j] = zero(eltype(C))
    else
        for k in 1:nz
            @inbounds A[k, i, j] = zero(eltype(A))
            @inbounds C[k, i, j] = zero(eltype(C))
        end
    end
end

end # module

using .YBJNormal: sumB!, compute_sigma, compute_A!
