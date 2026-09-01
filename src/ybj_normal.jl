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

FORTRAN CORRESPONDENCE:
-----------------------
- sumB! → sumB in derivatives.f90
- compute_sigma → compute_sigma in derivatives.f90
- compute_A! → compute_A in derivatives.f90

================================================================================
=#

module YBJNormal

using ..QGYBJplus: RuntimeGeometry
using ..QGYBJplus: local_to_global, get_local_dims, z_is_local
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: local_to_global_z, allocate_z_pencil
using ..QGYBJplus: with_z_local, z_scratch

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

Since ∂²A/∂z² must integrate to zero (boundary conditions), B should have
zero vertical mean. This function enforces that constraint.

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ) within the dealiasing mask:
1. Compute vertical mean: B̄(kₓ,kᵧ) = (1/nz) Σₖ B(kₓ,kᵧ,k)
2. Subtract mean: B(kₓ,kᵧ,k) ← B(kₓ,kᵧ,k) - B̄

For wavenumbers outside the mask or kh² = 0, set B = 0.

# Arguments
- `B::Array{Complex,3}`: Wave envelope (modified in-place)
- `G::RuntimeGeometry`: RuntimeGeometry structure with wavenumbers
- `Lmask`: Optional dealiasing mask (default: all modes kept)
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
Modified B array with zero vertical mean at each (kₓ, kᵧ).

# Fortran Correspondence
Matches `sumB` in derivatives.f90.
"""
function sumB!(B::AbstractArray{<:Complex,3}, G::RuntimeGeometry; Lmask=nothing, workspace=nothing)
    with_z_local(G, (B,), (:inout,);
                 scratch=z_scratch(workspace, :B_z)) do B_z
        _sumB!(B_z, G, Lmask)
    end
    return B
end

"""
Remove the vertical mean of `B` at each retained horizontal wavenumber, and
zero the modes the dealiasing mask drops. Requires a fully local vertical
dimension, which `sumB!` arranges.
"""
function _sumB!(B::AbstractArray{<:Complex,3}, G::RuntimeGeometry, Lmask)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask

    B_arr = parent(B)
    nz_local, nx_local, ny_local = size(B_arr)
    @assert nz_local == nz "Vertical dimension must be fully local"

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, B)
        j_global = local_to_global(j, 3, B)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += B_arr[k, i, j]
            end
            aveij = s / nz
            for k in 1:nz
                B_arr[k, i, j] -= aveij
            end
        else
            for k in 1:nz
                B_arr[k, i, j] = 0
            end
        end
    end
    return B
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
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
2D complex array sigma(nx_local, ny_local) with the constraint values.

# Fortran Correspondence
Matches `compute_sigma` in derivatives.f90.

# Note
In MPI mode with 2D decomposition, this requires z to be fully local.
Transpose operations are handled internally if needed.
"""
function compute_sigma(f::Real, G::RuntimeGeometry, nBk, rBk;
                       Lmask=nothing, workspace=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ sigma requires a finite, nonzero Coriolis frequency"))
    return with_z_local(G, (nBk, rBk), (:in, :in);
                        scratch=z_scratch(workspace, :B_z, :work_z)) do nB_z, rB_z
        _compute_sigma_complex(f, G, nB_z, rB_z, Lmask)
    end
end

"""
Sigma is the solvability condition of the vertical integration: one value per
horizontal wavenumber, accumulated over the full column. Requires a fully local
vertical dimension, which `compute_sigma` arranges.
"""
function _compute_sigma_complex(f, G::RuntimeGeometry, nBk, rBk, Lmask)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx, ny) : Lmask
    nB_arr = parent(nBk)
    rB_arr = parent(rBk)
    nz_local, nx_local, ny_local = size(nB_arr)
    @assert nz_local == nz "Vertical dimension must be fully local"

    σ = zeros(eltype(nB_arr), nx_local, ny_local)
    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, nBk)
        j_global = local_to_global(j, 3, nBk)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2
        if L[i_global, j_global] && kₕ² > 0
            value = zero(eltype(nB_arr))
            for k in 1:nz
                value += (rB_arr[k, i, j] - 2im*nB_arr[k, i, j]) / kₕ²
            end
            σ[i, j] = value
        end
    end

    σ ./= f
    return σ
end

function compute_sigma(f::Real, G::RuntimeGeometry,
                       nBRk, nBIk, rBRk, rBIk;
                       Lmask=nothing, workspace=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ sigma requires a finite, nonzero Coriolis frequency"))
    return with_z_local(G, (nBRk, nBIk, rBRk, rBIk), (:in, :in, :in, :in);
                        scratch=z_scratch(workspace, :B_z, :work_z, :A_z, :C_z)) do nBR, nBI, rBR, rBI
        _compute_sigma(f, G, nBR, nBI, rBR, rBI, Lmask)
    end
end

"""
Split-component sigma constraint. Requires a fully local vertical dimension,
which `compute_sigma` arranges.
"""
function _compute_sigma(
    f, G::RuntimeGeometry, nBRk, nBIk, rBRk, rBIk, Lmask)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask

    nBRk_arr = parent(nBRk)
    nz_local, nx_local, ny_local = size(nBRk_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    σ = zeros(ComplexF64, nx_local, ny_local)

    nBIk_arr = parent(nBIk)
    rBRk_arr = parent(rBRk)
    rBIk_arr = parent(rBIk)

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, nBRk)
        j_global = local_to_global(j, 3, nBRk)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += ( rBRk_arr[k, i, j] + 2*nBIk_arr[k, i, j] + im*( rBIk_arr[k, i, j] - 2*nBRk_arr[k, i, j] ) )/kₕ²
            end
            σ[i,j] = s
        else
            σ[i,j] = 0
        end
    end

    # Enforce zero vertical mean of the dimensional YBJ tendency, whose
    # dispersion coefficient is f₀/2.
    σ ./= f

    return σ
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

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ):

**Stage 1: Cumulative Integration**
```julia
Ã[1] = 0
Ã[k] = Ã[k-1] + (Σⱼ₌₁ᵏ⁻¹ B[j]) × N²[k-1]/f₀² × dz²
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
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `N2_profile`: Optional N²(z) profile for variable stratification. If not provided,
  uses a unit profile.

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
    with_z_local(G, (A, C, B), (:out, :out, :in);
                 scratch=z_scratch(workspace, :A_z, :C_z, :B_z)) do A_z, C_z, B_z
        _compute_A_complex!(A_z, C_z, B_z, sigma, f, G, Lmask, N2_profile)
    end
    return A, C
end

"""
Recover `A` and `C = A_z` from the normal-YBJ envelope by upward integration.
Requires a fully local vertical dimension, which `compute_A!` arranges. `sigma`
is indexed on the same pencil the kernel sees, which `compute_sigma` matches.
"""
function _compute_A_complex!(A, C, B, σ, f, G, Lmask, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx, ny) : Lmask
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ?
        N2_profile : ones(nz)
    inv_f² = inv(float(f))^2
    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    A_arr = parent(A)
    C_arr = parent(C)
    B_arr = parent(B)
    nz_local, nx_local, ny_local = size(A_arr)
    @assert nz_local == nz "Vertical dimension must be fully local"

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, B)
        j_global = local_to_global(j, 3, B)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            cumulative_B = zero(eltype(B_arr))
            A_arr[1, i, j] = 0
            for k in 2:nz
                cumulative_B += B_arr[k-1, i, j]
                A_arr[k, i, j] = A_arr[k-1, i, j] +
                    cumulative_B * N²[k-1] * inv_f² * Δz^2
            end
            sum_A = zero(eltype(A_arr))
            for k in 1:nz
                sum_A += A_arr[k, i, j]
            end
            adjustment = (σ[i, j] - sum_A) / nz
            for k in 1:nz
                A_arr[k, i, j] += adjustment
            end
            for k in 1:(nz-1)
                C_arr[k, i, j] =
                    (A_arr[k+1, i, j] - A_arr[k, i, j]) / Δz
            end
            C_arr[nz, i, j] = 0
        else
            for k in 1:nz
                A_arr[k, i, j] = 0
                C_arr[k, i, j] = 0
            end
        end
    end
    return A, C
end

function compute_A!(A::AbstractArray{<:Complex,3}, C::AbstractArray{<:Complex,3},
                    BRk::AbstractArray{<:Complex,3}, BIk::AbstractArray{<:Complex,3},
                    sigma::AbstractArray{<:Complex,2}, G::RuntimeGeometry;
                    f::Real, Lmask=nothing, workspace=nothing,
                    N2_profile=nothing)
    isfinite(f) && !iszero(f) ||
        throw(ArgumentError("normal-YBJ recovery requires a finite, nonzero Coriolis frequency"))
    with_z_local(G, (A, C, BRk, BIk), (:out, :out, :in, :in);
                 scratch=z_scratch(workspace, :A_z, :C_z, :B_z, :work_z)) do A_z, C_z, BR_z, BI_z
        _compute_A!(A_z, C_z, BR_z, BI_z, sigma, f, G, Lmask, N2_profile)
    end
    return A, C
end

"""
Split-component form of the normal-YBJ recovery. Requires a fully local
vertical dimension, which `compute_A!` arranges.
"""
function _compute_A!(A, C, BRk, BIk, σ, f, G, Lmask, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ? N2_profile : ones(nz)
    inv_f² = inv(float(f))^2
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    A_arr = parent(A)
    C_arr = parent(C)
    BRk_arr = parent(BRk)
    BIk_arr = parent(BIk)
    nz_local, nx_local, ny_local = size(A_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, BRk)
        j_global = local_to_global(j, 3, BRk)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            # Stage 1: build Ã by cumulative vertical integration
            sBR = 0.0 + 0.0im
            sBI = 0.0 + 0.0im
            A_arr[1, i, j] = 0
            for k in 2:nz
                sBR += BRk_arr[k-1, i, j]
                sBI += BIk_arr[k-1, i, j]
                A_arr[k, i, j] = A_arr[k-1, i, j] +
                    (sBR + im*sBI) * N²[k-1] * inv_f² * Δz^2
            end
            # Stage 2: compute vertical sum
            sumA = 0.0 + 0.0im
            for k in 1:nz
                sumA += A_arr[k, i, j]
            end
            # Adjust to enforce mean(A) = σ(i,j)/nz
            adj = (σ[i,j] - sumA)/nz
            for k in 1:nz
                A_arr[k, i, j] += adj
            end
            # C = A_z, forward diff; top C=0
            for k in 1:nz-1
                C_arr[k, i, j] = (A_arr[k+1, i, j] - A_arr[k, i, j])/Δz
            end
            C_arr[nz, i, j] = 0
        else
            for k in 1:nz
                A_arr[k, i, j] = 0; C_arr[k, i, j] = 0
            end
        end
    end
    return A, C
end

end # module

using .YBJNormal: sumB!, compute_sigma, compute_A!
