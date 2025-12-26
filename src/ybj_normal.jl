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
   - B = N² ∂²A/∂z² (simplified relation)
   - Recover A from B via vertical INTEGRATION
   - Computationally simpler but less accurate for high-kz

PHYSICAL CONTEXT:
-----------------
The normal YBJ equation describes near-inertial wave evolution:

    ∂B/∂t + J(ψ, B) = i·αdisp·kh²·A - (i/2)ζ·B

where:
- αdisp = N²/(2f) is the dispersion coefficient
- ζ = ∇²ψ is the relative vorticity
- B is related to the wave amplitude A by: B ≈ N² ∂²A/∂z² (simplified)

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

using ..QGYBJplus: Grid, QGParams
using ..QGYBJplus: N2_ut
using ..QGYBJplus: local_to_global, get_local_dims, z_is_local
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: local_to_global_z, allocate_z_pencil

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

    B = N² ∂²A/∂z²

Since ∂²A/∂z² must integrate to zero (boundary conditions), B should have
zero vertical mean. This function enforces that constraint.

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ) within the dealiasing mask:
1. Compute vertical mean: B̄(kₓ,kᵧ) = (1/nz) Σₖ B(kₓ,kᵧ,k)
2. Subtract mean: B(kₓ,kᵧ,k) ← B(kₓ,kᵧ,k) - B̄

For wavenumbers outside the mask or kh² = 0, set B = 0.

# Arguments
- `B::Array{Complex,3}`: Wave envelope (modified in-place)
- `G::Grid`: Grid structure with wavenumbers
- `Lmask`: Optional dealiasing mask (default: all modes kept)
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
Modified B array with zero vertical mean at each (kₓ, kᵧ).

# Fortran Correspondence
Matches `sumB` in derivatives.f90.
"""
function sumB!(B::AbstractArray{<:Complex,3}, G::Grid; Lmask=nothing, workspace=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(G)

    if need_transpose
        _sumB_2d!(B, G, Lmask, workspace)
    else
        _sumB_direct!(B, G, Lmask)
    end
    return B
end

# Direct computation when z is fully local
function _sumB_direct!(B::AbstractArray{<:Complex,3}, G::Grid, Lmask)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask

    B_arr = parent(B)
    nz_local, nx_local, ny_local = size(B_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, G)
        j_global = local_to_global(j, 3, G)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
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
end

# 2D decomposition version with transposes
function _sumB_2d!(B::AbstractArray{<:Complex,3}, G::Grid, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask

    # Transpose to z-pencil for vertical operations
    B_z = workspace !== nothing && hasfield(typeof(workspace), :B_z) ? workspace.B_z : allocate_z_pencil(G, ComplexF64)
    transpose_to_z_pencil!(B_z, B, G)

    B_z_arr = parent(B_z)
    nz_local, nx_local_z, ny_local_z = size(B_z_arr)

    @assert nz_local == nz "After transpose, z must be fully local"

    @inbounds for j in 1:ny_local_z, i in 1:nx_local_z
        i_global = local_to_global_z(i, 2, G)
        j_global = local_to_global_z(j, 3, G)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += B_z_arr[k, i, j]
            end
            aveij = s / nz
            for k in 1:nz
                B_z_arr[k, i, j] -= aveij
            end
        else
            for k in 1:nz
                B_z_arr[k, i, j] = 0
            end
        end
    end

    # Transpose back to xy-pencil
    transpose_to_xy_pencil!(B, B_z, G)
end

#=
================================================================================
                    SIGMA CONSTRAINT COMPUTATION
================================================================================
Sigma is the solvability condition for the vertical integration.
================================================================================
=#

"""
    compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask=nothing, workspace=nothing, N2_profile=nothing) -> sigma

Compute the sigma constraint for normal YBJ A recovery.

# Physical Background
When recovering A from B via vertical integration, we need to determine the
vertical mean of A. The sigma parameter provides this constraint from the
nonlinear forcing terms.

# Mathematical Formula
For each horizontal wavenumber (kₓ, kᵧ):

```
σ(kₓ,kᵧ) = Σₖ [(rBRk + 2·nBIk)/kh² + i(rBIk - 2·nBRk)/kh²]
```

where:
- nBRk, nBIk: Real and imaginary parts of nonlinear advection term
- rBRk, rBIk: Real and imaginary parts of refraction term
- kh² = kₓ² + kᵧ²

# Arguments
- `par::QGParams`: Model parameters
- `G::Grid`: Grid with wavenumbers
- `nBRk, nBIk`: Real/imaginary parts of advection forcing
- `rBRk, rBIk`: Real/imaginary parts of refraction forcing
- `Lmask`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `N2_profile`: Optional N²(z) profile for consistent stratification physics.
  If not provided, uses N2_ut(par, G) based on par.stratification.

# Returns
2D complex array sigma(nx_local, ny_local) with the constraint values.

# Fortran Correspondence
Matches `compute_sigma` in derivatives.f90.

# Note
In MPI mode with 2D decomposition, this requires z to be fully local.
Transpose operations are handled internally if needed.
"""
function compute_sigma(par::QGParams, G::Grid,
                       nBRk, nBIk, rBRk, rBIk; Lmask=nothing, workspace=nothing, N2_profile=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(G)

    if need_transpose
        return _compute_sigma_2d(par, G, nBRk, nBIk, rBRk, rBIk, Lmask, workspace, N2_profile)
    else
        return _compute_sigma_direct(par, G, nBRk, nBIk, rBRk, rBIk, Lmask, N2_profile)
    end
end

# Direct computation when z is fully local
function _compute_sigma_direct(par::QGParams, G::Grid, nBRk, nBIk, rBRk, rBIk, Lmask, N2_profile)
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
        i_global = local_to_global(i, 2, G)
        j_global = local_to_global(j, 3, G)
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

    # Scale by f/N² (inverse of dispersion coefficient factor)
    # Use provided N2_profile if available, otherwise compute from params
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ? N2_profile : N2_ut(par, G)
    N²_mean = sum(N²) / nz
    σ .*= (par.f₀ / N²_mean)

    return σ
end

# 2D decomposition version with transposes
function _compute_sigma_2d(par::QGParams, G::Grid, nBRk, nBIk, rBRk, rBIk, Lmask, workspace, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask

    # Transpose to z-pencil for vertical summation
    nBRk_z = allocate_z_pencil(G, ComplexF64)
    nBIk_z = allocate_z_pencil(G, ComplexF64)
    rBRk_z = allocate_z_pencil(G, ComplexF64)
    rBIk_z = allocate_z_pencil(G, ComplexF64)

    transpose_to_z_pencil!(nBRk_z, nBRk, G)
    transpose_to_z_pencil!(nBIk_z, nBIk, G)
    transpose_to_z_pencil!(rBRk_z, rBRk, G)
    transpose_to_z_pencil!(rBIk_z, rBIk, G)

    nBRk_z_arr = parent(nBRk_z)
    nBIk_z_arr = parent(nBIk_z)
    rBRk_z_arr = parent(rBRk_z)
    rBIk_z_arr = parent(rBIk_z)

    nz_local, nx_local_z, ny_local_z = size(nBRk_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    σ = zeros(ComplexF64, nx_local_z, ny_local_z)

    @inbounds for j in 1:ny_local_z, i in 1:nx_local_z
        i_global = local_to_global_z(i, 2, G)
        j_global = local_to_global_z(j, 3, G)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            s = 0.0 + 0.0im
            for k in 1:nz
                s += ( rBRk_z_arr[k, i, j] + 2*nBIk_z_arr[k, i, j] + im*( rBIk_z_arr[k, i, j] - 2*nBRk_z_arr[k, i, j] ) )/kₕ²
            end
            σ[i,j] = s
        else
            σ[i,j] = 0
        end
    end

    # Scale by f/N² (inverse of dispersion coefficient factor)
    # Use provided N2_profile if available, otherwise compute from params
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ? N2_profile : N2_ut(par, G)
    N²_mean = sum(N²) / nz
    σ .*= (par.f₀ / N²_mean)

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
    compute_A!(A, C, BRk, BIk, sigma, par, G; Lmask=nothing, workspace=nothing, N2_profile=nothing)

Recover wave amplitude A from envelope B using normal YBJ vertical integration.

# Physical Background
In normal YBJ, B and A are related by:

    B = N² ∂²A/∂z²

To recover A from B, we integrate twice:
1. First integral: ∂A/∂z = ∫ B/N² dz + c₁
2. Second integral: A = ∫∫ B/N² dz² + c₁z + c₂

The constants are determined by:
- Boundary condition: ∂A/∂z = 0 at top (Neumann)
- Mean constraint: ∫A dz = σ (from sigma)

# Algorithm
For each horizontal wavenumber (kₓ, kᵧ):

**Stage 1: Cumulative Integration**
```julia
Ã[1] = 0
Ã[k] = Ã[k-1] + (Σⱼ₌₁ᵏ⁻¹ B[j]) × N²[k-1] × dz²
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
- `BRk, BIk`: Real/imaginary parts of wave envelope B
- `sigma::Array{Complex,2}`: Sigma constraint from compute_sigma
- `par::QGParams`: Model parameters
- `G::Grid`: Grid structure
- `Lmask`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `N2_profile`: Optional N²(z) profile for variable stratification. If not provided,
  uses `N2_ut(par, G)`.

# Returns
Tuple (A, C) with recovered amplitude and its vertical derivative.

# Fortran Correspondence
Matches `compute_A` in derivatives.f90.

# Note
This is the NORMAL YBJ recovery method. For YBJ+, use `invert_B_to_A!` instead,
which solves the full L⁺A = B elliptic problem via tridiagonal solve.
"""
function compute_A!(A::AbstractArray{<:Complex,3}, C::AbstractArray{<:Complex,3},
                    BRk::AbstractArray{<:Complex,3}, BIk::AbstractArray{<:Complex,3},
                    sigma::AbstractArray{<:Complex,2}, par::QGParams, G::Grid;
                    Lmask=nothing, workspace=nothing, N2_profile=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(G)

    if need_transpose
        _compute_A_2d!(A, C, BRk, BIk, sigma, par, G, Lmask, workspace, N2_profile)
    else
        _compute_A_direct!(A, C, BRk, BIk, sigma, par, G, Lmask, N2_profile)
    end
    return A, C
end

# Direct computation when z is fully local
function _compute_A_direct!(A, C, BRk, BIk, σ, par, G, Lmask, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ? N2_profile : N2_ut(par, G)
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    A_arr = parent(A)
    C_arr = parent(C)
    BRk_arr = parent(BRk)
    BIk_arr = parent(BIk)
    nz_local, nx_local, ny_local = size(A_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, G)
        j_global = local_to_global(j, 3, G)
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
                A_arr[k, i, j] = A_arr[k-1, i, j] + ( sBR + im*sBI ) * N²[k-1]*Δz*Δz
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
end

# 2D decomposition version with transposes
function _compute_A_2d!(A, C, BRk, BIk, σ, par, G, Lmask, workspace, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    N² = (N2_profile !== nothing && length(N2_profile) == nz) ? N2_profile : N2_ut(par, G)
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    # Transpose inputs to z-pencil
    BRk_z = allocate_z_pencil(G, ComplexF64)
    BIk_z = allocate_z_pencil(G, ComplexF64)
    A_z = allocate_z_pencil(G, ComplexF64)
    C_z = allocate_z_pencil(G, ComplexF64)

    transpose_to_z_pencil!(BRk_z, BRk, G)
    transpose_to_z_pencil!(BIk_z, BIk, G)

    BRk_z_arr = parent(BRk_z)
    BIk_z_arr = parent(BIk_z)
    A_z_arr = parent(A_z)
    C_z_arr = parent(C_z)

    nz_local, nx_local_z, ny_local_z = size(BRk_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    @inbounds for j in 1:ny_local_z, i in 1:nx_local_z
        i_global = local_to_global_z(i, 2, G)
        j_global = local_to_global_z(j, 3, G)
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2

        if L[i_global, j_global] && kₕ² > 0
            # Stage 1: cumulative vertical integration
            sBR = 0.0 + 0.0im
            sBI = 0.0 + 0.0im
            A_z_arr[1, i, j] = 0
            for k in 2:nz
                sBR += BRk_z_arr[k-1, i, j]
                sBI += BIk_z_arr[k-1, i, j]
                A_z_arr[k, i, j] = A_z_arr[k-1, i, j] + ( sBR + im*sBI ) * N²[k-1]*Δz*Δz
            end
            # Stage 2: apply σ constraint
            # Note: σ is computed in z-pencil configuration, so indexing matches
            sumA = 0.0 + 0.0im
            for k in 1:nz
                sumA += A_z_arr[k, i, j]
            end
            adj = (σ[i,j] - sumA)/nz
            for k in 1:nz
                A_z_arr[k, i, j] += adj
            end
            # Stage 3: vertical derivative
            for k in 1:nz-1
                C_z_arr[k, i, j] = (A_z_arr[k+1, i, j] - A_z_arr[k, i, j])/Δz
            end
            C_z_arr[nz, i, j] = 0
        else
            for k in 1:nz
                A_z_arr[k, i, j] = 0; C_z_arr[k, i, j] = 0
            end
        end
    end

    # Transpose results back to xy-pencil
    transpose_to_xy_pencil!(A, A_z, G)
    transpose_to_xy_pencil!(C, C_z, G)
end

end # module

using .YBJNormal: sumB!, compute_sigma, compute_A!
