#=
================================================================================
                    operators.jl - Velocity Field Computation
================================================================================

This file computes diagnostic velocity fields from the model's prognostic
variables (streamfunction ψ and wave amplitude A).

PHYSICAL BACKGROUND:
--------------------
In QG-YBJ+ dynamics, velocities arise from two sources:

1. GEOSTROPHIC FLOW (from streamfunction ψ):
   - Horizontal: u = -∂ψ/∂y, v = ∂ψ/∂x (geostrophic balance)
   - Vertical: w from the QG omega equation (ageostrophic correction)

2. WAVE-INDUCED FLOW (from wave amplitude A):
   - Horizontal: Stokes drift from wave envelope gradients
   - Vertical: YBJ formulation involving ∂A_z/∂x, ∂A_z/∂y

GEOSTROPHIC VELOCITIES:
-----------------------
The horizontal geostrophic velocities follow from:

    f u = -∂p'/∂y = -f ∂ψ/∂y  →  u = -∂ψ/∂y
    f v =  ∂p'/∂x =  f ∂ψ/∂x  →  v =  ∂ψ/∂x

In spectral space:
    û(k) = -i kᵧ ψ̂(k)
    v̂(k) =  i kₓ ψ̂(k)

QG OMEGA EQUATION:
------------------
The ageostrophic vertical velocity w comes from the omega equation:

    N² ∇²w + f² ∂²w/∂z² = 2f J(ψ_z, ∇²ψ)

or equivalently (dividing by N²):

    ∇²w + (f²/N²) ∂²w/∂z² = (2f/N²) J(ψ_z, ∇²ψ)

This is a 3D elliptic PDE solved via:
- Horizontal: spectral differentiation (kₓ², kᵧ²)
- Vertical: tridiagonal solver at each (kₓ, kᵧ)

Physical meaning: w arises from ageostrophic convergence/divergence required
to maintain geostrophic balance as the flow evolves.

YBJ VERTICAL VELOCITY:
----------------------
For wave-induced vertical motion, following Asselin & Young (2019) equation (2.10):

    w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.

where:
- A is the wave amplitude (recovered from B via L⁺A = B)
- A_{zs} = ∂_s(A_z) = (1/2)(∂_x - i∂_y)(A_z) is the complex horizontal derivative
- The complex coordinate s = x + iy, so ∂_s = (1/2)(∂_x - i∂_y)
- e^{-ift} is the inertial oscillation factor
- c.c. denotes complex conjugate (ensures real result)

Expanding this gives the oscillating vertical velocity:
    w = -(f²/N²) * [cos(ft)·w_cos + sin(ft)·w_sin]
where:
    w_cos = Re(∂A_z/∂x) + Im(∂A_z/∂y)
    w_sin = Im(∂A_z/∂x) - Re(∂A_z/∂y)

This represents vertical motion induced by wave envelope modulation, oscillating
at the inertial frequency f.

WAVE VELOCITY AND STOKES DRIFT:
-------------------------------
Near-inertial waves contribute to particle advection through two mechanisms:

1. WAVE VELOCITY (Asselin & Young 2019, eq. 1.2):
   The backrotated wave velocity is LA, where L = ∂_z(f²/N²)∂_z:
       u_wave = Re(LA)
       v_wave = Im(LA)

   For YBJ+: B = L⁺A where L⁺ = L + (1/4)Δ, so LA = B - (1/4)ΔA
   In spectral space: LA = B + (k_h²/4)A (since Δ → -k_h²)

2. STOKES DRIFT (Wagner & Young 2016, eq. 3.16a-3.20):
   The horizontal Stokes drift uses the full Jacobian form:

       J₀ = (LA)* ∂_{s*}(LA) - (f₀²/N²)(∂_{s*} A_z*) ∂_z(LA)

   where ∂_{s*} = (1/2)(∂_x + i∂_y) is the complex horizontal derivative.

   From eq. (3.18): if₀ U^S = J₀, giving:
       u_S = Im(J₀)/f₀
       v_S = -Re(J₀)/f₀

   The vertical Stokes drift (eq. 3.19-3.20) uses:
       K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}   where M = (f₀²/N²)A_z
       w_S = -2·Im(K₀)/f₀

   with:
       M*_z = a_z·A_z* + a·A_{zz}*     (a = f₀²/N², a_z = ∂_z(f₀²/N²))
       M_{ss*} = (a/4)·Δ_H(A_z)
       M*_{s*} = a·(A_{zs})*
       M_{sz} = a_z·A_{zs} + a·A_{zzs}

The total velocity for particle advection is:
    u_total = u_QG + u_wave + u_S
    v_total = v_QG + v_wave + v_S
    w_total = w_QG + w_S

SPECTRAL DIFFERENTIATION:
-------------------------
All derivatives are computed in spectral space:
    ∂f/∂x → i kₓ f̂(k)
    ∂f/∂y → i kᵧ f̂(k)

Vertical derivatives use second-order finite differences.

================================================================================
=#
module Operators

using LinearAlgebra
using ..QGYBJplus: Grid, State, local_to_global, z_is_local
using ..QGYBJplus: fft_backward!, plan_transforms!
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: local_to_global_z, allocate_z_pencil
using ..QGYBJplus: invert_B_to_A!
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
import PencilArrays: PencilArray
const PARENT = Base.parentmodule(@__MODULE__)

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

# Access invert_B_to_A! through the Elliptic submodule via PARENT
# (Direct import via `using ..QGYBJplus: invert_B_to_A!` can fail in some loading contexts)
# @inline invert_B_to_A!(args...; kwargs...) = PARENT.Elliptic.invert_B_to_A!(args...; kwargs...)

function _coerce_N2_profile(N2_profile, N2_const, nz, G::Grid)
    N2_type = float(promote_type(eltype(G.z), typeof(N2_const)))
    N2_const_T = N2_type(N2_const)

    if N2_profile === nothing
        return fill(N2_const_T, nz)
    end

    if length(N2_profile) != nz
        @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=$(N2_const)"
        return fill(N2_const_T, nz)
    end

    if !(eltype(N2_profile) <: Real)
        N2_profile = real.(N2_profile)
    end
    if eltype(N2_profile) != N2_type
        N2_profile = N2_type.(N2_profile)
    end

    return N2_profile
end

#=
================================================================================
                    GEOSTROPHIC VELOCITY COMPUTATION
================================================================================
The primary diagnostic: horizontal and vertical velocities from streamfunction.
================================================================================
=#

"""
    compute_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing, dealias_mask=nothing)

Compute geostrophic velocities from the spectral streamfunction ψ̂.

# Physical Equations
Horizontal velocities from geostrophic balance:
```
u = -∂ψ/∂y  →  û(k) = -i kᵧ ψ̂(k)
v =  ∂ψ/∂x  →  v̂(k) =  i kₓ ψ̂(k)
```

Vertical velocity from QG omega equation:
```
∇²w + (f²/N²) ∂²w/∂z² = (2f/N²) J(ψ_z, ∇²ψ)
```
or YBJ formulation:
```
w = -(f²/N²) [(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
```

# Algorithm
1. Compute û = -i kᵧ ψ̂ and v̂ = i kₓ ψ̂ in spectral space
2. Transform to physical space via inverse FFT
3. Optionally solve omega equation or use YBJ formula for w

# Arguments
- `S::State`: State with ψ (input) and u, v, w (output)
- `G::Grid`: Grid with wavenumbers kx, ky
- `plans`: FFT plans (auto-generated if nothing)
- `params`: Model parameters (for f₀, N²)
- `compute_w::Bool`: If true, compute vertical velocity
- `use_ybj_w::Bool`: If true, use YBJ formula instead of omega equation
- `N2_profile::Vector`: Optional N²(z) profile for vertical velocity computation
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `dealias_mask`: Optional 2D dealiasing mask for omega equation RHS (quadratic term).
  Should be the same mask used for other nonlinear terms (typically 2/3 rule).

# Returns
Modified State with updated u, v, w fields.

# Note
This computes ONLY QG velocities. For Lagrangian advection including wave
effects, use `compute_total_velocities!` instead.

# Fortran Correspondence
Matches `compute_velo` in derivatives.f90.
"""
function compute_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing, dealias_mask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    ψk_arr = parent(S.psi)
    u_arr = parent(S.u)
    v_arr = parent(S.v)
    nz_local, nx_local, ny_local = size(ψk_arr)

    # Spectral differentiation: û = -i ky ψ̂, v̂ = i kx ψ̂
    ψk = S.psi
    uk = similar(ψk)
    vk = similar(ψk)
    uk_arr = parent(uk)
    vk_arr = parent(vk)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, ψk)
        j_global = local_to_global(j_local, 3, ψk)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        uk_arr[k, i_local, j_local] = -ikᵧ * ψk_arr[k, i_local, j_local]
        vk_arr[k, i_local, j_local] =  ikₓ * ψk_arr[k, i_local, j_local]
    end

    # Inverse FFT to real space
    if plans === nothing
        # Use unified transform planning (handles both serial and parallel)
        plans = plan_transforms!(G)
    end

    # Allocate destination arrays on correct pencil for fft_backward!
    # For MPI: must be on input_pencil (physical space), not output_pencil
    tmpu = _allocate_fft_dst(uk, plans)
    tmpv = _allocate_fft_dst(vk, plans)
    fft_backward!(tmpu, uk, plans)
    fft_backward!(tmpv, vk, plans)

    tmpu_arr = parent(tmpu)
    tmpv_arr = parent(tmpv)

    # Note on normalization:
    # - FFTW.ifft and PencilFFTs ldiv! are NORMALIZED (divide by N internally)
    # - After fft_backward!, tmpu and tmpv contain correct physical-space values
    # - NO additional normalization is needed here
    #
    # This differs from pseudo-spectral convolutions (nonlinear.jl) where:
    # - We do: IFFT → multiply → FFT → divide by N
    # - The division compensates for unnormalized FFT output
    # Here we're only doing IFFT, so no extra division.
    #
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(tmpu_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        u_arr[k, i_local, j_local] = real(tmpu_arr[k, i_local, j_local])
        v_arr[k, i_local, j_local] = real(tmpv_arr[k, i_local, j_local])
    end

    # Compute vertical velocity if requested
    if compute_w
        if use_ybj_w
            # Use YBJ vertical velocity formulation (equation 4)
            compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=N2_profile, workspace=workspace)
        else
            # Use standard QG omega equation with dealiasing
            # The omega equation RHS is a quadratic term J(ψ_z, ∇²ψ) that needs dealiasing
            compute_vertical_velocity!(S, G, plans, params; N2_profile=N2_profile, 
                                workspace=workspace, dealias_mask=dealias_mask)
        end
    else
        # Set w to zero (leading-order QG approximation)
        w_arr = parent(S.w)
        fill!(w_arr, zero(eltype(w_arr)))
    end

    return S
end

#=
================================================================================
                    QG OMEGA EQUATION SOLVER
================================================================================
Computes the ageostrophic vertical velocity from the QG omega equation.
This is a 3D elliptic problem solved via tridiagonal systems.
================================================================================
=#

"""
    compute_vertical_velocity!(S, G, plans, params; N2_profile=nothing, workspace=nothing, dealias_mask=nothing)

Solve the QG omega equation for ageostrophic vertical velocity.

# Physical Background
In quasi-geostrophic dynamics, the leading-order horizontal flow is non-divergent
(∇·u_g = 0). Vertical motion arises from ageostrophic corrections that maintain
thermal wind balance as the flow evolves.

The omega equation relates w to the horizontal flow:
```
N² ∇²w + f² ∂²w/∂z² = 2f J(ψ_z, ∇²ψ)
```

or equivalently (dividing by N²):
```
∇²w + (f²/N²) ∂²w/∂z² = (2f/N²) J(ψ_z, ∇²ψ)
```

where:
- Left side: 3D Laplacian (horizontal + stratification-weighted vertical)
- Right side: Jacobian forcing from vertical shear and vorticity
- f²/N² << 1: stratification suppresses vertical motion relative to horizontal

# Physical Interpretation
The RHS forcing J(ψ_z, ∇²ψ) represents:
- Thermal wind tilting: vertical shear ψ_z interacting with vorticity ∇²ψ
- Frontogenesis/frontolysis: differential advection of temperature gradients

Strong w occurs at:
- Fronts (sharp density gradients)
- Edges of eddies (strong vorticity gradients)

# Numerical Method
1. Compute RHS in spectral space via omega_eqn_rhs!
2. For each horizontal wavenumber (kₓ, kᵧ):
   - Set up tridiagonal system in z
   - Solve using LAPACK gtsv! (O(nz) per wavenumber)
3. Transform w to physical space

# Boundary Conditions
w = 0 at z = -Lz and z = 0 (rigid lid and bottom).

# Arguments
- `S::State`: State with ψ (input) and w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters (f₀)
- `N2_profile::Vector`: Optional N²(z) profile (default: constant N² = 1)
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `dealias_mask`: 2D dealiasing mask for omega equation RHS. If `nothing` (default),
  a standard 2/3-rule mask is computed automatically to avoid aliasing in the
  quadratic Jacobian term.

# Fortran Correspondence
Matches omega equation solver in the Fortran implementation.
"""
function compute_vertical_velocity!(S::State, G::Grid, plans, params; 
                            N2_profile=nothing, workspace=nothing, dealias_mask=nothing)
    # Compute default dealiasing mask if not provided
    # The omega equation involves a quadratic Jacobian J(ψ, ∇²ψ) that needs dealiasing
    if dealias_mask === nothing
        dealias_mask = PARENT.dealias_mask(G)
    end

    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(S.psi, G)

    if need_transpose
        _compute_vertical_velocity_2d!(S, G, plans, params, N2_profile, workspace, dealias_mask)
    else
        _compute_vertical_velocity_direct!(S, G, plans, params, N2_profile, dealias_mask)
    end
    return S
end

# Direct computation when z is fully local (serial or 1D decomposition)
function _compute_vertical_velocity_direct!(S::State, G::Grid, plans, params, N2_profile, dealias_mask)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    w_arr = parent(S.w)
    nz_local, nx_local, ny_local = size(parent(S.psi))

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local for direct solve"

    # Get RHS of omega equation with proper dealiasing
    # Previous code never passed dealias_mask, causing aliasing in the quadratic RHS term
    rhsk = similar(S.psi)
    PARENT.Diagnostics.omega_eqn_rhs!(rhsk, S.psi, G, plans; Lmask=dealias_mask)
    rhsk_arr = parent(rhsk)

    # Get stratification parameters
    if params !== nothing && hasfield(typeof(params), :f₀)
        f = params.f₀
    else
        f = 1.0  # Default
    end

    # Get N² value from params (default to 1.0 if not available)
    N2_const = if params !== nothing && hasfield(typeof(params), :N²)
        params.N²
    else
        1.0
    end

    # Get N² profile - use provided profile, or create constant profile from params.N²
    N2_profile = _coerce_N2_profile(N2_profile, N2_const, nz, G)

    # Solve the full omega equation: ∇²w + (f²/N²)(∂²w/∂z²) = (2f/N²) J(ψ_z, ∇²ψ)
    # Note: The RHS from omega_eqn_rhs! is 2 J(ψ_z, ∇²ψ), so we multiply by f/N² below
    wk = similar(S.psi)
    wk_arr = parent(wk)
    fill!(wk_arr, 0.0)

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f² = f^2

    # Pre-allocate work arrays outside loop to reduce GC pressure
    n_interior = nz - 2  # Interior points (constant for all wavenumbers)
    if n_interior > 0
        d = zeros(Float64, n_interior)
        dₗ = zeros(Float64, n_interior-1)
        dᵤ = zeros(Float64, n_interior-1)
        rhs = zeros(eltype(S.psi), n_interior)
        dₗ_work = zeros(Float64, n_interior-1)
        d_work = zeros(Float64, n_interior)
        dᵤ_work = zeros(Float64, n_interior-1)
        rhsᵣ = zeros(Float64, n_interior)
        rhsᵢ = zeros(Float64, n_interior)
        solᵣ = zeros(Float64, n_interior)
        solᵢ = zeros(Float64, n_interior)
    end

    # For each LOCAL horizontal wavenumber (kₓ, kᵧ), solve tridiagonal system
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, wk)
        j_global = local_to_global(j_local, 3, wk)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        if kₕ² > 0 && nz > 2  # Need at least 3 levels for tridiagonal
            if n_interior > 0
                # Fill tridiagonal system (reusing pre-allocated arrays)
                # ∇²w + (f²/N²)(∂²w/∂z²) = (2f/N²) J(ψ_z, ∇²ψ)
                # In spectral space: -kₕ²·w + (f²/N²)·∂²w/∂z² = RHS
                # Centered second derivative: ∂²w/∂z² ≈ (w[k+1] - 2w[k] + w[k-1])/Δz²
                fill!(dₗ, 0.0); fill!(dᵤ, 0.0)
                for iz in 1:n_interior
                    k = iz + 1  # Actual z-level (2 to nz-1)
                    # Correct coefficient: f²/N² (not N²/f²)
                    coeff_z = (f²/N2_profile[k])/(Δz*Δz)
                    d[iz] = -2*coeff_z - kₕ²  # Diagonal
                    if iz > 1
                        dₗ[iz-1] = coeff_z    # Sub-diagonal
                    end
                    if iz < n_interior
                        dᵤ[iz] = coeff_z      # Super-diagonal
                    end
                    # RHS scaling: omega_eqn_rhs! gives 2 J(...), we need (2f/N²) J(...)
                    # So multiply by f/N²
                    rhs[iz] = (f/N2_profile[k]) * rhsk_arr[k, i_local, j_local]
                end

                # Solve tridiagonal system - real and imaginary parts separately
                dₗ_work .= dₗ
                d_work .= d
                dᵤ_work .= dᵤ
                @inbounds for iz in 1:n_interior
                    rhsᵣ[iz] = real(rhs[iz])
                    rhsᵢ[iz] = imag(rhs[iz])
                end

                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵣ)
                    solᵣ .= rhsᵣ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (real part) at kx=$(kₓ), ky=$(kᵧ): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                dₗ_work .= dₗ
                d_work .= d
                dᵤ_work .= dᵤ
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵢ)
                    solᵢ .= rhsᵢ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (imag part) at kx=$(kₓ), ky=$(kᵧ): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                for iz in 1:n_interior
                    k = iz + 1
                    wk_arr[k, i_local, j_local] = solᵣ[iz] + im * solᵢ[iz]
                end
            end

        elseif kₕ² > 0 && nz <= 2
            # With rigid lid BCs (w=0 at top and bottom), there are no interior points
            # for nz <= 2. The only consistent solution is w=0 everywhere.
            # Previous code incorrectly assigned w = -rhs/kh², violating BCs.
            for k in 1:nz
                wk_arr[k, i_local, j_local] = 0.0
            end
        end
    end

    # Transform to real space
    tmpw = _allocate_fft_dst(wk, plans)
    fft_backward!(tmpw, wk, plans)
    tmpw_arr = parent(tmpw)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[k, i_local, j_local] = real(tmpw_arr[k, i_local, j_local])
    end
end

# 2D decomposition version with transposes
function _compute_vertical_velocity_2d!(S::State, G::Grid, plans, params, N2_profile, workspace, dealias_mask)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get stratification parameters
    if params !== nothing && hasfield(typeof(params), :f₀)
        f = params.f₀
    else
        f = 1.0
    end

    # Get N² value from params (default to 1.0 if not available)
    N2_const = if params !== nothing && hasfield(typeof(params), :N²)
        params.N²
    else
        1.0
    end

    # Get N² profile - use provided profile, or create constant profile from params.N²
    N2_profile = _coerce_N2_profile(N2_profile, N2_const, nz, G)

    # Allocate z-pencil workspace
    work_z = workspace !== nothing && hasfield(typeof(workspace), :work_z) ? workspace.work_z : allocate_z_pencil(G, ComplexF64)
    wk_z = allocate_z_pencil(G, ComplexF64)

    # Step 1: Compute RHS in xy-pencil configuration with proper dealiasing
    rhsk = similar(S.psi)
    PARENT.Diagnostics.omega_eqn_rhs!(rhsk, S.psi, G, plans; Lmask=dealias_mask)

    # Step 2: Transpose RHS to z-pencil
    transpose_to_z_pencil!(work_z, rhsk, G)

    # Step 3: Solve tridiagonal system on z-pencil (z now fully local)
    rhsk_z_arr = parent(work_z)
    wk_z_arr = parent(wk_z)
    fill!(wk_z_arr, 0.0)

    nz_local, nx_local_z, ny_local_z = size(rhsk_z_arr)
    @assert nz_local == nz "Z must be fully local in z-pencil"

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f² = f^2

    # Pre-allocate work arrays outside loop to reduce GC pressure
    n_interior = nz - 2  # Interior points (constant for all wavenumbers)
    if n_interior > 0
        d  = zeros(Float64, n_interior)
        dₗ = zeros(Float64, n_interior-1)
        dᵤ = zeros(Float64, n_interior-1)

        rhs = zeros(ComplexF64, n_interior)

        dₗ_work = zeros(Float64, n_interior-1)
        d_work  = zeros(Float64, n_interior)
        dᵤ_work = zeros(Float64, n_interior-1)

        rhsᵣ = zeros(Float64, n_interior)
        rhsᵢ = zeros(Float64, n_interior)
        solᵣ = zeros(Float64, n_interior)
        solᵢ = zeros(Float64, n_interior)
    end

    @inbounds for j_local in 1:ny_local_z, i_local in 1:nx_local_z
        i_global = local_to_global_z(i_local, 2, G)
        j_global = local_to_global_z(j_local, 3, G)
        
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        if kₕ² > 0 && nz > 2
            if n_interior > 0
                # Fill tridiagonal system (reusing pre-allocated arrays)
                # ∇²w + (f²/N²)(∂²w/∂z²) = (2f/N²) J(ψ_z, ∇²ψ)
                # In spectral space: -kₕ²·w + (f²/N²)·∂²w/∂z² = RHS
                # Centered second derivative: ∂²w/∂z² ≈ (w[k+1] - 2w[k] + w[k-1])/Δz²
                fill!(dₗ, 0.0); fill!(dᵤ, 0.0)
                for iz in 1:n_interior
                    k = iz + 1
                    # Correct coefficient: f²/N² (not N²/f²)
                    coeff_z = (f²/N2_profile[k])/(Δz*Δz)
                    d[iz] = -2*coeff_z - kₕ²  # Diagonal
                    if iz > 1
                        dₗ[iz-1] = coeff_z    # Sub-diagonal
                    end
                    if iz < n_interior
                        dᵤ[iz] = coeff_z      # Super-diagonal
                    end
                    # RHS scaling: omega_eqn_rhs! gives 2 J(...), we need (2f/N²) J(...)
                    rhs[iz] = (f/N2_profile[k]) * rhsk_z_arr[k, i_local, j_local]
                end

                dₗ_work .= dₗ
                d_work .= d
                dᵤ_work .= dᵤ
                @inbounds for iz in 1:n_interior
                    rhsᵣ[iz] = real(rhs[iz])
                    rhsᵢ[iz] = imag(rhs[iz])
                end

                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵣ)
                    solᵣ .= rhsᵣ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (real part, 2D decomp) at kx=$(G.kx[i_global]), ky=$(G.ky[j_global]): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                dₗ_work .= dₗ
                d_work .= d
                dᵤ_work .= dᵤ
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵢ)
                    solᵢ .= rhsᵢ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (imag part, 2D decomp) at kx=$(G.kx[i_global]), ky=$(G.ky[j_global]): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                for iz in 1:n_interior
                    k = iz + 1
                    wk_z_arr[k, i_local, j_local] = solᵣ[iz] + im * solᵢ[iz]
                end
            end

        elseif kₕ² > 0 && nz <= 2
            # With rigid lid BCs (w=0 at top and bottom), there are no interior points
            # for nz <= 2. The only consistent solution is w=0 everywhere.
            # Previous code incorrectly assigned w = -rhs/kh², violating BCs.
            for k in 1:nz
                wk_z_arr[k, i_local, j_local] = 0.0
            end
        end
    end

    # Step 4: Transpose result back to xy-pencil
    wk = similar(S.psi)
    transpose_to_xy_pencil!(wk, wk_z, G)

    # Step 5: Transform to real space
    tmpw = _allocate_fft_dst(wk, plans)
    fft_backward!(tmpw, wk, plans)
    tmpw_arr = parent(tmpw)
    w_arr = parent(S.w)
    nz_local, nx_local, ny_local = size(tmpw_arr)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    # Use nz_local (not global nz) for MPI-distributed arrays
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[k, i_local, j_local] = real(tmpw_arr[k, i_local, j_local])
    end
end

#=
================================================================================
                    YBJ VERTICAL VELOCITY
================================================================================
Wave-induced vertical motion from the YBJ+ formulation.
================================================================================
=#

"""
    compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=nothing, workspace=nothing, skip_inversion=false, t=nothing)

Compute vertical velocity from near-inertial wave envelope using YBJ+ formulation.

# Physical Background
Near-inertial waves induce vertical motion through the modulation of their
envelope. Following Asselin & Young (2019) equation (2.10):

```
w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.
```

where:
- A is the true wave amplitude (recovered from evolved B via L⁺A = B)
- A_{zs} = ∂_s(A_z) = (1/2)(∂_x - i∂_y)(A_z) is the complex horizontal derivative
- The complex derivative uses s = x + iy, so ∂_s = (1/2)(∂_x - i∂_y)
- c.c. denotes complex conjugate

Expanding this gives:
```
w = -(f²/N²) * [cos(ft)·(Re(∂A_z/∂x) + Im(∂A_z/∂y)) + sin(ft)·(Im(∂A_z/∂x) - Re(∂A_z/∂y))]
```

# Physical Interpretation
This represents vertical motion induced by:
- Horizontal gradients in the wave envelope's vertical structure
- Wave packet propagation and refraction
- Strong w occurs where wave amplitude varies both horizontally and vertically
- The velocity oscillates at inertial frequency f

# Algorithm
1. **A Recovery**: Solve L⁺A = B using invert_B_to_A!
   - L⁺ is the YBJ+ elliptic operator
   - Tridiagonal solver in z for each horizontal wavenumber

2. **Vertical Derivative**: Compute A_z = ∂A/∂z
   - Uses second-order finite differences

3. **Horizontal Gradients**: Compute ∂(A_z)/∂x, ∂(A_z)/∂y
   - Spectral differentiation: multiply by i kₓ, i kᵧ

4. **Combine**: Apply equation (2.10) with oscillating e^{-ift} factor

# Arguments
- `S::State`: State with B (input) and w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters (f₀)
- `N2_profile::Vector`: Optional N²(z) profile (default: constant N² = 1)
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `skip_inversion::Bool`: If true, skip B→A re-inversion and use existing S.A, S.C.
  Use this when A/C were already computed in the timestep with the correct stratification.
  Default: false (re-inverts for safety).
- `t::Real`: Current time for computing the oscillating velocity e^{-ift}.
  If not provided (default), computes only the cosine component (t=0 snapshot).

# Fortran Correspondence
Matches YBJ vertical velocity computation in the Fortran implementation.

# Important Note
When `skip_inversion=false` (default), this function re-inverts B→A using the provided
N2_profile (or constant N² fallback). If the timestep computed A/C with a different
stratification, this will give inconsistent results. For runs with nonuniform N²,
either:
1. Pass `skip_inversion=true` and ensure A/C are already computed correctly
2. Pass the exact same N2_profile that was used in the timestep

# References
- Asselin & Young (2019), J. Fluid Mech. 876, 428-448, equation (2.10)
"""
function compute_ybj_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing, workspace=nothing, skip_inversion=false, t=nothing)
    # Warn about potential stratification inconsistency
    # If skip_inversion=false and no N2_profile provided, we'll re-invert B→A with constant N².
    # This can give inconsistent results if the simulation uses variable stratification.
    if !skip_inversion && N2_profile === nothing
        @warn "compute_ybj_vertical_velocity!: Re-inverting B→A with constant N² (no N2_profile provided). " *
              "If your simulation uses variable stratification, pass N2_profile or use skip_inversion=true " *
              "to avoid inconsistent vertical velocity." maxlog=1
    end

    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(S.A, G)

    if need_transpose
        _compute_ybj_vertical_velocity_2d!(S, G, plans, params, N2_profile, workspace, skip_inversion, t)
    else
        _compute_ybj_vertical_velocity_direct!(S, G, plans, params, N2_profile, skip_inversion, t)
    end
    return S
end

# Direct computation when z is fully local (serial or 1D decomposition)
function _compute_ybj_vertical_velocity_direct!(S::State, G::Grid, plans, params, N2_profile, skip_inversion, t)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    w_arr = parent(S.w)

    # Get parameters - need f and N² profile
    if params !== nothing && hasfield(typeof(params), :f₀)
        f = params.f₀
    else
        f = 1.0  # Default
    end

    # Get N² value from params (default to 1.0 if not available)
    N2_const = if params !== nothing && hasfield(typeof(params), :N²)
        params.N²
    else
        1.0
    end

    # Get N² profile - use provided profile, or create constant profile from params.N²
    N2_profile = _coerce_N2_profile(N2_profile, N2_const, nz, G)

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A using centralized YBJ+ inversion
    # a(z) = f²/N²(z) is the elliptic coefficient
    if skip_inversion
        # Use existing S.A and S.C computed by the timestep with correct stratification.
        # This avoids re-inverting with a potentially different (constant) N² profile.
        # Note: S.A being all zeros is valid (e.g., no waves), so we just warn if unexpected
        if all(iszero, parent(S.A))
            @warn "skip_inversion=true but S.A is all zeros - vertical velocity will be zero" maxlog=1
        end
    else
        # Re-invert B→A with the given N² profile.
        # WARNING: If the timestep computed A with a different stratification,
        # this will give inconsistent results.
        a_vec = similar(G.z)
        f_sq = f^2
        @inbounds for k in eachindex(a_vec)
            a_vec[k] = f_sq / N2_profile[k]  # a = f²/N²
        end
        invert_B_to_A!(S, G, params, a_vec)
    end
    # Step 2: Compute vertical derivative A_z using finite differences
    Aₖ_z = S.C  # C was set to A_z by invert_B_to_A!
    Aₖ_z_arr = parent(Aₖ_z)
    nz_spec, nx_spec, ny_spec = size(Aₖ_z_arr)

    # Verify z is fully local on the spectral pencil (MPI input/output pencils can differ)
    @assert nz_spec == nz "Vertical dimension must be fully local for direct solve"

    # Step 3: Compute horizontal derivatives of A_z
    dAz_dxₖ = similar(Aₖ_z)
    dAz_dyₖ = similar(Aₖ_z)
    dAz_dxₖ_arr = parent(dAz_dxₖ)
    dAz_dyₖ_arr = parent(dAz_dyₖ)

    # Compute derivatives for k = 1:(nz-1) where A_z is defined
    @inbounds for k in 1:(nz-1), j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, dAz_dxₖ)
        j_global = local_to_global(j_local, 3, dAz_dxₖ)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        dAz_dxₖ_arr[k, i_local, j_local] = ikₓ * Aₖ_z_arr[k, i_local, j_local]
        dAz_dyₖ_arr[k, i_local, j_local] = ikᵧ * Aₖ_z_arr[k, i_local, j_local]
    end

    # Zero the top slice (k=nz) to avoid garbage from similar() affecting fft_backward!
    # Without this, the uninitialized data can inject NaNs/noise into the transform.
    @inbounds for j_local in 1:ny_spec, i_local in 1:nx_spec
        dAz_dxₖ_arr[nz, i_local, j_local] = 0
        dAz_dyₖ_arr[nz, i_local, j_local] = 0
    end

    # Step 4: Compute YBJ vertical velocity in PHYSICAL space
    # From Asselin & Young (2019) equation (2.10):
    #   w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.
    #
    # where A_{zs} = ∂_s(A_z) = (1/2)(∂_x - i∂_y)(A_z) = (1/2)[(∂A_z/∂x) - i(∂A_z/∂y)]
    #
    # Expanding A_{zs} = A_r + i*A_i where:
    #   A_r = (1/2)[Re(∂A_z/∂x) + Im(∂A_z/∂y)]  (real part of A_{zs})
    #   A_i = (1/2)[Im(∂A_z/∂x) - Re(∂A_z/∂y)]  (imaginary part of A_{zs})
    #
    # The full oscillating velocity is:
    #   w = -(f²/N²) * 2 * Re(A_{zs} * e^{-ift})
    #     = -(f²/N²) * [cos(ft)·(Re(∂A_z/∂x) + Im(∂A_z/∂y)) + sin(ft)·(Im(∂A_z/∂x) - Re(∂A_z/∂y))]

    # Transform horizontal derivatives to physical space
    dAz_dx_phys = _allocate_fft_dst(dAz_dxₖ, plans)
    dAz_dy_phys = _allocate_fft_dst(dAz_dyₖ, plans)
    fft_backward!(dAz_dx_phys, dAz_dxₖ, plans)
    fft_backward!(dAz_dy_phys, dAz_dyₖ, plans)
    dAz_dx_phys_arr = parent(dAz_dx_phys)
    dAz_dy_phys_arr = parent(dAz_dy_phys)
    nz_phys, nx_phys, ny_phys = size(dAz_dx_phys_arr)

    @assert size(w_arr) == (nz_phys, nx_phys, ny_phys) "Physical pencils for w and FFT output must match"

    # Compute oscillation factors if time is provided
    if t !== nothing
        cos_ft = cos(f * t)
        sin_ft = sin(f * t)
    else
        # If no time provided, use t=0 (cosine term only)
        cos_ft = 1.0
        sin_ft = 0.0
    end

    # Compute w in physical space using equation (2.10)
    @inbounds for k in 1:(nz_phys-1), j_local in 1:ny_phys, i_local in 1:nx_phys
        k_out = k + 1  # Shift to match output grid
        N²ₗ = N2_profile[k_out]
        ybj_factor = -(f^2) / N²ₗ

        # Get derivatives in physical space
        dAz_dx = dAz_dx_phys_arr[k, i_local, j_local]
        dAz_dy = dAz_dy_phys_arr[k, i_local, j_local]

        # Cosine coefficient: Re(∂A_z/∂x) + Im(∂A_z/∂y)
        w_cos = real(dAz_dx) + imag(dAz_dy)
        # Sine coefficient: Im(∂A_z/∂x) - Re(∂A_z/∂y)
        w_sin = imag(dAz_dx) - real(dAz_dy)

        # Full oscillating velocity from eq (2.10)
        w_arr[k_out, i_local, j_local] = ybj_factor * (cos_ft * w_cos + sin_ft * w_sin)
    end

    # Apply boundary conditions: w = 0 at top and bottom
    @inbounds for j_local in 1:ny_phys, i_local in 1:nx_phys
        w_arr[1, i_local, j_local] = 0.0
        if nz > 1
            w_arr[nz, i_local, j_local] = 0.0
        end
    end
end

# 2D decomposition version with transposes
function _compute_ybj_vertical_velocity_2d!(S::State, G::Grid, plans, params, N2_profile, workspace, skip_inversion, t)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get parameters
    if params !== nothing && hasfield(typeof(params), :f₀)
        f = params.f₀
    else
        f = 1.0
    end

    # Get N² value from params (default to 1.0 if not available)
    N2_const = if params !== nothing && hasfield(typeof(params), :N²)
        params.N²
    else
        1.0
    end

    # Get N² profile - use provided profile, or create constant profile from params.N²
    N2_profile = _coerce_N2_profile(N2_profile, N2_const, nz, G)

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A (invert_B_to_A! handles 2D decomposition internally)
    # a(z) = f²/N²(z) is the elliptic coefficient
    if skip_inversion
        # Use existing S.A and S.C computed by the timestep with correct stratification.
        # Note: S.A being all zeros is valid (e.g., no waves), so we just warn if unexpected
        if all(iszero, parent(S.A))
            @warn "skip_inversion=true but S.A is all zeros - vertical velocity will be zero" maxlog=1
        end
    else
        # Re-invert B→A with the given N² profile.
        # WARNING: If the timestep computed A with a different stratification,
        # this will give inconsistent results.
        a_vec = similar(G.z)
        f_sq = f^2
        @inbounds for k in eachindex(a_vec)
            a_vec[k] = f_sq / N2_profile[k]  # a = f²/N²
        end
        # Pass workspace if available
        invert_B_to_A!(S, G, params, a_vec; workspace=workspace)
    end

    # Now A and C (A_z) are in xy-pencil form.
    # invert_B_to_A! already computed A_z and stored it in S.C during the z-pencil phase.
    # Use S.C directly instead of recomputing - this ensures MPI and serial paths match,
    # and respects skip_inversion=true which promises to reuse precomputed A/C.
    Aₖ_z = S.C
    Aₖ_z_arr = parent(Aₖ_z)

    # Compute horizontal derivatives of A_z in xy-pencil
    nz_local, nx_local, ny_local = size(Aₖ_z_arr)
    @assert nz_local == nz "Vertical dimension must be fully local for YBJ vertical velocity"
    dAz_dxₖ = similar(Aₖ_z)
    dAz_dyₖ = similar(Aₖ_z)
    dAz_dxₖ_arr = parent(dAz_dxₖ)
    dAz_dyₖ_arr = parent(dAz_dyₖ)

    # Compute derivatives for k = 1:(nz-1) where A_z is defined
    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, dAz_dxₖ)
        j_global = local_to_global(j_local, 3, dAz_dxₖ)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        dAz_dxₖ_arr[k, i_local, j_local] = ikₓ * Aₖ_z_arr[k, i_local, j_local]
        dAz_dyₖ_arr[k, i_local, j_local] = ikᵧ * Aₖ_z_arr[k, i_local, j_local]
    end

    # Zero the top slice (k=nz) to avoid garbage from similar() affecting fft_backward!
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        dAz_dxₖ_arr[nz, i_local, j_local] = 0
        dAz_dyₖ_arr[nz, i_local, j_local] = 0
    end

    # Compute YBJ vertical velocity in PHYSICAL space (2D decomposition version)
    # From Asselin & Young (2019) equation (2.10):
    #   w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.
    #
    # where A_{zs} = (1/2)[(∂A_z/∂x) - i(∂A_z/∂y)]
    #
    # Full formula:
    #   w = -(f²/N²) * [cos(ft)·(Re(∂A_z/∂x) + Im(∂A_z/∂y)) + sin(ft)·(Im(∂A_z/∂x) - Re(∂A_z/∂y))]

    # Transform horizontal derivatives to physical space
    dAz_dx_phys = _allocate_fft_dst(dAz_dxₖ, plans)
    dAz_dy_phys = _allocate_fft_dst(dAz_dyₖ, plans)
    fft_backward!(dAz_dx_phys, dAz_dxₖ, plans)
    fft_backward!(dAz_dy_phys, dAz_dyₖ, plans)
    dAz_dx_phys_arr = parent(dAz_dx_phys)
    dAz_dy_phys_arr = parent(dAz_dy_phys)

    w_arr = parent(S.w)

    # Compute oscillation factors if time is provided
    if t !== nothing
        cos_ft = cos(f * t)
        sin_ft = sin(f * t)
    else
        # If no time provided, use t=0 (cosine term only)
        cos_ft = 1.0
        sin_ft = 0.0
    end

    # Compute w in physical space using equation (2.10)
    nz_phys, nx_phys, ny_phys = size(dAz_dx_phys_arr)
    @inbounds for k in 1:(nz_phys-1), j_local in 1:ny_phys, i_local in 1:nx_phys
        k_out = k + 1
        N²ₗ = N2_profile[k_out]
        ybj_factor = -(f^2) / N²ₗ

        # Get derivatives in physical space
        dAz_dx = dAz_dx_phys_arr[k, i_local, j_local]
        dAz_dy = dAz_dy_phys_arr[k, i_local, j_local]

        # Cosine coefficient: Re(∂A_z/∂x) + Im(∂A_z/∂y)
        w_cos = real(dAz_dx) + imag(dAz_dy)
        # Sine coefficient: Im(∂A_z/∂x) - Re(∂A_z/∂y)
        w_sin = imag(dAz_dx) - real(dAz_dy)

        # Full oscillating velocity from eq (2.10)
        w_arr[k_out, i_local, j_local] = ybj_factor * (cos_ft * w_cos + sin_ft * w_sin)
    end

    # Apply boundary conditions
    @inbounds for j_local in 1:ny_phys, i_local in 1:nx_phys
        w_arr[1, i_local, j_local] = 0.0
        if nz > 1
            w_arr[nz, i_local, j_local] = 0.0
        end
    end
end

#=
================================================================================
                    TOTAL VELOCITY FOR LAGRANGIAN ADVECTION
================================================================================
For particle tracking, we need the complete velocity field including both
geostrophic flow and wave-induced motion.
================================================================================
=#

"""
    compute_total_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing, dealias_mask=nothing, include_wave_velocity=true)

Compute the TOTAL velocity field for Lagrangian particle advection.

# Physical Background
In QG-YBJ+ dynamics, a particle is advected by:
1. **Geostrophic flow**: u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x
2. **Wave velocity**: From YBJ+ equation (1.2): u + iv = e^{-ift} LA
3. **Wave-induced Stokes drift**: Second-order drift from near-inertial waves

The total velocity is:
```
u_total = u_QG + u_wave + u_S
v_total = v_QG + v_wave + v_S
w_total = w_QG + w_S (from omega equation or YBJ, plus vertical Stokes drift)
```

# Wave Velocity (Asselin & Young 2019, eq. 1.2)
The backrotated wave velocity is LA, where L = ∂_z(f²/N²)∂_z:
```
u_wave = Re(LA)
v_wave = Im(LA)
```
For YBJ+: B = L⁺A where L⁺ = L + (1/4)Δ, so LA = B - (1/4)ΔA.
In spectral space: LA = B + (k_h²/4)A

# Wave-Induced Stokes Drift
Following Wagner & Young (2016) equations (3.16a)-(3.20), the Stokes drift
uses the full Jacobian formulation:

Horizontal Stokes drift (eq. 3.16a, 3.18):
```
J₀ = (LA)* ∂_{s*}(LA) - (f₀²/N²)(∂_{s*} A_z*) ∂_z(LA)
u_S = Im(J₀)/f₀
v_S = -Re(J₀)/f₀
```
where ∂_{s*} = (1/2)(∂_x + i∂_y).

Vertical Stokes drift (eq. 3.19-3.20):
```
K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}   where M = (f₀²/N²)A_z
w_S = -2·Im(K₀)/f₀
```
with M*_z = a_z·A_z* + a·A_{zz}*, M_{ss*} = (a/4)·Δ_H(A_z),
M*_{s*} = a·(A_{zs})*, M_{sz} = a_z·A_{zs} + a·A_{zzs}, and a = f₀²/N².

# Usage
For Lagrangian particle advection, always use this function rather than
`compute_velocities!` to include wave effects.

# Arguments
- `S::State`: State with ψ, A, B (input) and u, v, w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters
- `compute_w::Bool`: If true, compute vertical velocity
- `use_ybj_w::Bool`: If true, use YBJ formula for w
- `N2_profile::Vector`: Optional N²(z) profile for vertical velocity computation
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `dealias_mask`: Optional 2D dealiasing mask for omega equation RHS
- `include_wave_velocity::Bool`: If true (default), include wave velocity Re(LA), Im(LA)

# Returns
Modified State with total velocity fields u, v, w.
"""
function compute_total_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing, dealias_mask=nothing, include_wave_velocity=true)
    # First compute QG velocities (pass dealias_mask for omega equation RHS dealiasing)
    compute_velocities!(S, G; plans=plans, params=params, compute_w=compute_w, use_ybj_w=use_ybj_w, N2_profile=N2_profile, workspace=workspace, dealias_mask=dealias_mask)

    # Add wave velocity and Stokes drift (respecting compute_w for vertical component)
    # Pass N2_profile for the second term in the Jacobian (f²/N²)
    compute_wave_velocities!(S, G; plans=plans, params=params, compute_w=compute_w, include_wave_velocity=include_wave_velocity, N2_profile=N2_profile)

    return S
end

#=
================================================================================
                    WAVE-INDUCED STOKES DRIFT
================================================================================
Near-inertial waves induce a Lagrangian drift in the direction of wave
propagation. This is the Stokes drift correction.
================================================================================
=#

"""
    compute_wave_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, include_wave_velocity=true, N2_profile=nothing)

Compute wave velocities and Stokes drift, adding them to existing QG velocities.

# Operator Definitions (from PDF)
    L  (YBJ operator):   L  = ∂/∂z(f²/N² ∂/∂z)              [eq. (4)]
    L⁺ (YBJ+ operator):  L⁺ = L - k_h²/4                     [spectral space]

Key relation: L = L⁺ + k_h²/4

# Physical Background
Near-inertial waves contribute to particle advection through two mechanisms:

1. **Wave velocity**: Following Asselin & Young (2019) YBJ+ equation (1.2):
   u + iv = e^{-ift} (LA)                                    [eq. (3)]
   where L is the YBJ operator (NOT L⁺).

   The backrotated velocity (phase-averaged) is LA:
   - u_wave = Re(LA)
   - v_wave = Im(LA)

   Since B = L⁺A and L = L⁺ + k_h²/4:
   LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A                      [spectral space]

2. **Stokes drift**: Following Wagner & Young (2016) equation (3.16a), the
   horizontal Stokes drift is computed from the full Jacobian:

   J₀ = ∂(M*, M_z̃)/∂(z̃, s*) = (LA)* ∂_{s*}(LA) - M*_{s*} (M_z̃)_{z̃}

   where:
   - M = (f²/N²) A_z is the "buoyancy action"
   - M_z̃ = LA is the wave velocity
   - ∂_{s*} = (1/2)(∂_x + i∂_y) is the complex horizontal derivative
   - s* = x - iy is the complex conjugate horizontal coordinate

   The full form (expanding M*_{s*} = (f²/N²) ∂_{s*}(A_z*)):
   J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA)

   From equation (3.18): if₀ U^S = J₀, so:
   - u_S = Im(J₀)/f₀
   - v_S = -Re(J₀)/f₀

   For vertical Stokes drift, equation (3.19)-(3.20) gives the full Jacobian form:
   - if₀w^S = K₀* - K₀, where K₀ = ∂(M*, M_s)/∂(z̃, s*) and M = (f₀²/N²)A_z
   - Expanding: K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}
   - With: M*_z = a_z·A_z* + a·A_{zz}*, M_{ss*} = a·(1/4)Δ_H(A_z),
           M*_{s*} = a·(A_{zs})*, M_{sz} = a_z·A_{zs} + a·A_{zzs}
   - Where a = f₀²/N² and a_z = ∂_z(f₀²/N²)
   - Final: w^S = -2·Im(K₀)/f₀

# Physical Interpretation
- Wave velocity: Direct advection by the wave orbital motion (from backrotated velocity LA)
- Stokes drift: Net drift in the direction of wave propagation
- The second term in J₀ accounts for vertical structure of the wave envelope
- Both contributions are important for accurate Lagrangian transport

# Algorithm
1. Compute LA = B + (k_h²/4)A in spectral space (YBJ+ relation)
2. Compute ∂_{s*}(LA) = (1/2)(∂_x + i∂_y)(LA) in spectral space
3. Compute ∂_{s*}(A_z*) = conj((1/2)(∂_x - i∂_y)(A_z)) using A_z = S.C
4. Compute ∂_z(LA) using finite differences
5. For vertical Stokes drift: compute A_{zz}, Δ_H(A_z), A_{zs}, A_{zzs}, and a_z profile
6. Transform all fields to physical space
7. Compute wave velocity: u_w, v_w = Re(LA), Im(LA)
8. Compute full Jacobian: J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA)
9. Extract horizontal Stokes drift: u_S = Im(J₀)/f₀, v_S = -Re(J₀)/f₀
10. Compute K₀ and extract vertical Stokes drift: w_S = -2·Im(K₀)/f₀
11. Add contributions to existing u, v, w fields (in-place modification)

# Arguments
- `S::State`: State with A, B, C (input) and u, v, w modified (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters (requires f₀ for Stokes drift normalization, N² for stratification)
- `compute_w::Bool`: If true (default), compute and add vertical wave Stokes drift
- `include_wave_velocity::Bool`: If true (default), include wave velocity Re(LA), Im(LA)
- `N2_profile::Vector`: Optional N²(z) profile for Jacobian second term (default: constant from params)

# Note
This function modifies u, v, w in-place by adding wave contributions.
Call after compute_velocities! to get total velocity.

# References
- Asselin & Young (2019), J. Fluid Mech. 876, 428-448, equation (1.2)
- Wagner & Young (2016), J. Fluid Mech. 802, 806-837, equations (3.16a), (3.17)-(3.20)
- Xie & Vanneste (2015), J. Fluid Mech. 774, 143-169
"""
function compute_wave_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, include_wave_velocity=true, N2_profile=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    u_arr = parent(S.u)
    v_arr = parent(S.v)
    w_arr = parent(S.w)
    Aₖ_arr = parent(S.A)
    L⁺Aₖ_arr = parent(S.L⁺A)
    Aₖ_z_arr = parent(S.C)  # A_z = ∂A/∂z computed by invert_L⁺A_to_A!
    nz_local, nx_local, ny_local = size(Aₖ_arr)

    # Get f₀ for Stokes drift normalization (Wagner & Young 2016, eq 3.18)
    f₀ = if params !== nothing && hasfield(typeof(params), :f₀)
        params.f₀
    else
        1.0  # Default fallback
    end

    # Get N² value from params (default to 1.0 if not available)
    N2_const = if params !== nothing && hasfield(typeof(params), :N²)
        params.N²
    else
        1.0
    end

    # Get N² profile - use provided profile, or create constant profile from params.N²
    N2_profile_local = _coerce_N2_profile(N2_profile, N2_const, nz, G)

    # Set up plans if needed
    if plans === nothing
        plans = plan_transforms!(G)
    end

    # Compute LA, ∂_{s*}(LA), and ∂_{s*}(A_z) in spectral space
    # YBJ+ relation: B = L⁺A = LA + (1/4)ΔA, so LA = B - (1/4)ΔA
    # In spectral space: Δ → -k_h², so LA = B + (k_h²/4)A
    #
    # ∂_{s*} = (1/2)(∂_x + i∂_y), so in spectral space:
    # ∂_{s*} → (1/2)(ikₓ + i·ikᵧ) = (i/2)(kₓ - kᵧ)... wait, let me recalculate
    # Actually: ∂_{s*}f → (1/2)(ikₓ f̂ + i·ikᵧ f̂) = (i/2)(kₓ + i kᵧ) f̂ = (i/2) k* f̂
    # where k* = kₓ + i kᵧ is the complex wavenumber conjugate
    LAₖ = similar(S.A)
    dLA_ds_conjₖ = similar(S.A)  # ∂_{s*}(LA)
    dAz_ds_conjₖ = similar(S.A)  # ∂_{s*}(A_z) - will take conj later for ∂_{s*}(A_z*)
    LAₖ_arr = parent(LAₖ)
    dLA_ds_conjₖ_arr = parent(dLA_ds_conjₖ)
    dAz_ds_conjₖ_arr = parent(dAz_ds_conjₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, LAₖ)
        j_global = local_to_global(j_local, 3, LAₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        # Wave velocity from eq (1.2): u + iv = e^{-ift} LA
        # LA = L⁺A + (k_h²/4)A in spectral space (YBJ+ relation)
        LA_val = L⁺Aₖ_arr[k, i_local, j_local] + (kₕ² / 4) * Aₖ_arr[k, i_local, j_local]
        LAₖ_arr[k, i_local, j_local] = LA_val

        # Complex derivative: ∂_{s*} = (1/2)(∂_x + i∂_y)
        # In spectral space: ∂_{s*} f̂ = (1/2)(ikₓ + i²kᵧ) f̂ = (i/2)(kₓ - kᵧ)...
        # Wait: ∂_{s*} = (1/2)(∂_x + i∂_y), so
        # ∂_{s*} f → (1/2)(ikₓ + i·ikᵧ) f̂ = (1/2)(ikₓ - kᵧ) f̂
        # Let's define: ds_conj_factor = (1/2)(ikₓ - kᵧ) = (i/2)(kₓ + ikᵧ)
        # No wait, that's wrong too. Let me be more careful:
        # ∂/∂s* = (1/2)(∂/∂x + i ∂/∂y)
        # Applied to f: ∂f/∂s* = (1/2)(∂f/∂x + i ∂f/∂y)
        # In spectral: = (1/2)(ikₓ f̂ + i·ikᵧ f̂) = (1/2)(ikₓ + i²kᵧ) f̂ = (1/2)(ikₓ - kᵧ) f̂
        ds_conj_factor = 0.5 * (im * kₓ - kᵧ)  # = (1/2)(ikₓ - kᵧ)
        dLA_ds_conjₖ_arr[k, i_local, j_local] = ds_conj_factor * LA_val

        # Horizontal derivative of A_z for second Jacobian term
        # Need ∂_{s*}(A_z*) = conj(∂_{s}(A_z)) where ∂_s = (1/2)(∂_x - i∂_y)
        # But it's easier: ∂_{s*}(A_z*) = (∂_{s*}(A_z))* where we use complex conj
        # Actually: ∂_{s*}(A_z*) = (1/2)(∂_x(A_z*) + i∂_y(A_z*))
        #                       = (1/2)(∂_x(A_z)* + i∂_y(A_z)*)  [derivatives of real coords are linear]
        #                       = ((1/2)(∂_x(A_z) - i∂_y(A_z)))*  [factor out conj]
        #                       = (∂_s(A_z))*
        # where ∂_s = (1/2)(∂_x - i∂_y) is the other complex derivative
        # In spectral: ∂_s f̂ = (1/2)(ikₓ - i·ikᵧ) f̂ = (1/2)(ikₓ + kᵧ) f̂
        ds_factor = 0.5 * (im * kₓ + kᵧ)  # = (1/2)(ikₓ + kᵧ)
        Az_val = Aₖ_z_arr[k, i_local, j_local]
        dAz_ds_conjₖ_arr[k, i_local, j_local] = ds_factor * Az_val  # Will take conj later
    end

    # For vertical Stokes drift (eq 3.19-3.20), we need additional derivatives:
    # K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz} where M = (f₀²/N²)A_z
    # This requires: A_{zz}, Δ_H(A_z), A_{zs}, A_{zzs}, and a_z = ∂_z(f₀²/N²)

    # Compute A_{zz} = ∂²A/∂z² using finite differences on A_z
    Aₖ_zz = similar(S.A)
    Aₖ_zz_arr = parent(Aₖ_zz)
    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        for k in 2:(nz_local-1)
            # Centered difference for A_{zz}
            Aₖ_zz_arr[k, i_local, j_local] = (Aₖ_z_arr[k+1, i_local, j_local] - 2*Aₖ_z_arr[k, i_local, j_local] + Aₖ_z_arr[k-1, i_local, j_local]) / (Δz^2)
        end
        # One-sided at boundaries (second-order forward/backward)
        if nz_local >= 3
            Aₖ_zz_arr[1, i_local, j_local] = (Aₖ_z_arr[3, i_local, j_local] - 2*Aₖ_z_arr[2, i_local, j_local] + Aₖ_z_arr[1, i_local, j_local]) / (Δz^2)
            Aₖ_zz_arr[nz_local, i_local, j_local] = (Aₖ_z_arr[nz_local, i_local, j_local] - 2*Aₖ_z_arr[nz_local-1, i_local, j_local] + Aₖ_z_arr[nz_local-2, i_local, j_local]) / (Δz^2)
        elseif nz_local == 2
            Aₖ_zz_arr[1, i_local, j_local] = 0.0
            Aₖ_zz_arr[2, i_local, j_local] = 0.0
        else
            Aₖ_zz_arr[1, i_local, j_local] = 0.0
        end
    end

    # Compute horizontal Laplacian Δ_H(A_z) and A_{zs}, A_{zzs} in spectral space
    # Δ_H(A_z) = -k_h² · Â_z in spectral space
    # A_{zs} = ∂_s(A_z) = (1/2)(ikₓ + kᵧ) · Â_z  [already computed as dAz_ds_conjₖ]
    # A_{zzs} = ∂_s(A_{zz}) = (1/2)(ikₓ + kᵧ) · Â_{zz}
    Δ_H_Azₖ = similar(S.A)
    A_zzsₖ = similar(S.A)
    Δ_H_Azₖ_arr = parent(Δ_H_Azₖ)
    A_zzsₖ_arr = parent(A_zzsₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, Δ_H_Azₖ)
        j_global = local_to_global(j_local, 3, Δ_H_Azₖ)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        Az_val = Aₖ_z_arr[k, i_local, j_local]
        Azz_val = Aₖ_zz_arr[k, i_local, j_local]

        # Horizontal Laplacian: Δ_H → -k_h² in spectral space
        Δ_H_Azₖ_arr[k, i_local, j_local] = -kₕ² * Az_val

        # ∂_s(A_{zz}) in spectral space: ∂_s → (1/2)(ikₓ + kᵧ)
        ds_factor = 0.5 * (im * kₓ + kᵧ)
        A_zzsₖ_arr[k, i_local, j_local] = ds_factor * Azz_val
    end

    # Compute a_z = ∂_z(f₀²/N²) profile
    # a = f₀²/N², so a_z = -f₀² · (∂N²/∂z) / N⁴
    a_profile = similar(N2_profile_local)
    a_z_profile = similar(N2_profile_local)
    for k in 1:nz
        a_profile[k] = f₀² / N2_profile_local[k]
    end
    # Compute a_z using finite differences
    for k in 2:(nz-1)
        a_z_profile[k] = (a_profile[k+1] - a_profile[k-1]) / (2 * Δz)
    end
    if nz >= 2
        a_z_profile[1] = (a_profile[2] - a_profile[1]) / Δz
        a_z_profile[nz] = (a_profile[nz] - a_profile[nz-1]) / Δz
    else
        a_z_profile[1] = 0.0
    end

    # Compute vertical derivative of LA using finite differences
    # ∂(LA)/∂z for full Jacobian and vertical Stokes drift
    dLA_dzₖ = similar(S.A)
    dLA_dzₖ_arr = parent(dLA_dzₖ)
    # Note: Δz already computed above for A_{zz}

    # Second-order centered differences for interior, one-sided at boundaries
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        for k in 2:(nz_local-1)
            # Centered difference
            dLA_dzₖ_arr[k, i_local, j_local] = (LAₖ_arr[k+1, i_local, j_local] - LAₖ_arr[k-1, i_local, j_local]) / (2 * Δz)
        end
        # One-sided at boundaries
        if nz_local >= 2
            dLA_dzₖ_arr[1, i_local, j_local] = (LAₖ_arr[2, i_local, j_local] - LAₖ_arr[1, i_local, j_local]) / Δz
            dLA_dzₖ_arr[nz_local, i_local, j_local] = (LAₖ_arr[nz_local, i_local, j_local] - LAₖ_arr[nz_local-1, i_local, j_local]) / Δz
        else
            dLA_dzₖ_arr[1, i_local, j_local] = 0.0
        end
    end

    # Transform all fields to physical space
    # The Jacobian J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA) is a product of fields
    # and MUST be computed in physical space, not spectral space
    LAᵣ = _allocate_fft_dst(LAₖ, plans)
    dLA_ds_conjᵣ = _allocate_fft_dst(dLA_ds_conjₖ, plans)
    dAz_ds_conjᵣ = _allocate_fft_dst(dAz_ds_conjₖ, plans)
    dLA_dzᵣ = _allocate_fft_dst(dLA_dzₖ, plans)

    fft_backward!(LAᵣ, LAₖ, plans)
    fft_backward!(dLA_ds_conjᵣ, dLA_ds_conjₖ, plans)
    fft_backward!(dAz_ds_conjᵣ, dAz_ds_conjₖ, plans)
    fft_backward!(dLA_dzᵣ, dLA_dzₖ, plans)

    # Transform additional fields for vertical Stokes drift (eq 3.19-3.20)
    Azᵣ = _allocate_fft_dst(S.C, plans)  # A_z in physical space
    Azzᵣ = _allocate_fft_dst(Aₖ_zz, plans)  # A_{zz} in physical space
    Δ_H_Azᵣ = _allocate_fft_dst(Δ_H_Azₖ, plans)  # Δ_H(A_z) in physical space
    A_zzsᵣ = _allocate_fft_dst(A_zzsₖ, plans)  # A_{zzs} in physical space

    fft_backward!(Azᵣ, S.C, plans)
    fft_backward!(Azzᵣ, Aₖ_zz, plans)
    fft_backward!(Δ_H_Azᵣ, Δ_H_Azₖ, plans)
    fft_backward!(A_zzsᵣ, A_zzsₖ, plans)

    LAᵣ_arr = parent(LAᵣ)
    dLA_ds_conjᵣ_arr = parent(dLA_ds_conjᵣ)
    dAz_ds_conjᵣ_arr = parent(dAz_ds_conjᵣ)
    dLA_dzᵣ_arr = parent(dLA_dzᵣ)
    Azᵣ_arr = parent(Azᵣ)
    Azzᵣ_arr = parent(Azzᵣ)
    Δ_H_Azᵣ_arr = parent(Δ_H_Azᵣ)
    A_zzsᵣ_arr = parent(A_zzsᵣ)

    # Compute wave velocity and Stokes drift in physical space
    # Following Wagner & Young (2016) equation (3.16a):
    #   J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA)
    #
    # From equation (3.18): if₀ U^S = J₀, so:
    #   u_S = Im(J₀)/f₀
    #   v_S = -Re(J₀)/f₀
    #
    # For vertical Stokes drift from eq (3.19)-(3.20):
    #   if₀w^S = K₀* - K₀, where K₀ = ∂(M*, M_s)/∂(z̃, s*) and M = (f₀²/N²)A_z
    #   This expands to: w^S = -2·Im(K₀)/f₀
    inv_f₀ = 1.0 / f₀
    f₀² = f₀^2

    # Add wave velocity and Stokes drift to existing QG velocities in physical space
    nz_phys, nx_phys, ny_phys = size(LAᵣ_arr)
    @inbounds for k in 1:nz_phys, j_local in 1:ny_phys, i_local in 1:nx_phys
        LA_phys = LAᵣ_arr[k, i_local, j_local]
        dLA_ds_conj_phys = dLA_ds_conjᵣ_arr[k, i_local, j_local]
        dAz_ds_phys = dAz_ds_conjᵣ_arr[k, i_local, j_local]  # This is ∂_s(A_z), need conj for ∂_{s*}(A_z*)
        dLA_dz_phys = dLA_dzᵣ_arr[k, i_local, j_local]
        N²ₖ = N2_profile_local[k]

        # Wave velocity from YBJ+ eq (1.2): u + iv = e^{-ift} LA
        # Backrotated velocity (phase-averaged): (u,v)_wave = (Re(LA), Im(LA))
        if include_wave_velocity
            u_arr[k, i_local, j_local] += real(LA_phys)
            v_arr[k, i_local, j_local] += imag(LA_phys)
        end

        # Full Jacobian from Wagner & Young (2016) eq (3.16a):
        # J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA)
        #
        # First term: (LA)* ∂_{s*}(LA)
        term1 = conj(LA_phys) * dLA_ds_conj_phys

        # Second term: (f²/N²)(∂_{s*} A_z*) ∂_z(LA)
        # where ∂_{s*}(A_z*) = conj(∂_s(A_z))
        dAz_ds_conj_phys = conj(dAz_ds_phys)  # ∂_{s*}(A_z*) = (∂_s(A_z))*
        term2 = (f₀² / N²ₖ) * dAz_ds_conj_phys * dLA_dz_phys

        # Full Jacobian
        J₀ = term1 - term2

        # Stokes drift from eq (3.18): if₀ U^S = J₀
        # U^S = u_S + i v_S = J₀ / (if₀) = -i J₀ / f₀
        # So: u_S = Im(J₀)/f₀, v_S = -Re(J₀)/f₀
        u_stokes = inv_f₀ * imag(J₀)
        v_stokes = -inv_f₀ * real(J₀)

        u_arr[k, i_local, j_local] += u_stokes
        v_arr[k, i_local, j_local] += v_stokes

        # Only add vertical Stokes drift if compute_w is requested
        # Following Wagner & Young (2016) equation (3.19)-(3.20):
        #   if₀w^S = K₀* - K₀, where K₀ = ∂(M*, M_s)/∂(z̃, s*) and M = (f₀²/N²)A_z
        #   Since K₀* - K₀ = -2i·Im(K₀), we have: w^S = -2·Im(K₀)/f₀
        #
        # K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}
        # where:
        #   M = a·A_z with a = f₀²/N²
        #   M*_z = a_z·A_z* + a·A_{zz}*
        #   M_{ss*} = a·(1/4)Δ_H(A_z)
        #   M*_{s*} = a·(A_{zs})*
        #   M_{sz} = a_z·A_{zs} + a·A_{zzs}
        if compute_w
            # Get local field values
            Az_phys = Azᵣ_arr[k, i_local, j_local]
            Azz_phys = Azzᵣ_arr[k, i_local, j_local]
            Δ_H_Az_phys = Δ_H_Azᵣ_arr[k, i_local, j_local]
            Azs_phys = dAz_ds_phys  # Already computed: ∂_s(A_z)
            Azzs_phys = A_zzsᵣ_arr[k, i_local, j_local]

            # Get stratification factors
            aₖ = a_profile[k]  # f₀²/N²
            a_zₖ = a_z_profile[k]  # ∂_z(f₀²/N²)

            # Compute K₀ components
            # M*_z = a_z·A_z* + a·A_{zz}*
            M_star_z = a_zₖ * conj(Az_phys) + aₖ * conj(Azz_phys)

            # M_{ss*} = a·(1/4)Δ_H(A_z)
            M_ss_star = aₖ * 0.25 * Δ_H_Az_phys

            # M*_{s*} = a·(A_{zs})*
            M_star_s_star = aₖ * conj(Azs_phys)

            # M_{sz} = a_z·A_{zs} + a·A_{zzs}
            M_sz = a_zₖ * Azs_phys + aₖ * Azzs_phys

            # K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}
            K₀ = M_star_z * M_ss_star - M_star_s_star * M_sz

            # w^S = -2·Im(K₀)/f₀
            w_stokes = -2.0 * inv_f₀ * imag(K₀)
            w_arr[k, i_local, j_local] += w_stokes
        end
    end

    return S
end

#=
================================================================================
                    WAVE DISPLACEMENT FOR GLM PARTICLE ADVECTION
================================================================================
In the GLM (Generalized Lagrangian Mean) framework for particle advection,
particles are advected by the QG (Lagrangian-mean) velocity, and the wave
contribution appears as a displacement ξ rather than an additional velocity.

OPERATOR DEFINITIONS (from PDF):
--------------------------------
L  (YBJ operator):   L  = ∂/∂z(f²/N² ∂/∂z)           [equation (4)]
L⁺ (YBJ+ operator):  L⁺ = L - k_h²/4

KEY RELATIONS:
--------------
- YBJ+ envelope:     B = L⁺A
- Wave velocity:     u + iv = (LA) × e^{-ift}         [equation (3)]

Since L = L⁺ + k_h²/4, the wave velocity amplitude is:
    LA = (L⁺ + k_h²/4)A = L⁺A + (k_h²/4)A = B + (k_h²/4)A

The wave displacement is then:
    ξx + iξy = Re{(LA / (-if)) × e^{-ift}}           [equation (6)]

This function computes LA in physical space and stores it in state.LA_real
and state.LA_imag for interpolation to particle positions.
================================================================================
=#

"""
    compute_wave_displacement!(S, G; plans=nothing, params=nothing)

Compute wave velocity amplitude LA for GLM particle advection.

# Operator Definitions (from PDF)
- L  (YBJ):  L  = ∂/∂z(f²/N² ∂/∂z)                    [equation (4)]
- L⁺ (YBJ+): L⁺ = L - k_h²/4

# Wave Velocity Amplitude
The instantaneous wave velocity is (equation 3):
    u + iv = (LA) × e^{-ift}

where L is the YBJ operator (NOT L⁺). Since B = L⁺A and L = L⁺ + k_h²/4:
    LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A

# Wave Displacement
Particles compute wave displacement via (equation 6):
    ξx + iξy = Re{(LA / (-if)) × e^{-ift}}

# Arguments
- `S::State`: State with B, A (input) and LA_real, LA_imag (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters

# Note
This function should be called after invert_B_to_A! has computed A from B.
The particle advection code then interpolates LA to particle positions and
computes the time-dependent wave displacement ξ.
"""
function compute_wave_displacement!(S::State, G::Grid; plans=nothing, params=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    Aₖ_arr = parent(S.A)
    L⁺Aₖ_arr = parent(S.L⁺A)
    LA_real_arr = parent(S.LA_real)
    LA_imag_arr = parent(S.LA_imag)
    nz_local, nx_local, ny_local = size(Aₖ_arr)

    # Set up plans if needed
    if plans === nothing
        plans = plan_transforms!(G)
    end

    # Compute LA = B + (k_h²/4)A in spectral space
    # YBJ+ relation: B = L⁺A = LA + (1/4)ΔA, so LA = B - (1/4)ΔA
    # In spectral space: Δ → -k_h², so LA = B + (k_h²/4)A
    LAₖ = similar(S.A)
    LAₖ_arr = parent(LAₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, S.A)
        j_global = local_to_global(j_local, 3, S.A)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        # LA = L⁺A + (k_h²/4)A
        LAₖ_arr[k, i_local, j_local] = L⁺Aₖ_arr[k, i_local, j_local] + (kₕ² / 4) * Aₖ_arr[k, i_local, j_local]
    end

    # Transform LA to physical space
    LA_phys = allocate_fft_backward_dst(LAₖ, plans)
    fft_backward!(LA_phys, LAₖ, plans)
    LA_phys_arr = parent(LA_phys)

    # Store real and imaginary parts in state
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(LA_phys_arr)
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        LA_real_arr[k, i, j] = real(LA_phys_arr[k, i, j])
        LA_imag_arr[k, i, j] = imag(LA_phys_arr[k, i, j])
    end

    return S
end

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!, compute_total_velocities!, compute_wave_velocities!, compute_wave_displacement!
