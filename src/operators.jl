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

    ∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)

This is a 3D elliptic PDE solved via:
- Horizontal: spectral differentiation (kₓ², kᵧ²)
- Vertical: tridiagonal solver at each (kₓ, kᵧ)

Physical meaning: w arises from ageostrophic convergence/divergence required
to maintain geostrophic balance as the flow evolves.

YBJ VERTICAL VELOCITY:
----------------------
For wave-induced vertical motion (equation 4 in YBJ papers):

    w = -(f²/N²) [(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.

where:
- A is the wave amplitude (recovered from B via L⁺A = B)
- A_z = ∂A/∂z is the vertical derivative
- c.c. ensures real result

This represents vertical motion induced by wave envelope modulation.

WAVE-INDUCED HORIZONTAL VELOCITIES:
-----------------------------------
The Stokes drift from near-inertial waves:

    u_wave = Re[(∂A*/∂x)A + A*(∂A/∂x)] = 2 Re[A* ∂A/∂x]
    v_wave = Re[(∂A*/∂y)A + A*(∂A/∂y)] = 2 Re[A* ∂A/∂y]

These wave corrections are important for Lagrangian particle advection.

SPECTRAL DIFFERENTIATION:
-------------------------
All derivatives are computed in spectral space:
    ∂f/∂x → i kₓ f̂(k)
    ∂f/∂y → i kᵧ f̂(k)

Vertical derivatives use second-order finite differences.

FORTRAN CORRESPONDENCE:
-----------------------
- compute_velocities! → compute_velo in derivatives.f90
- compute_vertical_velocity! → solve_omega_eqn
- compute_wave_velocities! → wave velocity terms in compute_velo

================================================================================
=#
module Operators

using LinearAlgebra
using ..QGYBJ: Grid, State, local_to_global, get_local_dims
using ..QGYBJ: fft_forward!, fft_backward!, plan_transforms!, compute_wavenumbers!
using ..QGYBJ: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJ: local_to_global_z, allocate_z_pencil
const PARENT = Base.parentmodule(@__MODULE__)

#=
================================================================================
                    GEOSTROPHIC VELOCITY COMPUTATION
================================================================================
The primary diagnostic: horizontal and vertical velocities from streamfunction.
================================================================================
=#

"""
    compute_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing)

Compute geostrophic velocities from the spectral streamfunction ψ̂.

# Physical Equations
Horizontal velocities from geostrophic balance:
```
u = -∂ψ/∂y  →  û(k) = -i kᵧ ψ̂(k)
v =  ∂ψ/∂x  →  v̂(k) =  i kₓ ψ̂(k)
```

Vertical velocity from QG omega equation:
```
∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)
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

# Returns
Modified State with updated u, v, w fields.

# Note
This computes ONLY QG velocities. For Lagrangian advection including wave
effects, use `compute_total_velocities!` instead.

# Fortran Correspondence
Matches `compute_velo` in derivatives.f90.
"""
function compute_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    ψk_arr = parent(S.psi)
    u_arr = parent(S.u)
    v_arr = parent(S.v)
    nx_local, ny_local, nz_local = size(ψk_arr)

    # Spectral differentiation: û = -i ky ψ̂, v̂ = i kx ψ̂
    ψk = S.psi
    uk = similar(ψk)
    vk = similar(ψk)
    uk_arr = parent(uk)
    vk_arr = parent(vk)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        uk_arr[i_local, j_local, k] = -ikᵧ * ψk_arr[i_local, j_local, k]
        vk_arr[i_local, j_local, k] =  ikₓ * ψk_arr[i_local, j_local, k]
    end

    # Inverse FFT to real space
    if plans === nothing
        # Use unified transform planning (handles both serial and parallel)
        plans = plan_transforms!(G)
    end
    tmpu = similar(ψk)
    tmpv = similar(ψk)
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
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        u_arr[i_local, j_local, k] = real(tmpu_arr[i_local, j_local, k])
        v_arr[i_local, j_local, k] = real(tmpv_arr[i_local, j_local, k])
    end

    # Compute vertical velocity if requested
    if compute_w
        if use_ybj_w
            # Use YBJ vertical velocity formulation (equation 4)
            compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=N2_profile, workspace=workspace)
        else
            # Use standard QG omega equation
            compute_vertical_velocity!(S, G, plans, params; N2_profile=N2_profile, workspace=workspace)
        end
    else
        # Set w to zero (leading-order QG approximation)
        w_arr = parent(S.w)
        fill!(w_arr, 0.0)
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
    compute_vertical_velocity!(S, G, plans, params; N2_profile=nothing, workspace=nothing)

Solve the QG omega equation for ageostrophic vertical velocity.

# Physical Background
In quasi-geostrophic dynamics, the leading-order horizontal flow is non-divergent
(∇·u_g = 0). Vertical motion arises from ageostrophic corrections that maintain
thermal wind balance as the flow evolves.

The omega equation relates w to the horizontal flow:
```
∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)
```

where:
- Left side: 3D Laplacian (horizontal + stratification-weighted vertical)
- Right side: Jacobian forcing from vertical shear and vorticity

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
w = 0 at z = 0 and z = Lz (rigid lid and bottom).

# Arguments
- `S::State`: State with ψ (input) and w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters (f₀)
- `N2_profile::Vector`: Optional N²(z) profile (default: constant N² = 1)
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Fortran Correspondence
Matches omega equation solver in the Fortran implementation.
"""
function compute_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing, workspace=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

    if need_transpose
        _compute_vertical_velocity_2d!(S, G, plans, params, N2_profile, workspace)
    else
        _compute_vertical_velocity_direct!(S, G, plans, params, N2_profile)
    end
    return S
end

# Direct computation when z is fully local (serial or 1D decomposition)
function _compute_vertical_velocity_direct!(S::State, G::Grid, plans, params, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    w_arr = parent(S.w)
    nx_local, ny_local, nz_local = size(parent(S.psi))

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local for direct solve"

    # Get RHS of omega equation
    rhsk = similar(S.psi)
    PARENT.Diagnostics.omega_eqn_rhs!(rhsk, S.psi, G, plans)
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
    if N2_profile === nothing
        # Use params.N² as constant profile (not hardcoded 1.0)
        N2_profile = fill(eltype(S.psi)(N2_const), nz)
    else
        # Ensure N2_profile has correct length and type
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=$(N2_const)"
            N2_profile = fill(eltype(S.psi)(N2_const), nz)
        end
    end

    # Solve the full omega equation: ∇²w + (N²/f²)(∂²w/∂z²) = RHS
    wk = similar(S.psi)
    wk_arr = parent(wk)
    fill!(wk_arr, 0.0)

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f² = f^2

    # For each LOCAL horizontal wavenumber (kₓ, kᵧ), solve tridiagonal system
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        if kₕ² > 0 && nz > 2  # Need at least 3 levels for tridiagonal
            n_interior = nz - 2  # Interior points

            if n_interior > 0
                # Tridiagonal matrix coefficients (real-valued)
                d = zeros(Float64, n_interior)      # diagonal
                dₗ = zeros(Float64, n_interior-1)   # lower diagonal
                dᵤ = zeros(Float64, n_interior-1)   # upper diagonal
                rhs = zeros(eltype(S.psi), n_interior)  # RHS vector

                # Fill tridiagonal system
                for iz in 1:n_interior
                    k = iz + 1  # Actual z-level (2 to nz-1)
                    d[iz] = -(N2_profile[k]/f²)/(Δz*Δz) - kₕ²
                    if iz > 1
                        dₗ[iz-1] = (N2_profile[k]/f²)/(Δz*Δz)
                    end
                    if iz < n_interior
                        dᵤ[iz] = (N2_profile[k]/f²)/(Δz*Δz)
                    end
                    rhs[iz] = rhsk_arr[i_local, j_local, k]
                end

                # Solve tridiagonal system - real and imaginary parts separately
                dₗ_work = copy(dₗ)
                d_work = copy(d)
                dᵤ_work = copy(dᵤ)
                rhsᵣ = real.(rhs)
                rhsᵢ = imag.(rhs)

                solᵣ = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵣ)
                    solᵣ .= rhsᵣ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (real part) at kx=$(kₓ), ky=$(kᵧ): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                dₗ_work = copy(dₗ)
                d_work = copy(d)
                dᵤ_work = copy(dᵤ)
                solᵢ = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵢ)
                    solᵢ .= rhsᵢ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (imag part) at kx=$(kₓ), ky=$(kᵧ): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                solution = solᵣ .+ im .* solᵢ
                for iz in 1:n_interior
                    k = iz + 1
                    wk_arr[i_local, j_local, k] = solution[iz]
                end
            end

        elseif kₕ² > 0 && nz <= 2
            for k in 1:nz
                wk_arr[i_local, j_local, k] = -rhsk_arr[i_local, j_local, k] / kₕ²
            end
        end
    end

    # Transform to real space
    tmpw = similar(wk)
    fft_backward!(tmpw, wk, plans)
    tmpw_arr = parent(tmpw)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[i_local, j_local, k] = real(tmpw_arr[i_local, j_local, k])
    end
end

# 2D decomposition version with transposes
function _compute_vertical_velocity_2d!(S::State, G::Grid, plans, params, N2_profile, workspace)
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
    if N2_profile === nothing
        N2_profile = fill(N2_const, nz)
    elseif length(N2_profile) != nz
        @warn "N2_profile length mismatch, using constant N²=$(N2_const)"
        N2_profile = fill(N2_const, nz)
    end

    # Allocate z-pencil workspace
    work_z = workspace !== nothing && hasfield(typeof(workspace), :work_z) ? workspace.work_z : allocate_z_pencil(G, ComplexF64)
    wk_z = allocate_z_pencil(G, ComplexF64)

    # Step 1: Compute RHS in xy-pencil configuration
    rhsk = similar(S.psi)
    PARENT.Diagnostics.omega_eqn_rhs!(rhsk, S.psi, G, plans)

    # Step 2: Transpose RHS to z-pencil
    transpose_to_z_pencil!(work_z, rhsk, G)

    # Step 3: Solve tridiagonal system on z-pencil (z now fully local)
    rhsk_z_arr = parent(work_z)
    wk_z_arr = parent(wk_z)
    fill!(wk_z_arr, 0.0)

    nx_local_z, ny_local_z, nz_local = size(rhsk_z_arr)
    @assert nz_local == nz "Z must be fully local in z-pencil"

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f² = f^2

    @inbounds for j_local in 1:ny_local_z, i_local in 1:nx_local_z
        i_global = local_to_global_z(i_local, 1, G)
        j_global = local_to_global_z(j_local, 2, G)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        if kₕ² > 0 && nz > 2
            n_interior = nz - 2

            if n_interior > 0
                d = zeros(Float64, n_interior)
                dₗ = zeros(Float64, n_interior-1)
                dᵤ = zeros(Float64, n_interior-1)
                rhs = zeros(ComplexF64, n_interior)

                for iz in 1:n_interior
                    k = iz + 1
                    d[iz] = -(N2_profile[k]/f²)/(Δz*Δz) - kₕ²
                    if iz > 1
                        dₗ[iz-1] = (N2_profile[k]/f²)/(Δz*Δz)
                    end
                    if iz < n_interior
                        dᵤ[iz] = (N2_profile[k]/f²)/(Δz*Δz)
                    end
                    rhs[iz] = rhsk_z_arr[i_local, j_local, k]
                end

                dₗ_work = copy(dₗ)
                d_work = copy(d)
                dᵤ_work = copy(dᵤ)
                rhsᵣ = real.(rhs)
                rhsᵢ = imag.(rhs)

                solᵣ = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵣ)
                    solᵣ .= rhsᵣ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (real part, 2D decomp) at kx=$(G.kx[i_global]), ky=$(G.ky[j_global]): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                dₗ_work = copy(dₗ)
                d_work = copy(d)
                dᵤ_work = copy(dᵤ)
                solᵢ = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dₗ_work, d_work, dᵤ_work, rhsᵢ)
                    solᵢ .= rhsᵢ
                catch e
                    error("LAPACK gtsv failed for vertical velocity (imag part, 2D decomp) at kx=$(G.kx[i_global]), ky=$(G.ky[j_global]): $e. " *
                          "This may indicate singular matrix due to N²≈0 or ill-conditioned system.")
                end

                solution = solᵣ .+ im .* solᵢ
                for iz in 1:n_interior
                    k = iz + 1
                    wk_z_arr[i_local, j_local, k] = solution[iz]
                end
            end

        elseif kₕ² > 0 && nz <= 2
            for k in 1:nz
                wk_z_arr[i_local, j_local, k] = -rhsk_z_arr[i_local, j_local, k] / kₕ²
            end
        end
    end

    # Step 4: Transpose result back to xy-pencil
    wk = similar(S.psi)
    transpose_to_xy_pencil!(wk, wk_z, G)

    # Step 5: Transform to real space
    tmpw = similar(wk)
    fft_backward!(tmpw, wk, plans)
    tmpw_arr = parent(tmpw)
    w_arr = parent(S.w)
    nx_local, ny_local, _ = size(tmpw_arr)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[i_local, j_local, k] = real(tmpw_arr[i_local, j_local, k])
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
    compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=nothing, L=nothing, workspace=nothing)

Compute vertical velocity from near-inertial wave envelope using YBJ+ formulation.

# Physical Background
Near-inertial waves induce vertical motion through the modulation of their
envelope. The YBJ vertical velocity (equation 4 in Asselin & Young 2019):

```
w = -(f²/N²) [(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
```

where:
- A is the true wave amplitude (recovered from evolved B via L⁺A = B)
- A_z = ∂A/∂z is the vertical derivative
- c.c. denotes complex conjugate (ensures real result)

# Physical Interpretation
This represents vertical motion induced by:
- Horizontal gradients in the wave envelope's vertical structure
- Wave packet propagation and refraction
- Strong w occurs where wave amplitude varies both horizontally and vertically

# Algorithm
1. **A Recovery**: Solve L⁺A = B using invert_B_to_A!
   - L⁺ is the YBJ+ elliptic operator
   - Tridiagonal solver in z for each horizontal wavenumber

2. **Vertical Derivative**: Compute A_z = ∂A/∂z
   - Uses second-order finite differences

3. **Horizontal Gradients**: Compute ∂(A_z)/∂x, ∂(A_z)/∂y
   - Spectral differentiation: multiply by i kₓ, i kᵧ

4. **Combine**: Apply YBJ formula with c.c. for real result

# Arguments
- `S::State`: State with B (input) and w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters (f₀)
- `N2_profile::Vector`: Optional N²(z) profile (default: constant N² = 1)
- `L`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Fortran Correspondence
Matches YBJ vertical velocity computation in the Fortran implementation.
"""
function compute_ybj_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing, L=nothing, workspace=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

    if need_transpose
        _compute_ybj_vertical_velocity_2d!(S, G, plans, params, N2_profile, workspace)
    else
        _compute_ybj_vertical_velocity_direct!(S, G, plans, params, N2_profile)
    end
    return S
end

# Direct computation when z is fully local (serial or 1D decomposition)
function _compute_ybj_vertical_velocity_direct!(S::State, G::Grid, plans, params, N2_profile)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    w_arr = parent(S.w)
    nx_local, ny_local, nz_local = size(w_arr)

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local for direct solve"

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
    if N2_profile === nothing
        N2_profile = fill(eltype(S.psi)(N2_const), nz)
    else
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=$(N2_const)"
            N2_profile = fill(eltype(S.psi)(N2_const), nz)
        end
    end

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A using centralized YBJ+ inversion
    # a(z) = f²/N²(z) is the elliptic coefficient
    # NOTE: This re-inverts B→A with the given N² profile. If the timestep already
    # computed A with a different stratification, this will overwrite S.A and S.C.
    # Ensure N2_profile matches what was used in the timestep!
    a_vec = similar(G.z)
    f_sq = f^2
    @inbounds for k in eachindex(a_vec)
        a_vec[k] = f_sq / N2_profile[k]  # a = f²/N²
    end
    invert_B_to_A!(S, G, params, a_vec)
    Aₖ = S.A
    Aₖ_arr = parent(Aₖ)

    # Step 2: Compute vertical derivative A_z using finite differences
    Aₖ_z = S.C  # C was set to A_z by invert_B_to_A!
    Aₖ_z_arr = parent(Aₖ_z)

    # Step 3: Compute horizontal derivatives of A_z
    dAz_dxₖ = similar(Aₖ_z)
    dAz_dyₖ = similar(Aₖ_z)
    dAz_dxₖ_arr = parent(dAz_dxₖ)
    dAz_dyₖ_arr = parent(dAz_dyₖ)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        dAz_dxₖ_arr[i_local, j_local, k] = ikₓ * Aₖ_z_arr[i_local, j_local, k]
        dAz_dyₖ_arr[i_local, j_local, k] = ikᵧ * Aₖ_z_arr[i_local, j_local, k]
    end

    # Step 4: Compute YBJ vertical velocity
    # w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
    wₖ_ybj = similar(S.psi)
    wₖ_ybj_arr = parent(wₖ_ybj)
    fill!(wₖ_ybj_arr, 0.0)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        k_out = k + 1  # Shift to match output grid
        N²ₗ = N2_profile[k_out]

        # YBJ formula
        ybj_factor = -(f^2) / N²ₗ
        complex_term = dAz_dxₖ_arr[i_local, j_local, k] - im * dAz_dyₖ_arr[i_local, j_local, k]

        # Apply the + c.c. operation to get real result
        wₖ_ybj_arr[i_local, j_local, k_out] = ybj_factor * (complex_term + conj(complex_term))
    end

    # Apply boundary conditions: w = 0 at top and bottom
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        wₖ_ybj_arr[i_local, j_local, 1] = 0.0
        if nz > 1
            wₖ_ybj_arr[i_local, j_local, nz] = 0.0
        end
    end

    # Transform to real space
    tmpw = similar(wₖ_ybj)
    fft_backward!(tmpw, wₖ_ybj, plans)
    tmpw_arr = parent(tmpw)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[i_local, j_local, k] = real(tmpw_arr[i_local, j_local, k])
    end
end

# 2D decomposition version with transposes
function _compute_ybj_vertical_velocity_2d!(S::State, G::Grid, plans, params, N2_profile, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get parameters
    if params !== nothing && hasfield(typeof(params), :f₀)
        f = params.f₀
    else
        f = 1.0
    end

    # Get N² profile
    if N2_profile === nothing
        N2_profile = ones(Float64, nz)
    elseif length(N2_profile) != nz
        @warn "N2_profile length mismatch, using constant N²=1.0"
        N2_profile = ones(Float64, nz)
    end

    Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A (invert_B_to_A! handles 2D decomposition internally)
    # a(z) = f²/N²(z) is the elliptic coefficient
    a_vec = similar(G.z)
    f_sq = f^2
    @inbounds for k in eachindex(a_vec)
        a_vec[k] = f_sq / N2_profile[k]  # a = f²/N² (was incorrectly 1/N²)
    end
    # Pass workspace if available
    invert_B_to_A!(S, G, params, a_vec; workspace=workspace)

    # Now A and C (A_z) are in xy-pencil form
    # For vertical derivative computation in YBJ w, we need z local
    # However, invert_B_to_A! already computed A_z and stored it in S.C
    # We need to transpose A to z-pencil to compute proper vertical derivative

    # Allocate z-pencil workspace
    A_z_pencil = workspace !== nothing && hasfield(typeof(workspace), :A_z) ? workspace.A_z : allocate_z_pencil(G, ComplexF64)
    Az_z_pencil = allocate_z_pencil(G, ComplexF64)

    # Transpose A to z-pencil for vertical derivative
    transpose_to_z_pencil!(A_z_pencil, S.A, G)

    # Compute vertical derivative A_z on z-pencil (z now fully local)
    A_z_arr = parent(A_z_pencil)
    Az_z_arr = parent(Az_z_pencil)
    fill!(Az_z_arr, 0.0)

    nx_local_z, ny_local_z, _ = size(A_z_arr)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local_z, i_local in 1:nx_local_z
        Az_z_arr[i_local, j_local, k] = (A_z_arr[i_local, j_local, k+1] - A_z_arr[i_local, j_local, k]) / Δz
    end

    # Transpose A_z back to xy-pencil for horizontal derivatives
    Aₖ_z = similar(S.A)
    transpose_to_xy_pencil!(Aₖ_z, Az_z_pencil, G)
    Aₖ_z_arr = parent(Aₖ_z)

    # Compute horizontal derivatives of A_z in xy-pencil
    nx_local, ny_local, _ = size(Aₖ_z_arr)
    dAz_dxₖ = similar(Aₖ_z)
    dAz_dyₖ = similar(Aₖ_z)
    dAz_dxₖ_arr = parent(dAz_dxₖ)
    dAz_dyₖ_arr = parent(dAz_dyₖ)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        dAz_dxₖ_arr[i_local, j_local, k] = ikₓ * Aₖ_z_arr[i_local, j_local, k]
        dAz_dyₖ_arr[i_local, j_local, k] = ikᵧ * Aₖ_z_arr[i_local, j_local, k]
    end

    # Compute YBJ vertical velocity
    wₖ_ybj = similar(S.psi)
    wₖ_ybj_arr = parent(wₖ_ybj)
    fill!(wₖ_ybj_arr, 0.0)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        k_out = k + 1
        N²ₗ = N2_profile[k_out]
        ybj_factor = -(f^2) / N²ₗ
        complex_term = dAz_dxₖ_arr[i_local, j_local, k] - im * dAz_dyₖ_arr[i_local, j_local, k]
        wₖ_ybj_arr[i_local, j_local, k_out] = ybj_factor * (complex_term + conj(complex_term))
    end

    # Apply boundary conditions
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        wₖ_ybj_arr[i_local, j_local, 1] = 0.0
        if nz > 1
            wₖ_ybj_arr[i_local, j_local, nz] = 0.0
        end
    end

    # Transform to real space
    tmpw = similar(wₖ_ybj)
    fft_backward!(tmpw, wₖ_ybj, plans)
    tmpw_arr = parent(tmpw)
    w_arr = parent(S.w)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here
    @inbounds for k in 1:nz, j_local in 1:ny_local, i_local in 1:nx_local
        w_arr[i_local, j_local, k] = real(tmpw_arr[i_local, j_local, k])
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
    compute_total_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing)

Compute the TOTAL velocity field for Lagrangian particle advection.

# Physical Background
In QG-YBJ+ dynamics, a particle is advected by:
1. **Geostrophic flow**: u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x
2. **Wave-induced drift**: Stokes drift from near-inertial waves

The total velocity is:
```
u_total = u_QG + u_wave
v_total = v_QG + v_wave
w_total = w (from omega equation or YBJ)
```

# Wave-Induced Horizontal Velocities
The Stokes drift from the wave envelope:
```
u_wave = Re[(∂A*/∂x)A + A*(∂A/∂x)] = 2 Re[A* ∂A/∂x]
v_wave = Re[(∂A*/∂y)A + A*(∂A/∂y)] = 2 Re[A* ∂A/∂y]
```

These wave corrections can be significant in regions of strong wave gradients.

# Usage
For Lagrangian particle advection, always use this function rather than
`compute_velocities!` to include wave effects.

# Arguments
- `S::State`: State with ψ, A (input) and u, v, w (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters
- `compute_w::Bool`: If true, compute vertical velocity
- `use_ybj_w::Bool`: If true, use YBJ formula for w
- `N2_profile::Vector`: Optional N²(z) profile for vertical velocity computation
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
Modified State with total velocity fields u, v, w.
"""
function compute_total_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false, N2_profile=nothing, workspace=nothing)
    # First compute QG velocities
    compute_velocities!(S, G; plans=plans, params=params, compute_w=compute_w, use_ybj_w=use_ybj_w, N2_profile=N2_profile, workspace=workspace)

    # Add wave-induced velocities
    compute_wave_velocities!(S, G; plans=plans, params=params)

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
    compute_wave_velocities!(S, G; plans=nothing, params=nothing)

Compute wave-induced Stokes drift velocities and add to existing QG velocities.

# Physical Background
Near-inertial waves induce a net Lagrangian drift (Stokes drift) due to the
correlation between wave orbital velocity and wave-induced displacement.
For wave amplitude A:

```
u_wave = Re[(∂A*/∂x)A + A*(∂A/∂x)] = 2 Re[A* ∂A/∂x]
v_wave = Re[(∂A*/∂y)A + A*(∂A/∂y)] = 2 Re[A* ∂A/∂y]
w_wave = Re[(∂A*/∂z)A + A*(∂A/∂z)] = 2 Re[A* ∂A/∂z]
```

# Physical Interpretation
- The Stokes drift is proportional to the gradient of |A|²
- Particles drift from regions of low to high wave amplitude
- Horizontal drift: particles move toward regions of high wave energy
- Vertical drift: particles move along vertical wave energy gradients
- Important for Lagrangian dispersion in NIW-active regions

# Algorithm
1. Compute horizontal gradients: ∂A/∂x, ∂A/∂y in spectral space
2. Use S.C = A_z = ∂A/∂z (computed by invert_B_to_A!)
3. Transform A, ∂A/∂x, ∂A/∂y, ∂A/∂z to physical space
4. Compute wave velocities: u_wave, v_wave, w_wave = 2 Re[A* ∂A/∂(x,y,z)]
5. Add to existing u, v, w fields (in-place modification)

# Arguments
- `S::State`: State with A, C (input) and u, v, w modified (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters

# Note
This function modifies u, v, w in-place by adding wave contributions.
Call after compute_velocities! to get total velocity.
S.C must contain A_z (set by invert_B_to_A!) before calling this function.
"""
function compute_wave_velocities!(S::State, G::Grid; plans=nothing, params=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    u_arr = parent(S.u)
    v_arr = parent(S.v)
    w_arr = parent(S.w)
    Aₖ_arr = parent(S.A)
    # S.C = A_z = ∂A/∂z is already computed by invert_B_to_A!
    nx_local, ny_local, nz_local = size(Aₖ_arr)

    # Set up plans if needed
    if plans === nothing
        plans = plan_transforms!(G)
    end

    # Compute horizontal derivatives of A: ∂A/∂x, ∂A/∂y in spectral space
    dA_dxₖ = similar(S.A)
    dA_dyₖ = similar(S.A)
    dA_dxₖ_arr = parent(dA_dxₖ)
    dA_dyₖ_arr = parent(dA_dyₖ)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikₓ = im * G.kx[i_global]
        ikᵧ = im * G.ky[j_global]
        dA_dxₖ_arr[i_local, j_local, k] = ikₓ * Aₖ_arr[i_local, j_local, k]
        dA_dyₖ_arr[i_local, j_local, k] = ikᵧ * Aₖ_arr[i_local, j_local, k]
    end

    # Transform A, ∂A/∂x, ∂A/∂y, ∂A/∂z to physical space
    # The Stokes drift formula u_wave = 2*Re[A* ∂A/∂x] is a product of fields
    # and MUST be computed in physical space, not spectral space
    Aᵣ = similar(S.A)
    dA_dxᵣ = similar(S.A)
    dA_dyᵣ = similar(S.A)
    dA_dzᵣ = similar(S.A)

    fft_backward!(Aᵣ, S.A, plans)
    fft_backward!(dA_dxᵣ, dA_dxₖ, plans)
    fft_backward!(dA_dyᵣ, dA_dyₖ, plans)
    fft_backward!(dA_dzᵣ, S.C, plans)  # S.C = A_z in spectral space

    Aᵣ_arr = parent(Aᵣ)
    dA_dxᵣ_arr = parent(dA_dxᵣ)
    dA_dyᵣ_arr = parent(dA_dyᵣ)
    dA_dzᵣ_arr = parent(dA_dzᵣ)

    # Compute Stokes drift in physical space:
    # u_wave = 2 * Re[A* ∂A/∂x] = ∂|A|²/∂x (gradient of wave energy density)
    # v_wave = 2 * Re[A* ∂A/∂y] = ∂|A|²/∂y
    # w_wave = 2 * Re[A* ∂A/∂z] = ∂|A|²/∂z (vertical Stokes drift)
    # Note: fft_backward! is normalized, so Aᵣ and derivatives are in physical space

    # Add wave velocities to existing QG velocities directly in physical space
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        A_phys = Aᵣ_arr[i_local, j_local, k]
        dAdx_phys = dA_dxᵣ_arr[i_local, j_local, k]
        dAdy_phys = dA_dyᵣ_arr[i_local, j_local, k]
        dAdz_phys = dA_dzᵣ_arr[i_local, j_local, k]

        # Stokes drift: (u,v,w)_wave = 2 * Re[conj(A) * ∂A/∂(x,y,z)]
        u_wave = 2.0 * real(conj(A_phys) * dAdx_phys)
        v_wave = 2.0 * real(conj(A_phys) * dAdy_phys)
        w_wave = 2.0 * real(conj(A_phys) * dAdz_phys)

        u_arr[i_local, j_local, k] += u_wave
        v_arr[i_local, j_local, k] += v_wave
        w_arr[i_local, j_local, k] += w_wave
    end

    return S
end

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!, compute_total_velocities!, compute_wave_velocities!
