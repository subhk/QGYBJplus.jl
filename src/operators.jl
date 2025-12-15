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
    compute_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)

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

# Returns
Modified State with updated u, v, w fields.

# Note
This computes ONLY QG velocities. For Lagrangian advection including wave
effects, use `compute_total_velocities!` instead.

# Fortran Correspondence
Matches `compute_velo` in derivatives.f90.
"""
function compute_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)
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
        ikx = im * G.kx[i_global]
        iky = im * G.ky[j_global]
        uk_arr[i_local, j_local, k] = -iky * ψk_arr[i_local, j_local, k]
        vk_arr[i_local, j_local, k] =  ikx * ψk_arr[i_local, j_local, k]
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
            compute_ybj_vertical_velocity!(S, G, plans, params)
        else
            # Use standard QG omega equation
            compute_vertical_velocity!(S, G, plans, params)
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
    if params !== nothing && hasfield(typeof(params), :f0)
        f = params.f0
    else
        f = 1.0  # Default
    end

    # Get N² profile - use provided profile or default to constant
    if N2_profile === nothing
        N2_profile = ones(eltype(S.psi), nz)
    else
        # Ensure N2_profile has correct length and type
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=1.0"
            N2_profile = ones(eltype(S.psi), nz)
        end
    end

    # Solve the full omega equation: ∇²w + (N²/f²)(∂²w/∂z²) = RHS
    wk = similar(S.psi)
    wk_arr = parent(wk)
    fill!(wk_arr, 0.0)

    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f2 = f^2

    # For each LOCAL horizontal wavenumber (kx, ky), solve tridiagonal system
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2

        if kh2 > 0 && nz > 2  # Need at least 3 levels for tridiagonal
            n_interior = nz - 2  # Interior points

            if n_interior > 0
                # Tridiagonal matrix coefficients (real-valued)
                d = zeros(Float64, n_interior)      # diagonal
                dl = zeros(Float64, n_interior-1)   # lower diagonal
                du = zeros(Float64, n_interior-1)   # upper diagonal
                rhs = zeros(eltype(S.psi), n_interior)  # RHS vector

                # Fill tridiagonal system
                for iz in 1:n_interior
                    k = iz + 1  # Actual z-level (2 to nz-1)
                    d[iz] = -(N2_profile[k]/f2)/(dz*dz) - kh2
                    if iz > 1
                        dl[iz-1] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    if iz < n_interior
                        du[iz] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    rhs[iz] = rhsk_arr[i_local, j_local, k]
                end

                # Solve tridiagonal system - real and imaginary parts separately
                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                rhs_real = real.(rhs)
                rhs_imag = imag.(rhs)

                sol_real = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_real)
                    sol_real .= rhs_real
                catch e
                    @warn "LAPACK gtsv failed for real part: $e"
                end

                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                sol_imag = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_imag)
                    sol_imag .= rhs_imag
                catch e
                    @warn "LAPACK gtsv failed for imag part: $e"
                end

                solution = sol_real .+ im .* sol_imag
                for iz in 1:n_interior
                    k = iz + 1
                    wk_arr[i_local, j_local, k] = solution[iz]
                end
            end

        elseif kh2 > 0 && nz <= 2
            for k in 1:nz
                wk_arr[i_local, j_local, k] = -rhsk_arr[i_local, j_local, k] / kh2
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
    if params !== nothing && hasfield(typeof(params), :f0)
        f = params.f0
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

    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f2 = f^2

    @inbounds for j_local in 1:ny_local_z, i_local in 1:nx_local_z
        i_global = local_to_global_z(i_local, 1, G)
        j_global = local_to_global_z(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2

        if kh2 > 0 && nz > 2
            n_interior = nz - 2

            if n_interior > 0
                d = zeros(Float64, n_interior)
                dl = zeros(Float64, n_interior-1)
                du = zeros(Float64, n_interior-1)
                rhs = zeros(ComplexF64, n_interior)

                for iz in 1:n_interior
                    k = iz + 1
                    d[iz] = -(N2_profile[k]/f2)/(dz*dz) - kh2
                    if iz > 1
                        dl[iz-1] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    if iz < n_interior
                        du[iz] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    rhs[iz] = rhsk_z_arr[i_local, j_local, k]
                end

                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                rhs_real = real.(rhs)
                rhs_imag = imag.(rhs)

                sol_real = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_real)
                    sol_real .= rhs_real
                catch; end

                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                sol_imag = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_imag)
                    sol_imag .= rhs_imag
                catch; end

                solution = sol_real .+ im .* sol_imag
                for iz in 1:n_interior
                    k = iz + 1
                    wk_z_arr[i_local, j_local, k] = solution[iz]
                end
            end

        elseif kh2 > 0 && nz <= 2
            for k in 1:nz
                wk_z_arr[i_local, j_local, k] = -rhsk_z_arr[i_local, j_local, k] / kh2
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
    if params !== nothing && hasfield(typeof(params), :f0)
        f = params.f0
    else
        f = 1.0  # Default
    end

    # Get N² profile - use provided profile or default to constant
    if N2_profile === nothing
        N2_profile = ones(eltype(S.psi), nz)
    else
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=1.0"
            N2_profile = ones(eltype(S.psi), nz)
        end
    end

    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A using centralized YBJ+ inversion
    a_vec = similar(G.z)
    if N2_profile === nothing
        fill!(a_vec, one(eltype(a_vec)))
    else
        @inbounds for k in eachindex(a_vec)
            a_vec[k] = one(eltype(a_vec)) / N2_profile[k]
        end
    end
    invert_B_to_A!(S, G, params, a_vec)
    Ak = S.A
    Ak_arr = parent(Ak)

    # Step 2: Compute vertical derivative A_z using finite differences
    Ask_z = S.C  # C was set to A_z by invert_B_to_A!
    Ask_z_arr = parent(Ask_z)

    # Step 3: Compute horizontal derivatives of A_z
    dAz_dx_k = similar(Ask_z)
    dAz_dy_k = similar(Ask_z)
    dAz_dx_k_arr = parent(dAz_dx_k)
    dAz_dy_k_arr = parent(dAz_dy_k)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikx = im * G.kx[i_global]
        iky = im * G.ky[j_global]
        dAz_dx_k_arr[i_local, j_local, k] = ikx * Ask_z_arr[i_local, j_local, k]
        dAz_dy_k_arr[i_local, j_local, k] = iky * Ask_z_arr[i_local, j_local, k]
    end

    # Step 4: Compute YBJ vertical velocity
    # w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
    wk_ybj = similar(S.psi)
    wk_ybj_arr = parent(wk_ybj)
    fill!(wk_ybj_arr, 0.0)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        k_out = k + 1  # Shift to match output grid
        N2_level = N2_profile[k_out]

        # YBJ formula
        ybj_factor = -(f^2) / N2_level
        complex_term = dAz_dx_k_arr[i_local, j_local, k] - im * dAz_dy_k_arr[i_local, j_local, k]

        # Apply the + c.c. operation to get real result
        wk_ybj_arr[i_local, j_local, k_out] = ybj_factor * (complex_term + conj(complex_term))
    end

    # Apply boundary conditions: w = 0 at top and bottom
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        wk_ybj_arr[i_local, j_local, 1] = 0.0
        if nz > 1
            wk_ybj_arr[i_local, j_local, nz] = 0.0
        end
    end

    # Transform to real space
    tmpw = similar(wk_ybj)
    fft_backward!(tmpw, wk_ybj, plans)
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
    if params !== nothing && hasfield(typeof(params), :f0)
        f = params.f0
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

    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0

    # Step 1: Recover A from B = L⁺A (invert_B_to_A! handles 2D decomposition internally)
    a_vec = similar(G.z)
    @inbounds for k in eachindex(a_vec)
        a_vec[k] = one(eltype(a_vec)) / N2_profile[k]
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
        Az_z_arr[i_local, j_local, k] = (A_z_arr[i_local, j_local, k+1] - A_z_arr[i_local, j_local, k]) / dz
    end

    # Transpose A_z back to xy-pencil for horizontal derivatives
    Ask_z = similar(S.A)
    transpose_to_xy_pencil!(Ask_z, Az_z_pencil, G)
    Ask_z_arr = parent(Ask_z)

    # Compute horizontal derivatives of A_z in xy-pencil
    nx_local, ny_local, _ = size(Ask_z_arr)
    dAz_dx_k = similar(Ask_z)
    dAz_dy_k = similar(Ask_z)
    dAz_dx_k_arr = parent(dAz_dx_k)
    dAz_dy_k_arr = parent(dAz_dy_k)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikx = im * G.kx[i_global]
        iky = im * G.ky[j_global]
        dAz_dx_k_arr[i_local, j_local, k] = ikx * Ask_z_arr[i_local, j_local, k]
        dAz_dy_k_arr[i_local, j_local, k] = iky * Ask_z_arr[i_local, j_local, k]
    end

    # Compute YBJ vertical velocity
    wk_ybj = similar(S.psi)
    wk_ybj_arr = parent(wk_ybj)
    fill!(wk_ybj_arr, 0.0)

    @inbounds for k in 1:(nz-1), j_local in 1:ny_local, i_local in 1:nx_local
        k_out = k + 1
        N2_level = N2_profile[k_out]
        ybj_factor = -(f^2) / N2_level
        complex_term = dAz_dx_k_arr[i_local, j_local, k] - im * dAz_dy_k_arr[i_local, j_local, k]
        wk_ybj_arr[i_local, j_local, k_out] = ybj_factor * (complex_term + conj(complex_term))
    end

    # Apply boundary conditions
    @inbounds for j_local in 1:ny_local, i_local in 1:nx_local
        wk_ybj_arr[i_local, j_local, 1] = 0.0
        if nz > 1
            wk_ybj_arr[i_local, j_local, nz] = 0.0
        end
    end

    # Transform to real space
    tmpw = similar(wk_ybj)
    fft_backward!(tmpw, wk_ybj, plans)
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
    compute_total_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)

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

# Returns
Modified State with total velocity fields u, v, w.
"""
function compute_total_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)
    # First compute QG velocities
    compute_velocities!(S, G; plans=plans, params=params, compute_w=compute_w, use_ybj_w=use_ybj_w)

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
```

# Physical Interpretation
- The Stokes drift is proportional to the gradient of |A|²
- Particles drift from regions of low to high wave amplitude
- Important for Lagrangian dispersion in NIW-active regions

# Algorithm
1. Compute horizontal gradients: ∂A/∂x, ∂A/∂y in spectral space
2. Compute wave velocity: u_wave = 2 Re[A* ∂A/∂x]
3. Transform to physical space
4. Add to existing u, v fields (in-place modification)

# Arguments
- `S::State`: State with A (input) and u, v modified (output)
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `params`: Model parameters

# Note
This function modifies u, v in-place by adding wave contributions.
Call after compute_velocities! to get total velocity.
"""
function compute_wave_velocities!(S::State, G::Grid; plans=nothing, params=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get underlying arrays
    u_arr = parent(S.u)
    v_arr = parent(S.v)
    Ak_arr = parent(S.A)
    nx_local, ny_local, nz_local = size(Ak_arr)

    # Set up plans if needed
    if plans === nothing
        plans = plan_transforms!(G)
    end

    # Compute horizontal derivatives of A: ∂A/∂x, ∂A/∂y
    dA_dx_k = similar(S.A)
    dA_dy_k = similar(S.A)
    dA_dx_k_arr = parent(dA_dx_k)
    dA_dy_k_arr = parent(dA_dy_k)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        ikx = im * G.kx[i_global]
        iky = im * G.ky[j_global]
        dA_dx_k_arr[i_local, j_local, k] = ikx * Ak_arr[i_local, j_local, k]
        dA_dy_k_arr[i_local, j_local, k] = iky * Ak_arr[i_local, j_local, k]
    end

    # Compute wave velocity contributions in spectral space
    u_wave_k = similar(S.A)
    v_wave_k = similar(S.A)
    u_wave_k_arr = parent(u_wave_k)
    v_wave_k_arr = parent(v_wave_k)

    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)
        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2
        if kh2 > 0  # Dealias
            u_wave_k_arr[i_local, j_local, k] = 2.0 * real(conj(Ak_arr[i_local, j_local, k]) * dA_dx_k_arr[i_local, j_local, k])
            v_wave_k_arr[i_local, j_local, k] = 2.0 * real(conj(Ak_arr[i_local, j_local, k]) * dA_dy_k_arr[i_local, j_local, k])
        else
            u_wave_k_arr[i_local, j_local, k] = 0.0
            v_wave_k_arr[i_local, j_local, k] = 0.0
        end
    end

    # Transform to real space
    u_wave_real = similar(S.u)
    v_wave_real = similar(S.v)

    fft_backward!(u_wave_real, u_wave_k, plans)
    fft_backward!(v_wave_real, v_wave_k, plans)

    u_wave_real_arr = parent(u_wave_real)
    v_wave_real_arr = parent(v_wave_real)

    # Note: fft_backward! is normalized (FFTW.ifft / PencilFFTs ldiv!)
    # No additional normalization needed here

    # Add wave velocities to existing QG velocities
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        u_arr[i_local, j_local, k] += real(u_wave_real_arr[i_local, j_local, k])
        v_arr[i_local, j_local, k] += real(v_wave_real_arr[i_local, j_local, k])
    end

    return S
end

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!, compute_total_velocities!, compute_wave_velocities!
