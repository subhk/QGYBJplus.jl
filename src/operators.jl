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
using ..QGYBJ: Grid, State
using ..QGYBJ: fft_forward!, fft_backward!, plan_transforms!, compute_wavenumbers!
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
    # Spectral differentiation: û = -i ky ψ̂, v̂ = i kx ψ̂
    ψk = S.psi
    uk = similar(ψk)
    vk = similar(ψk)
    @inbounds for k in axes(ψk,3), j in 1:G.ny, i in 1:G.nx
        ikx = im * G.kx[i]
        iky = im * G.ky[j]
        uk[i,j,k] = -iky * ψk[i,j,k]
        vk[i,j,k] =  ikx * ψk[i,j,k]
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
    # Normalization: FFTW.ifft returns unnormalized; divide by (nx*ny)
    norm = (G.nx * G.ny)
    @inbounds for k in axes(S.u,3)
        # Real part
        S.u[:,:,k] .= real.(tmpu[:,:,k]) ./ norm
        S.v[:,:,k] .= real.(tmpv[:,:,k]) ./ norm
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
        fill!(S.w, 0.0)
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
    compute_vertical_velocity!(S, G, plans, params; N2_profile=nothing)

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

# Fortran Correspondence
Matches omega equation solver in the Fortran implementation.
"""
function compute_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    
    # Get RHS of omega equation
    rhsk = similar(S.psi)
    PARENT.Diagnostics.omega_eqn_rhs!(rhsk, S.psi, G, plans)
    
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
    # This is a 3D elliptic equation solved as a tridiagonal system for each (kx,ky)
    
    wk = similar(S.psi)
    fill!(wk, 0.0)
    
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    f2 = f^2
    
    # For each horizontal wavenumber (kx, ky), solve tridiagonal system
    @inbounds for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        
        if kh2 > 0 && nz > 2  # Need at least 3 levels for tridiagonal
            # Set up tridiagonal system for vertical levels 2:(nz-1)
            # Boundary conditions: w = 0 at k=1 and k=nz
            n_interior = nz - 2  # Interior points
            
            if n_interior > 0
                # Tridiagonal matrix coefficients (real-valued)
                d = zeros(Float64, n_interior)      # diagonal
                dl = zeros(Float64, n_interior-1)   # lower diagonal  
                du = zeros(Float64, n_interior-1)   # upper diagonal
                rhs = zeros(eltype(S.psi), n_interior)  # RHS vector (complex eltype already)
                
                # Fill tridiagonal system based on Fortran implementation
                for iz in 1:n_interior
                    k = iz + 1  # Actual z-level (2 to nz-1)
                    
                    # N² coefficient: a_ell_ut = 1.0/N² (normalized Bu=1)
                    a_ell = 1.0 / N2_profile[k]
                    
                    # Diagonal term: -(N²/f²)/dz² - kh2
                    # In Fortran: d(iz) = -(... + dz*dz*kh2/a_ell_ut(iz))
                    # Here: d(iz) = -(N²/f²)/dz² - kh2
                    d[iz] = -(N2_profile[k]/f2)/(dz*dz) - kh2
                    
                    # Off-diagonal terms (simplified density weighting = 1)
                    if iz > 1
                        dl[iz-1] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    if iz < n_interior
                        du[iz] = (N2_profile[k]/f2)/(dz*dz)
                    end
                    
                    # RHS
                    rhs[iz] = rhsk[i,j,k]
                end
                
                # Solve tridiagonal system using LAPACK (same as Fortran code)
                # For complex RHS, solve real and imaginary parts separately
                
                # Prepare arrays for LAPACK gtsv! (modifies input arrays)
                dl_work = copy(dl)  # Sub-diagonal
                d_work = copy(d)    # Diagonal  
                du_work = copy(du)  # Super-diagonal
                
                # Split complex RHS into real and imaginary parts
                rhs_real = real.(rhs)
                rhs_imag = imag.(rhs)
                
                # Solve real part: A * x_real = rhs_real
                sol_real = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_real)
                    sol_real .= rhs_real  # gtsv! overwrites RHS with solution
                catch e
                    @warn "LAPACK gtsv failed for real part: $e, using zeros"
                end
                
                # Reset arrays for imaginary part (gtsv! modifies them)
                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                
                # Solve imaginary part: A * x_imag = rhs_imag  
                # Reset arrays for imaginary part (gtsv! modifies them)
                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                sol_imag = zeros(Float64, n_interior)
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_imag)
                    sol_imag .= rhs_imag  # gtsv! overwrites RHS with solution
                catch e
                    @warn "LAPACK gtsv failed for imaginary part: $e, using zeros"
                end
                
                # Combine real and imaginary solutions
                solution = sol_real .+ im .* sol_imag
                
                # Store solution in wk (interior points)
                for iz in 1:n_interior
                    k = iz + 1
                    wk[i,j,k] = solution[iz]
                end
            end
            
        elseif kh2 > 0 && nz <= 2
            # Simple 2D case - approximate with ∇²w = RHS
            for k in 1:nz
                wk[i,j,k] = -rhsk[i,j,k] / kh2
            end
        end
        
        # Boundary conditions are automatically satisfied (wk initialized to 0)
        # wk[i,j,1] = 0 and wk[i,j,nz] = 0
    end
    
    # Transform to real space
    tmpw = similar(wk)
    fft_backward!(tmpw, wk, plans)
    
    # Store in state (real part, normalized)
    norm = nx * ny
    @inbounds for k in 1:nz
        S.w[:,:,k] .= real.(tmpw[:,:,k]) ./ norm
    end
    
    return S
end

#=
================================================================================
                    YBJ VERTICAL VELOCITY
================================================================================
Wave-induced vertical motion from the YBJ+ formulation.
================================================================================
=#

"""
    compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=nothing, L=nothing)

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

# Fortran Correspondence
Matches YBJ vertical velocity computation in the Fortran implementation.
"""
function compute_ybj_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing, L=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    
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
        # Ensure N2_profile has correct length and type
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=1.0"
            N2_profile = ones(eltype(S.psi), nz)
        end
    end
    
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    
    # Step 1: Recover A from B = L⁺A using centralized YBJ+ inversion
    # Build a = 1/N² if provided, otherwise use ones
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
    
    # Step 2: Compute vertical derivative A_z using finite differences
    Ask_z = S.C  # C was set to A_z by invert_B_to_A!
    
    # Step 3: Compute horizontal derivatives of A_z
    dAz_dx_k = similar(Ask_z)
    dAz_dy_k = similar(Ask_z)
    
    @inbounds for k in 1:(nz-1), j in 1:ny, i in 1:nx
        ikx = im * G.kx[i]
        iky = im * G.ky[j]
        dAz_dx_k[i,j,k] = ikx * Ask_z[i,j,k]
        dAz_dy_k[i,j,k] = iky * Ask_z[i,j,k]
    end
    
    # Step 4: Compute YBJ vertical velocity
    # w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
    # The c.c. (complex conjugate) makes the result real
    wk_ybj = similar(S.psi)
    fill!(wk_ybj, 0.0)
    
    @inbounds for k in 1:(nz-1), j in 1:ny, i in 1:nx
        k_out = k + 1  # Shift to match output grid (intermediate to full levels)
        # Get N² at this level
        N2_level = N2_profile[k_out]
        
        # YBJ formula: w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
        ybj_factor = -(f^2) / N2_level
        complex_term = dAz_dx_k[i,j,k] - im * dAz_dy_k[i,j,k]
        
        # Apply the + c.c. operation to get real result
        wk_ybj[i,j,k_out] = ybj_factor * (complex_term + conj(complex_term))
    end
    
    # Apply boundary conditions: w = 0 at top and bottom
    wk_ybj[:,:,1] .= 0.0
    if nz > 1
        wk_ybj[:,:,nz] .= 0.0
    end
    
    # Transform to real space
    tmpw = similar(wk_ybj)
    fft_backward!(tmpw, wk_ybj, plans)
    
    # Store in state (real part, normalized)
    norm = nx * ny
    @inbounds for k in 1:nz
        S.w[:,:,k] .= real.(tmpw[:,:,k]) ./ norm
    end
    
    return S
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
    
    # Set up plans if needed
    if plans === nothing
        plans = plan_transforms!(G)
    end
    
    # Use wave amplitude A for wave velocity computation
    Ak = S.A  # Wave amplitude in spectral space
    
    # Compute horizontal derivatives of A: ∂A/∂x, ∂A/∂y
    dA_dx_k = similar(Ak)
    dA_dy_k = similar(Ak)
    
    @inbounds for k in axes(Ak,3), j in 1:ny, i in 1:nx
        ikx = im * G.kx[i]
        iky = im * G.ky[j]
        dA_dx_k[i,j,k] = ikx * Ak[i,j,k]
        dA_dy_k[i,j,k] = iky * Ak[i,j,k]
    end
    
    # Compute wave velocity contributions in spectral space
    # u_wave = Real[(∂A*/∂x)A + A*(∂A/∂x)] = Real[2 * Real(A* ∂A/∂x)]
    # v_wave = Real[(∂A*/∂y)A + A*(∂A/∂y)] = Real[2 * Real(A* ∂A/∂y)]
    u_wave_k = similar(Ak)
    v_wave_k = similar(Ak)
    
    @inbounds for k in axes(Ak,3), j in 1:ny, i in 1:nx
        if G.kh2[i,j] > 0  # Dealias
            # Wave velocity contributions
            u_wave_k[i,j,k] = 2.0 * real(conj(Ak[i,j,k]) * dA_dx_k[i,j,k])
            v_wave_k[i,j,k] = 2.0 * real(conj(Ak[i,j,k]) * dA_dy_k[i,j,k])
        else
            u_wave_k[i,j,k] = 0.0
            v_wave_k[i,j,k] = 0.0
        end
    end
    
    # Transform to real space
    u_wave_real = similar(S.u)
    v_wave_real = similar(S.v)
    
    fft_backward!(u_wave_real, u_wave_k, plans)
    fft_backward!(v_wave_real, v_wave_k, plans)
    
    # Normalization
    norm = nx * ny
    
    # Add wave velocities to existing QG velocities
    @inbounds for k in 1:nz
        S.u[:,:,k] .+= real.(u_wave_real[:,:,k]) ./ norm
        S.v[:,:,k] .+= real.(v_wave_real[:,:,k]) ./ norm
    end
    
    return S
end

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!, compute_total_velocities!, compute_wave_velocities!
