"""
Operators and utilities mapping to key Fortran routines in derivatives.f90:
 - compute_streamfunction (q -> psi inversion via elliptic solver; in elliptic.jl)
 - compute_velo (psi -> u,v,b,w diagnostics)
This file wires spectral-to-real conversions and simple diagnostic operators.
"""
module Operators

using ..QGYBJ: Grid, State
using ..QGYBJ: fft_forward!, fft_backward!, plan_transforms!, compute_wavenumbers!

"""
    compute_velocities!(S, G)

Given spectral streamfunction `psi(kx,ky,z)`, compute velocities:
- Horizontal: `u = -∂ψ/∂y`, `v = ∂ψ/∂x` using spectral differentiation
- Vertical: `w` from either QG omega equation or YBJ formulation

Vertical velocity options:
1. QG omega equation: ∇²w + (N²/f²)(∂²w/∂z²) = 2 J(ψ_z, ∇²ψ)
2. YBJ formulation: w₁ = -(1/(r₁r₂))(Fr/Ro)² D/Dt ψ_z

Set `use_ybj_w=true` and provide `S_old` and `dt` for YBJ vertical velocity.
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
        if use_ybj_w && S_old !== nothing && dt !== nothing
            # Use YBJ vertical velocity formulation
            compute_ybj_vertical_velocity!(S, S_old, G, plans, params, dt)
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

"""
    compute_vertical_velocity!(S, G, plans, params)

Compute QG ageostrophic vertical velocity by solving the omega equation:
∇²w + (N²/f²)(∂²w/∂z²) = 2 J(ψ_z, ∇²ψ)

This is the diagnostic vertical velocity from quasi-geostrophic theory.
"""
function compute_vertical_velocity!(S::State, G::Grid, plans, params)
    nx, ny, nz = G.nx, G.ny, G.nz
    
    # For simple implementation, use existing omega_eqn_rhs! to get RHS
    # Then solve the elliptic equation for w
    using ..QGYBJ: omega_eqn_rhs!
    
    rhsk = similar(S.psi)
    omega_eqn_rhs!(rhsk, S.psi, G, plans)
    
    # Solve ∇²w + (N²/f²)(∂²w/∂z²) = RHS
    # For now, implement a simplified version focusing on horizontal structure
    wk = similar(S.psi)
    
    # Get stratification parameters
    if params !== nothing && hasfield(typeof(params), :Bu)
        Bu = params.Bu  # Burger number = N²H²/(f²L²)
        Ro = params.Ro  # Rossby number
        Fr = params.Fr  # Froude number
    else
        # Default values if params not provided
        Bu = 1.0
        Ro = 0.1
        Fr = 0.1
    end
    
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        
        if kh2 > 0  # Avoid division by zero at k=0
            # Simplified omega equation solver
            # ∇²w ≈ RHS (neglecting vertical structure for now)
            wk[i,j,k] = -rhsk[i,j,k] / kh2
        else
            wk[i,j,k] = 0.0
        end
    end
    
    # Apply boundary conditions: w = 0 at top and bottom
    if nz > 1
        wk[:,:,1] .= 0.0    # Bottom
        wk[:,:,nz] .= 0.0   # Top
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

"""
    compute_ybj_vertical_velocity!(S, G, plans, params)

Compute YBJ vertical velocity using equation (4) from QG_YBJp.pdf:
w₀ = -(f²/N²)e^(-ift)(∂ₓ - i∂ᵧ)Aᵤ + c.c.

This simplifies to the real part of:
w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z]

where A is the wave envelope and the subscript z denotes vertical derivative.
"""
function compute_ybj_vertical_velocity!(S::State, G::Grid, plans, params)
    nx, ny, nz = G.nx, G.ny, G.nz
    
    # Get parameters - need f and N² profile
    if params !== nothing && hasfield(typeof(params), :f0)
        f = params.f0
        # Get N² profile (simplified - would use actual stratification)
        N2 = ones(nz)  # Would be actual N²(z) profile
    else
        f = 1.0  # Default
        N2 = ones(nz)
    end
    
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    
    # Need to recover A from B = L⁺A
    # For now, use A ≈ B for simplicity (this would be improved)
    # In full implementation, would solve L⁺A = B for A
    Ak = S.A  # Wave envelope in spectral space
    
    # Compute vertical derivative of A: A_z
    Ask_z = similar(Ak, Complex{eltype(Ak)}, nx, ny, nz-1)
    
    # Vertical finite difference for A_z (on intermediate levels)
    @inbounds for k in 1:nz-1, j in 1:ny, i in 1:nx
        if G.kh2[i,j] > 0  # Dealias
            Ask_z[i,j,k] = (Ak[i,j,k+1] - Ak[i,j,k]) / dz
        else
            Ask_z[i,j,k] = 0.0
        end
    end
    
    # Compute horizontal derivatives of A_z: (∂A/∂x)_z and (∂A/∂y)_z
    dAz_dx_k = similar(Ask_z)
    dAz_dy_k = similar(Ask_z)
    
    @inbounds for k in 1:size(Ask_z,3), j in 1:ny, i in 1:nx
        ikx = im * G.kx[i]
        iky = im * G.ky[j]
        dAz_dx_k[i,j,k] = ikx * Ask_z[i,j,k]
        dAz_dy_k[i,j,k] = iky * Ask_z[i,j,k]
    end
    
    # Compute YBJ vertical velocity in spectral space
    # w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z]
    wk_ybj = similar(S.psi)
    fill!(wk_ybj, 0.0)
    
    @inbounds for k in 1:size(Ask_z,3), j in 1:ny, i in 1:nx
        k_out = k + 1  # Shift to match output grid (intermediate to full levels)
        if k_out <= nz
            # Get N² at this level
            N2_level = N2[k_out]
            
            # YBJ formula: w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z]
            ybj_factor = -(f^2) / N2_level
            complex_deriv = dAz_dx_k[i,j,k] - im * dAz_dy_k[i,j,k]
            
            # Take real part (the c.c. in equation 4 makes it real)
            wk_ybj[i,j,k_out] = ybj_factor * real(complex_deriv)
        end
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

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!

