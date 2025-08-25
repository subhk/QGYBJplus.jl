"""
Operators and utilities mapping to key Fortran routines in derivatives.f90:
 - compute_streamfunction (q -> psi inversion via elliptic solver; in elliptic.jl)
 - compute_velo (psi -> u,v,b,w diagnostics)
This file wires spectral-to-real conversions and simple diagnostic operators.
"""
module Operators

using LinearAlgebra
using ..QGYBJ: Grid, State
using ..QGYBJ: fft_forward!, fft_backward!, plan_transforms!, compute_wavenumbers!

"""
    compute_velocities!(S, G)

Given spectral streamfunction `psi(kx,ky,z)`, compute QG velocities only:
- Horizontal: `u = -∂ψ/∂y`, `v = ∂ψ/∂x` using spectral differentiation
- Vertical: `w` from either QG omega equation or YBJ formulation

This computes ONLY QG velocities. For particle advection, use `compute_total_velocities!`
to get the full QG + wave velocity field.

Vertical velocity options:
1. QG omega equation: ∇²w + (N²/f²)(∂²w/∂z²) = 2 J(ψ_z, ∇²ψ)
2. YBJ formulation: w₀ = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] (equation 4)

Set `use_ybj_w=true` for YBJ vertical velocity from wave envelope gradients.
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

"""
    compute_vertical_velocity!(S, G, plans, params; N2_profile=nothing)

Compute QG ageostrophic vertical velocity by solving the omega equation:
∇²w + (N²/f²)(∂²w/∂z²) = 2 J(ψ_z, ∇²ψ)

This is the diagnostic vertical velocity from quasi-geostrophic theory.
The full 3D elliptic equation is solved using LAPACK's tridiagonal solver (gtsv!)
for each horizontal wavenumber, matching the Fortran implementation.

Optional parameters:
- N2_profile: Vector of N²(z) values. If not provided, uses constant N² = 1.0.
"""
function compute_vertical_velocity!(S::State, G::Grid, plans, params; N2_profile=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    
    # Get RHS of omega equation
    using ..QGYBJ: omega_eqn_rhs!
    
    rhsk = similar(S.psi)
    omega_eqn_rhs!(rhsk, S.psi, G, plans)
    
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
                # Tridiagonal matrix coefficients
                d = zeros(eltype(S.psi), n_interior)      # diagonal
                dl = zeros(eltype(S.psi), n_interior-1)   # lower diagonal  
                du = zeros(eltype(S.psi), n_interior-1)   # upper diagonal
                rhs = zeros(Complex{eltype(S.psi)}, n_interior)  # RHS vector
                
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
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_real)
                    sol_real = rhs_real  # gtsv! overwrites RHS with solution
                catch e
                    @warn "LAPACK gtsv failed for real part: $e, using zeros"
                    sol_real = zeros(eltype(d), n_interior)
                end
                
                # Reset arrays for imaginary part (gtsv! modifies them)
                dl_work = copy(dl)
                d_work = copy(d)
                du_work = copy(du)
                
                # Solve imaginary part: A * x_imag = rhs_imag  
                try
                    LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_imag)
                    sol_imag = rhs_imag  # gtsv! overwrites RHS with solution
                catch e
                    @warn "LAPACK gtsv failed for imaginary part: $e, using zeros"
                    sol_imag = zeros(eltype(d), n_interior)
                end
                
                # Combine real and imaginary solutions
                solution = complex.(sol_real, sol_imag)
                
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

"""
    compute_ybj_vertical_velocity!(S, G, plans, params; N2_profile=nothing, L=nothing)

Compute YBJ vertical velocity using the complete YBJ+ formulation:

1. **A Recovery**: Solve L⁺A = B for the true wave envelope A
   - Uses tridiagonal solver: a_ell(z) d²A/dz² + b_ell(z) dA/dz - kh²A/4 = B
   - Based on A_solver_ybj_plus from Fortran implementation

2. **Vertical Velocity**: Apply YBJ equation (4)
   - w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
   - where A_z is the vertical derivative of the recovered A

This is the complete YBJ vertical velocity including proper A recovery
from the evolved field B = L⁺A, matching the Fortran implementation.

Optional parameters:
- N2_profile: Vector of N²(z) values. If not provided, uses constant N² = 1.0
- L: Dealiasing mask. If not provided, no dealiasing applied
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
    
    # Step 1: Recover A from B = L⁺A using YBJ+ elliptic solver
    # This solves: a_ell(z) d²A/dz² + b_ell(z) dA/dz - kh²A/4 = B
    # where B is stored in S.A (confusing naming in original code)
    Bk = S.A  # Actually B = L⁺A in spectral space
    Ak = similar(Bk)  # True A will be computed here
    
    f2 = f^2
    
    # Solve for A from B using tridiagonal system (following A_solver_ybj_plus)
    @inbounds for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        
        # Check dealiasing mask if provided
        if L !== nothing && size(L) == (nx, ny)
            dealias = (L[i,j] == 1)
        else
            dealias = true  # No dealiasing mask provided
        end
        
        if kh2 > 0 && dealias && nz > 2
            # Set up tridiagonal system for A recovery
            d = zeros(eltype(S.psi), nz)      # diagonal
            dl = zeros(eltype(S.psi), nz-1)   # lower diagonal
            du = zeros(eltype(S.psi), nz-1)   # upper diagonal
            rhs = zeros(Complex{eltype(S.psi)}, nz)  # RHS vector
            
            # Fill tridiagonal system (YBJ+ version with kh²/4 factor)
            for k in 1:nz
                # Coefficient: a_ell_ut = 1.0/N² (normalized)
                a_ell = 1.0 / N2_profile[k]
                
                if k == 1
                    # Bottom boundary
                    d[k] = -(a_ell + kh2 * dz^2 / 4.0)
                    if nz > 1
                        du[k] = a_ell
                    end
                elseif k == nz
                    # Top boundary
                    d[k] = -(a_ell + kh2 * dz^2 / 4.0)
                    dl[k-1] = a_ell
                else
                    # Interior points
                    a_ell_k = 1.0 / N2_profile[k]
                    d[k] = -(a_ell_k + kh2 * dz^2 / 4.0)
                    du[k] = a_ell_k
                    dl[k-1] = a_ell_k
                end
                
                # RHS = B (normalized by Bu, but Bu=1 in normalized system)
                rhs[k] = dz^2 * Bk[i,j,k]  # Factor from Fortran: dz*dz*Bu*B
            end
            
            # Solve tridiagonal system using LAPACK
            # For complex RHS, solve real and imaginary parts separately
            
            # Prepare arrays for LAPACK gtsv! (modifies input arrays)
            dl_work = copy(dl)
            d_work = copy(d) 
            du_work = copy(du)
            
            # Split complex RHS into real and imaginary parts
            rhs_real = real.(rhs)
            rhs_imag = imag.(rhs)
            
            # Solve real part
            try
                LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_real)
                sol_real = rhs_real
            catch e
                @warn "LAPACK gtsv failed for YBJ A recovery (real): $e, using zeros"
                sol_real = zeros(eltype(d), nz)
            end
            
            # Reset arrays for imaginary part
            dl_work = copy(dl)
            d_work = copy(d)
            du_work = copy(du)
            
            # Solve imaginary part
            try
                LinearAlgebra.LAPACK.gtsv!(dl_work, d_work, du_work, rhs_imag)
                sol_imag = rhs_imag
            catch e
                @warn "LAPACK gtsv failed for YBJ A recovery (imag): $e, using zeros"
                sol_imag = zeros(eltype(d), nz)
            end
            
            # Store recovered A
            for k in 1:nz
                Ak[i,j,k] = complex(sol_real[k], sol_imag[k])
            end
            
        else
            # Set to zero for kh²=0 modes or when dealiased
            for k in 1:nz
                Ak[i,j,k] = 0.0
            end
        end
    end
    
    # Step 2: Compute vertical derivative A_z using finite differences
    Ask_z = similar(Ak, Complex{eltype(Ak)}, nx, ny, nz-1)
    
    @inbounds for k in 1:nz-1, j in 1:ny, i in 1:nx
        Ask_z[i,j,k] = (Ak[i,j,k+1] - Ak[i,j,k]) / dz
    end
    
    # Step 3: Compute horizontal derivatives of A_z
    dAz_dx_k = similar(Ask_z)
    dAz_dy_k = similar(Ask_z)
    
    @inbounds for k in 1:size(Ask_z,3), j in 1:ny, i in 1:nx
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
    
    @inbounds for k in 1:size(Ask_z,3), j in 1:ny, i in 1:nx
        k_out = k + 1  # Shift to match output grid (intermediate to full levels)
        if k_out <= nz
            # Get N² at this level
            N2_level = N2_profile[k_out]
            
            # YBJ formula: w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
            ybj_factor = -(f^2) / N2_level
            complex_term = dAz_dx_k[i,j,k] - im * dAz_dy_k[i,j,k]
            
            # Apply the + c.c. operation to get real result
            wk_ybj[i,j,k_out] = ybj_factor * (complex_term + conj(complex_term))
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

"""
    compute_total_velocities!(S, G; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)

Compute TOTAL velocity field for particle advection: QG velocity + wave velocity.
This is the proper velocity field for advecting particles in QG-YBJ simulations.

Total velocity components:
- Horizontal: u_total = u_QG + u_wave, v_total = v_QG + v_wave  
- Vertical: w_total from QG omega equation or YBJ formulation

The wave velocities come from the Stokes drift and wave-induced corrections:
u_wave = Real[(∂A*/∂x)A + A*(∂A/∂x)], v_wave = Real[(∂A*/∂y)A + A*(∂A/∂y)]

For YBJ formulation, see equations in QG_YBJp.pdf.
"""
function compute_total_velocities!(S::State, G::Grid; plans=nothing, params=nothing, compute_w=true, use_ybj_w=false)
    # First compute QG velocities
    compute_velocities!(S, G; plans=plans, params=params, compute_w=compute_w, use_ybj_w=use_ybj_w)
    
    # Add wave-induced velocities
    compute_wave_velocities!(S, G; plans=plans, params=params)
    
    return S
end

"""
    compute_wave_velocities!(S, G; plans=nothing, params=nothing)

Compute wave-induced horizontal velocities and add them to the existing QG velocities.
Based on the YBJ formulation for wave-mean flow interaction.

Wave velocities from Stokes drift and wave corrections:
u_wave = Real[(∂A*/∂x)A + A*(∂A/∂x)]
v_wave = Real[(∂A*/∂y)A + A*(∂A/∂y)]

where A is the wave envelope amplitude.
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

