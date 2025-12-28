#=
================================================================================
                timestep_imex.jl - IMEX Time Integration
================================================================================

IMEX (Implicit-Explicit) time stepping for YBJ+ equation with operator splitting.

The YBJ+ equation for B:
    ∂B/∂t + J(ψ,B) + (i/2)ζB = i·αdisp·kₕ²·A

where A = (L⁺)⁻¹·B (elliptic inversion).

OPERATOR SPLITTING + IMEX-CN:
-----------------------------
Step 1 (Exact refraction): B* = B^n × exp(-i·dt·ζ/2)  [in physical space]
Step 2 (IMEX-CN for advection + dispersion):
    - EXPLICIT: N(B) = -J(ψ,B)  (advection only)
    - IMPLICIT: L(B) = i·αdisp·kₕ²·(L⁺)⁻¹·B  (dispersion)

The refraction term is handled exactly via integrating factor:
    dB/dt = -(i/2)ζB  →  B(t) = B(0)·exp(-i·ζ·t/2)

This is energy-preserving since |exp(-i·ζ·t/2)| = 1, unlike forward Euler
which amplifies: |1 - i·dt·ζ/2| = sqrt(1 + (dt·ζ/2)²) > 1.

STABILITY:
---------
- Refraction: Exactly energy-preserving (integrating factor)
- Dispersion: Unconditionally stable (implicit CN)
- Advection: CFL limited by dt × U_max / dx < 1
- For U = 0.335 m/s, dx ≈ 550m: dt_max ≈ 1600s

================================================================================
=#

using LinearAlgebra

"""
    IMEXThreadLocal

Per-thread workspace for IMEX tridiagonal solves.
Each thread needs its own working arrays to avoid data races.
"""
struct IMEXThreadLocal
    tri_diag::Vector{ComplexF64}     # Main diagonal
    tri_upper::Vector{ComplexF64}    # Upper diagonal
    tri_lower::Vector{ComplexF64}    # Lower diagonal
    tri_rhs::Vector{ComplexF64}      # RHS for solve
    tri_sol::Vector{ComplexF64}      # Solution
    tri_c_prime::Vector{ComplexF64}  # Work array for forward elimination
    tri_d_prime::Vector{ComplexF64}  # Work array for forward elimination
    B_col::Vector{ComplexF64}        # Column work vector
    A_col::Vector{ComplexF64}        # Column work vector
    RHS_col::Vector{ComplexF64}      # Column work vector
end

function init_thread_local(nz::Int)
    # Use zeros instead of undef to avoid garbage values causing NaN
    return IMEXThreadLocal(
        zeros(ComplexF64, nz),      # tri_diag
        zeros(ComplexF64, nz-1),    # tri_upper
        zeros(ComplexF64, nz-1),    # tri_lower
        zeros(ComplexF64, nz),      # tri_rhs
        zeros(ComplexF64, nz),      # tri_sol
        zeros(ComplexF64, nz),      # tri_c_prime
        zeros(ComplexF64, nz),      # tri_d_prime
        zeros(ComplexF64, nz),      # B_col
        zeros(ComplexF64, nz),      # A_col
        zeros(ComplexF64, nz)       # RHS_col
    )
end

"""
    IMEXWorkspace

Pre-allocated workspace for IMEX time stepping with threading support.
"""
struct IMEXWorkspace{CT, RT}
    # Tendency arrays (shared, written before parallel region)
    nBk::CT      # J(ψ, B) advection
    rBk::CT      # ζ × B refraction
    nqk::CT      # J(ψ, q) advection
    dqk::CT      # Vertical diffusion

    # Temporary arrays for IMEX
    RHS::CT      # Right-hand side for elliptic solve
    Atemp::CT    # Temporary A storage
    αdisp_profile::Vector{Float64}  # αdisp(z) cache (length nz)
    r_ut::Vector{Float64}           # Unstaggered density weights (length nz)
    r_st::Vector{Float64}           # Staggered density weights (length nz)

    # Per-thread workspace for tridiagonal solves
    thread_local::Vector{IMEXThreadLocal}

    # For q equation (same as original)
    qtemp::CT

    # Grid size for reference
    nz::Int
end

"""
    init_imex_workspace(S, G; nthreads=Threads.maxthreadid())

Initialize workspace for IMEX time stepping with threading support.

NOTE: All work arrays are pre-allocated here to avoid heap corruption
from repeated allocation/deallocation in tight loops during time stepping.
Per-thread workspaces are created for the tridiagonal solves to enable
parallel processing of horizontal modes.
"""
function init_imex_workspace(S, G; nthreads=Threads.maxthreadid())
    CT = typeof(S.B)
    RT = typeof(S.u)

    nBk = similar(S.B)
    rBk = similar(S.B)
    nqk = similar(S.q)
    dqk = similar(S.B)
    RHS = similar(S.B)
    Atemp = similar(S.A)
    # Initialize with zeros to avoid garbage values
    αdisp_profile = zeros(Float64, G.nz)
    r_ut = ones(Float64, G.nz)  # Default to 1 for Boussinesq
    r_st = ones(Float64, G.nz)  # Default to 1 for Boussinesq
    qtemp = similar(S.q)

    nz = G.nz

    # Create per-thread workspaces
    # Use max(1, nthreads) to ensure at least one workspace exists
    n_workspaces = max(1, nthreads)
    thread_local = [init_thread_local(nz) for _ in 1:n_workspaces]

    return IMEXWorkspace{CT, RT}(
        nBk, rBk, nqk, dqk, RHS, Atemp, αdisp_profile, r_ut, r_st,
        thread_local,
        qtemp,
        nz
    )
end

"""
    apply_refraction_exact!(Bk_out, Bk_in, ψk, G, par, plans)

Apply exact refraction using integrating factor (operator splitting step 1).

Solves: dB/dt = -(i/2)ζB  exactly over one timestep dt.
Solution: B(dt) = B(0) × exp(-i·dt·ζ/2)

This is energy-preserving since |exp(-i·dt·ζ/2)| = 1 for real ζ.

# Arguments
- `Bk_out`: Output wave envelope in spectral space
- `Bk_in`: Input wave envelope in spectral space
- `ψk`: Streamfunction in spectral space (for computing ζ = ∇²ψ)
- `G`: Grid
- `par`: Parameters (uses par.dt and par.passive_scalar)
- `plans`: FFT plans

# Notes
If par.passive_scalar is true, refraction is skipped (just copies Bk_in to Bk_out).
"""
function apply_refraction_exact!(Bk_out, Bk_in, ψk, G, par, plans)
    # Skip refraction for passive scalar mode
    if par.passive_scalar
        parent(Bk_out) .= parent(Bk_in)
        return Bk_out
    end

    dt = par.dt
    nx, ny, nz = G.nx, G.ny, G.nz

    ψ_arr = parent(ψk)
    nz_spec, nx_spec, ny_spec = size(ψ_arr)

    # Compute vorticity ζ = -kₕ²ψ in spectral space
    ζk = similar(ψk)
    ζk_arr = parent(ζk)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, ψk)
        j_global = local_to_global(j_local, 3, ψk)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        ζk_arr[k, i_local, j_local] = -kₕ² * ψ_arr[k, i_local, j_local]
    end

    # Transform to physical space
    ζ_phys = allocate_fft_backward_dst(ζk, plans)
    B_phys = allocate_fft_backward_dst(Bk_in, plans)
    fft_backward!(ζ_phys, ζk, plans)
    fft_backward!(B_phys, Bk_in, plans)

    ζ_phys_arr = parent(ζ_phys)
    B_phys_arr = parent(B_phys)
    nz_phys, nx_phys, ny_phys = size(ζ_phys_arr)

    # Apply exact integrating factor: B* = B × exp(-i·dt·ζ/2)
    # The factor exp(-i·dt·ζ/2) has magnitude 1 since ζ is real, ensuring energy conservation
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        ζ_val = real(ζ_phys_arr[k, i, j])  # Vorticity is real
        phase_factor = exp(-im * dt * ζ_val / 2)
        B_phys_arr[k, i, j] *= phase_factor
    end

    # Transform back to spectral space
    fft_forward!(Bk_out, B_phys, plans)

    return Bk_out
end

"""
    solve_modified_elliptic!(A, B, G, par, a, kₕ², β_scale, αdisp_profile, r_ut, r_st,
                             tl::IMEXThreadLocal)

Solve the modified elliptic problem for IMEX dispersion:
    (L⁺ - β)·A = B

where L⁺ = (1/ρ)∂/∂z(ρ a(z) ∂/∂z) - kₕ²/4 and β = (dt/2)·i·αdisp(z)·kₕ²

This is a tridiagonal system for each horizontal mode.

# Arguments
- `A`: Output wave amplitude (nz vector for this horizontal mode)
- `B`: Input wave envelope (nz vector)
- `G`: Grid
- `par`: Parameters
- `a`: a_ell = f²/N² array
- `kₕ²`: Horizontal wavenumber squared
- `β_scale`: Scalar factor = (dt/2)·i·kₕ² (multiplied by αdisp_profile[k] per level)
- `αdisp_profile`: αdisp(z) profile, length nz
- `r_ut`: Unstaggered density weights (length nz)
- `r_st`: Staggered density weights (length nz)
- `tl`: IMEXThreadLocal with pre-allocated tridiagonal arrays (thread-local)
"""
function solve_modified_elliptic!(A_out::AbstractVector, B_in::AbstractVector,
                                   G, par, a, kₕ², β_scale, αdisp_profile::AbstractVector,
                                   r_ut::AbstractVector, r_st::AbstractVector,
                                   tl::IMEXThreadLocal)
    nz = G.nz
    # G.dz is a vector of layer thicknesses; assume uniform grid and use first element
    dz = G.dz[1]
    dz² = dz * dz

    # Build tridiagonal system: (L⁺ - β)·A = B
    # where L⁺ = (1/ρ)∂/∂z(ρ a(z) ∂A/∂z) - kₕ²/4·A is the YBJ+ elliptic operator
    # and β = (dt/2)·i·αdisp(z)·kₕ² is the implicit dispersion coefficient
    # With Neumann BCs: ∂A/∂z = 0 at z = 0, Lz

    # This matches the discretization used by invert_B_to_A! (including density weights).
    # The matrix is scaled by dz², so RHS is dz² * B.
    @inbounds for k in 1:nz
        βₖ = β_scale * αdisp_profile[k]
        tl.tri_rhs[k] = dz² * B_in[k]

        if k == 1
            # Bottom boundary (Neumann): A_z = 0 → A[0] = A[1]
            coeff = (r_ut[1] * a[1]) / r_st[1]
            tl.tri_diag[1] = -(coeff + (kₕ² * dz²) / 4.0) - βₖ * dz²
            tl.tri_upper[1] = coeff
        elseif k == nz
            # Top boundary (Neumann): A_z = 0 → A[nz+1] = A[nz]
            coeff = (r_ut[nz-1] * a[nz-1]) / r_st[nz]
            tl.tri_lower[nz-1] = coeff
            tl.tri_diag[nz] = -(coeff + (kₕ² * dz²) / 4.0) - βₖ * dz²
        else
            coeff_down = (r_ut[k-1] * a[k-1]) / r_st[k]
            coeff_up = (r_ut[k] * a[k]) / r_st[k]
            tl.tri_lower[k-1] = coeff_down
            tl.tri_diag[k] = -(coeff_up + coeff_down + (kₕ² * dz²) / 4.0) - βₖ * dz²
            tl.tri_upper[k] = coeff_up
        end
    end

    # Handle near-mean modes (very small kₕ²): operator is nearly singular
    # For small kₕ², the operator L⁺ - β can become ill-conditioned because
    # β ∝ kₕ² becomes comparable to the L⁺ eigenvalues.
    # Use a threshold based on machine precision and typical operator magnitude.
    # For kₕ² < 1e-12, use explicit forward step instead of implicit solve.
    if kₕ² < 1e-12
        fill!(A_out, zero(eltype(A_out)))
        return A_out
    end

    # Solve tridiagonal system using Thomas algorithm
    # Use pre-allocated work arrays from thread-local workspace
    solve_tridiagonal_complex!(A_out, tl.tri_lower, tl.tri_diag, tl.tri_upper, tl.tri_rhs, nz,
                                tl.tri_c_prime, tl.tri_d_prime)

    return A_out
end

"""
    solve_tridiagonal_complex!(x, a, b, c, d, n, c_prime, d_prime)

Solve tridiagonal system with complex coefficients using Thomas algorithm.
    a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]

# Arguments
- `x`: Solution vector (output)
- `a`: Lower diagonal (length n-1)
- `b`: Main diagonal (length n)
- `c`: Upper diagonal (length n-1)
- `d`: RHS (length n)
- `n`: System size
- `c_prime`: Pre-allocated work vector (length n)
- `d_prime`: Pre-allocated work vector (length n)

NOTE: c_prime and d_prime must be pre-allocated to avoid heap corruption
from repeated allocation/deallocation in tight loops.
"""
function solve_tridiagonal_complex!(x::AbstractVector{ComplexF64},
                                     a::AbstractVector{ComplexF64},
                                     b::AbstractVector{ComplexF64},
                                     c::AbstractVector{ComplexF64},
                                     d::AbstractVector{ComplexF64}, n::Int,
                                     c_prime::AbstractVector{ComplexF64},
                                     d_prime::AbstractVector{ComplexF64})
    # Forward elimination (using pre-allocated work arrays)
    c_prime[1] = c[1] / b[1]
    d_prime[1] = d[1] / b[1]

    @inbounds for i in 2:n-1
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    end

    # Last row
    denom = b[n] - a[n-1] * c_prime[n-1]
    d_prime[n] = (d[n] - a[n-1] * d_prime[n-1]) / denom

    # Back substitution
    x[n] = d_prime[n]
    @inbounds for i in n-1:-1:1
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    end

    return x
end

"""
    first_imex_step!(S, G, par, plans, imex_ws; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

First-order forward Euler step to initialize IMEX time stepping.
Similar to first_projection_step! but uses IMEX structure.
"""
function first_imex_step!(S::State, G::Grid, par::QGParams, plans, imex_ws::IMEXWorkspace;
                          a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)
    # For first step, just use explicit forward Euler (same as projection step)
    # The IMEX structure kicks in from step 2 onwards
    first_projection_step!(S, G, par, plans; a=a, dealias_mask=dealias_mask,
                           workspace=workspace, N2_profile=N2_profile)
    return S
end

"""
    imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

IMEX Crank-Nicolson time step for YBJ+ equation with operator splitting for refraction.

Uses a Lie splitting approach:
1. **Stage 1 (Exact Refraction)**: B* = B^n × exp(-i·dt·ζ/2)
   - Solves dB/dt = -(i/2)ζB exactly using integrating factor
   - Energy-preserving since |exp(-i·dt·ζ/2)| = 1
2. **Stage 2 (IMEX-CN for Advection + Dispersion)**: Starting from B*
   - Advection treated explicitly: N(B) = -J(ψ,B)
   - Dispersion treated implicitly via Crank-Nicolson

This fixes the instability of forward Euler for refraction, which amplifies:
|1 - i·dt·ζ/2| = sqrt(1 + (dt·ζ/2)²) > 1.

# Algorithm
1. Apply exact refraction: B* = B^n × exp(-i·dt·ζ/2) in physical space
2. Compute advection tendency: N = -J(ψ,B^n)
3. For each spectral mode (kx, ky):
   a. Compute A* = (L⁺)⁻¹B* (essential for IMEX-CN consistency!)
   b. Build RHS = B* + (dt/2)·i·αdisp·kₕ²·A* + dt·N
   c. Solve modified elliptic: (L⁺ - (dt/2)·i·αdisp·kₕ²)·A^{n+1} = RHS
   d. Recover B^{n+1} = RHS + (dt/2)·i·αdisp·kₕ²·A^{n+1}

# Arguments
- `Snp1::State`: State at time n+1 (output)
- `Sn::State`: State at time n (input)
- `G::Grid`: Grid
- `par::QGParams`: Parameters
- `plans`: FFT plans
- `imex_ws::IMEXWorkspace`: Pre-allocated workspace
- `a`: Elliptic coefficient a_ell = f²/N²
- `dealias_mask`: Dealiasing mask
- `workspace`: Additional workspace for elliptic solvers
- `N2_profile`: N²(z) profile

# Stability
- Refraction: Exactly energy-preserving (integrating factor)
- Dispersion: Unconditionally stable (implicit CN)
- Advection CFL: dt < dx/U_max ≈ 1600s for this problem
"""
function imex_cn_step!(Snp1::State, Sn::State, G::Grid, par::QGParams, plans, imex_ws::IMEXWorkspace;
                        a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

    if !par.ybj_plus
        error("IMEX time stepping only implemented for ybj_plus=true")
    end

    # Get arrays
    qn_arr = parent(Sn.q)
    Bn_arr = parent(Sn.B)
    An_arr = parent(Sn.A)
    qnp1_arr = parent(Snp1.q)
    Bnp1_arr = parent(Snp1.B)
    Anp1_arr = parent(Snp1.A)

    nz_local, nx_local, ny_local = size(qn_arr)
    nz = G.nz
    dt = par.dt

    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    # Ensure we have enough thread-local workspaces for any thread id in use
    # (thread count at init time may differ from runtime, especially with MPI)
    nthreads = Threads.maxthreadid()
    if length(imex_ws.thread_local) < nthreads
        for _ in (length(imex_ws.thread_local)+1):nthreads
            push!(imex_ws.thread_local, init_thread_local(nz))
        end
    end

    #= Step 1: Update diagnostics ψ, u, v, A for current state =#
    if !par.fixed_flow
        invert_q_to_psi!(Sn, G; a=a, par=par, workspace=workspace)
    end
    # Only compute u, v - not w (omega equation is expensive and not needed for advection)
    compute_velocities!(Sn, G; plans=plans, params=par, N2_profile=N2_profile,
                        workspace=workspace, dealias_mask=L, compute_w=false)
    invert_B_to_A!(Sn, G, par, a; workspace=workspace)

    #= Step 2: Compute explicit tendencies =#
    nqk_arr = parent(imex_ws.nqk)
    nBk_arr = parent(imex_ws.nBk)
    dqk_arr = parent(imex_ws.dqk)
    qtemp_arr = parent(imex_ws.qtemp)

    # Advection - skip q advection if flow is fixed (saves FFTs)
    if !par.fixed_flow
        convol_waqg_q!(imex_ws.nqk, Sn.u, Sn.v, Sn.q, G, plans; Lmask=L)
    end
    convol_waqg_B!(imex_ws.nBk, Sn.u, Sn.v, Sn.B, G, plans; Lmask=L)

    # Note: Refraction is now handled exactly via operator splitting in Step 3.5,
    # so we skip the explicit refraction_waqg_B! computation.

    # Vertical diffusion for q - skip if flow is fixed
    if !par.fixed_flow
        dissipation_q_nv!(imex_ws.dqk, Sn.q, par, G; workspace=workspace)
    end

    #= Step 3: Apply physics switches =#
    if par.inviscid; imex_ws.dqk .= 0; end
    if par.linear; imex_ws.nqk .= 0; imex_ws.nBk .= 0; end
    # Note: passive_scalar flag is handled in apply_refraction_exact!

    # Determine if dispersion is active (needed for IMEX implicit solve)
    dispersion_active = !par.no_dispersion && !par.passive_scalar

    #= Step 3.5: Apply exact refraction via operator splitting =#
    # This is the key fix for stability: instead of treating refraction with
    # forward Euler (which amplifies), we solve it exactly using an integrating factor.
    # B* = B^n × exp(-i·dt·ζ/2) is energy-preserving since |exp(...)| = 1.
    # Store B* in imex_ws.RHS (reused later for IMEX RHS)
    Bstar = imex_ws.RHS  # Temporary storage for refraction-applied B
    apply_refraction_exact!(Bstar, Sn.B, Sn.psi, G, par, plans)

    # Get B* array for use in IMEX loop
    Bstar_arr = parent(Bstar)

    # IMPORTANT: We must compute A* = (L⁺)⁻¹B* for consistency.
    # Using A^n with B* breaks the relation A = (L⁺)⁻¹B that IMEX-CN relies on,
    # causing instability. A* is computed per-mode in the IMEX loop below.

    #= Step 4: Update q with explicit Euler (or could use CN for diffusion) =#
    if par.fixed_flow
        # Just copy q - no evolution when flow is fixed
        qnp1_arr .= qn_arr
    else
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, Sn.q)
            j_global = local_to_global(j, 3, Sn.q)

            if L[i_global, j_global]
                kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
                λₑ = int_factor(kₓ, kᵧ, par; waves=false)
                # Forward Euler with integrating factor for hyperdiffusion
                qnp1_arr[k, i, j] = (qn_arr[k, i, j] - dt*nqk_arr[k, i, j] + dt*dqk_arr[k, i, j]) * exp(-λₑ)
            else
                qnp1_arr[k, i, j] = 0
            end
        end
    end

    #= Step 5: IMEX Crank-Nicolson for B equation =#
    # IMPORTANT: For IMEX-CN to be consistent, the implicit and explicit parts
    # must use the SAME αdisp profile. Use N²(z) when provided; otherwise derive
    # αdisp from a_ell to stay consistent with the elliptic operator.
    #
    # The dispersion coefficient αdisp = N²/(2f₀) appears in the dispersion term:
    #   ∂B/∂t = ... + i·αdisp·kₕ²·A
    αdisp_profile = imex_ws.αdisp_profile
    if length(αdisp_profile) != nz
        resize!(αdisp_profile, nz)
    end
    T = eltype(αdisp_profile)
    if N2_profile !== nothing && length(N2_profile) == nz
        inv_two_f0 = T(1) / (T(2) * T(par.f₀))
        @inbounds for k in 1:nz
            αdisp_profile[k] = T(N2_profile[k]) * inv_two_f0
        end
    else
        half_f0 = T(par.f₀) / T(2)
        @inbounds for k in 1:nz
            αdisp_profile[k] = half_f0 / a[k]
        end
    end

    # Density weights (default to unity for Boussinesq)
    r_ut = imex_ws.r_ut
    r_st = imex_ws.r_st
    if length(r_ut) != nz
        resize!(r_ut, nz)
    end
    if length(r_st) != nz
        resize!(r_st, nz)
    end
    if par.ρ_ut_profile !== nothing
        @inbounds for k in 1:nz
            r_ut[k] = par.ρ_ut_profile[k]
        end
    else
        fill!(r_ut, one(eltype(r_ut)))
    end
    if par.ρ_st_profile !== nothing
        @inbounds for k in 1:nz
            r_st[k] = par.ρ_st_profile[k]
        end
    else
        fill!(r_st, one(eltype(r_st)))
    end

    # Process each horizontal mode
    # For IMEX-CN: (I - dt/2·L)·B^{n+1} = (I + dt/2·L)·B^n + dt·N
    # where L·B = i·αdisp·kₕ²·A and A = (L⁺)⁻¹·B

    # Reformulated: solve for A^{n+1} from modified elliptic problem
    # Then B^{n+1} = L⁺·A^{n+1}

    # Process horizontal modes in parallel using threads
    # Each thread uses its own workspace to avoid data races
    n_modes = nx_local * ny_local

    # Threading disabled by default: benchmarks show serial is faster even for large grids
    # (thread synchronization and cache contention overhead exceed benefits)
    # This may change with different hardware or Julia versions
    use_threading = false

    @inbounds if use_threading
    Threads.@threads for mode_idx in 1:n_modes
        # Convert linear index to (i, j)
        i = ((mode_idx - 1) % nx_local) + 1
        j = ((mode_idx - 1) ÷ nx_local) + 1

        # Get thread-local workspace
        tid = Threads.threadid()
        tl = imex_ws.thread_local[tid]

        i_global = local_to_global(i, 2, Sn.q)
        j_global = local_to_global(j, 3, Sn.q)

        if !L[i_global, j_global]
            # Zero out dealiased modes
            @inbounds for k in 1:nz_local
                Bnp1_arr[k, i, j] = 0
                Anp1_arr[k, i, j] = 0
            end
            continue
        end

        kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)
        hyperdiff_factor = exp(-λʷ)

        # Handle cases where dispersion is disabled or kₕ² ≈ 0
        if !dispersion_active || kₕ² < 1e-14
            # Pure explicit step: B^{n+1} = B* + dt·N^n (with hyperdiffusion)
            # where N = -J(ψ,B) (advection only - refraction already applied exactly)
            # Note: B* = B^n × exp(-i·dt·ζ/2) from operator splitting
            @inbounds for k in 1:nz_local
                N_k = -nBk_arr[k, i, j]  # Advection only (refraction handled by splitting)
                Bnp1_arr[k, i, j] = (Bstar_arr[k, i, j] + dt * N_k) * hyperdiff_factor
                Anp1_arr[k, i, j] = 0  # No dispersion means A is meaningless
            end
        else
            # IMEX-CN for B equation with dispersion (after exact refraction)
            # NOTE: nz_local == nz for xy-pencil decomposition (z is not distributed)

            # Step 1: Compute A* = (L⁺)⁻¹B* for consistency with B*
            # This is critical: using A^n with B* breaks IMEX-CN stability!
            @inbounds for k in 1:nz_local
                tl.B_col[k] = Bstar_arr[k, i, j]
            end
            # Use β_scale=0 to get standard L⁺ inversion (A* = (L⁺)⁻¹B*)
            solve_modified_elliptic!(tl.A_col, tl.B_col, G, par, a, kₕ²,
                                     complex(0.0), αdisp_profile, r_ut, r_st, tl)

            # Step 2: Build RHS for IMEX-CN using B* and A*
            @inbounds for k in 1:nz_local
                N_k = -nBk_arr[k, i, j]  # Advection only (refraction handled by splitting)
                disp_star = im * αdisp_profile[k] * kₕ² * tl.A_col[k]  # Uses A*, not A^n!
                tl.RHS_col[k] = Bstar_arr[k, i, j] + (dt/2) * disp_star + dt * N_k
            end

            # Step 3: Solve modified elliptic problem for A^{n+1}
            # IMEX-CN formulation: (L⁺ - β)·A^{n+1} = RHS where β = (dt/2)·i·αdisp·kₕ²
            β_scale = (dt/2) * im * kₕ²

            solve_modified_elliptic!(tl.A_col, tl.RHS_col, G, par, a, kₕ²,
                                     β_scale, αdisp_profile, r_ut, r_st, tl)

            # Recover B using L⁺·A = RHS + β·A (consistent with the modified solve)
            @inbounds for k in 1:nz_local
                βₖ = β_scale * αdisp_profile[k]
                Bnp1_arr[k, i, j] = (tl.RHS_col[k] + βₖ * tl.A_col[k]) * hyperdiff_factor
                Anp1_arr[k, i, j] = tl.A_col[k] * hyperdiff_factor
            end
        end
    end  # end threaded for
    else
    # Serial path for small grids (threading overhead exceeds benefit)
    for mode_idx in 1:n_modes
        i = ((mode_idx - 1) % nx_local) + 1
        j = ((mode_idx - 1) ÷ nx_local) + 1

        tid = 1  # Use first workspace
        tl = imex_ws.thread_local[tid]

        i_global = local_to_global(i, 2, Sn.q)
        j_global = local_to_global(j, 3, Sn.q)

        if !L[i_global, j_global]
            for k in 1:nz_local
                Bnp1_arr[k, i, j] = 0
                Anp1_arr[k, i, j] = 0
            end
            continue
        end

        kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)
        hyperdiff_factor = exp(-λʷ)

        if !dispersion_active || kₕ² < 1e-14
            # Explicit update for mean mode or when dispersion is disabled
            # Use B* (after exact refraction) and advection only
            for k in 1:nz_local
                N_k = -nBk_arr[k, i, j]  # Advection only (refraction handled by splitting)
                Bnp1_arr[k, i, j] = (Bstar_arr[k, i, j] + dt * N_k) * hyperdiff_factor
                Anp1_arr[k, i, j] = 0
            end
        else
            # Step 1: Compute A* = (L⁺)⁻¹B* for consistency with the refraction-rotated B*
            # This is critical: using A^n with B* breaks IMEX-CN stability!
            # We solve L⁺·A* = B* using the elliptic solver with β=0
            for k in 1:nz_local
                tl.B_col[k] = Bstar_arr[k, i, j]
            end
            # Use β_scale=0 to get standard L⁺ inversion (A* = (L⁺)⁻¹B*)
            solve_modified_elliptic!(tl.A_col, tl.B_col, G, par, a, kₕ²,
                                     complex(0.0), αdisp_profile, r_ut, r_st, tl)

            # Step 2: Build RHS for IMEX solve using B* and A*
            for k in 1:nz_local
                N_k = -nBk_arr[k, i, j]  # Advection only (refraction handled by splitting)
                disp_star = im * αdisp_profile[k] * kₕ² * tl.A_col[k]  # Uses A*, not A^n!
                tl.RHS_col[k] = Bstar_arr[k, i, j] + (dt/2) * disp_star + dt * N_k
            end

            # Check if RHS is too small for stable implicit solve
            rhs_max = maximum(k -> abs(tl.RHS_col[k]), 1:nz_local)
            if rhs_max < 1e-20
                # Explicit forward Euler: B^{n+1} = B* + dt*(N + disp*)
                for k in 1:nz_local
                    N_k = -nBk_arr[k, i, j]
                    disp_k = im * αdisp_profile[k] * kₕ² * tl.A_col[k]  # Uses A*
                    Bnp1_arr[k, i, j] = (Bstar_arr[k, i, j] + dt * (N_k + disp_k)) * hyperdiff_factor
                    Anp1_arr[k, i, j] = tl.A_col[k] * hyperdiff_factor
                end
                continue
            end

            # Step 3: Solve modified elliptic problem for A^{n+1}
            β_scale = (dt/2) * im * kₕ²

            solve_modified_elliptic!(tl.A_col, tl.RHS_col, G, par, a, kₕ²,
                                     β_scale, αdisp_profile, r_ut, r_st, tl)

            # Recover B from the IMEX-CN relation: B = RHS + β*A
            for k in 1:nz_local
                βₖ = β_scale * αdisp_profile[k]
                Bnp1_arr[k, i, j] = (tl.RHS_col[k] + βₖ * tl.A_col[k]) * hyperdiff_factor
                Anp1_arr[k, i, j] = tl.A_col[k] * hyperdiff_factor
            end
        end
    end  # end serial for
    end  # end if use_threading

    #= Step 6: Wave feedback on mean flow =#
    wave_feedback_enabled = !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback
    if wave_feedback_enabled
        # Reuse qtemp from workspace to avoid allocation every timestep
        # (allocation inside tight loops causes heap corruption in MPI)
        qwk = imex_ws.qtemp
        qwk_arr = parent(qwk)
        compute_qw_complex!(qwk, Snp1.B, par, G, plans; Lmask=L)

        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, Snp1.q)
            j_global = local_to_global(j, 3, Snp1.q)
            if L[i_global, j_global]
                qnp1_arr[k, i, j] -= qwk_arr[k, i, j]
            end
        end
    end

    #= Step 7: Update diagnostics for new state =#
    if !par.fixed_flow
        invert_q_to_psi!(Snp1, G; a=a, par=par, workspace=workspace)
    else
        # Copy psi from Sn
        parent(Snp1.psi) .= parent(Sn.psi)
    end

    # A is already computed above for most modes
    # For kₕ² ≈ 0 modes, do standard inversion
    # Actually, let's just do full inversion to be safe
    invert_B_to_A!(Snp1, G, par, a; workspace=workspace)

    # Only compute u, v for diagnostics - w can be computed separately if needed
    compute_velocities!(Snp1, G; plans=plans, params=par, N2_profile=N2_profile,
                        workspace=workspace, dealias_mask=L, compute_w=false)

    return Snp1
end
