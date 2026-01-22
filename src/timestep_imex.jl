#=
================================================================================
                timestep_imex.jl - IMEX Time Integration
================================================================================

Second-order IMEX (Implicit-Explicit) time stepping for YBJ+ equation with
Strang operator splitting and Adams-Bashforth 2 for advection.

OPERATOR DEFINITIONS (from PDF):
--------------------------------
    L  (YBJ operator):   L  = ∂/∂z(f²/N² ∂/∂z)              [eq. (4)]
    L⁺ (YBJ+ operator):  L⁺ = L - k_h²/4                     [spectral space]

Key relation: L = L⁺ + k_h²/4

YBJ+ EQUATION:
--------------
The prognostic variable is B = L⁺A. The YBJ+ equation for B:
    ∂B/∂t + J(ψ,L⁺A) = i·αdisp·kₕ²·A - (i/2)ζB

where:
- A = (L⁺)⁻¹·B (elliptic inversion)
- αdisp = f₀/2 (dispersion coefficient)
- ζ = ∇²ψ (relative vorticity)

Note: The dispersion term uses A (not LA), because the full wave equation
dispersion +i(f/2)k²LA reduces to +i(f/2)k²A when using B = L⁺A as the
prognostic variable.

SECOND-ORDER SCHEME: STRANG SPLITTING + IMEX-CNAB
-------------------------------------------------
Step 1: First half-refraction (Strang)
    B* = B^n × exp(-i·(dt/2)·ζ/2)

Step 2: IMEX-CNAB for advection + dispersion
    - EXPLICIT (AB2): (3/2)N^n - (1/2)N^{n-1}  where N = -J(ψ,B*) and
      B* is the half-refraction state
    - IMPLICIT (CN):  (1/2)[L(B*) + L(B^{n+1})]  where L(B) = i·(f/2)·kₕ²·A

Step 3: Second half-refraction (Strang)
    B^{n+1} = B** × exp(-i·(dt/2)·ζ/2)

The refraction term is handled exactly via integrating factor:
    dB/dt = -(i/2)ζB  →  B(t) = B(0)·exp(-i·ζ·t/2)

This is energy-preserving since |exp(-i·ζ·t/2)| = 1.

TEMPORAL ACCURACY:
-----------------
- Strang splitting: Second-order
- Crank-Nicolson for dispersion: Second-order
- Adams-Bashforth 2 for advection: Second-order
- Overall: SECOND-ORDER in time

STABILITY:
---------
- Refraction: Exactly energy-preserving (integrating factor)
- Dispersion: Unconditionally stable (implicit CN)
- Advection: CFL limited by dt × U_max / dx < 1
- For U = 0.335 m/s, dx ≈ 550m: dt_max ≈ 1600s

NOTE: First time step uses forward Euler for advection (AB2 bootstrap).

================================================================================
=#

using LinearAlgebra

const IMEX_KH2_EPS = 1e-12

function _prefilter_spectral!(dst, src, G::Grid, Lmask)
    nx, ny = G.nx, G.ny
    src_arr = parent(src)
    dst_arr = parent(dst)
    nz_local, nx_local, ny_local = size(src_arr)

    use_inline_dealias = isnothing(Lmask)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, src)
        j_global = local_to_global(j_local, 3, src)
        keep = use_inline_dealias ? is_dealiased(i_global, j_global, nx, ny) : Lmask[i_global, j_global]
        dst_arr[k, i_local, j_local] = keep ? src_arr[k, i_local, j_local] : zero(eltype(dst_arr))
    end
    return dst
end

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
    L⁺A_col::Vector{ComplexF64}        # Column work vector
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
        zeros(ComplexF64, nz),      # L⁺A_col
        zeros(ComplexF64, nz),      # A_col
        zeros(ComplexF64, nz)       # RHS_col
    )
end

"""
    IMEXWorkspace

Pre-allocated workspace for IMEX time stepping with threading support.

For second-order IMEX-CNAB with Strang splitting, we store the previous
tendencies for both q and B to enable Adams-Bashforth 2 extrapolation.
"""
struct IMEXWorkspace{CT, RT}
    # Tendency arrays (shared, written before parallel region)
    nL⁺Ak::CT      # J(ψ, L⁺A) advection at time n
    nL⁺Ak_prev::CT # J(ψ, L⁺A) advection at time n-1 (for AB2)
    rL⁺Ak::CT      # ζ × B refraction
    nqk::CT      # J(ψ, q) advection
    dqk::CT      # Vertical diffusion
    tqk_prev::CT # q advection tendency at time n-1 (for AB2)

    # Temporary arrays for IMEX
    RHS::CT      # Right-hand side for elliptic solve
    L⁺Astar::CT    # B after first half-refraction (for Strang splitting)
    Atemp::CT    # Temporary A storage
    αdisp_profile::Vector{Float64}  # αdisp(z) cache (length nz)

    # Per-thread workspace for tridiagonal solves
    thread_local::Vector{IMEXThreadLocal}

    # For q equation (same as original)
    qtemp::CT

    # Grid size for reference
    nz::Int

    # Flag to track if previous tendency is valid (for AB2 bootstrap)
    has_prev_tendency::Base.RefValue{Bool}
end

"""
    init_imex_workspace(S, G; nthreads=Threads.maxthreadid())

Initialize workspace for IMEX time stepping with threading support.

NOTE: All work arrays are pre-allocated here to avoid heap corruption
from repeated allocation/deallocation in tight loops during time stepping.
Per-thread workspaces are created for the tridiagonal solves to enable
parallel processing of horizontal modes.

The workspace includes storage for Adams-Bashforth 2 (previous tendency)
and Strang splitting (intermediate B* state).
"""
function init_imex_workspace(S, G; nthreads=Threads.maxthreadid())
    CT = typeof(S.L⁺A)
    RT = typeof(S.u)

    nL⁺Ak = similar(S.L⁺A)
    nL⁺Ak_prev = similar(S.L⁺A)  # For AB2: stores N^{n-1}
    rL⁺Ak = similar(S.L⁺A)
    nqk = similar(S.q)
    dqk = similar(S.L⁺A)
    tqk_prev = similar(S.q)  # For AB2: stores N^{n-1} = -J(ψ,q)
    RHS = similar(S.L⁺A)
    L⁺Astar = similar(S.L⁺A)     # For Strang: B after first half-refraction
    Atemp = similar(S.A)
    # Initialize with zeros to avoid garbage values
    αdisp_profile = zeros(Float64, G.nz)
    qtemp = similar(S.q)

    nz = G.nz

    # Create per-thread workspaces
    # Use max(1, nthreads) to ensure at least one workspace exists
    n_workspaces = max(1, nthreads)
    thread_local = [init_thread_local(nz) for _ in 1:n_workspaces]

    # Flag for AB2 bootstrap (first step can't use AB2)
    has_prev_tendency = Ref(false)

    return IMEXWorkspace{CT, RT}(
        nL⁺Ak, nL⁺Ak_prev, rL⁺Ak, nqk, dqk, tqk_prev, RHS, L⁺Astar, Atemp, αdisp_profile,
        thread_local,
        qtemp,
        nz,
        has_prev_tendency
    )
end

"""
    apply_refraction_exact!(L⁺Ak_out, L⁺Ak_in, ψk, G, par, plans; dt_fraction=1.0, dealias_mask=nothing)

Apply exact refraction using integrating factor (operator splitting).

Solves: dB/dt = -(i/2)ζB  exactly over time `dt_fraction * par.dt`.
Solution: B(Δt) = B(0) × exp(-i·Δt·ζ/2)

This is energy-preserving since |exp(-i·Δt·ζ/2)| = 1 for real ζ.

# Arguments
- `L⁺Ak_out`: Output wave envelope in spectral space
- `L⁺Ak_in`: Input wave envelope in spectral space
- `ψk`: Streamfunction in spectral space (for computing ζ = ∇²ψ)
- `G`: Grid
- `par`: Parameters (uses par.dt and par.passive_scalar)
- `plans`: FFT plans
- `dt_fraction`: Fraction of timestep to apply (default 1.0). Use 0.5 for Strang splitting.
- `dealias_mask`: Dealiasing mask (true = keep mode, false = zero)

# Notes
- If par.passive_scalar is true, refraction is skipped (just copies L⁺Ak_in to L⁺Ak_out).
- For Strang splitting (second-order), call with dt_fraction=0.5 before and after the IMEX step.
- For Lie splitting (first-order), call with dt_fraction=1.0 before the IMEX step only.
"""
function apply_refraction_exact!(L⁺Ak_out, L⁺Ak_in, ψk, G, par, plans;
                                 dt_fraction::Real=1.0, dealias_mask=nothing)
    dt = par.dt * dt_fraction
    nx, ny, nz = G.nx, G.ny, G.nz

    ψ_arr = parent(ψk)
    nz_spec, nx_spec, ny_spec = size(ψ_arr)

    use_inline_dealias = isnothing(dealias_mask)
    @inline should_keep(i_g, j_g) = use_inline_dealias ? is_dealiased(i_g, j_g, nx, ny) : dealias_mask[i_g, j_g]

    # Skip refraction for passive scalar mode, but still enforce dealiasing
    if par.passive_scalar
        parent(L⁺Ak_out) .= parent(L⁺Ak_in)
        L⁺Ak_out_arr = parent(L⁺Ak_out)
        nz_spec, nx_spec, ny_spec = size(L⁺Ak_out_arr)
        @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
            i_global = local_to_global(i_local, 2, L⁺Ak_out)
            j_global = local_to_global(j_local, 3, L⁺Ak_out)
            if !should_keep(i_global, j_global)
                L⁺Ak_out_arr[k, i_local, j_local] = 0
            end
        end
        return L⁺Ak_out
    end

    # Compute vorticity ζ = -kₕ²ψ in spectral space
    ζk = similar(ψk)
    ζk_arr = parent(ζk)

    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, ψk)
        j_global = local_to_global(j_local, 3, ψk)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        if should_keep(i_global, j_global)
            ζk_arr[k, i_local, j_local] = -kₕ² * ψ_arr[k, i_local, j_local]
        else
            ζk_arr[k, i_local, j_local] = 0
        end
    end

    # Transform to physical space
    ζ_phys = allocate_fft_backward_dst(ζk, plans)
    L⁺A_phys = allocate_fft_backward_dst(L⁺Ak_in, plans)
    fft_backward!(ζ_phys, ζk, plans)
    L⁺Ak_f = similar(L⁺Ak_in)
    _prefilter_spectral!(L⁺Ak_f, L⁺Ak_in, G, dealias_mask)
    fft_backward!(L⁺A_phys, L⁺Ak_f, plans)

    ζ_phys_arr = parent(ζ_phys)
    L⁺A_phys_arr = parent(L⁺A_phys)
    nz_phys, nx_phys, ny_phys = size(ζ_phys_arr)

    # Apply exact integrating factor: B* = B × exp(-i·dt·ζ/2)
    # From YBJ+ equation (1.4): refraction term is -(i/2)ζB, so solution is B(t) = B(0)·exp(-iζt/2)
    # The factor exp(-i·dt·ζ/2) has magnitude 1 since ζ is real, ensuring energy conservation
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        ζ_val = real(ζ_phys_arr[k, i, j])  # Vorticity is real
        phase_factor = exp(-im * dt * ζ_val / 2)
        L⁺A_phys_arr[k, i, j] *= phase_factor
    end

    # Transform back to spectral space
    fft_forward!(L⁺Ak_out, L⁺A_phys, plans)

    # Apply dealiasing mask to remove quadratic aliasing from ζ·B product
    L⁺Ak_out_arr = parent(L⁺Ak_out)
    nz_spec, nx_spec, ny_spec = size(L⁺Ak_out_arr)
    @inbounds for k in 1:nz_spec, j_local in 1:ny_spec, i_local in 1:nx_spec
        i_global = local_to_global(i_local, 2, L⁺Ak_out)
        j_global = local_to_global(j_local, 3, L⁺Ak_out)
        if !should_keep(i_global, j_global)
            L⁺Ak_out_arr[k, i_local, j_local] = 0
        end
    end

    return L⁺Ak_out
end

"""
    solve_modified_elliptic!(A, B, G, par, a, kₕ², β_scale, αdisp_profile, tl::IMEXThreadLocal)

Solve the modified elliptic problem for IMEX dispersion:
    (L⁺ - β)·A = B

where L⁺ = ∂/∂z(a(z) ∂/∂z) - kₕ²/4 and β = (dt/2)·i·αdisp(z)·kₕ² (Boussinesq)

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
- `tl`: IMEXThreadLocal with pre-allocated tridiagonal arrays (thread-local)
"""
function solve_modified_elliptic!(A_out::AbstractVector, B_in::AbstractVector,
                                   G, par, a, kₕ², β_scale, αdisp_profile::AbstractVector,
                                   tl::IMEXThreadLocal)
    nz = G.nz
    if nz == 1
        β₁ = β_scale * αdisp_profile[1]
        denom = (kₕ² / 4.0) + β₁
        if abs(denom) < IMEX_KH2_EPS
            A_out[1] = zero(eltype(A_out))
        else
            A_out[1] = -B_in[1] / denom
        end
        return A_out
    end
    # G.dz is a vector of layer thicknesses; assume uniform grid and use first element
    dz = G.dz[1]
    dz² = dz * dz

    # Build tridiagonal system: (L⁺ - β)·A = B
    # where L⁺ = ∂/∂z(a(z) ∂A/∂z) - kₕ²/4·A is the YBJ+ elliptic operator (Boussinesq)
    # and β = (dt/2)·i·αdisp(z)·kₕ² is the implicit dispersion coefficient
    # With Neumann BCs: ∂A/∂z = 0 at z = -Lz, 0

    # This matches the discretization used by invert_L⁺A_to_A!
    # The matrix is scaled by dz², so RHS is dz² * B.
    @inbounds for k in 1:nz
        βₖ = β_scale * αdisp_profile[k]
        tl.tri_rhs[k] = dz² * B_in[k]

        if k == 1
            # Bottom boundary (Neumann): A_z = 0 → A[0] = A[1]
            tl.tri_diag[1] = -(a[1] + (kₕ² * dz²) / 4.0) - βₖ * dz²
            tl.tri_upper[1] = a[1]
        elseif k == nz
            # Top boundary (Neumann): A_z = 0 → A[nz+1] = A[nz]
            tl.tri_lower[nz-1] = a[nz]
            tl.tri_diag[nz] = -(a[nz] + (kₕ² * dz²) / 4.0) - βₖ * dz²
        else
            tl.tri_lower[k-1] = a[k]
            tl.tri_diag[k] = -(a[k+1] + a[k] + (kₕ² * dz²) / 4.0) - βₖ * dz²
            tl.tri_upper[k] = a[k+1]
        end
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

This function:
1. Performs a first-order forward Euler step (same as first_projection_step!)
2. Initializes the AB2 state by computing and storing the q and B advection tendencies

After this function is called, subsequent calls to imex_cn_step! will use
second-order Adams-Bashforth 2 for advection.
"""
function first_imex_step!(S::State, G::Grid, par::QGParams, plans, imex_ws::IMEXWorkspace;
                          a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing,
                          particle_tracker=nothing, current_time=nothing)
    # Reset AB2 state - first step must use forward Euler
    imex_ws.has_prev_tendency[] = false

    # Initialize AB2 history at t=0 before advancing the state
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    if !par.fixed_flow
        invert_q_to_psi!(S, G; a=a, par=par, workspace=workspace)
    end
    compute_velocities!(S, G; plans=plans, params=par, N2_profile=N2_profile,
                        workspace=workspace, dealias_mask=L, compute_w=false)

    # Use half-step refraction state for AB2 history to stay consistent with Strang splitting
    apply_refraction_exact!(imex_ws.L⁺Astar, S.L⁺A, S.psi, G, par, plans;
                            dt_fraction=0.5, dealias_mask=L)
    if par.linear
        fill!(parent(imex_ws.nL⁺Ak_prev), zero(eltype(imex_ws.nL⁺Ak_prev)))
    else
        convol_waqg_L⁺A!(imex_ws.nL⁺Ak_prev, S.u, S.v, imex_ws.L⁺Astar, G, plans; Lmask=L)
    end

    # Initialize q advection history for AB2
    tqk_prev_arr = parent(imex_ws.tqk_prev)
    if par.fixed_flow
        fill!(tqk_prev_arr, zero(eltype(tqk_prev_arr)))
    else
        convol_waqg_q!(imex_ws.nqk, S.u, S.v, S.q, G, plans; Lmask=L)
        if par.linear; imex_ws.nqk .= 0; end
        nqk_arr = parent(imex_ws.nqk)
        nz_local, nx_local, ny_local = size(tqk_prev_arr)
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, S.q)
            j_global = local_to_global(j, 3, S.q)
            if L[i_global, j_global]
                tqk_prev_arr[k, i, j] = -nqk_arr[k, i, j]
            else
                tqk_prev_arr[k, i, j] = 0
            end
        end
    end

    imex_ws.has_prev_tendency[] = true

    # For first step, just use explicit forward Euler (same as projection step)
    # The IMEX structure kicks in from step 2 onwards
    # Pass particle_tracker so particles also advance during the first step
    first_projection_step!(S, G, par, plans; a=a, dealias_mask=dealias_mask,
                           workspace=workspace, N2_profile=N2_profile,
                           particle_tracker=particle_tracker, current_time=current_time)

    return S
end

"""
    imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

Second-order IMEX-CNAB time step for YBJ+ equation with Strang splitting for refraction.

Uses Strang splitting with Adams-Bashforth 2 for advection (waves and mean flow) and
Crank-Nicolson for linear PV diffusion/hyperdiffusion:
1. **Stage 1 (First Half-Refraction)**: B* = B^n × exp(-i·(dt/2)·ζ/2)
2. **Stage 2 (IMEX-CNAB for Advection + Dispersion)**:
   - Advection (AB2): (3/2)N^n - (1/2)N^{n-1} where N = -J(ψ,L⁺A)
   - Dispersion (CN): (1/2)[L(B*) + L(B^{n+1})]
3. **Stage 3 (Second Half-Refraction)**: B^{n+1} = B** × exp(-i·(dt/2)·ζ/2) using ψ^{n+1} predictor

This achieves second-order temporal accuracy through:
- Strang splitting (second-order) instead of Lie splitting (first-order)
- Adams-Bashforth 2 (second-order) for advection (q and B)
- Crank-Nicolson for dispersion (second-order)

# Algorithm
1. Compute q advection at time n: Q^n = -J(ψ^n, q^n)
2. Apply first half-refraction: B* = B^n × exp(-i·(dt/2)·ζ/2)
3. Compute B advection using B*: N^n = -J(ψ^n, B*)
4. For each spectral mode (kx, ky) solve IMEX-CNAB for B:
   a. Compute A* = (L⁺)⁻¹B* (essential for IMEX-CN consistency!)
   b. Build RHS = B* + (dt/2)·i·αdisp·kₕ²·A* + (3dt/2)·N^n - (dt/2)·N^{n-1}
   c. Solve modified elliptic: (L⁺ - β)·A^{n+1} = RHS where β = (dt/2)·i·αdisp·kₕ²
   d. Recover B** = RHS + β·A^{n+1}
5. Update q with CNAB: (I - dt/2·L)·q^{n+1} = (I + dt/2·L)·q^n + dt·[ (3/2)Q^n - (1/2)Q^{n-1} ]
   where L = νz∂zz - λ_h (vertical diffusion + hyperdiffusion)
6. Compute ψ^{n+1} predictor from q^{n+1} (and q^w predictor when enabled)
7. Apply second half-refraction using ψ^{n+1} predictor
8. Store tendencies for next step (AB2)

# Arguments
- `Snp1::State`: State at time n+1 (output)
- `Sn::State`: State at time n (input)
- `G::Grid`: Grid
- `par::QGParams`: Parameters
- `plans`: FFT plans
- `imex_ws::IMEXWorkspace`: Pre-allocated workspace (stores N^{n-1} for AB2)
- `a`: Elliptic coefficient a_ell = f²/N²
- `dealias_mask`: Dealiasing mask
- `workspace`: Additional workspace for elliptic solvers
- `N2_profile`: N²(z) profile

# Temporal Accuracy
- Overall: Second-order in time
- First step: Falls back to forward Euler for advection (AB2 bootstrap)

# Stability
- Refraction: Exactly energy-preserving (integrating factor)
- Dispersion: Unconditionally stable (implicit CN)
- Advection CFL: dt < dx/U_max ≈ 1600s for this problem
"""
function imex_cn_step!(Snp1::State, Sn::State, G::Grid, par::QGParams, plans, imex_ws::IMEXWorkspace;
                        a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing,
                        particle_tracker=nothing, current_time=nothing)

    if !par.ybj_plus
        error("IMEX time stepping only implemented for ybj_plus=true")
    end

    # Get arrays
    qn_arr = parent(Sn.q)
    L⁺An_arr = parent(Sn.L⁺A)
    An_arr = parent(Sn.A)
    qnp1_arr = parent(Snp1.q)
    L⁺Anp1_arr = parent(Snp1.L⁺A)
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
    invert_L⁺A_to_A!(Sn, G, par, a; workspace=workspace)

    #= Step 2: Compute explicit tendencies =#
    nqk_arr = parent(imex_ws.nqk)
    nL⁺Ak_arr = parent(imex_ws.nL⁺Ak)
    dqk_arr = parent(imex_ws.dqk)
    qtemp_arr = parent(imex_ws.qtemp)

    # Advection - skip q advection if flow is fixed (saves FFTs)
    if !par.fixed_flow
        convol_waqg_q!(imex_ws.nqk, Sn.u, Sn.v, Sn.q, G, plans; Lmask=L)
    end
    # Vertical diffusion for q - skip if flow is fixed
    if !par.fixed_flow
        dissipation_q_nv!(imex_ws.dqk, Sn.q, par, G; workspace=workspace)
    end

    #= Step 3: Apply physics switches =#
    if par.inviscid; imex_ws.dqk .= 0; end
    if par.linear; imex_ws.nqk .= 0; end
    # Note: passive_scalar flag is handled in apply_refraction_exact!

    # Determine if dispersion is active (needed for IMEX implicit solve)
    dispersion_active = !par.no_dispersion && !par.passive_scalar

    # Determine AB2 coefficients based on whether we have a valid previous tendency
    # First step (bootstrap): use forward Euler (c_n=1, c_nm1=0)
    # Subsequent steps: use AB2 (c_n=3/2, c_nm1=-1/2)
    use_ab2 = imex_ws.has_prev_tendency[]
    if use_ab2
        c_n = 1.5      # 3/2 for AB2
        c_nm1 = -0.5   # -1/2 for AB2
    else
        c_n = 1.0      # Forward Euler
        c_nm1 = 0.0
    end

    # Get previous tendency arrays (for AB2)
    nL⁺Ak_prev_arr = parent(imex_ws.nL⁺Ak_prev)
    tqk_prev_arr = parent(imex_ws.tqk_prev)

    #= Step 3.5: Apply FIRST HALF refraction via Strang splitting =#
    # For second-order Strang splitting, we apply half the refraction before
    # and half after the IMEX step. This gives O(dt²) splitting error.
    # B* = B^n × exp(-i·(dt/2)·ζ/2) is energy-preserving since |exp(...)| = 1.
    L⁺Astar = imex_ws.L⁺Astar  # Dedicated storage for first-half refraction result
    apply_refraction_exact!(L⁺Astar, Sn.L⁺A, Sn.psi, G, par, plans;
                            dt_fraction=0.5, dealias_mask=L)

    # Get B* array for use in IMEX loop
    L⁺Astar_arr = parent(L⁺Astar)

    # IMPORTANT: Compute B advection using the refraction-rotated state (B*).
    # This keeps the Strang split consistent and preserves second-order accuracy.
    if par.linear
        fill!(nL⁺Ak_arr, zero(eltype(nL⁺Ak_arr)))
    else
        convol_waqg_L⁺A!(imex_ws.nL⁺Ak, Sn.u, Sn.v, L⁺Astar, G, plans; Lmask=L)
    end

    # IMPORTANT: We must compute A* = (L⁺)⁻¹B* for consistency.
    # Using A^n with B* breaks the relation A = (L⁺)⁻¹B that IMEX-CN relies on,
    # causing instability. A* is computed per-mode in the IMEX loop below.

    #= Step 4: Update q with CNAB (advection AB2 + CN for diffusion/hyperdiffusion) =#
    if par.fixed_flow
        # Just copy q - no evolution when flow is fixed
        qnp1_arr .= qn_arr
    else
        nz = G.nz
        dt = par.dt
        Treal = eltype(G.z)
        νz = par.inviscid ? zero(Treal) : Treal(par.νz)
        has_vert_diff = (νz != 0) && (nz > 1)
        Δz = has_vert_diff ? (G.z[2] - G.z[1]) : one(Treal)
        α = has_vert_diff ? (νz / (Δz * Δz)) : zero(Treal)
        off = has_vert_diff ? (-0.5 * dt * α) : zero(Treal)

        tl = imex_ws.thread_local[1]
        tri_diag = tl.tri_diag
        tri_upper = tl.tri_upper
        tri_lower = tl.tri_lower
        tri_rhs = tl.tri_rhs
        tri_sol = tl.tri_sol
        tri_c_prime = tl.tri_c_prime
        tri_d_prime = tl.tri_d_prime

        @inbounds for j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, Sn.q)
            j_global = local_to_global(j, 3, Sn.q)

            if L[i_global, j_global]
                kₓ = G.kx[i_global]
                kᵧ = G.ky[j_global]
                λ_dt = int_factor(kₓ, kᵧ, par; waves=false)
                λ = dt > 0 ? (λ_dt / dt) : zero(Treal)
                denom = one(Treal) + 0.5 * dt * λ

                if !has_vert_diff
                    @inbounds for k in 1:nz_local
                        N_n = -nqk_arr[k, i, j]
                        N_nm1 = use_ab2 ? tqk_prev_arr[k, i, j] : zero(eltype(tqk_prev_arr))
                        rhs = (one(Treal) - 0.5 * dt * λ) * qn_arr[k, i, j] +
                              dt * (c_n * N_n + c_nm1 * N_nm1)
                        qnp1_arr[k, i, j] = rhs / denom
                    end
                else
                    @inbounds for k in 1:nz
                        diag = (k == 1 || k == nz) ?
                               (one(Treal) + 0.5 * dt * α + 0.5 * dt * λ) :
                               (one(Treal) + dt * α + 0.5 * dt * λ)
                        tri_diag[k] = Complex(diag, 0)
                        if k < nz
                            tri_upper[k] = Complex(off, 0)
                            tri_lower[k] = Complex(off, 0)
                        end
                        N_n = -nqk_arr[k, i, j]
                        N_nm1 = use_ab2 ? tqk_prev_arr[k, i, j] : zero(eltype(tqk_prev_arr))
                        tri_rhs[k] = (one(Treal) - 0.5 * dt * λ) * qn_arr[k, i, j] +
                                     dt * (c_n * N_n + c_nm1 * N_nm1) +
                                     0.5 * dt * dqk_arr[k, i, j]
                    end

                    solve_tridiagonal_complex!(
                        tri_sol, tri_lower, tri_diag, tri_upper, tri_rhs, nz,
                        tri_c_prime, tri_d_prime
                    )
                    @inbounds for k in 1:nz
                        qnp1_arr[k, i, j] = tri_sol[k]
                    end
                end
            else
                @inbounds for k in 1:nz_local
                    qnp1_arr[k, i, j] = 0
                end
            end
        end
    end

    #= Step 5: IMEX Crank-Nicolson for B equation =#
    # From YBJ+ equation (1.4): dispersion term is +i(f/2)kₕ²A
    # The dispersion coefficient αdisp = f/2 is CONSTANT (independent of N²)
    # per Asselin & Young (2019)
    αdisp_profile = imex_ws.αdisp_profile
    if length(αdisp_profile) != nz
        resize!(αdisp_profile, nz)
    end
    T = eltype(αdisp_profile)
    αdisp_const = T(par.f₀) / T(2)
    fill!(αdisp_profile, αdisp_const)

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
                L⁺Anp1_arr[k, i, j] = 0
                Anp1_arr[k, i, j] = 0
            end
            continue
        end

        kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)
        hyperdiff_factor = exp(-λʷ)

        # Handle cases where dispersion is disabled or kₕ² ≈ 0
        if !dispersion_active || kₕ² < IMEX_KH2_EPS
            # Explicit update for mean mode or when dispersion is disabled
            # Use B* (after first half-refraction) and AB2 advection
            @inbounds for k in 1:nz_local
                N_n = -nL⁺Ak_arr[k, i, j]
                N_nm1 = use_ab2 ? -nL⁺Ak_prev_arr[k, i, j] : zero(eltype(nL⁺Ak_prev_arr))
                advection_term = (c_n * dt) * N_n + (c_nm1 * dt) * N_nm1
                L⁺Anp1_arr[k, i, j] = (L⁺Astar_arr[k, i, j] + advection_term) * hyperdiff_factor
                Anp1_arr[k, i, j] = 0
            end
        else
            # IMEX-CNAB: CN for dispersion, AB2 for advection
            # Step 1: Compute A* = (L⁺)⁻¹B* for consistency with B*
            @inbounds for k in 1:nz_local
                tl.L⁺A_col[k] = L⁺Astar_arr[k, i, j]
            end
            solve_modified_elliptic!(tl.A_col, tl.L⁺A_col, G, par, a, kₕ²,
                                     complex(0.0), αdisp_profile, tl)

            # Step 2: Build RHS for IMEX-CNAB using B*, A*, and AB2 advection
            @inbounds for k in 1:nz_local
                N_n = -nL⁺Ak_arr[k, i, j]
                N_nm1 = use_ab2 ? -nL⁺Ak_prev_arr[k, i, j] : zero(eltype(nL⁺Ak_prev_arr))
                disp_star = im * αdisp_profile[k] * kₕ² * tl.A_col[k]
                advection_term = (c_n * dt) * N_n + (c_nm1 * dt) * N_nm1
                tl.RHS_col[k] = L⁺Astar_arr[k, i, j] + (dt/2) * disp_star + advection_term
            end

            # Step 3: Solve modified elliptic problem for A^{n+1}
            β_scale = (dt/2) * im * kₕ²

            solve_modified_elliptic!(tl.A_col, tl.RHS_col, G, par, a, kₕ²,
                                     β_scale, αdisp_profile, tl)

            # Recover B** from the IMEX-CN relation
            @inbounds for k in 1:nz_local
                βₖ = β_scale * αdisp_profile[k]
                L⁺Anp1_arr[k, i, j] = (tl.RHS_col[k] + βₖ * tl.A_col[k]) * hyperdiff_factor
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
                L⁺Anp1_arr[k, i, j] = 0
                Anp1_arr[k, i, j] = 0
            end
            continue
        end

        kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)
        hyperdiff_factor = exp(-λʷ)

        if !dispersion_active || kₕ² < IMEX_KH2_EPS
            # Explicit update for mean mode or when dispersion is disabled
            # Use B* (after first half-refraction) and AB2 advection
            # AB2: (c_n * dt) * N^n + (c_nm1 * dt) * N^{n-1}
            for k in 1:nz_local
                N_n = -nL⁺Ak_arr[k, i, j]
                N_nm1 = use_ab2 ? -nL⁺Ak_prev_arr[k, i, j] : zero(eltype(nL⁺Ak_prev_arr))
                advection_term = (c_n * dt) * N_n + (c_nm1 * dt) * N_nm1
                # Store L⁺A** (before second half-refraction) in L⁺Anp1 temporarily
                L⁺Anp1_arr[k, i, j] = (L⁺Astar_arr[k, i, j] + advection_term) * hyperdiff_factor
                Anp1_arr[k, i, j] = 0
            end
        else
            # IMEX-CNAB: CN for dispersion, AB2 for advection
            # Step 1: Compute A* = (L⁺)⁻¹B* for consistency with the refraction-rotated B*
            # This is critical: using A^n with B* breaks IMEX-CN stability!
            for k in 1:nz_local
                tl.L⁺A_col[k] = L⁺Astar_arr[k, i, j]
            end
            # Use β_scale=0 to get standard L⁺ inversion (A* = (L⁺)⁻¹B*)
            solve_modified_elliptic!(tl.A_col, tl.L⁺A_col, G, par, a, kₕ²,
                                     complex(0.0), αdisp_profile, tl)

            # Step 2: Build RHS for IMEX-CNAB using B*, A*, and AB2 advection
            # RHS = B* + (dt/2)·i·αdisp·kₕ²·A* + (c_n·dt)·N^n + (c_nm1·dt)·N^{n-1}
            for k in 1:nz_local
                N_n = -nL⁺Ak_arr[k, i, j]
                N_nm1 = use_ab2 ? -nL⁺Ak_prev_arr[k, i, j] : zero(eltype(nL⁺Ak_prev_arr))
                disp_star = im * αdisp_profile[k] * kₕ² * tl.A_col[k]
                advection_term = (c_n * dt) * N_n + (c_nm1 * dt) * N_nm1
                tl.RHS_col[k] = L⁺Astar_arr[k, i, j] + (dt/2) * disp_star + advection_term
            end

            # Step 3: Solve modified elliptic problem for A^{n+1}
            # IMEX-CN: (L⁺ - β)·A^{n+1} = RHS where β = (dt/2)·i·αdisp·kₕ²
            β_scale = (dt/2) * im * kₕ²

            solve_modified_elliptic!(tl.A_col, tl.RHS_col, G, par, a, kₕ²,
                                     β_scale, αdisp_profile, tl)

            # Recover B** from the IMEX-CN relation: B** = RHS + β*A^{n+1}
            for k in 1:nz_local
                βₖ = β_scale * αdisp_profile[k]
                L⁺Anp1_arr[k, i, j] = (tl.RHS_col[k] + βₖ * tl.A_col[k]) * hyperdiff_factor
                Anp1_arr[k, i, j] = tl.A_col[k] * hyperdiff_factor
            end
        end
    end  # end serial for
    end  # end if use_threading

    #= Step 5.5: Apply SECOND HALF refraction with updated ψ =#
    # Use ψ^{n+1} (from the updated mean flow) for the second half-step to
    # keep the coupled system formally second-order in time.
    RHS_temp = imex_ws.RHS  # Reuse RHS as temporary storage
    parent(RHS_temp) .= L⁺Anp1_arr  # Copy B** to temporary

    wave_feedback_enabled = !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback
    psi_pred_valid = false

    if par.fixed_flow
        # Mean flow is fixed: use ψ^n for refraction
        apply_refraction_exact!(Snp1.L⁺A, RHS_temp, Sn.psi, G, par, plans;
                                dt_fraction=0.5, dealias_mask=L)
        psi_pred_valid = true
    else
        if wave_feedback_enabled
            # Predictor: use ψ^n to get B_pred and q^w_pred for ψ^{n+1}
            apply_refraction_exact!(Snp1.L⁺A, RHS_temp, Sn.psi, G, par, plans;
                                    dt_fraction=0.5, dealias_mask=L)

            # Backup q^{n+1} (base, before wave feedback)
            qtemp_arr .= qnp1_arr

            # Compute q^w_pred into qnp1_arr, then form q_pred = q_base - q^w_pred
            compute_qw_complex!(Snp1.q, Snp1.L⁺A, par, G, plans; Lmask=L)
            @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
                i_global = local_to_global(i, 2, Snp1.q)
                j_global = local_to_global(j, 3, Snp1.q)
                if L[i_global, j_global]
                    qnp1_arr[k, i, j] = qtemp_arr[k, i, j] - qnp1_arr[k, i, j]
                else
                    qnp1_arr[k, i, j] = 0
                end
            end

            # Compute ψ^{n+1} predictor from q_pred
            invert_q_to_psi!(Snp1, G; a=a, par=par, workspace=workspace)

            # Restore q^{n+1} base state before final wave feedback
            qnp1_arr .= qtemp_arr

            # Corrector: refraction with ψ^{n+1} predictor
            apply_refraction_exact!(Snp1.L⁺A, RHS_temp, Snp1.psi, G, par, plans;
                                    dt_fraction=0.5, dealias_mask=L)
        else
            # No wave feedback: compute ψ^{n+1} from q^{n+1} base
            invert_q_to_psi!(Snp1, G; a=a, par=par, workspace=workspace)
            apply_refraction_exact!(Snp1.L⁺A, RHS_temp, Snp1.psi, G, par, plans;
                                    dt_fraction=0.5, dealias_mask=L)
            psi_pred_valid = true
        end
    end

    # Update L⁺Anp1_arr to point to the final result
    L⁺Anp1_arr = parent(Snp1.L⁺A)

    #= Step 5.6: Store current advection tendencies for next step (AB2) =#
    parent(imex_ws.nL⁺Ak_prev) .= parent(imex_ws.nL⁺Ak)
    if par.fixed_flow
        fill!(tqk_prev_arr, zero(eltype(tqk_prev_arr)))
    else
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, Sn.q)
            j_global = local_to_global(j, 3, Sn.q)
            if L[i_global, j_global]
                tqk_prev_arr[k, i, j] = -nqk_arr[k, i, j]
            else
                tqk_prev_arr[k, i, j] = 0
            end
        end
    end
    imex_ws.has_prev_tendency[] = true

    #= Step 6: Wave feedback on mean flow =#
    if wave_feedback_enabled
        # Reuse qtemp from workspace to avoid allocation every timestep
        # (allocation inside tight loops causes heap corruption in MPI)
        qwk = imex_ws.qtemp
        qwk_arr = parent(qwk)
        compute_qw_complex!(qwk, Snp1.L⁺A, par, G, plans; Lmask=L)

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
        if !psi_pred_valid
            invert_q_to_psi!(Snp1, G; a=a, par=par, workspace=workspace)
        end
    else
        # Copy psi from Sn
        parent(Snp1.psi) .= parent(Sn.psi)
    end

    # A is already computed above for most modes
    # For kₕ² ≈ 0 modes, do standard inversion
    # Actually, let's just do full inversion to be safe
    invert_L⁺A_to_A!(Snp1, G, par, a; workspace=workspace)

    # Only compute u, v for diagnostics - w can be computed separately if needed
    compute_velocities!(Snp1, G; plans=plans, params=par, N2_profile=N2_profile,
                        workspace=workspace, dealias_mask=L, compute_w=false)

    #= Step 8: Advect particles (if tracker provided) =#
    # Particles co-evolve with the wave and mean flow equations using the same dt
    if particle_tracker !== nothing
        advect_particles!(particle_tracker, Snp1, G, par.dt, current_time;
                          params=par, N2_profile=N2_profile)
    end

    return Snp1
end
