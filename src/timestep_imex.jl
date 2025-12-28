#=
================================================================================
                timestep_imex.jl - IMEX Time Integration
================================================================================

IMEX (Implicit-Explicit) Crank-Nicolson time stepping for YBJ+ equation.

The YBJ+ equation for B:
    ∂B/∂t + J(ψ,B) + (i/2)ζB = i·αdisp·kₕ²·A

where A = (L⁺)⁻¹·B (elliptic inversion).

IMEX splits this into:
- EXPLICIT: N(B) = -J(ψ,B) - (i/2)ζB  (advection + refraction)
- IMPLICIT: L(B) = i·αdisp·kₕ²·(L⁺)⁻¹·B  (dispersion)

The Crank-Nicolson scheme:
    (I - dt/2·L)·B^{n+1} = (I + dt/2·L)·B^n + dt·N^n

Since L involves (L⁺)⁻¹, we reformulate as a modified elliptic problem:
    (L⁺ - dt/2·i·αdisp·kₕ²)·A^{n+1} = B^n + dt/2·i·αdisp·kₕ²·A^n + dt·N^n

This is a tridiagonal system with complex diagonal modification.

STABILITY:
---------
- Dispersion: Unconditionally stable (implicit treatment)
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
    return IMEXThreadLocal(
        Vector{ComplexF64}(undef, nz),      # tri_diag
        Vector{ComplexF64}(undef, nz-1),    # tri_upper
        Vector{ComplexF64}(undef, nz-1),    # tri_lower
        Vector{ComplexF64}(undef, nz),      # tri_rhs
        Vector{ComplexF64}(undef, nz),      # tri_sol
        Vector{ComplexF64}(undef, nz),      # tri_c_prime
        Vector{ComplexF64}(undef, nz),      # tri_d_prime
        Vector{ComplexF64}(undef, nz),      # B_col
        Vector{ComplexF64}(undef, nz),      # A_col
        Vector{ComplexF64}(undef, nz)       # RHS_col
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

    # Per-thread workspace for tridiagonal solves
    thread_local::Vector{IMEXThreadLocal}

    # For q equation (same as original)
    qtemp::CT

    # Grid size for reference
    nz::Int
end

"""
    init_imex_workspace(S, G; nthreads=Threads.nthreads())

Initialize workspace for IMEX time stepping with threading support.

NOTE: All work arrays are pre-allocated here to avoid heap corruption
from repeated allocation/deallocation in tight loops during time stepping.
Per-thread workspaces are created for the tridiagonal solves to enable
parallel processing of horizontal modes.
"""
function init_imex_workspace(S, G; nthreads=Threads.nthreads())
    CT = typeof(S.B)
    RT = typeof(S.u)

    nBk = similar(S.B)
    rBk = similar(S.B)
    nqk = similar(S.q)
    dqk = similar(S.B)
    RHS = similar(S.B)
    Atemp = similar(S.A)
    qtemp = similar(S.q)

    nz = G.nz

    # Create per-thread workspaces
    # Use max(1, nthreads) to ensure at least one workspace exists
    n_workspaces = max(1, nthreads)
    thread_local = [init_thread_local(nz) for _ in 1:n_workspaces]

    return IMEXWorkspace{CT, RT}(
        nBk, rBk, nqk, dqk, RHS, Atemp,
        thread_local,
        qtemp,
        nz
    )
end

"""
    solve_modified_elliptic!(A, B, G, par, a, kₕ², β, tl::IMEXThreadLocal)

Solve the modified elliptic problem for IMEX dispersion:
    (L⁺ - β)·A = B

where L⁺ = (f²/N²)∂²/∂z² - kₕ²/4 and β = (dt/2)·i·αdisp·kₕ²

This is a tridiagonal system for each horizontal mode.

# Arguments
- `A`: Output wave amplitude (nz vector for this horizontal mode)
- `B`: Input wave envelope (nz vector)
- `G`: Grid
- `par`: Parameters
- `a`: a_ell = f²/N² array
- `kₕ²`: Horizontal wavenumber squared
- `β`: Implicit dispersion coefficient = (dt/2)·i·αdisp·kₕ²
- `tl`: IMEXThreadLocal with pre-allocated tridiagonal arrays (thread-local)
"""
function solve_modified_elliptic!(A_out::AbstractVector, B_in::AbstractVector,
                                   G, par, a, kₕ², β, tl::IMEXThreadLocal)
    nz = G.nz
    # G.dz is a vector of layer thicknesses; assume uniform grid and use first element
    dz = G.dz[1]
    dz² = dz * dz

    # Build tridiagonal system: (L⁺ - β)·A = B
    # where L⁺ = a(z)·∂²A/∂z² - kₕ²/4·A  is the YBJ+ elliptic operator
    # and β = (dt/2)·i·αdisp·kₕ² is the implicit dispersion coefficient
    # With Neumann BCs: ∂A/∂z = 0 at z = 0, Lz

    # Interior points: a[k]/dz²·(A[k+1] - 2A[k] + A[k-1]) - kₕ²/4·A[k] - β·A[k] = B[k]
    # Rearranging: a[k]/dz²·A[k-1] + (-2a[k]/dz² - kₕ²/4 - β)·A[k] + a[k]/dz²·A[k+1] = B[k]

    @inbounds for k in 1:nz
        aₖ = a[k]  # f²/N²(z)

        # Main diagonal: -2a/dz² - kₕ²/4 - β
        tl.tri_diag[k] = -2.0 * aₖ / dz² - kₕ² / 4.0 - β

        # Off-diagonals: a/dz²
        if k < nz
            tl.tri_upper[k] = aₖ / dz²
        end
        if k > 1
            tl.tri_lower[k-1] = a[k] / dz²  # Use a at current level
        end

        tl.tri_rhs[k] = B_in[k]
    end

    # Apply Neumann BCs by modifying first and last rows
    # At k=1 (bottom): ∂A/∂z = 0 → A[0] = A[2] (ghost point)
    # Row becomes: (-2a/dz² - kₕ²/4 - β)·A[1] + 2a/dz²·A[2] = B[1]
    tl.tri_diag[1] = -2.0 * a[1] / dz² - kₕ² / 4.0 - β
    tl.tri_upper[1] = 2.0 * a[1] / dz²

    # At k=nz (top): ∂A/∂z = 0 → A[nz+1] = A[nz-1] (ghost point)
    # Row becomes: 2a/dz²·A[nz-1] + (-2a/dz² - kₕ²/4 - β)·A[nz] = B[nz]
    tl.tri_lower[nz-1] = 2.0 * a[nz] / dz²
    tl.tri_diag[nz] = -2.0 * a[nz] / dz² - kₕ² / 4.0 - β

    # Handle mean mode (kₕ² = 0): operator is singular
    # Set A = 0 for mean mode (consistent with original code)
    if kₕ² < 1e-14 && abs(β) < 1e-14
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

IMEX Crank-Nicolson time step for YBJ+ equation.

Treats dispersion implicitly for unconditional stability, advection/refraction explicitly.

# Algorithm
1. Compute explicit tendencies N(B) = -J(ψ,B) - (i/2)ζB at time n
2. Compute RHS = B^n + (dt/2)·i·αdisp·kₕ²·A^n + dt·N^n
3. Solve modified elliptic: (L⁺ - (dt/2)·i·αdisp·kₕ²)·A^{n+1} = (some function of RHS)
4. Recover B^{n+1} = L⁺·A^{n+1}

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
- Dispersion: Unconditionally stable
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

    # Ensure we have enough thread-local workspaces for the actual thread count
    # (Threads.nthreads() at init time may differ from runtime, especially with MPI)
    nthreads = Threads.nthreads()
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
    rBk_arr = parent(imex_ws.rBk)
    dqk_arr = parent(imex_ws.dqk)
    qtemp_arr = parent(imex_ws.qtemp)

    # Advection
    convol_waqg_q!(imex_ws.nqk, Sn.u, Sn.v, Sn.q, G, plans; Lmask=L)
    convol_waqg_B!(imex_ws.nBk, Sn.u, Sn.v, Sn.B, G, plans; Lmask=L)

    # Refraction
    refraction_waqg_B!(imex_ws.rBk, Sn.B, Sn.psi, G, plans; Lmask=L)

    # Vertical diffusion for q
    dissipation_q_nv!(imex_ws.dqk, Sn.q, par, G; workspace=workspace)

    #= Step 3: Apply physics switches =#
    if par.inviscid; imex_ws.dqk .= 0; end
    if par.linear; imex_ws.nqk .= 0; imex_ws.nBk .= 0; end
    if par.passive_scalar; imex_ws.rBk .= 0; end  # No refraction
    if par.fixed_flow; imex_ws.nqk .= 0; end

    # Determine if dispersion is active (needed for IMEX implicit solve)
    dispersion_active = !par.no_dispersion && !par.passive_scalar

    #= Step 4: Update q with explicit Euler (or could use CN for diffusion) =#
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, Sn.q)
        j_global = local_to_global(j, 3, Sn.q)

        if L[i_global, j_global]
            kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
            λₑ = int_factor(kₓ, kᵧ, par; waves=false)

            if par.fixed_flow
                qnp1_arr[k, i, j] = qn_arr[k, i, j]
            else
                # Forward Euler with integrating factor for hyperdiffusion
                qnp1_arr[k, i, j] = (qn_arr[k, i, j] - dt*nqk_arr[k, i, j] + dt*dqk_arr[k, i, j]) * exp(-λₑ)
            end
        else
            qnp1_arr[k, i, j] = 0
        end
    end

    #= Step 5: IMEX Crank-Nicolson for B equation =#
    # IMPORTANT: For IMEX-CN to be consistent, the implicit and explicit parts
    # must use the SAME αdisp value. We use constant αdisp = N²/(2f₀) based on
    # the reference stratification par.N². This is consistent with the YBJ+
    # formulation which assumes slowly-varying stratification.
    #
    # The dispersion coefficient αdisp = N²/(2f₀) appears in the dispersion term:
    #   ∂B/∂t = ... + i·αdisp·kₕ²·A
    αdisp = par.N² / (2.0 * par.f₀)

    # Process each horizontal mode
    # For IMEX-CN: (I - dt/2·L)·B^{n+1} = (I + dt/2·L)·B^n + dt·N
    # where L·B = i·αdisp·kₕ²·A and A = (L⁺)⁻¹·B

    # Reformulated: solve for A^{n+1} from modified elliptic problem
    # Then B^{n+1} = L⁺·A^{n+1}

    # Pre-compute grid spacing for use in parallel loop
    dz = G.dz[1]  # Uniform grid: use first element
    dz² = dz * dz

    # Process horizontal modes in parallel using threads
    # Each thread uses its own workspace to avoid data races
    n_modes = nx_local * ny_local

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
            # Pure explicit step: B^{n+1} = B^n + dt·N^n (with hyperdiffusion)
            # where N = -J(ψ,B) - (i/2)ζB
            @inbounds for k in 1:nz_local
                N_k = -nBk_arr[k, i, j] - 0.5im * rBk_arr[k, i, j]
                Bnp1_arr[k, i, j] = (Bn_arr[k, i, j] + dt * N_k) * hyperdiff_factor
                Anp1_arr[k, i, j] = 0  # No dispersion means A is meaningless
            end
        else
            # IMEX-CN for B equation with dispersion
            # Build RHS for IMEX-CN: B^n + (dt/2)·disp^n + dt·N^n
            # where disp = i·αdisp·kₕ²·A and N = -J(ψ,B) - (i/2)ζB
            #
            # NOTE: nz_local == nz for xy-pencil decomposition (z is not distributed)
            @inbounds for k in 1:nz_local
                # Explicit tendency: N = -J(ψ,B) - (i/2)ζB
                N_k = -nBk_arr[k, i, j] - 0.5im * rBk_arr[k, i, j]

                # Dispersion term at time n: disp^n = i·αdisp·kₕ²·A^n
                # Uses the SAME αdisp as the implicit solve for consistency
                disp_n = im * αdisp * kₕ² * An_arr[k, i, j]

                # RHS for IMEX-CN (without hyperdiffusion - applied later)
                tl.RHS_col[k] = Bn_arr[k, i, j] + (dt/2) * disp_n + dt * N_k
            end

            # Solve modified elliptic problem for A^{n+1}
            # IMEX-CN formulation:
            #   B^{n+1} - (dt/2)·i·αdisp·kₕ²·A^{n+1} = RHS
            # Since B = L⁺·A:
            #   L⁺·A^{n+1} - (dt/2)·i·αdisp·kₕ²·A^{n+1} = RHS
            #   (L⁺ - β)·A^{n+1} = RHS
            # where β = (dt/2)·i·αdisp·kₕ²
            β = (dt/2) * im * αdisp * kₕ²

            # Solve (L⁺ - β)·A = RHS using thread-local workspace
            solve_modified_elliptic!(tl.A_col, tl.RHS_col, G, par, a, kₕ², β, tl)

            # Recover B = L⁺·A (apply L⁺ operator)
            # L⁺ = a·∂²/∂z² - kₕ²/4  where a = f²/N²
            # NOTE: nz_local == nz for xy-pencil decomposition (z is not distributed)
            @inbounds for k in 1:nz_local
                # Second derivative with Neumann BCs: ∂A/∂z = 0 at boundaries
                if k == 1
                    d2A = a[1] * (2*tl.A_col[2] - 2*tl.A_col[1]) / dz²
                elseif k == nz_local
                    d2A = a[nz_local] * (2*tl.A_col[nz_local-1] - 2*tl.A_col[nz_local]) / dz²
                else
                    d2A = a[k] * (tl.A_col[k+1] - 2*tl.A_col[k] + tl.A_col[k-1]) / dz²
                end

                # B = L⁺·A, then apply hyperdiffusion as post-processing
                B_raw = d2A - (kₕ²/4) * tl.A_col[k]
                Bnp1_arr[k, i, j] = B_raw * hyperdiff_factor
                Anp1_arr[k, i, j] = tl.A_col[k] * hyperdiff_factor
            end
        end
    end

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
