#=
================================================================================
                    timestep.jl - Time Integration
================================================================================

This file implements the time stepping schemes for the QG-YBJ+ model:

1. FORWARD EULER (Projection Step):
   Used for the first time step to initialize the leapfrog scheme.
   Simple but only first-order accurate.

2. LEAPFROG with ROBERT-ASSELIN FILTER:
   The main time integration scheme. Second-order accurate in time.
   The Robert-Asselin filter damps the computational mode that can
   grow with leapfrog schemes.

TIME INTEGRATION ALGORITHM:
---------------------------
For each time step from n to n+1:

1. Compute tendencies at time n:
   - Advection: J(ψ, q), J(ψ, B)
   - Refraction: B × ζ
   - Vertical diffusion: νz ∂²q/∂z²

2. Apply physics switches:
   - linear: zero nonlinear terms
   - inviscid: zero dissipation
   - passive_scalar: zero dispersion and refraction
   - fixed_flow: zero q tendency

3. Time step with hyperdiffusion integrating factors:
   - Leapfrog: φ^(n+1) = φ^(n-1) × e^(-2λdt) + 2dt × tendency^n × e^(-λdt)
   - Forward Euler: φ^(n+1) = (φ^n + dt × tendency) × e^(-λdt)
   where tendency = -advection + diffusion (evaluated at time n)

4. Robert-Asselin filter (leapfrog only):
   φ̃^n = φ^n + γ(φ^(n-1) - 2φ^n + φ^(n+1))
   where γ ~ 10⁻³ is small to minimize damping

5. Wave feedback on mean flow:
   q* = q - qʷ (if wave feedback is enabled)

6. Diagnostic updates:
   - Invert q → ψ
   - Invert B → A (YBJ+) or compute A directly (normal YBJ)
   - Compute velocities from ψ

FORTRAN CORRESPONDENCE:
----------------------
The time stepping matches main_waqg.f90 in the Fortran QG_YBJp code.
The integrating factor approach for hyperdiffusion is from the Fortran.

STABILITY:
----------
- Leapfrog CFL condition: dt < min(dx/|u|, dy/|v|)
- Hyperdiffusion CFL: dt × ν × k^(2n) < 1 for largest k
- Robert-Asselin γ too large → excessive damping
- Robert-Asselin γ too small → computational mode growth

PENCILARRAY COMPATIBILITY:
--------------------------
All loops use local indexing with local_to_global() for wavenumber access
based on the array pencil (important for MPI input/output pencil mismatch).
The vertical dimension (z) must be fully local for proper operation.

================================================================================
=#

#=
================================================================================
                    HELPER FUNCTIONS
================================================================================
=#

"""
    split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, L⁺A)

Split complex wave field L⁺A into real and imaginary parts stored as complex arrays.

This is a common operation in the time stepping code. The outputs L⁺ARk and L⁺AIk
are complex arrays where only the real part is used (imaginary part is zero).
This format is required for compatibility with the spectral derivative operations.

# Arguments
- `L⁺ARk`: Output array for real part of L⁺A (stored as Complex with imag=0)
- `L⁺AIk`: Output array for imaginary part of L⁺A (stored as Complex with imag=0)
- `L⁺A`: Input complex wave field
"""
function split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, L⁺A)
    L⁺A_arr = parent(L⁺A)
    L⁺ARk_arr = parent(L⁺ARk)
    L⁺AIk_arr = parent(L⁺AIk)

    @local_spectral_loop L⁺A begin
        L⁺ARk_arr[k, i, j] = Complex(real(L⁺A_arr[k, i, j]), 0)
        L⁺AIk_arr[k, i, j] = Complex(imag(L⁺A_arr[k, i, j]), 0)
    end
    return L⁺ARk, L⁺AIk
end

"""
    combine_real_imag_to_L⁺A!(L⁺A, L⁺ARk, L⁺AIk)

Combine real and imaginary parts back into complex wave field L⁺A.

The inverse of `split_L⁺A_to_real_imag!`. Takes L⁺ARk and L⁺AIk (complex arrays
with only real parts used) and combines them into L⁺A = L⁺AR + i*L⁺AI.

# Arguments
- `L⁺A`: Output complex wave field
- `L⁺ARk`: Real part of L⁺A (stored as Complex with imag=0)
- `L⁺AIk`: Imaginary part of L⁺A (stored as Complex with imag=0)
"""
function combine_real_imag_to_L⁺A!(L⁺A, L⁺ARk, L⁺AIk)
    L⁺A_arr = parent(L⁺A)
    L⁺ARk_arr = parent(L⁺ARk)
    L⁺AIk_arr = parent(L⁺AIk)

    @local_spectral_loop L⁺A begin
        L⁺A_arr[k, i, j] = Complex(real(L⁺ARk_arr[k, i, j]), 0) + im*Complex(real(L⁺AIk_arr[k, i, j]), 0)
    end
    return L⁺A
end

"""
    replace_q_with_wave_feedback_rhs!(S, G, par, plans, L; L⁺ARk=nothing, L⁺AIk=nothing)

Temporarily replace `S.q` by the inversion right-hand side `q* = q - qʷ`.

The prognostic PV remains the balanced-flow `q`. This helper returns a copy of
that prognostic `q`; callers must restore it after `invert_q_to_psi!`.
"""
function replace_q_with_wave_feedback_rhs!(S::State, G::Grid, par::QGParams, plans, L;
                                           L⁺ARk=nothing, L⁺AIk=nothing)
    q_base = copy(S.q)
    q_base_arr = parent(q_base)
    q_arr = parent(S.q)
    qwk = similar(S.q)
    qwk_arr = parent(qwk)

    if par.ybj_plus
        compute_qw_complex!(qwk, S.L⁺A, par, G, plans; Lmask=L)
    else
        if L⁺ARk === nothing || L⁺AIk === nothing
            L⁺ARk = similar(S.L⁺A)
            L⁺AIk = similar(S.L⁺A)
        end
        split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, S.L⁺A)
        compute_qw!(qwk, L⁺ARk, L⁺AIk, par, G, plans; Lmask=L)
    end

    @dealiased_spectral_loop S.q L begin
        q_arr[k, i, j] = q_base_arr[k, i, j] - qwk_arr[k, i, j]
    end begin
        q_arr[k, i, j] = 0
    end

    return q_base
end

restore_prognostic_q!(S::State, q_base) = (parent(S.q) .= parent(q_base); S)

#=
================================================================================
                    FORWARD EULER (Projection Step)
================================================================================
The projection step initializes the leapfrog scheme by providing values
at times n and n-1 from a single initial condition.

After this step:
- S contains fields at time n+1 (after one Euler step)
- The original S becomes the n-1 state for leapfrog
================================================================================
=#

"""
    first_projection_step!(S, G, par, plans; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

Forward Euler initialization step for the leapfrog time stepper.

# Purpose
The leapfrog scheme requires values at two time levels (n and n-1).
This function takes the initial state and advances it by one Forward Euler
step, providing the needed second time level.

# Algorithm
1. **Compute tendencies at time n:**
   - Advection of q and B by geostrophic flow
   - Wave refraction by vorticity
   - Vertical diffusion

2. **Apply physics switches:**
   - `linear`: Zero nonlinear advection
   - `inviscid`: Zero dissipation
   - `passive_scalar`: Zero dispersion and refraction
   - `fixed_flow`: Mean flow doesn't evolve

3. **Forward Euler update:**
   For each spectral mode:
   ```
   q^(n+1) = [q^n - dt × tendency_q + dt × diffusion] × exp(-λ_q × dt)
   B^(n+1) = [B^n - dt × tendency_B] × exp(-λ_B × dt)
   ```
   where λ is the hyperdiffusion factor.

4. **Wave feedback (optional):**
   ```
   q* = q - qʷ
   ```

5. **Diagnostic inversions:**
   - q → ψ (elliptic inversion)
   - B → A, C (YBJ+ inversion)
   - ψ → u, v (velocity computation)

# Arguments
- `S::State`: State to advance (modified in place)
- `G::Grid`: Grid struct
- `par::QGParams`: Model parameters
- `plans`: FFT plans
- `a`: Elliptic coefficient array a_ell(z) = f²/N²
- `dealias_mask`: Optional 2/3 dealiasing mask (nx × ny)
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `N2_profile`: Optional N²(z) profile for vertical velocity computation

# Returns
Modified state S at time n+1.

# Fortran Correspondence
This matches the projection step in main_waqg.f90.

# Example
```julia
# Initialize and run projection step
state = init_state(grid)
init_random_psi!(state, grid)
a = a_ell_ut(params, grid)
L = dealias_mask(grid)
first_projection_step!(state, grid, params, plans; a=a, dealias_mask=L)
```
"""
function first_projection_step!(S::State, G::Grid, par::QGParams, plans; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing,
                                particle_tracker=nothing, current_time=nothing)
    #= Setup - get local dimensions for PencilArray compatibility =#
    q_arr = parent(S.q)
    L⁺A_arr = parent(S.L⁺A)
    A_arr = parent(S.A)
    C_arr = parent(S.C)

    nz = G.nz

    # Note: In xy-pencil format, z is fully local (nz_local = nz).
    # In z-pencil format (after transpose), xy are distributed.
    # Functions that need z local (invert_q_to_psi!, dissipation_q_nv!, etc.)
    # handle transposes internally when using 2D decomposition.

    # Dealias mask - use global indices for lookup
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    # Allocate tendency arrays (same size as local arrays)
    nqk = similar(S.q)   # J(ψ, q) advection of PV
    dqk = similar(S.L⁺A)   # Vertical diffusion of q
    if par.ybj_plus
        nL⁺Ak = similar(S.L⁺A)   # J(ψ, L⁺A) advection (complex)
        rL⁺Ak = similar(S.L⁺A)   # ζ × L⁺A refraction (complex)
    else
        nL⁺ARk = similar(S.L⁺A)  # J(ψ, BR) advection of wave real part
        nL⁺AIk = similar(S.L⁺A)  # J(ψ, BI) advection of wave imaginary part
        rL⁺ARk = similar(S.L⁺A)  # BR × ζ refraction real part
        rL⁺AIk = similar(S.L⁺A)  # BI × ζ refraction imaginary part
    end

    # For normal YBJ, remove vertical mean of B before any diagnostics/tendencies.
    if !par.ybj_plus
        sumL⁺A!(S.L⁺A, G; Lmask=L)

        # Split B into real and imaginary parts for computation
        L⁺ARk = similar(S.L⁺A); L⁺AIk = similar(S.L⁺A)
        split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, S.L⁺A)
    end

    #= Step 1: Compute diagnostic fields ψ, velocities, and A =#
    invert_q_to_psi!(S, G; a, par=par, workspace=workspace)           # q → ψ
    compute_velocities!(S, G; plans, params=par, N2_profile=N2_profile, workspace=workspace, dealias_mask=L) # ψ → u, v

    # Compute A from B for dispersion term
    # Must use the same approach as the main integrator to avoid startup transients
    if par.ybj_plus
        # YBJ+: Solve elliptic problem B → A, C
        invert_L⁺A_to_A!(S, G, par, a; workspace=workspace)
    else
        # Normal YBJ: Use sumL⁺A!/compute_sigma/compute_A! path
        # For initial step, use zero tendencies for sigma computation
        nL⁺ARk_zero = similar(S.L⁺A); fill!(nL⁺ARk_zero, 0)
        nL⁺AIk_zero = similar(S.L⁺A); fill!(nL⁺AIk_zero, 0)
        rL⁺ARk_zero = similar(S.L⁺A); fill!(rL⁺ARk_zero, 0)
        rL⁺AIk_zero = similar(S.L⁺A); fill!(rL⁺AIk_zero, 0)
        sigma_init = compute_sigma(par, G, nL⁺ARk_zero, nL⁺AIk_zero, rL⁺ARk_zero, rL⁺AIk_zero; Lmask=L, N2_profile=N2_profile)
        compute_A!(S.A, S.C, L⁺ARk, L⁺AIk, sigma_init, par, G; Lmask=L, N2_profile=N2_profile)
    end

    #= Step 2: Compute nonlinear tendencies =#

    if par.ybj_plus
        # Advection: J(ψ, q), J(ψ, L⁺A)
        convol_waqg_q!(nqk, S.u, S.v, S.q, G, plans; Lmask=L)
        convol_waqg_L⁺A!(nL⁺Ak, S.u, S.v, S.L⁺A, G, plans; Lmask=L)

        # Wave refraction: L⁺A × ζ where ζ = ∇²ψ
        refraction_waqg_L⁺A!(rL⁺Ak, S.L⁺A, S.psi, G, plans; Lmask=L)
    else
        # Advection: J(ψ, q), J(ψ, BR), J(ψ, BI)
        convol_waqg!(nqk, nL⁺ARk, nL⁺AIk, S.u, S.v, S.q, L⁺ARk, L⁺AIk, G, plans; Lmask=L)

        # Wave refraction: B × ζ where ζ = ∇²ψ
        refraction_waqg!(rL⁺ARk, rL⁺AIk, L⁺ARk, L⁺AIk, S.psi, G, plans; Lmask=L)
    end

    # Vertical diffusion: νz ∂²q/∂z² (handles 2D decomposition transposes internally)
    dissipation_q_nv!(dqk, S.q, par, G; workspace=workspace)

    #= Step 3: Apply physics switches =#

    # inviscid: No dissipation
    if par.inviscid; dqk .= 0; end

    # linear: No nonlinear advection
    if par.linear
        nqk .= 0
        if par.ybj_plus
            nL⁺Ak .= 0
        else
            nL⁺ARk .= 0; nL⁺AIk .= 0
        end
    end

    # no_dispersion: Waves don't disperse (A = 0)
    if par.no_dispersion
        S.A .= 0; S.C .= 0
    end

    # passive_scalar: Waves are passive tracers (no dispersion, no refraction)
    if par.passive_scalar
        S.A .= 0; S.C .= 0
        if par.ybj_plus
            rL⁺Ak .= 0
        else
            rL⁺ARk .= 0; rL⁺AIk .= 0
        end
    end

    # fixed_flow: Mean flow doesn't evolve
    if par.fixed_flow; nqk .= 0; end

    #= Store old fields for time stepping =#
    qok  = copy(S.q)
    qok_arr = parent(qok)
    if par.ybj_plus
        L⁺Aok = copy(S.L⁺A)
        L⁺Aok_arr = parent(L⁺Aok)
    else
        L⁺ARok = similar(S.L⁺A); L⁺AIok = similar(S.L⁺A)
        L⁺ARok_arr = parent(L⁺ARok); L⁺AIok_arr = parent(L⁺AIok)
        split_L⁺A_to_real_imag!(L⁺ARok, L⁺AIok, S.L⁺A)
    end

    #= Step 4: Forward Euler with integrating factors =#
    # The integrating factor handles hyperdiffusion exactly:
    # φ^(n+1) = [φ^n - dt × F] × exp(-λ×dt)

    # Get parent arrays for tendency terms
    nqk_arr = parent(nqk)
    if par.ybj_plus
        nL⁺Ak_arr = parent(nL⁺Ak)
        rL⁺Ak_arr = parent(rL⁺Ak)
    else
        nL⁺ARk_arr = parent(nL⁺ARk); nL⁺AIk_arr = parent(nL⁺AIk)
        rL⁺ARk_arr = parent(rL⁺ARk); rL⁺AIk_arr = parent(rL⁺AIk)
    end
    dqk_arr = parent(dqk)

    # Precompute dispersion coefficient: αdisp = f₀/2
    # From YBJ+ equation (1.4): dispersion term is +i(f/2)kₕ²A
    # This is CONSTANT (independent of N²) per Asselin & Young (2019)
    αdisp_profile = Vector{Float64}(undef, nz)
    αdisp_const = par.f₀ / 2.0
    fill!(αdisp_profile, αdisp_const)

    @dealiased_wavenumber_loop S.q G L begin
        # Integrating factors for hyperdiffusion
        λₑ = int_factor(kₓ, kᵧ, par; waves=false)   # For mean flow
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)    # For waves

        #= Update q (QGPV) =#
        if par.fixed_flow
            # Keep q unchanged when mean flow is fixed
            q_arr[k, i, j] = qok_arr[k, i, j]
        else
            # q^(n+1) = [q^n - dt×J(ψ,q) + dt×diffusion] × exp(-λ×dt)
            q_arr[k, i, j] = ( qok_arr[k, i, j] - par.dt*nqk_arr[k, i, j] + par.dt*dqk_arr[k, i, j] ) * exp(-λₑ)
        end

        if par.ybj_plus
            #= Update B (wave envelope) - YBJ+ equation (1.4) from Asselin & Young (2019)
            ∂B/∂t = -J(ψ,B) - (i/2)ζ·B + i(f/2)kₕ²·A =#
            k_global = local_to_global(k, 1, S.q)
            αdisp = αdisp_profile[k_global]
            L⁺A_arr[k, i, j] = ( L⁺Aok_arr[k, i, j] - par.dt*nL⁺Ak_arr[k, i, j]
                               + par.dt*(im*αdisp*kₕ²*A_arr[k, i, j] - 0.5im*rL⁺Ak_arr[k, i, j]) ) * exp(-λʷ)
        else
            #= Update B (wave envelope) - Normal YBJ (PDF Eq. 45-46)
            In terms of real/imaginary parts (with αdisp = f/2):
                ∂BR/∂t = -J(ψ,BR) - αdisp·kₕ²·AI + (1/2)ζ·BI
                ∂BI/∂t = -J(ψ,BI) + αdisp·kₕ²·AR - (1/2)ζ·BR =#
            k_global = local_to_global(k, 1, S.q)
            αdisp = αdisp_profile[k_global]
            L⁺ARnew = ( L⁺ARok_arr[k, i, j] - par.dt*nL⁺ARk_arr[k, i, j]
                      - par.dt*αdisp*kₕ²*Complex(imag(A_arr[k, i, j]),0)
                      + par.dt*0.5*rL⁺AIk_arr[k, i, j] ) * exp(-λʷ)
            L⁺AInew = ( L⁺AIok_arr[k, i, j] - par.dt*nL⁺AIk_arr[k, i, j]
                      + par.dt*αdisp*kₕ²*Complex(real(A_arr[k, i, j]),0)
                      - par.dt*0.5*rL⁺ARk_arr[k, i, j] ) * exp(-λʷ)

            # Recombine into complex B
            L⁺A_arr[k, i, j] = Complex(real(L⁺ARnew), 0) + im*Complex(real(L⁺AInew), 0)
        end
    end begin
        # Zero out dealiased modes
        q_arr[k, i, j] = 0
        L⁺A_arr[k, i, j] = 0
    end

    #= Step 5: Wave feedback on mean flow =#
    # q* = q - qʷ is only the diagnostic RHS for ψ inversion. The prognostic
    # q field remains the balanced-flow PV, matching the QG-YBJp stepping.
    wave_feedback_enabled = !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback
    q_base = nothing
    if wave_feedback_enabled
        q_base = if par.ybj_plus
            replace_q_with_wave_feedback_rhs!(S, G, par, plans, L)
        else
            replace_q_with_wave_feedback_rhs!(S, G, par, plans, L; L⁺ARk, L⁺AIk)
        end
    end

    #= Step 6: Update diagnostic fields =#

    # Invert q → ψ (only if mean flow evolves)
    if !par.fixed_flow
        invert_q_to_psi!(S, G; a, par=par, workspace=workspace)
        if q_base !== nothing
            restore_prognostic_q!(S, q_base)
        end
    end

    # Recover A from B
    if par.passive_scalar
        fill!(A_arr, zero(eltype(A_arr)))
        fill!(C_arr, zero(eltype(C_arr)))
    elseif par.ybj_plus
        # YBJ+: Solve elliptic problem B → A, C (handles 2D decomposition internally)
        invert_L⁺A_to_A!(S, G, par, a; workspace=workspace)
    else
        # Normal YBJ: Different procedure
        sumL⁺A!(S.L⁺A, G; Lmask=L)  # Remove vertical mean
        split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, S.L⁺A)
        sigma = compute_sigma(par, G, nL⁺ARk, nL⁺AIk, rL⁺ARk, rL⁺AIk; Lmask=L, N2_profile=N2_profile)
        compute_A!(S.A, S.C, L⁺ARk, L⁺AIk, sigma, par, G; Lmask=L, N2_profile=N2_profile)
    end

    # Compute velocities from ψ (with dealiasing for omega equation RHS)
    compute_velocities!(S, G; plans, params=par, N2_profile=N2_profile, workspace=workspace, dealias_mask=L)

    #= Step 7: Advect particles (if tracker provided) =#
    # Particles co-evolve with the wave and mean flow equations using the same dt
    if particle_tracker !== nothing
        advect_particles!(particle_tracker, S, G, par.dt, current_time;
                          params=par, N2_profile=N2_profile)
    end

    return S
end

#=
================================================================================
                    LEAPFROG with ROBERT-ASSELIN FILTER
================================================================================
The leapfrog scheme is:
    φ^(n+1) = φ^(n-1) + 2dt × F^n

This is second-order accurate but has a computational mode that can grow.
The Robert-Asselin filter damps this mode:
    φ̃^n = φ^n + γ(φ^(n-1) - 2φ^n + φ^(n+1))

With the integrating factor for hyperdiffusion:
    φ^(n+1) = φ^(n-1) × e^(-2λdt) + 2dt × F^n × e^(-λdt)

This ensures exact treatment of the linear diffusion terms.
================================================================================
=#

"""
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing)

Advance the solution by one leapfrog time step with Robert-Asselin filtering.

# Algorithm

**1. Compute tendencies at time n:**
```
F_q^n = J(ψ^n, q^n) - νz∂²q^(n-1)/∂z²
F_B^n = J(ψ^n, B^n) + dispersion + refraction
```

**2. Leapfrog update with integrating factors:**
For each spectral mode (k):
```
q^(n+1) = q^(n-1) × e^(-2λdt) + 2dt × [-J(ψ,q)^n + diff^n] × e^(-λdt)
B^(n+1) = B^(n-1) × e^(-2λdt) + 2dt × [-J(ψ,B)^n + dispersion + refraction] × e^(-λdt)
```
Note: All tendencies are evaluated at time n and scaled by e^(-λdt) for second-order accuracy.

**3. Robert-Asselin filter:**
```
q̃^n = q^n + γ(q^(n-1) - 2q^n + q^(n+1))
B̃^n = B^n + γ(B^(n-1) - 2B^n + B^(n+1))
```
The filtered values are stored in Sn (which becomes Snm1 after rotation).

**4. Wave feedback (if enabled):**
```
q*^(n+1) = q^(n+1) - qʷ^(n+1)
```

**5. Diagnostic inversions:**
- q^(n+1) → ψ^(n+1)
- B^(n+1) → A^(n+1), C^(n+1)
- ψ^(n+1) → u^(n+1), v^(n+1)

# Arguments
- `Snp1::State`: State at time n+1 (output)
- `Sn::State`: State at time n (input, filter applied to Snm1)
- `Snm1::State`: State at time n-1 (input, receives filtered values)
- `G::Grid`: Grid struct
- `par::QGParams`: Model parameters
- `plans`: FFT plans
- `a`: Elliptic coefficient array
- `dealias_mask`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition
- `N2_profile`: Optional N²(z) profile for vertical velocity computation

# Returns
Modified Snp1 with solution at time n+1.

# Time Level Management
After this call:
- Snp1 contains fields at n+1
- Sn contains **filtered** fields at n (becomes new n-1 after rotation)
- Snm1 is unchanged (will be overwritten after rotation)

Typical loop structure:
```julia
for iter in 1:nsteps
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a)
    # Rotate: Snm1 ← Sn, Sn ← Snp1
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end
```

# Fortran Correspondence
This matches the main leapfrog loop in main_waqg.f90.

# Example
```julia
# After projection step, run leapfrog
for iter in 1:1000
    leapfrog_step!(Snp1, Sn, Snm1, grid, params, plans; a=a, dealias_mask=L)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end
```
"""
function leapfrog_step!(Snp1::State, Sn::State, Snm1::State,
                        G::Grid, par::QGParams, plans; a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing,
                        particle_tracker=nothing, current_time=nothing)
    #= Setup - get local dimensions for PencilArray compatibility =#
    qn_arr = parent(Sn.q)
    L⁺An_arr = parent(Sn.L⁺A)
    An_arr = parent(Sn.A)
    qnm1_arr = parent(Snm1.q)
    L⁺Anm1_arr = parent(Snm1.L⁺A)
    qnp1_arr = parent(Snp1.q)
    L⁺Anp1_arr = parent(Snp1.L⁺A)

    nz = G.nz

    # Note: In xy-pencil format, z is fully local (nz_local = nz).
    # In z-pencil format (after transpose), xy are distributed.
    # Functions that need z local (invert_q_to_psi!, dissipation_q_nv!, etc.)
    # handle transposes internally when using 2D decomposition.

    # Dealias mask - use global indices for lookup
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    #= Step 1: Update diagnostics for current state =#
    if !par.fixed_flow
        invert_q_to_psi!(Sn, G; a, par=par, workspace=workspace)
    end
    compute_velocities!(Sn, G; plans, params=par, N2_profile=N2_profile, workspace=workspace, dealias_mask=L)

    #= Step 2: Allocate and compute tendencies =#
    nqk = similar(Sn.q)   # Advection of q
    dqk = similar(Sn.L⁺A)   # Vertical diffusion
    if par.ybj_plus
        nL⁺Ak = similar(Sn.L⁺A)  # Advection of L⁺A
        rL⁺Ak = similar(Sn.L⁺A)  # Refraction of L⁺A

        # Compute tendencies
        convol_waqg_q!(nqk, Sn.u, Sn.v, Sn.q, G, plans; Lmask=L)
        convol_waqg_L⁺A!(nL⁺Ak, Sn.u, Sn.v, Sn.L⁺A, G, plans; Lmask=L)
        refraction_waqg_L⁺A!(rL⁺Ak, Sn.L⁺A, Sn.psi, G, plans; Lmask=L)
    else
        nL⁺ARk = similar(Sn.L⁺A)  # Advection of L⁺AR
        nL⁺AIk = similar(Sn.L⁺A)  # Advection of L⁺AI
        rL⁺ARk = similar(Sn.L⁺A)  # Refraction of L⁺AR
        rL⁺AIk = similar(Sn.L⁺A)  # Refraction of L⁺AI

        # Split L⁺A into real/imaginary
        L⁺ARk = similar(Sn.L⁺A); L⁺AIk = similar(Sn.L⁺A)
        split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, Sn.L⁺A)

        # Compute tendencies
        convol_waqg!(nqk, nL⁺ARk, nL⁺AIk, Sn.u, Sn.v, Sn.q, L⁺ARk, L⁺AIk, G, plans; Lmask=L)
        refraction_waqg!(rL⁺ARk, rL⁺AIk, L⁺ARk, L⁺AIk, Sn.psi, G, plans; Lmask=L)
    end

    # Vertical diffusion is lagged at time n-1, following QG-YBJp.
    dissipation_q_nv!(dqk, Snm1.q, par, G; workspace=workspace)

    #= Step 3: Apply physics switches =#
    if par.inviscid; dqk .= 0; end
    if par.linear
        nqk .= 0
        if par.ybj_plus
            nL⁺Ak .= 0
        else
            nL⁺ARk .= 0; nL⁺AIk .= 0
        end
    end
    if par.no_dispersion; Sn.A .= 0; Sn.C .= 0; end
    if par.passive_scalar
        Sn.A .= 0; Sn.C .= 0
        if par.ybj_plus
            rL⁺Ak .= 0
        else
            rL⁺ARk .= 0; rL⁺AIk .= 0
        end
    end
    if par.fixed_flow; nqk .= 0; end

    #= Step 4: Leapfrog update with integrating factors =#
    qtemp = similar(Sn.q)
    qtemp_arr = parent(qtemp)
    if par.ybj_plus
        L⁺Atemp = similar(Sn.L⁺A)
        L⁺Atemp_arr = parent(L⁺Atemp)
    else
        L⁺ARtemp = similar(Sn.L⁺A); L⁺AItemp = similar(Sn.L⁺A)
        L⁺ARtemp_arr = parent(L⁺ARtemp); L⁺AItemp_arr = parent(L⁺AItemp)
    end

    # Get parent arrays for tendency terms
    nqk_arr = parent(nqk)
    if par.ybj_plus
        nL⁺Ak_arr = parent(nL⁺Ak)
        rL⁺Ak_arr = parent(rL⁺Ak)
    else
        nL⁺ARk_arr = parent(nL⁺ARk); nL⁺AIk_arr = parent(nL⁺AIk)
        rL⁺ARk_arr = parent(rL⁺ARk); rL⁺AIk_arr = parent(rL⁺AIk)
    end
    dqk_arr = parent(dqk)

    # Precompute dispersion coefficient: αdisp = f₀/2
    # From YBJ+ equation (1.4): dispersion term is +i(f/2)kₕ²A
    # This is CONSTANT (independent of N²) per Asselin & Young (2019)
    αdisp_profile = Vector{Float64}(undef, nz)
    αdisp_const = par.f₀ / 2.0
    fill!(αdisp_profile, αdisp_const)

    @dealiased_wavenumber_loop Sn.q G L begin
        λₑ  = int_factor(kₓ, kᵧ, par; waves=false)
        λʷ = int_factor(kₓ, kᵧ, par; waves=true)

        #= Update q
        q^(n+1) = q^(n-1)×e^(-2λdt) - 2dt×J(ψ,q)^n×e^(-λdt)
                   + 2dt×diff^(n-1)×e^(-2λdt). =#
        if par.fixed_flow
            qtemp_arr[k, i, j] = qn_arr[k, i, j]  # Keep unchanged
        else
            qtemp_arr[k, i, j] = qnm1_arr[k, i, j]*exp(-2λₑ) +
                           2*par.dt*(-nqk_arr[k, i, j])*exp(-λₑ) +
                           2*par.dt*dqk_arr[k, i, j]*exp(-2λₑ)
        end

        if par.ybj_plus
            #= Update B (complex) - YBJ+ equation (1.4) from Asselin & Young (2019)
            ∂B/∂t = -J(ψ,B) - (i/2)ζ·B + i(f/2)kₕ²·A =#
            k_global = local_to_global(k, 1, Sn.q)
            αdisp = αdisp_profile[k_global]
            L⁺Atemp_arr[k, i, j] = L⁺Anm1_arr[k, i, j]*exp(-2λʷ) +
                           2*par.dt*( -nL⁺Ak_arr[k, i, j] +
                                      im*αdisp*kₕ²*An_arr[k, i, j] -
                                      0.5im*rL⁺Ak_arr[k, i, j] )*exp(-λʷ)
        else
            #= Update B (real and imaginary parts) - PDF Eq. 45-46
            BR^(n+1) = BR^(n-1)×e^(-2λdt) - 2dt×[J(ψ,BR) + αdisp·kₕ²·AI - (1/2)ζ·BI]×e^(-λdt)
            BI^(n+1) = BI^(n-1)×e^(-2λdt) - 2dt×[J(ψ,BI) - αdisp·kₕ²·AR + (1/2)ζ·BR]×e^(-λdt) =#
            k_global = local_to_global(k, 1, Sn.q)
            αdisp = αdisp_profile[k_global]
            L⁺ARtemp_arr[k, i, j] = Complex(real(L⁺Anm1_arr[k, i, j]),0)*exp(-2λʷ) -
                           2*par.dt*( nL⁺ARk_arr[k, i, j] +
                                      αdisp*kₕ²*Complex(imag(An_arr[k, i, j]),0) -
                                      0.5*rL⁺AIk_arr[k, i, j] )*exp(-λʷ)
            L⁺AItemp_arr[k, i, j] = Complex(imag(L⁺Anm1_arr[k, i, j]),0)*exp(-2λʷ) -
                           2*par.dt*( nL⁺AIk_arr[k, i, j] -
                                      αdisp*kₕ²*Complex(real(An_arr[k, i, j]),0) +
                                      0.5*rL⁺ARk_arr[k, i, j] )*exp(-λʷ)
        end
    end begin
        qtemp_arr[k, i, j] = 0
        if par.ybj_plus
            L⁺Atemp_arr[k, i, j] = 0
        else
            L⁺ARtemp_arr[k, i, j] = 0
            L⁺AItemp_arr[k, i, j] = 0
        end
    end

    #= Step 5: Robert-Asselin filter
    Damps the computational mode: φ̃^n = φ^n + γ(φ^(n-1) - 2φ^n + φ^(n+1))

    IMPORTANT: Store filtered values in Sn (not Snm1!), so that after the rotation
    (Snm1, Sn, Snp1) = (Sn, Snp1, Snm1), the filtered n state becomes the new n-1 state.
    Previous code stored in Snm1, but after rotation the old unfiltered Sn became the
    new Snm1, effectively leaving leapfrog unfiltered. =#
    γ = par.γ
    @dealiased_spectral_loop Sn.q L begin
        # Filter q - store in Sn so it becomes new Snm1 after rotation
        qn_arr[k, i, j] = qn_arr[k, i, j] + γ*( qnm1_arr[k, i, j] - 2qn_arr[k, i, j] + qtemp_arr[k, i, j] )

        # Filter B - store in Sn so it becomes new Snm1 after rotation
        if par.ybj_plus
            L⁺An_arr[k, i, j] = L⁺An_arr[k, i, j] + γ*( L⁺Anm1_arr[k, i, j] - 2L⁺An_arr[k, i, j] + L⁺Atemp_arr[k, i, j] )
        else
            L⁺Anp1_local = Complex(real(L⁺ARtemp_arr[k, i, j]),0) + im*Complex(real(L⁺AItemp_arr[k, i, j]),0)
            L⁺An_arr[k, i, j] = L⁺An_arr[k, i, j] + γ*( L⁺Anm1_arr[k, i, j] - 2L⁺An_arr[k, i, j] + L⁺Anp1_local )
        end
    end begin
        qn_arr[k, i, j] = 0
        L⁺An_arr[k, i, j] = 0
    end

    #= Step 6: Accept the new solution =#
    @local_spectral_loop Snp1.q begin
        qnp1_arr[k, i, j] = qtemp_arr[k, i, j]
        if par.ybj_plus
            L⁺Anp1_arr[k, i, j] = L⁺Atemp_arr[k, i, j]
        else
            L⁺Anp1_arr[k, i, j] = Complex(real(L⁺ARtemp_arr[k, i, j]),0) + im*Complex(real(L⁺AItemp_arr[k, i, j]),0)
        end
    end

    #= Step 7: Wave feedback on mean flow =#
    # q* = q - qʷ is only the diagnostic RHS for ψ inversion. The prognostic
    # q field remains the balanced-flow PV, matching the QG-YBJp stepping.
    wave_feedback_enabled = !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback
    qnp1_base = nothing
    if wave_feedback_enabled
        qnp1_base = replace_q_with_wave_feedback_rhs!(Snp1, G, par, plans, L)
    end

    #= Step 8: Update diagnostics for new state =#

    # Invert q → ψ (handles 2D decomposition transposes internally)
    if !par.fixed_flow
        invert_q_to_psi!(Snp1, G; a, par=par, workspace=workspace)
        if qnp1_base !== nothing
            restore_prognostic_q!(Snp1, qnp1_base)
        end
    end

    # Recover A from B
    if par.passive_scalar
        fill!(parent(Snp1.A), zero(eltype(parent(Snp1.A))))
        fill!(parent(Snp1.C), zero(eltype(parent(Snp1.C))))
    elseif par.ybj_plus
        # YBJ+: handles 2D decomposition transposes internally
        invert_L⁺A_to_A!(Snp1, G, par, a; workspace=workspace)
    else
        # Normal YBJ path
        sumL⁺A!(Snp1.L⁺A, G; Lmask=L)
        L⁺ARk3 = similar(Snp1.L⁺A); L⁺AIk3 = similar(Snp1.L⁺A)
        split_L⁺A_to_real_imag!(L⁺ARk3, L⁺AIk3, Snp1.L⁺A)
        sigma2 = compute_sigma(par, G, nL⁺ARk, nL⁺AIk, rL⁺ARk, rL⁺AIk; Lmask=L, N2_profile=N2_profile)
        compute_A!(Snp1.A, Snp1.C, L⁺ARk3, L⁺AIk3, sigma2, par, G; Lmask=L, N2_profile=N2_profile)
    end

    # Compute velocities
    compute_velocities!(Snp1, G; plans, params=par, N2_profile=N2_profile, workspace=workspace, dealias_mask=L)

    #= Step 9: Advect particles (if tracker provided) =#
    # Particles co-evolve with the wave and mean flow equations using the same dt
    if particle_tracker !== nothing
        advect_particles!(particle_tracker, Snp1, G, par.dt, current_time;
                          params=par, N2_profile=N2_profile)
    end

    return Snp1
end
