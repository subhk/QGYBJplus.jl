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
   - Leapfrog: φ^(n+1) = φ^(n-1) × e^(-2λdt) - 2dt × tendency × e^(-λdt)
   - Forward Euler: φ^(n+1) = φ^n × e^(-λdt) - dt × tendency

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
All loops use local indexing with local_to_global() for wavenumber access.
The vertical dimension (z) must be fully local for proper operation.

================================================================================
=#

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
    first_projection_step!(S, G, par, plans; a, dealias_mask=nothing, workspace=nothing)

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

# Returns
Modified state S at time n+1.

# Fortran Correspondence
This matches the projection step in main_waqg.f90.

# Example
```julia
# Initialize and run projection step
state = init_state(grid, params)
init_random_psi!(state, grid, params, plans)
a = a_ell_ut(params, grid)
L = dealias_mask(params, grid)
first_projection_step!(state, grid, params, plans; a=a, dealias_mask=L)
```
"""
function first_projection_step!(S::State, G::Grid, par::QGParams, plans; a, dealias_mask=nothing, workspace=nothing)
    #= Setup - get local dimensions for PencilArray compatibility =#
    q_arr = parent(S.q)
    B_arr = parent(S.B)
    psi_arr = parent(S.psi)
    A_arr = parent(S.A)
    C_arr = parent(S.C)

    nx_local, ny_local, nz_local = size(q_arr)
    nz = G.nz

    # Note: For 2D decomposition, nz_local may be < nz (z distributed in xy-pencil)
    # Functions that need z local (invert_q_to_psi!, dissipation_q_nv!, etc.)
    # handle transposes internally

    # Dealias mask - use global indices for lookup
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    # Allocate tendency arrays (same size as local arrays)
    nqk  = similar(S.q)    # J(ψ, q) advection of PV
    nBRk = similar(S.B)    # J(ψ, BR) advection of wave real part
    nBIk = similar(S.B)    # J(ψ, BI) advection of wave imaginary part
    rBRk = similar(S.B)    # BR × ζ refraction real part
    rBIk = similar(S.B)    # BI × ζ refraction imaginary part
    dqk  = similar(S.B)    # Vertical diffusion of q

    #= Split B into real and imaginary parts for computation
    The wave field B is complex; we work with BR = Re(B), BI = Im(B) =#
    BRk = similar(S.B); BIk = similar(S.B)
    BRk_arr = parent(BRk); BIk_arr = parent(BIk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        BRk_arr[i,j,k] = Complex(real(B_arr[i,j,k]), 0)
        BIk_arr[i,j,k] = Complex(imag(B_arr[i,j,k]), 0)
    end

    #= Step 1: Compute diagnostic fields ψ and velocities =#
    invert_q_to_psi!(S, G; a, par=par, workspace=workspace)           # q → ψ
    compute_velocities!(S, G; plans, params=par) # ψ → u, v

    #= Step 2: Compute nonlinear tendencies =#

    # Advection: J(ψ, q), J(ψ, BR), J(ψ, BI)
    convol_waqg!(nqk, nBRk, nBIk, S.u, S.v, S.q, BRk, BIk, G, plans; Lmask=L)

    # Wave refraction: B × ζ where ζ = ∇²ψ
    refraction_waqg!(rBRk, rBIk, BRk, BIk, S.psi, G, plans; Lmask=L)

    # Vertical diffusion: νz ∂²q/∂z² (handles 2D decomposition transposes internally)
    dissipation_q_nv!(dqk, S.q, par, G; workspace=workspace)

    #= Step 3: Apply physics switches =#

    # inviscid: No dissipation
    if par.inviscid; dqk .= 0; end

    # linear: No nonlinear advection
    if par.linear
        nqk .= 0; nBRk .= 0; nBIk .= 0
    end

    # no_dispersion: Waves don't disperse (A = 0)
    if par.no_dispersion
        S.A .= 0; S.C .= 0
    end

    # passive_scalar: Waves are passive tracers (no dispersion, no refraction)
    if par.passive_scalar
        S.A .= 0; S.C .= 0; rBRk .= 0; rBIk .= 0
    end

    # fixed_flow: Mean flow doesn't evolve
    if par.fixed_flow; nqk .= 0; end

    #= Store old fields for time stepping =#
    qok  = copy(S.q)
    qok_arr = parent(qok)
    BRok = similar(S.B); BIok = similar(S.B)
    BRok_arr = parent(BRok); BIok_arr = parent(BIok)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        BRok_arr[i,j,k] = Complex(real(B_arr[i,j,k]), 0)
        BIok_arr[i,j,k] = Complex(imag(B_arr[i,j,k]), 0)
    end

    #= Step 4: Forward Euler with integrating factors =#
    # The integrating factor handles hyperdiffusion exactly:
    # φ^(n+1) = [φ^n - dt × F] × exp(-λ×dt)

    # Get parent arrays for tendency terms
    nqk_arr = parent(nqk)
    nBRk_arr = parent(nBRk); nBIk_arr = parent(nBIk)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)
    dqk_arr = parent(dqk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        # Get global indices for wavenumber lookup
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)

        if L[i_global, j_global]
            kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
            # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
            kₕ² = kₓ^2 + kᵧ^2

            # Integrating factors for hyperdiffusion
            λₑ = int_factor(kₓ, kᵧ, par; waves=false)   # For mean flow
            λʷ = int_factor(kₓ, kᵧ, par; waves=true)    # For waves

            #= Update q (QGPV) =#
            if par.fixed_flow
                # Keep q unchanged when mean flow is fixed
                q_arr[i,j,k] = qok_arr[i,j,k]
            else
                # q^(n+1) = [q^n - dt×J(ψ,q) + dt×diffusion] × exp(-λ×dt)
                q_arr[i,j,k] = ( qok_arr[i,j,k] - par.dt*nqk_arr[i,j,k] + par.dt*dqk_arr[i,j,k] ) * exp(-λₑ)
            end

            #= Update B (wave envelope)
            The YBJ+ equation for B is:
                ∂B/∂t + J(ψ,B) = -i(kₕ²·N²/(2f))A + (1/2)B×ζ

            In terms of real/imaginary parts:
                ∂BR/∂t = -J(ψ,BR) - (kₕ²·N²/(2f))AI + (1/2)BI×ζ
                ∂BI/∂t = -J(ψ,BI) + (kₕ²·N²/(2f))AR - (1/2)BR×ζ =#
            αdisp = par.N² / (2.0 * par.f₀)  # Dispersion coefficient N²/(2f)
            BRnew = ( BRok_arr[i,j,k] - par.dt*nBRk_arr[i,j,k]
                      - par.dt*αdisp*kₕ²*Complex(imag(A_arr[i,j,k]),0)
                      + par.dt*0.5*rBIk_arr[i,j,k] ) * exp(-λʷ)
            BInew = ( BIok_arr[i,j,k] - par.dt*nBIk_arr[i,j,k]
                      + par.dt*αdisp*kₕ²*Complex(real(A_arr[i,j,k]),0)
                      - par.dt*0.5*rBRk_arr[i,j,k] ) * exp(-λʷ)

            # Recombine into complex B
            B_arr[i,j,k] = Complex(real(BRnew), 0) + im*Complex(real(BInew), 0)
        else
            # Zero out dealiased modes
            q_arr[i,j,k] = 0
            B_arr[i,j,k] = 0
        end
    end

    #= Step 5: Wave feedback on mean flow =#
    # q* = q - qʷ where qʷ is the wave feedback term
    wave_feedback_enabled = !par.no_feedback && !par.no_wave_feedback
    if wave_feedback_enabled
        qwk = similar(S.q)
        qwk_arr = parent(qwk)

        # Rebuild BR/BI from updated B
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            BRk_arr[i,j,k] = Complex(real(B_arr[i,j,k]), 0)
            BIk_arr[i,j,k] = Complex(imag(B_arr[i,j,k]), 0)
        end

        # Compute qʷ from B
        compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)

        # Subtract from q
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 1, G)
            j_global = local_to_global(j, 2, G)
            if L[i_global, j_global]
                q_arr[i,j,k] -= qwk_arr[i,j,k]
            else
                q_arr[i,j,k] = 0
            end
        end
    end

    #= Step 6: Update diagnostic fields =#

    # Invert q → ψ (only if mean flow evolves)
    if !par.fixed_flow
        invert_q_to_psi!(S, G; a, par=par, workspace=workspace)
    end

    # Recover A from B
    if par.ybj_plus
        # YBJ+: Solve elliptic problem B → A, C (handles 2D decomposition internally)
        invert_B_to_A!(S, G, par, a; workspace=workspace)
    else
        # Normal YBJ: Different procedure
        BRk2 = similar(S.B); BIk2 = similar(S.B)
        BRk2_arr = parent(BRk2); BIk2_arr = parent(BIk2)
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            BRk2_arr[i,j,k] = Complex(real(B_arr[i,j,k]), 0)
            BIk2_arr[i,j,k] = Complex(imag(B_arr[i,j,k]), 0)
        end
        sumB!(S.B, G; Lmask=L)  # Remove vertical mean
        sigma = compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask=L)
        compute_A!(S.A, S.C, BRk2, BIk2, sigma, par, G; Lmask=L)
    end

    # Compute velocities from ψ
    compute_velocities!(S, G; plans, params=par)

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
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a, dealias_mask=nothing, workspace=nothing)

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
q^(n+1) = q^(n-1) × e^(-2λdt) - 2dt × F_q^n × e^(-λdt) + 2dt × diff^(n-1) × e^(-2λdt)
B^(n+1) = B^(n-1) × e^(-2λdt) - 2dt × F_B^n × e^(-λdt)
```

**3. Robert-Asselin filter:**
```
q̃^n = q^n + γ(q^(n-1) - 2q^n + q^(n+1))
B̃^n = B^n + γ(B^(n-1) - 2B^n + B^(n+1))
```
The filtered values are stored in Snm1 for the next step.

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

# Returns
Modified Snp1 with solution at time n+1.

# Time Level Management
After this call:
- Snp1 contains fields at n+1
- Snm1 contains **filtered** fields at n (for next step's n-1)
- Sn is unchanged (use Snm1 as new Sn in next call)

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
                        G::Grid, par::QGParams, plans; a, dealias_mask=nothing, workspace=nothing)
    #= Setup - get local dimensions for PencilArray compatibility =#
    qn_arr = parent(Sn.q)
    Bn_arr = parent(Sn.B)
    An_arr = parent(Sn.A)
    qnm1_arr = parent(Snm1.q)
    Bnm1_arr = parent(Snm1.B)
    qnp1_arr = parent(Snp1.q)
    Bnp1_arr = parent(Snp1.B)

    nx_local, ny_local, nz_local = size(qn_arr)
    nz = G.nz

    # Note: For 2D decomposition, nz_local may be < nz (z distributed in xy-pencil)
    # Functions that need z local (invert_q_to_psi!, dissipation_q_nv!, etc.)
    # handle transposes internally

    # Dealias mask - use global indices for lookup
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    #= Step 1: Update diagnostics for current state =#
    if !par.fixed_flow
        invert_q_to_psi!(Sn, G; a, par=par, workspace=workspace)
    end
    compute_velocities!(Sn, G; plans, params=par)

    #= Step 2: Allocate and compute tendencies =#
    nqk  = similar(Sn.q)    # Advection of q
    nBRk = similar(Sn.B)    # Advection of BR
    nBIk = similar(Sn.B)    # Advection of BI
    rBRk = similar(Sn.B)    # Refraction of BR
    rBIk = similar(Sn.B)    # Refraction of BI
    dqk  = similar(Sn.B)    # Vertical diffusion

    # Split B into real/imaginary
    BRk = similar(Sn.B); BIk = similar(Sn.B)
    BRk_arr = parent(BRk); BIk_arr = parent(BIk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        BRk_arr[i,j,k] = Complex(real(Bn_arr[i,j,k]), 0)
        BIk_arr[i,j,k] = Complex(imag(Bn_arr[i,j,k]), 0)
    end

    # Compute tendencies
    convol_waqg!(nqk, nBRk, nBIk, Sn.u, Sn.v, Sn.q, BRk, BIk, G, plans; Lmask=L)
    refraction_waqg!(rBRk, rBIk, BRk, BIk, Sn.psi, G, plans; Lmask=L)

    # Vertical diffusion uses q at n-1 (for leapfrog stability)
    # Handles 2D decomposition transposes internally
    dissipation_q_nv!(dqk, Snm1.q, par, G; workspace=workspace)

    #= Step 3: Apply physics switches =#
    if par.inviscid; dqk .= 0; end
    if par.linear; nqk .= 0; nBRk .= 0; nBIk .= 0; end
    if par.no_dispersion; Sn.A .= 0; end
    if par.passive_scalar; Sn.A .= 0; rBRk .= 0; rBIk .= 0; end
    if par.fixed_flow; nqk .= 0; end

    #= Step 4: Leapfrog update with integrating factors =#
    qtemp = similar(Sn.q)
    BRtemp = similar(Sn.B); BItemp = similar(Sn.B)
    qtemp_arr = parent(qtemp)
    BRtemp_arr = parent(BRtemp); BItemp_arr = parent(BItemp)

    # Get parent arrays for tendency terms
    nqk_arr = parent(nqk)
    nBRk_arr = parent(nBRk); nBIk_arr = parent(nBIk)
    rBRk_arr = parent(rBRk); rBIk_arr = parent(rBIk)
    dqk_arr = parent(dqk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        # Get global indices for wavenumber lookup
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)

        if L[i_global, j_global]
            kₓ = G.kx[i_global]; kᵧ = G.ky[j_global]
            # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
            kₕ² = kₓ^2 + kᵧ^2
            λₑ  = int_factor(kₓ, kᵧ, par; waves=false)
            λʷ = int_factor(kₓ, kᵧ, par; waves=true)

            #= Update q
            q^(n+1) = q^(n-1)×e^(-2λdt) - 2dt×J(ψ,q)^n×e^(-λdt) + 2dt×diff^(n-1)×e^(-2λdt) =#
            if par.fixed_flow
                qtemp_arr[i,j,k] = qn_arr[i,j,k]  # Keep unchanged
            else
                qtemp_arr[i,j,k] = qnm1_arr[i,j,k]*exp(-2λₑ) -
                               2*par.dt*nqk_arr[i,j,k]*exp(-λₑ) +
                               2*par.dt*dqk_arr[i,j,k]*exp(-2λₑ)
            end

            #= Update B (real and imaginary parts)
            BR^(n+1) = BR^(n-1)×e^(-2λdt) - 2dt×[J(ψ,BR) + (kₕ²·N²/(2f))AI - (1/2)BI×ζ]×e^(-λdt)
            BI^(n+1) = BI^(n-1)×e^(-2λdt) - 2dt×[J(ψ,BI) - (kₕ²·N²/(2f))AR + (1/2)BR×ζ]×e^(-λdt) =#
            αdisp = par.N² / (2.0 * par.f₀)  # Dispersion coefficient N²/(2f)
            BRtemp_arr[i,j,k] = Complex(real(Bnm1_arr[i,j,k]),0)*exp(-2λʷ) -
                           2*par.dt*( nBRk_arr[i,j,k] +
                                     αdisp*kₕ²*Complex(imag(An_arr[i,j,k]),0) -
                                     0.5*rBIk_arr[i,j,k] )*exp(-λʷ)
            BItemp_arr[i,j,k] = Complex(imag(Bnm1_arr[i,j,k]),0)*exp(-2λʷ) -
                           2*par.dt*( nBIk_arr[i,j,k] -
                                     αdisp*kₕ²*Complex(real(An_arr[i,j,k]),0) +
                                     0.5*rBRk_arr[i,j,k] )*exp(-λʷ)
        else
            qtemp_arr[i,j,k] = 0; BRtemp_arr[i,j,k] = 0; BItemp_arr[i,j,k] = 0
        end
    end

    #= Step 5: Robert-Asselin filter
    Damps the computational mode: φ̃^n = φ^n + γ(φ^(n-1) - 2φ^n + φ^(n+1))
    Store filtered values in Snm1 (they become the "n-1" for next step) =#
    γ = par.γ
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        # Get global indices for dealias mask lookup
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)

        if L[i_global, j_global]
            # Filter q
            qnm1_arr[i,j,k] = qn_arr[i,j,k] + γ*( qnm1_arr[i,j,k] - 2qn_arr[i,j,k] + qtemp_arr[i,j,k] )

            # Filter B
            Bnp1 = Complex(real(BRtemp_arr[i,j,k]),0) + im*Complex(real(BItemp_arr[i,j,k]),0)
            Bnm1_arr[i,j,k] = Bn_arr[i,j,k] + γ*( Bnm1_arr[i,j,k] - 2Bn_arr[i,j,k] + Bnp1 )
        else
            qnm1_arr[i,j,k] = 0; Bnm1_arr[i,j,k] = 0
        end
    end

    #= Step 6: Accept the new solution =#
    Snp1.q .= qtemp
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        qnp1_arr[i,j,k] = qtemp_arr[i,j,k]
        Bnp1_arr[i,j,k] = Complex(real(BRtemp_arr[i,j,k]),0) + im*Complex(real(BItemp_arr[i,j,k]),0)
    end

    #= Step 7: Wave feedback on mean flow =#
    wave_feedback_enabled = !par.no_feedback && !par.no_wave_feedback
    if wave_feedback_enabled
        qwk = similar(Snp1.q)
        qwk_arr = parent(qwk)

        # Rebuild BR/BI from updated B
        BRk2 = similar(Snp1.B); BIk2 = similar(Snp1.B)
        BRk2_arr = parent(BRk2); BIk2_arr = parent(BIk2)
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            BRk2_arr[i,j,k] = Complex(real(Bnp1_arr[i,j,k]),0)
            BIk2_arr[i,j,k] = Complex(imag(Bnp1_arr[i,j,k]),0)
        end

        compute_qw!(qwk, BRk2, BIk2, par, G, plans; Lmask=L)

        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 1, G)
            j_global = local_to_global(j, 2, G)
            if L[i_global, j_global]
                qnp1_arr[i,j,k] -= qwk_arr[i,j,k]
            else
                qnp1_arr[i,j,k] = 0
            end
        end
    end

    #= Step 8: Update diagnostics for new state =#

    # Invert q → ψ (handles 2D decomposition transposes internally)
    if !par.fixed_flow
        invert_q_to_psi!(Snp1, G; a, par=par, workspace=workspace)
    end

    # Recover A from B
    if par.ybj_plus
        # YBJ+: handles 2D decomposition transposes internally
        invert_B_to_A!(Snp1, G, par, a; workspace=workspace)
    else
        # Normal YBJ path
        BRk3 = similar(Snp1.B); BIk3 = similar(Snp1.B)
        BRk3_arr = parent(BRk3); BIk3_arr = parent(BIk3)
        @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
            BRk3_arr[i,j,k] = Complex(real(Bnp1_arr[i,j,k]), 0)
            BIk3_arr[i,j,k] = Complex(imag(Bnp1_arr[i,j,k]), 0)
        end
        sumB!(Snp1.B, G; Lmask=L)
        sigma2 = compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask=L)
        compute_A!(Snp1.A, Snp1.C, BRk3, BIk3, sigma2, par, G; Lmask=L)
    end

    # Compute velocities
    compute_velocities!(Snp1, G; plans, params=par)

    return Snp1
end

#= ============================================================================
                    HIGH-LEVEL SIMULATION RUNNER
============================================================================ =#

"""
    run_simulation!(S, G, par, plans; kwargs...)

Run a complete QG-YBJ+ simulation with automatic time-stepping, output, and diagnostics.

This is the recommended high-level interface that handles all the details of:
- Leapfrog time integration with Robert-Asselin filter
- State array rotation (no manual management needed)
- Periodic output to NetCDF files
- Progress reporting and diagnostics

# Arguments
- `S::State`: Initial state (will be modified in-place)
- `G::Grid`: Spatial grid
- `par::QGParams`: Model parameters (includes dt, nt)
- `plans::Plans`: FFT plans

# Keyword Arguments
- `output_config::OutputConfig`: Output configuration (required for saving)
- `mpi_config=nothing`: MPI configuration for parallel runs
- `parallel_config=nothing`: Parallel I/O configuration
- `workspace=nothing`: Pre-allocated workspace arrays
- `print_progress::Bool=true`: Print progress to stdout
- `progress_interval::Int=0`: Steps between progress updates (0 = auto, based on nt)

# Returns
- Final `State` at the end of the simulation

# Example
```julia
# Setup
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, dt=100.0, nt=5000)
G, S, plans, a = setup_model(par)

# Initialize flow and waves
initialize_dipole!(S, G, par)
initialize_surface_waves!(S, G, par)

# Configure output
output_config = OutputConfig(
    output_dir = "output",
    psi_interval = 2π / par.f₀,     # Save every inertial period
    wave_interval = 2π / par.f₀
)

# Run simulation - all time-stepping handled automatically
S_final = run_simulation!(S, G, par, plans; output_config=output_config)
```

See also: [`OutputConfig`](@ref), [`leapfrog_step!`](@ref)
"""
function run_simulation!(S::State, G::Grid, par::QGParams, plans::Plans;
                         output_config::Union{OutputConfig,Nothing}=nothing,
                         mpi_config=nothing,
                         parallel_config=nothing,
                         workspace=nothing,
                         print_progress::Bool=true,
                         progress_interval::Int=0)

    # Determine if running in MPI mode
    is_mpi = mpi_config !== nothing
    is_root = !is_mpi || mpi_config.is_root

    # Setup workspace if not provided
    if workspace === nothing
        workspace = is_mpi ? init_mpi_workspace(G, mpi_config) : init_workspace(G)
    end

    # Setup parallel config if not provided (for I/O)
    if parallel_config === nothing && is_mpi
        parallel_config = ParallelConfig(
            use_mpi = true,
            comm = mpi_config.comm,
            parallel_io = false
        )
    elseif parallel_config === nothing
        parallel_config = ParallelConfig(use_mpi = false)
    end

    # Compute coefficients
    a_ell = a_ell_ut(par, G)
    L_mask = dealias_mask(G)

    # Initial velocity computation
    compute_velocities!(S, G; plans=plans, params=par, workspace=workspace)

    # Create output manager if config provided
    output_manager = nothing
    if output_config !== nothing
        output_manager = OutputManager(output_config, par, parallel_config)

        # Save initial state
        if is_root && print_progress
            println("Saving initial state...")
        end
        write_state_file(output_manager, S, G, plans, 0.0, parallel_config; params=par)
    end

    # First projection step (Forward Euler to initialize leapfrog)
    first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace)

    # Setup leapfrog states
    Sn = deepcopy(S)
    Snm1 = deepcopy(S)
    Snp1 = deepcopy(S)

    # Determine progress interval
    nt = par.nt
    dt = par.dt
    if progress_interval <= 0
        progress_interval = max(1, nt ÷ 20)  # ~20 progress updates
    end

    # Compute save intervals in steps
    psi_save_steps = output_config !== nothing && output_config.psi_interval > 0 ?
                     max(1, round(Int, output_config.psi_interval / dt)) : 0
    wave_save_steps = output_config !== nothing && output_config.wave_interval > 0 ?
                      max(1, round(Int, output_config.wave_interval / dt)) : 0
    save_steps = psi_save_steps > 0 ? psi_save_steps : wave_save_steps

    # Print header
    if is_root && print_progress
        println("\n" * "="^60)
        println("Starting time integration...")
        println("  Steps: $nt, dt: $dt")
        if save_steps > 0
            println("  Saving every $save_steps steps")
        end
        println("="^60 * "\n")
    end

    # Time integration loop
    for step in 1:nt
        # Leapfrog step
        leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                       a=a_ell, dealias_mask=L_mask, workspace=workspace)

        # Rotate states: (n-1) ← (n) ← (n+1) ← (n-1)
        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        current_time = step * dt

        # Progress output
        if is_root && print_progress && step % progress_interval == 0
            progress_pct = round(100 * step / nt, digits=1)
            @printf("  Step %d/%d (%.1f%%) - t = %.2e\n", step, nt, progress_pct, current_time)
        end

        # Save state
        if output_manager !== nothing && save_steps > 0 && step % save_steps == 0
            write_state_file(output_manager, Sn, G, plans, current_time, parallel_config; params=par)
        end
    end

    # Copy final state back to S
    copyto!(parent(S.psi), parent(Sn.psi))
    copyto!(parent(S.q), parent(Sn.q))
    copyto!(parent(S.B), parent(Sn.B))
    copyto!(parent(S.A), parent(Sn.A))
    copyto!(parent(S.u), parent(Sn.u))
    copyto!(parent(S.v), parent(Sn.v))
    copyto!(parent(S.w), parent(Sn.w))

    # Print completion message
    if is_root && print_progress
        println("\n" * "="^60)
        println("Simulation complete!")
        if output_manager !== nothing
            println("Output saved to: $(output_config.output_dir)/")
        end
        println("="^60)
    end

    return S
end
