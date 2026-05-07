#=
================================================================================
                    timestep.jl - Time Integration
================================================================================

This file implements time stepping for the QG-YBJ+ model.

The production time stepper is a second-order exponential Runge-Kutta method
(ETDRK2). Horizontal hyperdiffusion is handled exactly by integrating factors;
advection, refraction, dispersion, and vertical diffusion are evaluated
explicitly.

TIME INTEGRATION ALGORITHM:
---------------------------
For each time step from n to n+1:

1. Diagnose fields at time n:
   - If wave feedback is enabled, invert q* = q - qʷ for ψ
   - Restore prognostic q after the inversion

2. Compute tendencies at time n:
   - Advection: J(ψ, q), J(ψ, B)
   - Refraction: B × ζ
   - Vertical diffusion: νz ∂²q/∂z²

3. Apply physics switches:
   - linear: zero nonlinear terms
   - inviscid: zero dissipation
   - passive_scalar: zero dispersion and refraction
   - fixed_flow: zero q tendency

4. Time step with ETDRK2 and hyperdiffusion integrating factors:
   - a = Eφⁿ + Δt φ₁(LΔt) N(φⁿ)
   - φⁿ⁺¹ = a + Δt φ₂(LΔt) [N(a) - N(φⁿ)]
   where E = exp(LΔt) and L is the diagonal horizontal hyperdiffusion operator.

5. Diagnostic updates:
   - Invert q or q* → ψ
   - Invert B → A (YBJ+) or compute A directly (normal YBJ)
   - Compute velocities from ψ

FORTRAN CORRESPONDENCE:
----------------------
The time stepping matches main_waqg.f90 in the Fortran QG_YBJp code.
The integrating factor approach for hyperdiffusion is from the Fortran.

STABILITY:
----------
- Advective CFL condition: dt < min(dx/|u|, dy/|v|)
- Horizontal hyperdiffusion is integrated exactly.

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
                                           L⁺ARk=nothing, L⁺AIk=nothing,
                                           q_base=nothing, qwk=nothing,
                                           nonlinear_workspace=nothing)
    q_base = q_base === nothing ? copy(S.q) : q_base
    q_base_arr = parent(q_base)
    q_arr = parent(S.q)
    q_base_arr .= q_arr

    qwk = qwk === nothing ? similar(S.q) : qwk
    qwk_arr = parent(qwk)

    if par.ybj_plus
        compute_qw_complex!(qwk, S.L⁺A, par, G, plans; Lmask=L,
                            workspace=nonlinear_workspace)
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

_wave_feedback_enabled(par::QGParams) = !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback

"""
    ExpRK2Workspace(state, plans=nothing)

Reusable scratch storage for `exp_rk2_step!`.

Allocate this once per simulation and pass it as `timestep_workspace` to avoid
allocating RK stage states, tendency arrays, and velocity FFT scratch on every
time step. Pass `plans` when using MPI so physical FFT buffers are allocated on
the FFT input pencil.
"""
struct ExpRK2Workspace{S, A, P, N}
    stage::S
    rhsq₀::A
    rhsB₀::A
    rhsq₁::A
    rhsB₁::A
    nqk::A
    dqk::A
    nL⁺Ak::A
    rL⁺Ak::A
    q_base::A
    qwk::A
    uk::A
    vk::A
    tmpu::P
    tmpv::P
    nonlinear::N
end

function ExpRK2Workspace(S::State, plans=nothing)
    uk = similar(S.psi)
    vk = similar(S.psi)

    return ExpRK2Workspace(
        copy_state(S),
        similar(S.q), similar(S.L⁺A),
        similar(S.q), similar(S.L⁺A),
        similar(S.q), similar(S.q),
        similar(S.L⁺A), similar(S.L⁺A),
        similar(S.q), similar(S.q),
        uk, vk,
        allocate_fft_backward_dst(uk, plans),
        allocate_fft_backward_dst(vk, plans),
        NonlinearWorkspace(S.psi, plans),
    )
end

function _etd_coefficients(λdt, dt)
    E = exp(-λdt)

    if abs(λdt) < 1e-6
        x = λdt
        x2 = x * x
        hφ1 = dt * (1 - x / 2 + x2 / 6 - x2 * x / 24 + x2 * x2 / 120)
        hφ2 = dt * (1 / 2 - x / 6 + x2 / 24 - x2 * x / 120 + x2 * x2 / 720)
        return E, hφ1, hφ2
    else
        expm1_neg = expm1(-λdt)
        hφ1 = dt * (-expm1_neg) / λdt
        hφ2 = dt * (expm1_neg + λdt) / λdt^2
        return E, hφ1, hφ2
    end
end

function _update_diagnostics!(S::State, G::Grid, par::QGParams, plans, a, L;
                              workspace=nothing, N2_profile=nothing,
                              timestep_workspace=nothing, compute_w=true,
                              use_wave_feedback=false)
    if !par.fixed_flow
        q_base = nothing
        if use_wave_feedback && _wave_feedback_enabled(par)
            q_base = replace_q_with_wave_feedback_rhs!(S, G, par, plans, L;
                                                       q_base = timestep_workspace === nothing ? nothing : timestep_workspace.q_base,
                                                       qwk = timestep_workspace === nothing ? nothing : timestep_workspace.qwk,
                                                       nonlinear_workspace = timestep_workspace === nothing ? nothing : timestep_workspace.nonlinear)
        end

        invert_q_to_psi!(S, G; a, par=par, workspace=workspace)

        if q_base !== nothing
            restore_prognostic_q!(S, q_base)
        end
    end

    compute_velocities!(S, G; plans, params=par, N2_profile=N2_profile,
                        workspace=workspace, dealias_mask=L,
                        velocity_workspace=timestep_workspace,
                        compute_w=compute_w)

    if par.passive_scalar || par.no_dispersion
        fill!(parent(S.A), zero(eltype(parent(S.A))))
        fill!(parent(S.C), zero(eltype(parent(S.C))))
    elseif par.ybj_plus
        invert_L⁺A_to_A!(S, G, par, a; workspace=workspace)
    else
        error("The exponential RK2 time stepper currently requires ybj_plus=true.")
    end

    return S
end

function _compute_etdrk2_rhs!(rhsq, rhsB, S::State, G::Grid, par::QGParams, plans;
                              a, dealias_mask=nothing, workspace=nothing,
                              N2_profile=nothing, timestep_workspace=nothing)
    par.ybj_plus || error("The exponential RK2 time stepper currently requires ybj_plus=true.")

    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask
    _update_diagnostics!(S, G, par, plans, a, L; workspace=workspace,
                         N2_profile=N2_profile, timestep_workspace=timestep_workspace,
                         compute_w=false, use_wave_feedback=true)

    if timestep_workspace === nothing
        nqk = similar(S.q)
        dqk = similar(S.q)
        nL⁺Ak = similar(S.L⁺A)
        rL⁺Ak = similar(S.L⁺A)
    else
        nqk = timestep_workspace.nqk
        dqk = timestep_workspace.dqk
        nL⁺Ak = timestep_workspace.nL⁺Ak
        rL⁺Ak = timestep_workspace.rL⁺Ak
    end

    nonlinear_workspace = timestep_workspace === nothing ? nothing : timestep_workspace.nonlinear
    convol_waqg_q!(nqk, S.u, S.v, S.q, G, plans; Lmask=L, workspace=nonlinear_workspace)
    convol_waqg_L⁺A!(nL⁺Ak, S.u, S.v, S.L⁺A, G, plans; Lmask=L, workspace=nonlinear_workspace)
    refraction_waqg_L⁺A!(rL⁺Ak, S.L⁺A, S.psi, G, plans; Lmask=L, workspace=nonlinear_workspace)
    dissipation_q_nv!(dqk, S.q, par, G; workspace=workspace)

    if par.inviscid
        fill!(parent(dqk), zero(eltype(parent(dqk))))
    end

    if par.linear
        fill!(parent(nqk), zero(eltype(parent(nqk))))
        fill!(parent(nL⁺Ak), zero(eltype(parent(nL⁺Ak))))
    end

    if par.passive_scalar
        fill!(parent(rL⁺Ak), zero(eltype(parent(rL⁺Ak))))
    end

    rhsq_arr = parent(rhsq)
    rhsB_arr = parent(rhsB)
    nqk_arr = parent(nqk)
    dqk_arr = parent(dqk)
    nL⁺Ak_arr = parent(nL⁺Ak)
    rL⁺Ak_arr = parent(rL⁺Ak)
    A_arr = parent(S.A)

    @dealiased_wavenumber_loop S.q G L begin
        rhsq_arr[k, i, j] = par.fixed_flow ? zero(eltype(rhsq_arr)) :
                            -nqk_arr[k, i, j] + dqk_arr[k, i, j]

        αdisp = par.f₀ / 2
        rhsB_arr[k, i, j] = -nL⁺Ak_arr[k, i, j] +
                            im * αdisp * kₕ² * A_arr[k, i, j] -
                            0.5im * rL⁺Ak_arr[k, i, j]
    end begin
        rhsq_arr[k, i, j] = 0
        rhsB_arr[k, i, j] = 0
    end

    return rhsq, rhsB
end

"""
    exp_rk2_step!(Snp1, Sn, G, par, plans; a, dealias_mask=nothing,
                  workspace=nothing, N2_profile=nothing)

Advance one step with a second-order exponential Runge-Kutta method.

Horizontal hyperdiffusion is treated exactly with the same dimensional
separable integrating factor as QG-YBJp. All remaining tendencies are evaluated
explicitly with ETDRK2.
"""
function exp_rk2_step!(Snp1::State, Sn::State, G::Grid, par::QGParams, plans;
                       a, dealias_mask=nothing, workspace=nothing, N2_profile=nothing,
                       particle_tracker=nothing, current_time=nothing,
                       timestep_workspace=nothing)
    par.ybj_plus || error("exp_rk2_step! currently requires ybj_plus=true.")

    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    if timestep_workspace === nothing
        rhsq₀ = similar(Sn.q)
        rhsB₀ = similar(Sn.L⁺A)
        rhsq₁ = similar(Sn.q)
        rhsB₁ = similar(Sn.L⁺A)
        Sstage = copy_state(Sn)
    else
        rhsq₀ = timestep_workspace.rhsq₀
        rhsB₀ = timestep_workspace.rhsB₀
        rhsq₁ = timestep_workspace.rhsq₁
        rhsB₁ = timestep_workspace.rhsB₁
        Sstage = timestep_workspace.stage
    end

    _compute_etdrk2_rhs!(rhsq₀, rhsB₀, Sn, G, par, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile, timestep_workspace=timestep_workspace)

    qn_arr = parent(Sn.q)
    Bn_arr = parent(Sn.L⁺A)
    qstage_arr = parent(Sstage.q)
    Bstage_arr = parent(Sstage.L⁺A)
    rhsq₀_arr = parent(rhsq₀)
    rhsB₀_arr = parent(rhsB₀)

    @dealiased_wavenumber_loop Sn.q G L begin
        λq = int_factor(kₓ, kᵧ, par; waves=false)
        λB = int_factor(kₓ, kᵧ, par; waves=true)

        Eq, hφ1q, _ = _etd_coefficients(λq, par.dt)
        EB, hφ1B, _ = _etd_coefficients(λB, par.dt)

        qstage_arr[k, i, j] = par.fixed_flow ? qn_arr[k, i, j] :
                              Eq * qn_arr[k, i, j] + hφ1q * rhsq₀_arr[k, i, j]
        Bstage_arr[k, i, j] = EB * Bn_arr[k, i, j] + hφ1B * rhsB₀_arr[k, i, j]
    end begin
        qstage_arr[k, i, j] = 0
        Bstage_arr[k, i, j] = 0
    end

    _compute_etdrk2_rhs!(rhsq₁, rhsB₁, Sstage, G, par, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile, timestep_workspace=timestep_workspace)

    qnp1_arr = parent(Snp1.q)
    Bnp1_arr = parent(Snp1.L⁺A)
    rhsq₁_arr = parent(rhsq₁)
    rhsB₁_arr = parent(rhsB₁)

    @dealiased_wavenumber_loop Sn.q G L begin
        λq = int_factor(kₓ, kᵧ, par; waves=false)
        λB = int_factor(kₓ, kᵧ, par; waves=true)

        Eq, hφ1q, hφ2q = _etd_coefficients(λq, par.dt)
        EB, hφ1B, hφ2B = _etd_coefficients(λB, par.dt)

        qnp1_arr[k, i, j] = par.fixed_flow ? qn_arr[k, i, j] :
                            Eq * qn_arr[k, i, j] +
                            hφ1q * rhsq₀_arr[k, i, j] +
                            hφ2q * (rhsq₁_arr[k, i, j] - rhsq₀_arr[k, i, j])
        Bnp1_arr[k, i, j] = EB * Bn_arr[k, i, j] +
                            hφ1B * rhsB₀_arr[k, i, j] +
                            hφ2B * (rhsB₁_arr[k, i, j] - rhsB₀_arr[k, i, j])
    end begin
        qnp1_arr[k, i, j] = 0
        Bnp1_arr[k, i, j] = 0
    end

    _update_diagnostics!(Snp1, G, par, plans, a, L; workspace=workspace,
                         N2_profile=N2_profile, timestep_workspace=timestep_workspace,
                         use_wave_feedback=true)

    if particle_tracker !== nothing && current_time !== nothing
        advect_particles!(particle_tracker, Snp1, G, par.dt, current_time)
    end

    return Snp1
end
