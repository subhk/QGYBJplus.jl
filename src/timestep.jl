#=
================================================================================
                    timestep.jl - ETD-RK2 Time Integration
================================================================================

The production time stepper is a second-order exponential Runge-Kutta method
(ETD-RK2). Horizontal hyperdiffusion is integrated exactly; advection,
refraction, dispersion, and vertical diffusion are evaluated explicitly at two
Runge-Kutta stages.

For a semilinear equation u_t = L*u + N(u), one step is

    a       = exp(hL)u_n + h*phi_1(hL)N(u_n)
    u_{n+1} = a + h*phi_2(hL)(N(a) - N(u_n))

where L is the diagonal horizontal hyperdiffusion operator. The phi functions
are evaluated with cancellation-safe series near zero.
================================================================================
=#

"""Split the complex wave envelope into real-valued spectral components."""
function split_B_to_real_imag!(BRk, BIk, B)
    B_arr = parent(B)
    BRk_arr = parent(BRk)
    BIk_arr = parent(BIk)

    @local_spectral_loop B begin
        BRk_arr[k, i, j] = Complex(real(B_arr[k, i, j]), 0)
        BIk_arr[k, i, j] = Complex(imag(B_arr[k, i, j]), 0)
    end
    return BRk, BIk
end

"""Combine real-valued spectral components into the complex wave envelope."""
function combine_real_imag_to_B!(B, BRk, BIk)
    B_arr = parent(B)
    BRk_arr = parent(BRk)
    BIk_arr = parent(BIk)

    @local_spectral_loop B begin
        B_arr[k, i, j] = complex(real(BRk_arr[k, i, j]), real(BIk_arr[k, i, j]))
    end
    return B
end

"""
    replace_q_with_wave_feedback_rhs!(S, G, par, plans, L; kwargs...)

Temporarily replace prognostic PV by the inversion right-hand side
`q_effective = q - q_wave`. The returned copy must be restored after the
streamfunction inversion so wave feedback is not accumulated in prognostic q.
"""
function replace_q_with_wave_feedback_rhs!(S::State, G::Grid, par::QGParams, plans, L;
                                           BRk=nothing, BIk=nothing,
                                           q_base=nothing, qwk=nothing)
    q_base = q_base === nothing ? copy(S.q) : q_base
    parent(q_base) .= parent(S.q)

    qwk = qwk === nothing ? similar(S.q) : qwk
    if par.ybj_plus
        compute_qw_complex!(qwk, S.B, par, G, plans; Lmask=L)
    else
        BRk = BRk === nothing ? similar(S.B) : BRk
        BIk = BIk === nothing ? similar(S.B) : BIk
        split_B_to_real_imag!(BRk, BIk, S.B)
        compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)
    end

    q_arr = parent(S.q)
    q_base_arr = parent(q_base)
    qwk_arr = parent(qwk)
    @dealiased_spectral_loop S.q L begin
        q_arr[k, i, j] = q_base_arr[k, i, j] - qwk_arr[k, i, j]
    end begin
        q_arr[k, i, j] = 0
    end
    return q_base
end

restore_prognostic_q!(S::State, q_base) = (parent(S.q) .= parent(q_base); S)

_wave_feedback_enabled(par::QGParams) =
    !par.fixed_flow && !par.no_feedback && !par.no_wave_feedback

"""
    ExpRK2Workspace(state)

Reusable stage and tendency storage for [`exp_rk2_step!`](@ref). Allocate once
per simulation to avoid recreating the Runge-Kutta state and spectral tendency
arrays on every step.
"""
struct ExpRK2Workspace{S,A}
    stage::S
    rhsq0::A
    rhsB0::A
    rhsq1::A
    rhsB1::A
    nqk::A
    dqk::A
    nBk::A
    rBk::A
    nBRk::A
    nBIk::A
    rBRk::A
    rBIk::A
    BRk::A
    BIk::A
    q_base::A
    qwk::A
end

function ExpRK2Workspace(S::State, plans=nothing; G=nothing)
    return ExpRK2Workspace(
        copy_state(S),
        similar(S.q), similar(S.B), similar(S.q), similar(S.B),
        similar(S.q), similar(S.q), similar(S.B), similar(S.B),
        similar(S.B), similar(S.B), similar(S.B), similar(S.B),
        similar(S.B), similar(S.B), similar(S.q), similar(S.q),
    )
end

"""Return `exp(-x)`, `h*phi_1(-x)`, and `h*phi_2(-x)` accurately."""
function _etd_coefficients(x, h)
    E = exp(-x)
    if abs(x) < 1e-6
        x2 = x * x
        hphi1 = h * (1 - x / 2 + x2 / 6 - x2 * x / 24 + x2 * x2 / 120)
        hphi2 = h * (1 / 2 - x / 6 + x2 / 24 - x2 * x / 120 + x2 * x2 / 720)
        return E, hphi1, hphi2
    end

    expm1_neg = expm1(-x)
    hphi1 = h * (-expm1_neg) / x
    hphi2 = h * (expm1_neg + x) / x^2
    return E, hphi1, hphi2
end

function _diagnose_flow!(S::State, G::Grid, par::QGParams, plans, a, L;
                         workspace=nothing, N2_profile=nothing,
                         timestep_workspace=nothing, compute_w=false,
                         use_wave_feedback=true)
    if !par.fixed_flow
        q_base = nothing
        if use_wave_feedback && _wave_feedback_enabled(par)
            q_base = replace_q_with_wave_feedback_rhs!(
                S, G, par, plans, L;
                BRk=timestep_workspace === nothing ? nothing : timestep_workspace.BRk,
                BIk=timestep_workspace === nothing ? nothing : timestep_workspace.BIk,
                q_base=timestep_workspace === nothing ? nothing : timestep_workspace.q_base,
                qwk=timestep_workspace === nothing ? nothing : timestep_workspace.qwk,
            )
        end

        invert_q_to_psi!(S, G; a=a, par=par, workspace=workspace)
        q_base === nothing || restore_prognostic_q!(S, q_base)
    end

    compute_velocities!(S, G; plans=plans, params=par, compute_w=compute_w,
                        N2_profile=N2_profile, workspace=workspace,
                        dealias_mask=L)
    return S
end

function _etdrk2_arrays(S::State, timestep_workspace)
    if timestep_workspace === nothing
        return (
            nqk=similar(S.q), dqk=similar(S.q),
            nBk=similar(S.B), rBk=similar(S.B),
            nBRk=similar(S.B), nBIk=similar(S.B),
            rBRk=similar(S.B), rBIk=similar(S.B),
            BRk=similar(S.B), BIk=similar(S.B),
        )
    end
    return (
        nqk=timestep_workspace.nqk, dqk=timestep_workspace.dqk,
        nBk=timestep_workspace.nBk, rBk=timestep_workspace.rBk,
        nBRk=timestep_workspace.nBRk, nBIk=timestep_workspace.nBIk,
        rBRk=timestep_workspace.rBRk, rBIk=timestep_workspace.rBIk,
        BRk=timestep_workspace.BRk, BIk=timestep_workspace.BIk,
    )
end

function _compute_etdrk2_rhs!(rhsq, rhsB, S::State, G::Grid,
                              par::QGParams, plans;
                              a, dealias_mask=nothing, workspace=nothing,
                              N2_profile=nothing, timestep_workspace=nothing)
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask
    par.ybj_plus || sumB!(S.B, G; Lmask=L, workspace=workspace)
    _diagnose_flow!(S, G, par, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    timestep_workspace=timestep_workspace,
                    compute_w=false, use_wave_feedback=true)

    arrays = _etdrk2_arrays(S, timestep_workspace)
    nqk, dqk = arrays.nqk, arrays.dqk
    nBk, rBk = arrays.nBk, arrays.rBk
    nBRk, nBIk = arrays.nBRk, arrays.nBIk
    rBRk, rBIk = arrays.rBRk, arrays.rBIk
    BRk, BIk = arrays.BRk, arrays.BIk

    if par.ybj_plus
        if par.passive_scalar || par.no_dispersion
            fill!(parent(S.A), zero(eltype(parent(S.A))))
            fill!(parent(S.C), zero(eltype(parent(S.C))))
        else
            invert_B_to_A!(S, G, par, a; workspace=workspace)
        end

        convol_waqg_q!(nqk, S.u, S.v, S.q, G, plans; Lmask=L)
        convol_waqg_B!(nBk, S.u, S.v, S.B, G, plans; Lmask=L)
        refraction_waqg_B!(rBk, S.B, S.psi, G, plans; Lmask=L)
    else
        split_B_to_real_imag!(BRk, BIk, S.B)
        convol_waqg!(nqk, nBRk, nBIk, S.u, S.v, S.q, BRk, BIk,
                     G, plans; Lmask=L)
        refraction_waqg!(rBRk, rBIk, BRk, BIk, S.psi, G, plans; Lmask=L)

        if par.passive_scalar || par.no_dispersion
            fill!(parent(S.A), zero(eltype(parent(S.A))))
            fill!(parent(S.C), zero(eltype(parent(S.C))))
        else
            sigma = compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk;
                                  Lmask=L, N2_profile=N2_profile)
            compute_A!(S.A, S.C, BRk, BIk, sigma, par, G;
                       Lmask=L, N2_profile=N2_profile)
        end
    end

    dissipation_q_nv!(dqk, S.q, par, G; workspace=workspace)

    par.inviscid && fill!(parent(dqk), zero(eltype(parent(dqk))))
    if par.linear
        fill!(parent(nqk), zero(eltype(parent(nqk))))
        if par.ybj_plus
            fill!(parent(nBk), zero(eltype(parent(nBk))))
        else
            fill!(parent(nBRk), zero(eltype(parent(nBRk))))
            fill!(parent(nBIk), zero(eltype(parent(nBIk))))
        end
    end
    if par.passive_scalar
        if par.ybj_plus
            fill!(parent(rBk), zero(eltype(parent(rBk))))
        else
            fill!(parent(rBRk), zero(eltype(parent(rBRk))))
            fill!(parent(rBIk), zero(eltype(parent(rBIk))))
        end
    end

    rhsq_arr = parent(rhsq)
    rhsB_arr = parent(rhsB)
    nqk_arr = parent(nqk)
    dqk_arr = parent(dqk)
    A_arr = parent(S.A)
    alpha_disp = par.f₀ / 2

    if par.ybj_plus
        nBk_arr = parent(nBk)
        rBk_arr = parent(rBk)
        @dealiased_wavenumber_loop S.q G L begin
            rhsq_arr[k, i, j] = par.fixed_flow ? zero(eltype(rhsq_arr)) :
                                    -nqk_arr[k, i, j] + dqk_arr[k, i, j]
            rhsB_arr[k, i, j] = -nBk_arr[k, i, j] +
                                 im * alpha_disp * kₕ² * A_arr[k, i, j] -
                                 0.5im * rBk_arr[k, i, j]
        end begin
            rhsq_arr[k, i, j] = 0
            rhsB_arr[k, i, j] = 0
        end
    else
        nBRk_arr, nBIk_arr = parent(nBRk), parent(nBIk)
        rBRk_arr, rBIk_arr = parent(rBRk), parent(rBIk)
        @dealiased_wavenumber_loop S.q G L begin
            rhsq_arr[k, i, j] = par.fixed_flow ? zero(eltype(rhsq_arr)) :
                                    -nqk_arr[k, i, j] + dqk_arr[k, i, j]
            rhsBR = -real(nBRk_arr[k, i, j]) -
                    alpha_disp * kₕ² * imag(A_arr[k, i, j]) -
                    0.5 * real(rBIk_arr[k, i, j])
            rhsBI = -real(nBIk_arr[k, i, j]) +
                    alpha_disp * kₕ² * real(A_arr[k, i, j]) +
                    0.5 * real(rBRk_arr[k, i, j])
            rhsB_arr[k, i, j] = complex(rhsBR, rhsBI)
        end begin
            rhsq_arr[k, i, j] = 0
            rhsB_arr[k, i, j] = 0
        end
    end

    return rhsq, rhsB
end

function _finalize_etdrk2_state!(S::State, G::Grid, par::QGParams, plans, a, L;
                                 workspace=nothing, N2_profile=nothing,
                                 timestep_workspace=nothing)
    par.ybj_plus || sumB!(S.B, G; Lmask=L, workspace=workspace)
    _diagnose_flow!(S, G, par, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    timestep_workspace=timestep_workspace,
                    compute_w=true, use_wave_feedback=true)

    if par.passive_scalar || par.no_dispersion
        fill!(parent(S.A), zero(eltype(parent(S.A))))
        fill!(parent(S.C), zero(eltype(parent(S.C))))
    elseif par.ybj_plus
        invert_B_to_A!(S, G, par, a; workspace=workspace)
    else
        arrays = _etdrk2_arrays(S, timestep_workspace)
        split_B_to_real_imag!(arrays.BRk, arrays.BIk, S.B)
        if par.linear
            fill!(parent(arrays.nBRk), zero(eltype(parent(arrays.nBRk))))
            fill!(parent(arrays.nBIk), zero(eltype(parent(arrays.nBIk))))
        else
            convol_waqg!(arrays.nqk, arrays.nBRk, arrays.nBIk,
                         S.u, S.v, S.q, arrays.BRk, arrays.BIk,
                         G, plans; Lmask=L)
        end
        refraction_waqg!(arrays.rBRk, arrays.rBIk,
                         arrays.BRk, arrays.BIk, S.psi,
                         G, plans; Lmask=L)
        sigma = compute_sigma(par, G, arrays.nBRk, arrays.nBIk,
                              arrays.rBRk, arrays.rBIk;
                              Lmask=L, N2_profile=N2_profile)
        compute_A!(S.A, S.C, arrays.BRk, arrays.BIk, sigma, par, G;
                   Lmask=L, N2_profile=N2_profile)
    end
    return S
end

"""
    exp_rk2_step!(Snp1, Sn, G, par, plans; a, kwargs...)

Advance `Sn` by one second-order exponential Runge-Kutta step and write the
result into `Snp1`. Horizontal hyperdiffusion is handled exactly through ETD
phi functions. The method supports both YBJ+ and normal-YBJ formulations.
"""
function exp_rk2_step!(Snp1::State, Sn::State, G::Grid, par::QGParams, plans;
                       a, dealias_mask=nothing, workspace=nothing,
                       N2_profile=nothing, particle_tracker=nothing,
                       current_time=nothing, timestep_workspace=nothing)
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    if timestep_workspace === nothing
        rhsq0, rhsB0 = similar(Sn.q), similar(Sn.B)
        rhsq1, rhsB1 = similar(Sn.q), similar(Sn.B)
        Sstage = copy_state(Sn)
    else
        rhsq0, rhsB0 = timestep_workspace.rhsq0, timestep_workspace.rhsB0
        rhsq1, rhsB1 = timestep_workspace.rhsq1, timestep_workspace.rhsB1
        Sstage = timestep_workspace.stage
    end

    _compute_etdrk2_rhs!(rhsq0, rhsB0, Sn, G, par, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         timestep_workspace=timestep_workspace)

    qn_arr, Bn_arr = parent(Sn.q), parent(Sn.B)
    qstage_arr, Bstage_arr = parent(Sstage.q), parent(Sstage.B)
    rhsq0_arr, rhsB0_arr = parent(rhsq0), parent(rhsB0)

    @dealiased_wavenumber_loop Sn.q G L begin
        lambda_q = int_factor(kₓ, kᵧ, par; waves=false)
        lambda_B = int_factor(kₓ, kᵧ, par; waves=true)
        Eq, hphi1q, _ = _etd_coefficients(lambda_q, par.dt)
        EB, hphi1B, _ = _etd_coefficients(lambda_B, par.dt)

        qstage_arr[k, i, j] = par.fixed_flow ? qn_arr[k, i, j] :
                                  Eq * qn_arr[k, i, j] + hphi1q * rhsq0_arr[k, i, j]
        Bstage_arr[k, i, j] = EB * Bn_arr[k, i, j] + hphi1B * rhsB0_arr[k, i, j]
    end begin
        qstage_arr[k, i, j] = 0
        Bstage_arr[k, i, j] = 0
    end

    _compute_etdrk2_rhs!(rhsq1, rhsB1, Sstage, G, par, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         timestep_workspace=timestep_workspace)

    qnp1_arr, Bnp1_arr = parent(Snp1.q), parent(Snp1.B)
    rhsq1_arr, rhsB1_arr = parent(rhsq1), parent(rhsB1)

    @dealiased_wavenumber_loop Sn.q G L begin
        lambda_q = int_factor(kₓ, kᵧ, par; waves=false)
        lambda_B = int_factor(kₓ, kᵧ, par; waves=true)
        Eq, hphi1q, hphi2q = _etd_coefficients(lambda_q, par.dt)
        EB, hphi1B, hphi2B = _etd_coefficients(lambda_B, par.dt)

        qnp1_arr[k, i, j] = par.fixed_flow ? qn_arr[k, i, j] :
                                Eq * qn_arr[k, i, j] +
                                hphi1q * rhsq0_arr[k, i, j] +
                                hphi2q * (rhsq1_arr[k, i, j] - rhsq0_arr[k, i, j])
        Bnp1_arr[k, i, j] = EB * Bn_arr[k, i, j] +
                            hphi1B * rhsB0_arr[k, i, j] +
                            hphi2B * (rhsB1_arr[k, i, j] - rhsB0_arr[k, i, j])
    end begin
        qnp1_arr[k, i, j] = 0
        Bnp1_arr[k, i, j] = 0
    end

    _finalize_etdrk2_state!(Snp1, G, par, plans, a, L;
                            workspace=workspace, N2_profile=N2_profile,
                            timestep_workspace=timestep_workspace)

    if particle_tracker !== nothing
        advect_particles!(particle_tracker, Snp1, G, par.dt, current_time;
                          params=par, N2_profile=N2_profile)
    end
    return Snp1
end
