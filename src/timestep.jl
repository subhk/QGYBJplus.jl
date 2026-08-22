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

"""Typed physical and numerical choices consumed by the ETD-RK2 kernels."""
struct ETDModelOptions{F, B, W, D, N, P, C, T}
    f::T
    flow::F
    feedback::B
    formulation::W
    dissipation::D
    dynamics::N
    dispersion::P
    closure::C
    vertical_diffusivity::T
end

function ETDModelOptions(physics, numerics)
    f = float(physics.coriolis.f)
    return ETDModelOptions(
        f,
        physics.flow,
        physics.feedback,
        physics.formulation,
        numerics.dissipation,
        numerics.dynamics,
        numerics.dispersion,
        numerics.closure,
        typeof(f)(numerics.vertical_diffusion.coefficient),
    )
end

_fixed_flow(options::ETDModelOptions) = options.flow isa FixedFlow
_ybj_plus(options::ETDModelOptions) =
    options.formulation isa YBJPlus || options.formulation isa PassiveWave
_passive_wave(options::ETDModelOptions) = options.formulation isa PassiveWave
_inviscid(options::ETDModelOptions) = options.dissipation isa Inviscid
_linear(options::ETDModelOptions) = options.dynamics isa LinearDynamics
_no_dispersion(options::ETDModelOptions) = options.dispersion isa NoDispersion
_wave_feedback_enabled(options::ETDModelOptions) =
    !_fixed_flow(options) && options.feedback isa WaveMeanFeedback

"""Second-order exponential Runge-Kutta timestepper."""
mutable struct ExponentialRungeKutta2{T}
    Δt::T
    workspace::Any
end

function ExponentialRungeKutta2(; Δt::Real)
    value = float(Δt)
    isfinite(value) || throw(ArgumentError("Δt must be finite"))
    value > 0 || throw(ArgumentError("Δt must be positive"))
    return ExponentialRungeKutta2{typeof(value)}(value, nothing)
end

"""
Split a complex physical wave envelope into the separate spectra of its real
and imaginary components.

Taking `real` and `imag` of Fourier coefficients is not a valid field split:
in general `FFT(real(B)) != real(FFT(B))`. This helper therefore transforms to
physical space, separates the components, and transforms both back.
"""
function split_B_to_real_imag!(BRk, BIk, B, plans;
                               Bphysical=nothing,
                               component_physical=nothing)
    Bphysical = Bphysical === nothing ?
        allocate_fft_backward_dst(B, plans) : Bphysical
    component_physical = component_physical === nothing ?
        similar(Bphysical) : component_physical
    Bphysical === component_physical &&
        throw(ArgumentError("wave-component split requires distinct physical buffers"))

    fft_backward!(Bphysical, B, plans)
    Bphysical_arr = parent(Bphysical)
    component_arr = parent(component_physical)
    @inbounds for index in eachindex(Bphysical_arr, component_arr)
        value = Bphysical_arr[index]
        component_arr[index] = complex(real(value), 0)
        Bphysical_arr[index] = complex(imag(value), 0)
    end
    fft_forward!(BRk, component_physical, plans)
    fft_forward!(BIk, Bphysical, plans)
    return BRk, BIk
end

"""Combine separate real/imaginary-component spectra into `FFT(BR + i BI)`."""
function combine_real_imag_to_B!(B, BRk, BIk)
    B_arr = parent(B)
    BRk_arr = parent(BRk)
    BIk_arr = parent(BIk)

    @local_spectral_loop B begin
        B_arr[k, i, j] = BRk_arr[k, i, j] + im * BIk_arr[k, i, j]
    end
    return B
end

"""
    replace_q_with_wave_feedback_rhs!(S, G, options, plans, L; kwargs...)

Temporarily replace prognostic PV by the inversion right-hand side
`q_effective = q - q_wave`. The returned copy must be restored after the
streamfunction inversion so wave feedback is not accumulated in prognostic q.
"""
function replace_q_with_wave_feedback_rhs!(S::ModelFields, G::RuntimeGeometry,
                                           options::ETDModelOptions, plans, L;
                                           q_base=nothing, qwk=nothing)
    q_base = q_base === nothing ? copy(S.q) : q_base
    parent(q_base) .= parent(S.q)

    qwk = qwk === nothing ? similar(S.q) : qwk
    compute_qw_complex!(qwk, S.B, G, plans; f=options.f, Lmask=L)

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

restore_prognostic_q!(S::ModelFields, q_base) = (parent(S.q) .= parent(q_base); S)

"""
    ExponentialRungeKutta2Workspace(fields)

Reusable stage and tendency storage for model-owned [`step!`](@ref). Allocate
once per simulation to avoid recreating the Runge-Kutta state and spectral
tendency arrays on every step.
"""
mutable struct ExponentialRungeKutta2Workspace{S,A}
    next::S
    stage::S
    rhsq0::A
    rhsB0::A
    rhsq1::A
    rhsB1::A
    nqk::A
    dqk::A
    nBk::A
    rBk::A
    q_base::A
    qwk::A
end

function ExponentialRungeKutta2Workspace(S::ModelFields, plans=nothing; G=nothing)
    return ExponentialRungeKutta2Workspace(
        copy_fields(S),
        copy_fields(S),
        similar(S.q), similar(S.B), similar(S.q), similar(S.B),
        similar(S.q), similar(S.q), similar(S.B), similar(S.B),
        similar(S.q), similar(S.q),
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

function _diagnose_flow!(S::ModelFields, G::RuntimeGeometry,
                         options::ETDModelOptions, plans, a, L;
                         workspace=nothing, N2_profile=nothing,
                         timestep_workspace=nothing, compute_w=false,
                         use_wave_feedback=true)
    if !_fixed_flow(options)
        q_base = nothing
        if use_wave_feedback && _wave_feedback_enabled(options)
            q_base = replace_q_with_wave_feedback_rhs!(
                S, G, options, plans, L;
                q_base=timestep_workspace === nothing ? nothing : timestep_workspace.q_base,
                qwk=timestep_workspace === nothing ? nothing : timestep_workspace.qwk,
            )
        end

        invert_q_to_psi!(S, G; a, workspace)
        q_base === nothing || restore_prognostic_q!(S, q_base)
    end

    N2 = N2_profile === nothing ? 1.0 : first(N2_profile)
    compute_velocities!(S, G; plans, f=options.f, N2, compute_w,
        N2_profile, workspace, dealias_mask=L)
    return S
end

function _etdrk2_arrays(S::ModelFields, timestep_workspace)
    if timestep_workspace === nothing
        return (
            nqk=similar(S.q), dqk=similar(S.q),
            nBk=similar(S.B), rBk=similar(S.B),
        )
    end
    return (
        nqk=timestep_workspace.nqk, dqk=timestep_workspace.dqk,
        nBk=timestep_workspace.nBk, rBk=timestep_workspace.rBk,
    )
end

function _compute_etdrk2_rhs!(rhsq, rhsB, S::ModelFields, G::RuntimeGeometry,
                              options::ETDModelOptions, plans;
                              a, dealias_mask=nothing, workspace=nothing,
                              N2_profile=nothing,
                              timestep_workspace=nothing)
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask
    _ybj_plus(options) || sumB!(S.B, G; Lmask=L, workspace=workspace)
    _diagnose_flow!(S, G, options, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    timestep_workspace=timestep_workspace,
                    compute_w=false, use_wave_feedback=true)

    arrays = _etdrk2_arrays(S, timestep_workspace)
    nqk, dqk = arrays.nqk, arrays.dqk
    nBk, rBk = arrays.nBk, arrays.rBk

    # B is a complex physical field represented by its complex Fourier
    # transform. Both YBJ formulations therefore use complex advection and
    # refraction kernels; only the B -> A diagnostic relation differs.
    convol_waqg_q!(nqk, S.u, S.v, S.q, G, plans; Lmask=L)
    convol_waqg_B!(nBk, S.u, S.v, S.B, G, plans; Lmask=L)
    refraction_waqg_B!(rBk, S.B, S.psi, G, plans; Lmask=L)

    if _linear(options)
        fill!(parent(nqk), zero(eltype(parent(nqk))))
        fill!(parent(nBk), zero(eltype(parent(nBk))))
    end
    _passive_wave(options) &&
        fill!(parent(rBk), zero(eltype(parent(rBk))))

    if _passive_wave(options) || _no_dispersion(options)
        fill!(parent(S.A), zero(eltype(parent(S.A))))
        fill!(parent(S.C), zero(eltype(parent(S.C))))
    elseif _ybj_plus(options)
        invert_B_to_A!(S, G, a; workspace)
    else
        sigma = compute_sigma(options.f, G, nBk, rBk;
                              Lmask=L, workspace)
        compute_A!(S.A, S.C, S.B, sigma, G;
                   f=options.f, Lmask=L, workspace,
                   N2_profile=N2_profile)
    end

    dissipation_q_nv!(dqk, S.q, options.vertical_diffusivity, G;
        workspace=workspace)

    _inviscid(options) && fill!(parent(dqk), zero(eltype(parent(dqk))))
    rhsq_arr = parent(rhsq)
    rhsB_arr = parent(rhsB)
    nqk_arr = parent(nqk)
    dqk_arr = parent(dqk)
    nBk_arr = parent(nBk)
    rBk_arr = parent(rBk)
    A_arr = parent(S.A)
    alpha_disp = options.f / 2

    @dealiased_wavenumber_loop S.q G L begin
        rhsq_arr[k, i, j] = _fixed_flow(options) ? zero(eltype(rhsq_arr)) :
                                -nqk_arr[k, i, j] + dqk_arr[k, i, j]
        rhsB_arr[k, i, j] = -nBk_arr[k, i, j] +
                             im * alpha_disp * kₕ² * A_arr[k, i, j] -
                             0.5im * rBk_arr[k, i, j]
    end begin
        rhsq_arr[k, i, j] = 0
        rhsB_arr[k, i, j] = 0
    end

    return rhsq, rhsB
end

function _finalize_etdrk2_state!(S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing, N2_profile=nothing,
                                 timestep_workspace=nothing)
    _ybj_plus(options) || sumB!(S.B, G; Lmask=L, workspace=workspace)
    _diagnose_flow!(S, G, options, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    timestep_workspace=timestep_workspace,
                    compute_w=true, use_wave_feedback=true)

    if _passive_wave(options) || _no_dispersion(options)
        fill!(parent(S.A), zero(eltype(parent(S.A))))
        fill!(parent(S.C), zero(eltype(parent(S.C))))
    elseif _ybj_plus(options)
        invert_B_to_A!(S, G, a; workspace)
    else
        arrays = _etdrk2_arrays(S, timestep_workspace)
        if _linear(options)
            fill!(parent(arrays.nBk), zero(eltype(parent(arrays.nBk))))
        else
            convol_waqg_B!(arrays.nBk, S.u, S.v, S.B,
                           G, plans; Lmask=L)
        end
        refraction_waqg_B!(arrays.rBk, S.B, S.psi,
                           G, plans; Lmask=L)
        sigma = compute_sigma(options.f, G, arrays.nBk, arrays.rBk;
                              Lmask=L, workspace)
        compute_A!(S.A, S.C, S.B, sigma, G;
                   f=options.f, Lmask=L, workspace,
                   N2_profile=N2_profile)
    end
    return S
end

"""Internal ETD-RK2 kernel over explicit model components."""
function _advance_etdrk2!(Snp1::ModelFields, Sn::ModelFields, G::RuntimeGeometry,
                          options::ETDModelOptions, plans;
                          Δt::Real, a, dealias_mask=nothing, workspace=nothing,
                          N2_profile=nothing,
                          particle_tracker=nothing, particle_context=nothing,
                          current_time=nothing, timestep_workspace=nothing)
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask

    if timestep_workspace === nothing
        rhsq0, rhsB0 = similar(Sn.q), similar(Sn.B)
        rhsq1, rhsB1 = similar(Sn.q), similar(Sn.B)
        Sstage = copy_fields(Sn)
    else
        rhsq0, rhsB0 = timestep_workspace.rhsq0, timestep_workspace.rhsB0
        rhsq1, rhsB1 = timestep_workspace.rhsq1, timestep_workspace.rhsB1
        Sstage = timestep_workspace.stage
    end

    _compute_etdrk2_rhs!(rhsq0, rhsB0, Sn, G, options, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         timestep_workspace=timestep_workspace)

    qn_arr, Bn_arr = parent(Sn.q), parent(Sn.B)
    qstage_arr, Bstage_arr = parent(Sstage.q), parent(Sstage.B)
    rhsq0_arr, rhsB0_arr = parent(rhsq0), parent(rhsB0)

    @dealiased_wavenumber_loop Sn.q G L begin
        lambda_q = int_factor(kₓ, kᵧ, Δt, options.closure;
            waves=false, inviscid=_inviscid(options))
        lambda_B = int_factor(kₓ, kᵧ, Δt, options.closure;
            waves=true, inviscid=_inviscid(options))
        Eq, hphi1q, _ = _etd_coefficients(lambda_q, Δt)
        EB, hphi1B, _ = _etd_coefficients(lambda_B, Δt)

        qstage_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
                                  Eq * qn_arr[k, i, j] + hphi1q * rhsq0_arr[k, i, j]
        Bstage_arr[k, i, j] = EB * Bn_arr[k, i, j] + hphi1B * rhsB0_arr[k, i, j]
    end begin
        qstage_arr[k, i, j] = 0
        Bstage_arr[k, i, j] = 0
    end

    _compute_etdrk2_rhs!(rhsq1, rhsB1, Sstage, G, options, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         timestep_workspace=timestep_workspace)

    qnp1_arr, Bnp1_arr = parent(Snp1.q), parent(Snp1.B)
    rhsq1_arr, rhsB1_arr = parent(rhsq1), parent(rhsB1)

    @dealiased_wavenumber_loop Sn.q G L begin
        lambda_q = int_factor(kₓ, kᵧ, Δt, options.closure;
            waves=false, inviscid=_inviscid(options))
        lambda_B = int_factor(kₓ, kᵧ, Δt, options.closure;
            waves=true, inviscid=_inviscid(options))
        Eq, hphi1q, hphi2q = _etd_coefficients(lambda_q, Δt)
        EB, hphi1B, hphi2B = _etd_coefficients(lambda_B, Δt)

        qnp1_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
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

    _finalize_etdrk2_state!(Snp1, G, options, plans, a, L;
                            workspace=workspace, N2_profile=N2_profile,
                            timestep_workspace=timestep_workspace)

    if particle_tracker !== nothing
        advect_particles!(particle_tracker, Snp1, G, Δt, current_time;
                          params=particle_context, N2_profile=N2_profile)
    end
    return Snp1
end
