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

"""
Second-order exponential Runge-Kutta timestepper.

`workspace` starts as `nothing` and is filled with an
[`ExponentialRungeKutta2Workspace`](@ref) on the first `step!`, which is when
the field shapes are first known. A mutable field cannot be narrowed after that
assignment, so it stays untyped; `step!` re-establishes the concrete type with
an `isa` check before entering the kernels, and the field is read once per step
rather than inside any loop.
"""
mutable struct ExponentialRungeKutta2{T}
    Δt::T
    workspace::Any
    workspace_owner::Any
end

function ExponentialRungeKutta2(; Δt::Real)
    value = float(Δt)
    isfinite(value) || throw(ArgumentError("Δt must be finite"))
    value > 0 || throw(ArgumentError("Δt must be positive"))
    return ExponentialRungeKutta2{typeof(value)}(value, nothing, nothing)
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
                                           q_base=nothing, qwk=nothing,
                                           workspace=nothing,
                                           project_nonlinear_state::Bool=true)
    q_base = q_base === nothing ? copy(S.q) : q_base
    parent(q_base) .= parent(S.q)

    qwk = qwk === nothing ? similar(S.q) : qwk
    compute_qw_complex!(qwk, S.B, G, plans; f=options.f, Lmask=L, workspace)

    q_arr = parent(S.q)
    q_base_arr = parent(q_base)
    qwk_arr = parent(qwk)
    @dealiased_spectral_loop S.q L begin
        q_arr[k, i, j] = q_base_arr[k, i, j] - qwk_arr[k, i, j]
    end begin
        # The two-thirds cutoff belongs to quadratic stage tendencies, not to
        # the public elliptic inversion. qʷ is already zero outside the mask.
        q_arr[k, i, j] = project_nonlinear_state && !_linear(options) ?
                         0 : q_base_arr[k, i, j]
    end
    return q_base
end

restore_prognostic_q!(S::ModelFields, q_base) = (parent(S.q) .= parent(q_base); S)

"""Temporarily project prognostic PV onto the nonlinear Galerkin disk."""
function replace_q_with_dealiased_rhs!(S::ModelFields, L; q_base=nothing)
    q_base = q_base === nothing ? copy(S.q) : q_base
    parent(q_base) .= parent(S.q)
    q_arr = parent(S.q)
    q_base_arr = parent(q_base)
    @dealiased_spectral_loop S.q L begin
        q_arr[k, i, j] = q_base_arr[k, i, j]
    end begin
        q_arr[k, i, j] = 0
    end
    return q_base
end

"""Temporarily project prescribed ψ before diagnosing nonlinear velocities."""
function replace_psi_with_dealiased_diagnostic!(S::ModelFields, L;
                                                 psi_base=nothing)
    psi_base = psi_base === nothing ? copy(S.psi) : psi_base
    parent(psi_base) .= parent(S.psi)
    psi_arr = parent(S.psi)
    psi_base_arr = parent(psi_base)
    @dealiased_spectral_loop S.psi L begin
        psi_arr[k, i, j] = psi_base_arr[k, i, j]
    end begin
        psi_arr[k, i, j] = 0
    end
    return psi_base
end

restore_prescribed_psi!(S::ModelFields, psi_base) =
    (parent(S.psi) .= parent(psi_base); S)

"""
    _invert_total_q_to_psi!(S, G, options, plans, a, L; kwargs...)

Invert the balanced part of the prognostic generalized PV. When wave feedback
is active, `S.q` stores `q_balanced + q_wave`, whereas the elliptic inversion
expects only `q_balanced`. The temporary `q - q_wave` replacement is restored
even if the inversion throws.
"""
function _invert_total_q_to_psi!(S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing,
                                 timestep_workspace=nothing,
                                 use_wave_feedback=true,
                                 project_nonlinear_state::Bool=false)
    if use_wave_feedback && _wave_feedback_enabled(options)
        q_base = replace_q_with_wave_feedback_rhs!(
            S, G, options, plans, L;
            q_base=timestep_workspace === nothing ? nothing : timestep_workspace.q_base,
            qwk=timestep_workspace === nothing ? nothing : timestep_workspace.qwk,
            workspace=workspace,
            project_nonlinear_state,
        )
        try
            invert_q_to_psi!(S, G; a, workspace)
        finally
            restore_prognostic_q!(S, q_base)
        end
    elseif project_nonlinear_state && !_linear(options)
        # The divergence-form convolution filters q before multiplying, so ψ
        # (and therefore u/v) must be diagnosed from the same projected state.
        # Otherwise an out-of-cutoff initial q mode can alias into retained
        # modes during the very first nonlinear RHS evaluation.
        q_base = replace_q_with_dealiased_rhs!(
            S, L;
            q_base=timestep_workspace === nothing ? nothing : timestep_workspace.q_base,
        )
        try
            invert_q_to_psi!(S, G; a, workspace)
        finally
            restore_prognostic_q!(S, q_base)
        end
    else
        invert_q_to_psi!(S, G; a, workspace)
    end
    return S
end

"""
Per-horizontal-mode ETD-RK2 integrating factors.

`exp`/`expm1` depend only on (kₓ, kᵧ, Δt), never on z, but the stage loops run
over (k, j, i). Evaluating them inline therefore repeated every transcendental
`nz` times per stage. The table is rebuilt once per step, so a changing `Δt` is
still honoured.
"""
struct ETDCoefficientTable{M<:AbstractMatrix}
    Eq::M
    hphi1q::M
    hphi2q::M
    EB::M
    hphi1B::M
    hphi2B::M
end

function ETDCoefficientTable(S::ModelFields)
    _, nx_local, ny_local = size(parent(S.q))
    build() = zeros(Float64, nx_local, ny_local)
    return ETDCoefficientTable(build(), build(), build(),
                               build(), build(), build())
end

"""Refresh `table` for the current `Δt` and closure."""
function fill_etd_table!(table::ETDCoefficientTable, S::ModelFields,
                         G::RuntimeGeometry, options::ETDModelOptions, Δt)
    _, nx_local, ny_local = size(parent(S.q))
    inviscid = _inviscid(options)
    @inbounds for j in 1:ny_local, i in 1:nx_local
        kₓ = G.kx[local_to_global(i, 2, S.q)]
        kᵧ = G.ky[local_to_global(j, 3, S.q)]
        lambda_q = int_factor(kₓ, kᵧ, Δt, options.closure;
                              waves=false, inviscid=inviscid)
        lambda_B = int_factor(kₓ, kᵧ, Δt, options.closure;
                              waves=true, inviscid=inviscid)
        table.Eq[i, j], table.hphi1q[i, j], table.hphi2q[i, j] =
            _etd_coefficients(lambda_q, Δt)
        table.EB[i, j], table.hphi1B[i, j], table.hphi2B[i, j] =
            _etd_coefficients(lambda_B, Δt)
    end
    return table
end

"""
    ExponentialRungeKutta2Workspace(fields)

Reusable stage and tendency storage for model-owned [`step!`](@ref). Allocate
once per simulation to avoid recreating the Runge-Kutta state and spectral
tendency arrays on every step.
"""
mutable struct ExponentialRungeKutta2Workspace{S,A,E}
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
    etd::E
end

# `plans` and `G` are accepted for call-site compatibility; the workspace is
# derived entirely from the shape of `S`.
function ExponentialRungeKutta2Workspace(S::ModelFields, plans=nothing; G=nothing)
    return ExponentialRungeKutta2Workspace(
        copy_fields(S),
        copy_fields(S),
        similar(S.q), similar(S.B), similar(S.q), similar(S.B),
        similar(S.q), similar(S.q), similar(S.B), similar(S.B),
        similar(S.q), similar(S.q),
        ETDCoefficientTable(S),
    )
end

@inline function _etdrk2_array_layout_matches(cached::AbstractArray,
                                               live::AbstractArray)
    return typeof(cached) === typeof(live) &&
           size(parent(cached)) == size(parent(live))
end

@inline function _etdrk2_array_layout_matches(cached::PencilArray,
                                               live::PencilArray)
    return typeof(cached) === typeof(live) &&
           size(parent(cached)) == size(parent(live)) &&
           pencil(cached) === pencil(live)
end

@inline function _etdrk2_field_layout_matches(cached::ModelFields,
                                              live::ModelFields)
    return _etdrk2_array_layout_matches(cached.q, live.q) &&
           _etdrk2_array_layout_matches(cached.B, live.B) &&
           _etdrk2_array_layout_matches(cached.psi, live.psi) &&
           _etdrk2_array_layout_matches(cached.A, live.A) &&
           _etdrk2_array_layout_matches(cached.C, live.C) &&
           _etdrk2_array_layout_matches(cached.u, live.u) &&
           _etdrk2_array_layout_matches(cached.v, live.v) &&
           _etdrk2_array_layout_matches(cached.w, live.w)
end

"""Whether a reusable ETD workspace belongs to the live field layout."""
function _etdrk2_workspace_matches(workspace::ExponentialRungeKutta2Workspace,
                                    fields::ModelFields)
    _etdrk2_field_layout_matches(workspace.next, fields) || return false
    _etdrk2_field_layout_matches(workspace.stage, fields) || return false

    for array in (workspace.rhsq0, workspace.rhsq1, workspace.nqk,
                  workspace.dqk, workspace.q_base, workspace.qwk)
        _etdrk2_array_layout_matches(array, fields.q) || return false
    end
    for array in (workspace.rhsB0, workspace.rhsB1,
                  workspace.nBk, workspace.rBk)
        _etdrk2_array_layout_matches(array, fields.B) || return false
    end

    _, nx_local, ny_local = size(parent(fields.q))
    coefficient_size = (nx_local, ny_local)
    table = workspace.etd
    return all(array -> size(array) == coefficient_size,
               (table.Eq, table.hphi1q, table.hphi2q,
                table.EB, table.hphi1B, table.hphi2B))
end

"""Return `exp(-x)`, `h*phi_1(-x)`, and `h*phi_2(-x)` accurately."""
function _etd_coefficients(x, h)
    E = exp(-x)
    if abs(x) <= 1e-3
        x2 = x * x
        hphi1 = h * (1 - x / 2 + x2 / 6 - x2 * x / 24 + x2 * x2 / 120)
        hphi2 = h * (1 / 2 - x / 6 + x2 / 24 - x2 * x / 120 + x2 * x2 / 720)
        return E, hphi1, hphi2
    end

    expm1_neg = expm1(-x)
    hphi1 = h * (-expm1_neg) / x
    # This algebraically equivalent form avoids squaring x. In particular it
    # retains the h/x asymptote for very large finite x and evaluates to zero,
    # rather than NaN, in the infinitely stiff x=Inf limit.
    hphi2 = h * (1 + expm1_neg / x) / x
    return E, hphi1, hphi2
end

function _diagnose_flow!(S::ModelFields, G::RuntimeGeometry,
                         options::ETDModelOptions, plans, a, L;
                         workspace=nothing, N2_profile=nothing,
                         timestep_workspace=nothing, compute_w=false,
                         use_wave_feedback=true)
    psi_base = nothing
    if _fixed_flow(options) && !_linear(options)
        psi_base = replace_psi_with_dealiased_diagnostic!(
            S, L;
            psi_base=timestep_workspace === nothing ?
                     nothing : timestep_workspace.q_base,
        )
    elseif !_fixed_flow(options)
        _invert_total_q_to_psi!(S, G, options, plans, a, L;
            workspace, timestep_workspace, use_wave_feedback,
            project_nonlinear_state=true)
    end

    N2 = N2_profile === nothing ? 1.0 : first(N2_profile)
    try
        compute_velocities!(S, G; plans, f=options.f, N2, compute_w,
            N2_profile, workspace, dealias_mask=L)
    finally
        psi_base === nothing || restore_prescribed_psi!(S, psi_base)
    end
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
                              N2_face_profile=nothing,
                              timestep_workspace=nothing)
    L = isnothing(dealias_mask) ? trues(G.nx, G.ny) : dealias_mask
    # The cutoff is required by nonlinear products, but linear normal-YBJ
    # recovery is a one-column operation and is valid at every resolved mode.
    normal_recovery_mask = _linear(options) ? nothing : L
    _ybj_plus(options) ||
        sumB!(S.B, G; Lmask=normal_recovery_mask, workspace=workspace)
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
    convol_waqg_q!(nqk, S.u, S.v, S.q, G, plans; Lmask=L, workspace)
    convol_waqg_B!(nBk, S.u, S.v, S.B, G, plans; Lmask=L, workspace)
    refraction_waqg_B!(rBk, S.B, S.psi, G, plans; Lmask=L, workspace)

    if _linear(options)
        fill!(parent(nqk), zero(eltype(parent(nqk))))
        fill!(parent(nBk), zero(eltype(parent(nBk))))
    end
    _passive_wave(options) &&
        fill!(parent(rBk), zero(eltype(parent(rBk))))

    if _passive_wave(options)
        fill!(parent(S.A), zero(eltype(parent(S.A))))
        fill!(parent(S.C), zero(eltype(parent(S.C))))
    elseif !_no_dispersion(options)
        if _ybj_plus(options)
            invert_B_to_A!(S, G, a; workspace)
        else
            sigma = compute_sigma(options.f, G, nBk, rBk;
                                  Lmask=L, workspace)
            normal_N² = N2_face_profile === nothing ?
                        N2_profile : N2_face_profile
            compute_A!(S.A, S.C, S.B, sigma, G;
                       f=options.f, Lmask=normal_recovery_mask, workspace,
                       N2_profile=normal_N²)
        end
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
        dispersion = _no_dispersion(options) ?
                     zero(eltype(rhsB_arr)) :
                     im * alpha_disp * kₕ² * A_arr[k, i, j]
        rhsq_arr[k, i, j] = _fixed_flow(options) ? zero(eltype(rhsq_arr)) :
                                -nqk_arr[k, i, j] + dqk_arr[k, i, j]
        rhsB_arr[k, i, j] = -nBk_arr[k, i, j] +
                             dispersion -
                             0.5im * rBk_arr[k, i, j]
    end begin
        # Outside the nonlinear cutoff, a linear-QG mode still receives its
        # explicit vertical-diffusion tendency.  Nonlinear runs project these
        # modes out to keep every quadratic product alias-safe.
        rhsq_arr[k, i, j] = (!_fixed_flow(options) && _linear(options)) ?
                                dqk_arr[k, i, j] : 0
        # Linear wave dynamics may carry resolved modes outside the cutoff;
        # retain their pointwise dispersion tendency. Advection/refraction
        # products remain projected by their own kernels.
        rhsB_arr[k, i, j] = if _linear(options) && !_no_dispersion(options)
            im * alpha_disp * kₕ² * A_arr[k, i, j]
        else
            zero(eltype(rhsB_arr))
        end
    end

    return rhsq, rhsB
end

function _finalize_etdrk2_state!(S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing, N2_profile=nothing,
                                 N2_face_profile=nothing,
                                 timestep_workspace=nothing)
    normal_recovery_mask = _linear(options) ? nothing : L
    _ybj_plus(options) ||
        sumB!(S.B, G; Lmask=normal_recovery_mask, workspace=workspace)
    restore_full_fixed_diagnostics = _fixed_flow(options) && !_linear(options)
    _diagnose_flow!(S, G, options, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    timestep_workspace=timestep_workspace,
                    compute_w=!restore_full_fixed_diagnostics,
                    use_wave_feedback=true)

    if _passive_wave(options)
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
                           G, plans; Lmask=L, workspace)
        end
        refraction_waqg_B!(arrays.rBk, S.B, S.psi,
                           G, plans; Lmask=L, workspace)
        sigma = compute_sigma(options.f, G, arrays.nBk, arrays.rBk;
                              Lmask=L, workspace)
        normal_N² = N2_face_profile === nothing ?
                    N2_profile : N2_face_profile
        compute_A!(S.A, S.C, S.B, sigma, G;
                   f=options.f, Lmask=normal_recovery_mask, workspace,
                   N2_profile=normal_N²)
    end

    # Nonlinear fixed-flow stages use a projected prescribed ψ so both
    # operands of each quadratic product are alias-safe. The public final
    # diagnostics, however, must correspond to the restored prescribed ψ.
    if restore_full_fixed_diagnostics
        N2 = N2_profile === nothing ? 1.0 : first(N2_profile)
        compute_velocities!(S, G; plans, f=options.f, N2, compute_w=true,
            N2_profile, workspace, dealias_mask=L)
    end
    return S
end

"""Internal ETD-RK2 kernel over explicit model components."""
function _advance_etdrk2!(Snp1::ModelFields, Sn::ModelFields, G::RuntimeGeometry,
                          options::ETDModelOptions, plans;
                          Δt::Real, a, dealias_mask=nothing, workspace=nothing,
                          N2_profile=nothing, N2_face_profile=nothing,
                          timestep_workspace=nothing)
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

    etd = timestep_workspace === nothing ?
          ETDCoefficientTable(Sn) : timestep_workspace.etd
    fill_etd_table!(etd, Sn, G, options, Δt)
    Eq_table, hphi1q_table, hphi2q_table = etd.Eq, etd.hphi1q, etd.hphi2q
    EB_table, hphi1B_table, hphi2B_table = etd.EB, etd.hphi1B, etd.hphi2B

    _compute_etdrk2_rhs!(rhsq0, rhsB0, Sn, G, options, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         N2_face_profile=N2_face_profile,
                         timestep_workspace=timestep_workspace)

    qn_arr, Bn_arr = parent(Sn.q), parent(Sn.B)
    qstage_arr, Bstage_arr = parent(Sstage.q), parent(Sstage.B)
    rhsq0_arr, rhsB0_arr = parent(rhsq0), parent(rhsB0)

    @dealiased_wavenumber_loop Sn.q G L begin
        Eq, hphi1q = Eq_table[i, j], hphi1q_table[i, j]
        EB, hphi1B = EB_table[i, j], hphi1B_table[i, j]

        qstage_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
                                  Eq * qn_arr[k, i, j] + hphi1q * rhsq0_arr[k, i, j]
        Bstage_arr[k, i, j] = EB * Bn_arr[k, i, j] + hphi1B * rhsB0_arr[k, i, j]
    end begin
        qstage_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
                                  _linear(options) ?
                                  Eq_table[i, j] * qn_arr[k, i, j] +
                                  hphi1q_table[i, j] * rhsq0_arr[k, i, j] : 0
        Bstage_arr[k, i, j] = _linear(options) ?
                                  EB_table[i, j] * Bn_arr[k, i, j] +
                                  hphi1B_table[i, j] * rhsB0_arr[k, i, j] : 0
    end

    # FixedFlow diagnoses velocity from its prescribed ψ instead of inverting
    # q. Cached stage storage must therefore receive the current prescribed
    # field explicitly whenever a timestepper is reused after `set!`/restore.
    _fixed_flow(options) && copyto!(Sstage.psi, Sn.psi)

    _compute_etdrk2_rhs!(rhsq1, rhsB1, Sstage, G, options, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         N2_face_profile=N2_face_profile,
                         timestep_workspace=timestep_workspace)

    qnp1_arr, Bnp1_arr = parent(Snp1.q), parent(Snp1.B)
    rhsq1_arr, rhsB1_arr = parent(rhsq1), parent(rhsB1)

    @dealiased_wavenumber_loop Sn.q G L begin
        Eq, hphi1q, hphi2q = Eq_table[i, j], hphi1q_table[i, j], hphi2q_table[i, j]
        EB, hphi1B, hphi2B = EB_table[i, j], hphi1B_table[i, j], hphi2B_table[i, j]

        qnp1_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
                                Eq * qn_arr[k, i, j] +
                                hphi1q * rhsq0_arr[k, i, j] +
                                hphi2q * (rhsq1_arr[k, i, j] - rhsq0_arr[k, i, j])
        Bnp1_arr[k, i, j] = EB * Bn_arr[k, i, j] +
                            hphi1B * rhsB0_arr[k, i, j] +
                            hphi2B * (rhsB1_arr[k, i, j] - rhsB0_arr[k, i, j])
    end begin
        qnp1_arr[k, i, j] = _fixed_flow(options) ? qn_arr[k, i, j] :
                                _linear(options) ?
                                Eq_table[i, j] * qn_arr[k, i, j] +
                                hphi1q_table[i, j] * rhsq0_arr[k, i, j] +
                                hphi2q_table[i, j] *
                                (rhsq1_arr[k, i, j] - rhsq0_arr[k, i, j]) : 0
        Bnp1_arr[k, i, j] = _linear(options) ?
                                EB_table[i, j] * Bn_arr[k, i, j] +
                                hphi1B_table[i, j] * rhsB0_arr[k, i, j] +
                                hphi2B_table[i, j] *
                                (rhsB1_arr[k, i, j] - rhsB0_arr[k, i, j]) : 0
    end

    _fixed_flow(options) && copyto!(Snp1.psi, Sn.psi)

    _finalize_etdrk2_state!(Snp1, G, options, plans, a, L;
                            workspace=workspace, N2_profile=N2_profile,
                            N2_face_profile=N2_face_profile,
                            timestep_workspace=timestep_workspace)
    return Snp1
end
