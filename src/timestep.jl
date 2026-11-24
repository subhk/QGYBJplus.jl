#=
================================================================================
                    timestep.jl - ETD-RK2 Time Integration
================================================================================

The production time stepper is a second-order exponential Runge-Kutta method
(ETD-RK2). Horizontal hyperdiffusion is integrated exactly; advection,
refraction, dispersion, and vertical diffusion are evaluated explicitly at two
Runge-Kutta stages.

With wave feedback the prognostic q is the total PV q_g + qʷ, but the flow's
dissipation acts on the balanced PV q_g = q - qʷ only (qʷ belongs to the waves,
which have their own damping). The exact factor still damps q, and the
explicit part gains λqʷ, which the stage updates apply through the finite
weights hφ₁λ = 1 - E and hφ₂λ = 1 - hφ₁/h.

For a semilinear equation u_t = L*u + N(u), one step is

    a       = exp(hL)u_n + h*phi_1(hL)N(u_n)
    u_{n+1} = a + h*phi_2(hL)(N(a) - N(u_n))

where L is the diagonal horizontal hyperdiffusion operator. The phi functions
are evaluated with cancellation-safe series near zero.

The coefficient table, the right-hand-side assembly, and both stage updates
use KernelAbstractions kernels. ETD factors that depend on host-only scalar
types are calculated on the host and copied to the fields' backend.
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
# Passive waves are only advected, so they have no dispersion either.
_no_dispersion(options::ETDModelOptions) =
    options.dispersion isa NoDispersion || _passive_wave(options)
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

_real_component(value) = complex(real(value), zero(real(value)))
_imaginary_component(value) = complex(imag(value), zero(real(value)))
_combine_components(BR, BI) = BR + im * BI

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
    kernel_map!(_real_component, component_physical, Bphysical)
    kernel_map!(_imaginary_component, Bphysical, Bphysical)
    fft_forward!(BRk, component_physical, plans)
    fft_forward!(BIk, Bphysical, plans)
    return BRk, BIk
end

"""Combine separate real/imaginary-component spectra into `FFT(BR + i BI)`."""
function combine_real_imag_to_B!(B, BRk, BIk)
    kernel_map!(_combine_components, B, BRk, BIk)
    return B
end

"""Kernel rule for the stage-level dealiasing mask `L` (`nothing` keeps all)."""
_stage_mask(L, array) = kernel_mask(L, array, KeepAllModes())

@kernel function _balanced_pv_rhs_kernel!(rhs, @Const(q), mask, project_out)
    k, i, j = @index(Global, NTuple)
    # `rhs` holds qʷ on entry, which is already zero outside the mask.
    @inbounds if keep_mode(mask, i, j)
        rhs[k, i, j] = q[k, i, j] - rhs[k, i, j]
    else
        # The two-thirds cutoff belongs to quadratic stage tendencies, not to
        # the public elliptic inversion.
        rhs[k, i, j] = project_out ? zero(eltype(rhs)) : q[k, i, j]
    end
end

"""
    _balanced_pv_rhs!(rhs, S, G, options, plans, L; workspace=nothing,
                      project_nonlinear_state=true)

Write the right-hand side of the streamfunction inversion, the balanced PV
`q - qʷ`, into `rhs`. Prognostic `q` is total PV; it is only read. With
`project_nonlinear_state`, modes outside the nonlinear cutoff are zeroed.
"""
function _balanced_pv_rhs!(rhs, S::ModelFields, G::RuntimeGeometry,
                           options::ETDModelOptions, plans, L;
                           workspace=nothing, project_nonlinear_state::Bool=true)
    compute_qw_complex!(rhs, S.B, G, plans; f=options.f, Lmask=L, workspace)
    launch!(_balanced_pv_rhs_kernel!, rhs, size(rhs), rhs, S.q,
            _stage_mask(L, S.q), project_nonlinear_state && !_linear(options))
    return rhs
end

"""
    _invert_total_q_to_psi!(S, G, options, plans, a, L; kwargs...)

Invert the balanced part of the prognostic generalized PV. When wave feedback
is active, `S.q` stores `q_balanced + q_wave`, whereas the elliptic inversion
expects only `q_balanced`, so the inversion reads `q - q_wave` from a scratch
buffer and `S.q` is never modified.
"""
function _invert_total_q_to_psi!(S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing,
                                 timestep_workspace=nothing,
                                 use_wave_feedback=true,
                                 project_nonlinear_state::Bool=false,
                                 check_compatibility::Bool=false)
    _invert_total_q_to_psi!(S.psi, S, G, options, plans, a, L;
                            workspace, use_wave_feedback, project_nonlinear_state,
                            check_compatibility)
    return S
end

"""Balanced streamfunction of the PV in `S`, written into `psi`."""
function _invert_total_q_to_psi!(psi, S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing,
                                 use_wave_feedback=true,
                                 project_nonlinear_state::Bool=false,
                                 check_compatibility::Bool=false)
    @assert length(a) == G.nz "a must have length nz=$(G.nz)"
    invert(rhs) = Elliptic._invert_q_to_psi!(psi, rhs, G, a, workspace;
                                            check_compatibility)
    if use_wave_feedback && _wave_feedback_enabled(options)
        with_scratch(workspace) do
            rhs = scratch_like(workspace, S.q)
            _balanced_pv_rhs!(rhs, S, G, options, plans, L;
                              workspace, project_nonlinear_state)
            invert(rhs)
        end
    elseif project_nonlinear_state && !_linear(options)
        # The divergence-form convolution filters q before multiplying, so ψ
        # (and therefore u/v) must be diagnosed from the same projected state.
        # Otherwise an out-of-cutoff initial q mode can alias into retained
        # modes during the very first nonlinear RHS evaluation.
        with_scratch(workspace) do
            rhs = scratch_like(workspace, S.q)
            dealias!(rhs, S.q, _stage_mask(L, S.q))
            invert(rhs)
        end
    else
        invert(S.q)
    end
    return psi
end

"""
Per-horizontal-mode ETD-RK2 integrating factors.

`exp`/`expm1` depend only on (kₓ, kᵧ, Δt), never on z, but the stage kernels
run over (k, j, i). Evaluating them inline therefore repeated every
transcendental `nz` times per stage. The table is rebuilt once per step, so a
changing `Δt` is still honoured.
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
    _, nx, ny = size(S.q)
    function build()
        table = similar(S.q, real(eltype(S.q)), nx, ny)
        fill!(table, 0)
        return table
    end
    return ETDCoefficientTable(build(), build(), build(),
                               build(), build(), build())
end

# Hyperdiffusion exponents λΔt of the flow and wave equations for one closure.
@inline _flow_exponent(closure::HorizontalHyperdiffusivity, kₕ², Δt) =
    Nonlinear._hyperdiffusion_exponent(closure.flow, kₕ², Δt)
@inline _flow_exponent(closure::FlowHyperdiffusivity, kₕ², Δt) =
    Nonlinear._hyperdiffusion_exponent(closure, kₕ², Δt)
@inline _flow_exponent(::WaveHyperdiffusivity, kₕ², Δt) = 0.0
@inline _wave_exponent(closure::HorizontalHyperdiffusivity, kₕ², Δt) =
    Nonlinear._hyperdiffusion_exponent(closure.wave, kₕ², Δt)
@inline _wave_exponent(::FlowHyperdiffusivity, kₕ², Δt) = 0.0
@inline _wave_exponent(closure::WaveHyperdiffusivity, kₕ², Δt) =
    Nonlinear._hyperdiffusion_exponent(closure, kₕ², Δt)
@inline _flow_exponent(closure::GridScaleHyperdiffusivity, kₕ², Δt) =
    Nonlinear._grid_scale_exponent(closure, kₕ²)
@inline _wave_exponent(closure::GridScaleHyperdiffusivity, kₕ², Δt) =
    Nonlinear._grid_scale_exponent(closure, kₕ²)

@kernel function _etd_table_kernel!(Eq, hphi1q, hphi2q, EB, hphi1B, hphi2B,
                                    @Const(kx), @Const(ky), closure, Δt, inviscid)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    lambda_q = inviscid ? 0.0 : _flow_exponent(closure, kₕ², Δt)
    lambda_B = inviscid ? 0.0 : _wave_exponent(closure, kₕ², Δt)
    Eq_ij, hphi1q_ij, hphi2q_ij = _etd_coefficients(lambda_q, Δt)
    EB_ij, hphi1B_ij, hphi2B_ij = _etd_coefficients(lambda_B, Δt)
    @inbounds begin
        Eq[i, j] = Eq_ij
        hphi1q[i, j] = hphi1q_ij
        hphi2q[i, j] = hphi2q_ij
        EB[i, j] = EB_ij
        hphi1B[i, j] = hphi1B_ij
        hphi2B[i, j] = hphi2B_ij
    end
end

function _fill_etd_table!(table, kx, ky, closure, Δt, inviscid)
    launch!(_etd_table_kernel!, table.Eq, size(table.Eq),
            table.Eq, table.hphi1q, table.hphi2q,
            table.EB, table.hphi1B, table.hphi2B,
            kx, ky, closure, Δt, inviscid)
    return table
end

"""Refresh `table` for the current `Δt` and closure."""
function fill_etd_table!(table::ETDCoefficientTable, ::ModelFields,
                         G::RuntimeGeometry, options::ETDModelOptions, Δt)
    closure = options.closure
    h = float(Δt)
    inviscid = _inviscid(options)
    cpu = get_backend(table.Eq) isa KernelAbstractions.CPU
    if cpu || (isbitstype(typeof(closure)) && isbitstype(typeof(h)))
        return _fill_etd_table!(table, G.kx, G.ky, closure, h, inviscid)
    end

    # BigFloat and other non-isbits scalars cannot be passed to a device
    # kernel. Compute the horizontal factors in the same arithmetic as on
    # CPU, then copy the finished tables in the grid's precision.
    names = fieldnames(typeof(table))
    dims = size(table.Eq)
    T = eltype(table.Eq)
    host = ETDCoefficientTable(map(_ -> Matrix{T}(undef, dims), names)...)
    _fill_etd_table!(host, G.base.kx, G.base.ky, closure, h, inviscid)
    for name in names
        copy_from_host!(getfield(table, name), getfield(host, name))
    end
    return table
end

"""
    ExponentialRungeKutta2Workspace(fields, plans=nothing; G=nothing)

Reusable storage for model-owned [`step!`](@ref): the first-stage values of
`q` and `B` and their first-stage tendencies. Allocate it once per
simulation; `step!` reuses it on every later step.

A step computes its stage diagnostics (ψ, velocities, A) in the model's own
diagnostic fields, which the end of the step recomputes from the new `q` and
`B`. The second-stage tendencies are formed in scratch buffers and enter the
update directly, which writes `q` and `B` in place. Until that final kernel
`q` and `B` are only read, so a step that throws leaves them unchanged (the
diagnostic fields may then hold stage values).

Pass the model's runtime geometry as `G` to tie the workspace to that model:
`step!` then replaces it rather than reuse it for a different model, even one
whose fields have the same shape.
"""
mutable struct ExponentialRungeKutta2Workspace{A, E, G}
    stage_q::A
    stage_B::A
    rhsq0::A
    rhsB0::A
    etd::E
    geometry::G
end

# `plans` is accepted for call-site compatibility; the storage is derived
# entirely from the shape of `S`.
function ExponentialRungeKutta2Workspace(S::ModelFields, plans=nothing; G=nothing)
    return ExponentialRungeKutta2Workspace(
        similar(S.q), similar(S.B), similar(S.q), similar(S.B),
        ETDCoefficientTable(S), G)
end

"""Whether a workspace may serve the model whose runtime geometry is `G`."""
_etdrk2_workspace_belongs_to(workspace::ExponentialRungeKutta2Workspace, G) =
    workspace.geometry === nothing || workspace.geometry === G

@inline function _etdrk2_array_layout_matches(cached::AbstractArray,
                                               live::AbstractArray)
    return typeof(cached) === typeof(live) && size(cached) == size(live)
end

"""Whether a reusable ETD workspace belongs to the live field layout."""
function _etdrk2_workspace_matches(workspace::ExponentialRungeKutta2Workspace,
                                    fields::ModelFields)
    for array in (workspace.stage_q, workspace.rhsq0)
        _etdrk2_array_layout_matches(array, fields.q) || return false
    end
    for array in (workspace.stage_B, workspace.rhsB0)
        _etdrk2_array_layout_matches(array, fields.B) || return false
    end

    _, nx, ny = size(fields.q)
    coefficient_size = (nx, ny)
    table = workspace.etd
    return all(array -> size(array) == coefficient_size,
               (table.Eq, table.hphi1q, table.hphi2q,
                table.EB, table.hphi1B, table.hphi2B))
end

"""First-stage state: the workspace's `q` and `B` with the diagnostic arrays of `S`."""
function _stage_fields(S::ModelFields{T, RT, CT},
                       workspace::ExponentialRungeKutta2Workspace) where {T, RT, CT}
    return ModelFields{T, RT, CT}(workspace.stage_q, workspace.stage_B,
                                  S.psi, S.A, S.C, S.u, S.v, S.w)
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
                         N2_face_profile=nothing,
                         timestep_workspace=nothing, compute_w=false,
                         use_wave_feedback=true)
    # The scalar N² is only the vertical velocity's fallback for a missing
    # profile. Reading it from a device profile synchronizes the device, so
    # the stages, which skip w, never do.
    N2 = compute_w && N2_profile !== nothing ? first_value(N2_profile) : 1.0
    velocities!() = compute_velocities!(S, G; plans, f=options.f, N2, compute_w,
        N2_profile, N2_face_profile, workspace, dealias_mask=L)

    if _fixed_flow(options) && !_linear(options)
        # Nonlinear fixed-flow stages diagnose velocities from the projected
        # prescribed ψ so both operands of each quadratic product are
        # alias-safe; the prescribed ψ itself is restored afterwards.
        with_scratch(workspace) do
            psi_base = scratch_like(workspace, S.psi)
            copyto!(psi_base, S.psi)
            dealias!(S.psi, psi_base, _stage_mask(L, S.psi))
            try
                velocities!()
            finally
                copyto!(S.psi, psi_base)
            end
        end
    else
        _fixed_flow(options) || _invert_total_q_to_psi!(S, G, options, plans, a, L;
            workspace, use_wave_feedback, project_nonlinear_state=true)
        velocities!()
    end
    return S
end

"""
Balanced PV at `(k, i, j)` from the streamfunction: `∇²ψ + ∂z(a ∂zψ)` with the
flux stencil of the inversion, so it equals the inverted `q - qʷ`.
"""
@inline function _balanced_pv(psi, a, k, i, j, kₕ², Δz⁻², nz)
    @inbounds begin
        ψk = psi[k, i, j]
        up = k < nz ? a[k] * (psi[k + 1, i, j] - ψk) : zero(ψk)
        down = k > 1 ? a[k - 1] * (ψk - psi[k - 1, i, j]) : zero(ψk)
    end
    return (up - down) * Δz⁻² - kₕ² * ψk
end

"""
ETD-RK2 tendencies of `q` and `B` at `(k, i, j)` from the advection terms `nq`
and `nB`, the refraction term `rB`, and the amplitude `A` there, adding
vertical diffusion of `q` (read from the array `q`) and dispersion. With
`relax` (wave feedback and dissipation) the vertical diffusion acts on the
balanced PV, computed from `psi`.
"""
@inline function _etdrk2_tendencies(nq, nB, rB, A, q, psi, a, k, i, j, kx, ky,
                                    mask, alpha_disp, νz, Δz⁻², nz, diffuse,
                                    fixed_flow, linear, no_dispersion, relax)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    # Vertical diffusion νz ∂²/∂z² with Neumann boundaries.
    dq = if !diffuse
        zero(nq)
    elseif relax
        g = _balanced_pv(psi, a, k, i, j, kₕ², Δz⁻², nz)
        above = k < nz ? _balanced_pv(psi, a, k + 1, i, j, kₕ², Δz⁻², nz) : g
        below = k > 1 ? _balanced_pv(psi, a, k - 1, i, j, kₕ², Δz⁻², nz) : g
        νz * (above - 2g + below) * Δz⁻²
    else
        @inbounds if k == 1
            νz * (q[k + 1, i, j] - q[k, i, j]) * Δz⁻²
        elseif k == nz
            νz * (q[k - 1, i, j] - q[k, i, j]) * Δz⁻²
        else
            νz * (q[k + 1, i, j] - 2q[k, i, j] + q[k - 1, i, j]) * Δz⁻²
        end
    end
    if keep_mode(mask, i, j)
        dispersion = no_dispersion ? zero(nB) : im * alpha_disp * kₕ² * A
        tq = fixed_flow ? zero(nq) : -nq + dq
        tB = -nB + dispersion - im * rB / 2
    else
        # Outside the nonlinear cutoff, a linear-QG mode still receives its
        # explicit vertical-diffusion tendency. Nonlinear runs project these
        # modes out to keep every quadratic product alias-safe.
        tq = (!fixed_flow && linear) ? dq : zero(nq)
        # Linear wave dynamics may carry resolved modes outside the cutoff;
        # retain their pointwise dispersion tendency. Advection and
        # refraction products remain projected by their own kernels.
        tB = (linear && !no_dispersion) ? im * alpha_disp * kₕ² * A : zero(nB)
    end
    return tq, tB
end

"""First-stage tendencies, written over the advection terms in `rhsq`, `rhsB`."""
@kernel function _etdrk2_rhs_kernel!(rhsq, rhsB, @Const(q), @Const(rBk), @Const(A),
                                     @Const(psi), @Const(a), @Const(kx), @Const(ky),
                                     mask, alpha_disp, νz, Δz⁻², nz, diffuse,
                                     fixed_flow, linear, no_dispersion, relax)
    k, i, j = @index(Global, NTuple)
    @inbounds tq, tB = _etdrk2_tendencies(rhsq[k, i, j], rhsB[k, i, j],
        rBk[k, i, j], A[k, i, j], q, psi, a, k, i, j, kx, ky, mask, alpha_disp,
        νz, Δz⁻², nz, diffuse, fixed_flow, linear, no_dispersion, relax)
    @inbounds rhsq[k, i, j] = tq
    @inbounds rhsB[k, i, j] = tB
end

"""
Pointwise-kernel arguments that fix the ETD-RK2 tendency for `options`, with
the scalars in the grid's precision so single-precision kernels stay single.
"""
function _tendency_parameters(options::ETDModelOptions, G::RuntimeGeometry)
    Δz = vertical_spacing(G)
    T = typeof(Δz)
    return (T(options.f / 2), T(options.vertical_diffusivity), 1 / (Δz * Δz),
            G.nz, !_inviscid(options) && G.nz > 1, _fixed_flow(options),
            _linear(options), _no_dispersion(options), _relaxes_to_wave_pv(options))
end

"""
Whether the flow's dissipation must act on the balanced PV `q - qʷ` rather
than on the prognostic total PV: with wave feedback and active dissipation.
"""
_relaxes_to_wave_pv(options::ETDModelOptions) =
    _wave_feedback_enabled(options) && !_inviscid(options)

"""
Remove the depth mean of the horizontally uniform (kₕ = 0) column of `B`. No
wave amplitude produces it: at kₕ = 0, L⁺ = L, and the depth sum of LA
vanishes under the Neumann conditions. Other columns are left unchanged.
"""
function _remove_uniform_depth_mean!(B, G::RuntimeGeometry;
                                     reference_centered::Bool=false)
    nz, nx, ny = size(B)
    launch!(_uniform_depth_mean_kernel!, B, (nx, ny), B, G.kx, G.ky, nz,
            Val(reference_centered))
    return B
end

@kernel function _uniform_depth_mean_kernel!(B, @Const(kx), @Const(ky), nz,
                                             reference_centered)
    i, j = @index(Global, NTuple)
    @inbounds if kx[i]^2 + ky[j]^2 == 0
        _subtract_depth_mean!(B, i, j, nz, reference_centered)
    end
end

@inline _relative_to(value, ::Nothing) = value
@inline _relative_to(value, reference) = value - reference

# For initial data, work relative to one level so that a large constant
# offset cannot dominate the rounding of a small depth-varying signal.
@inline function _subtract_depth_mean!(B, i, j, nz,
                                       ::Val{reference_centered}) where reference_centered
    @inbounds reference = reference_centered ? B[1, i, j] : nothing
    offset = zero(eltype(B))
    for k in 1:nz
        @inbounds offset += _relative_to(B[k, i, j], reference)
    end
    offset /= nz
    for k in 1:nz
        @inbounds B[k, i, j] = _relative_to(B[k, i, j], reference) - offset
    end
    return nothing
end

"""
Restrict the wave envelope to the states a wave amplitude can produce:
remove the depth mean of every column for the original YBJ, where `B = LA`,
and of the horizontally uniform column for YBJ⁺. The refraction term would
otherwise feed these components, which carry no wave velocity.
"""
function _project_wave_envelope!(B, G::RuntimeGeometry, options::ETDModelOptions,
                                 L, workspace; reference_centered::Bool=false)
    if _ybj_plus(options)
        _remove_uniform_depth_mean!(B, G; reference_centered)
    else
        sumB!(B, G; Lmask=_linear(options) ? nothing : L, workspace,
              reference_centered)
    end
    return B
end

"""Diagnose ψ and the velocities of a stage state, after projecting its B."""
function _prepare_stage!(S, G, options, plans, a, L, workspace, N2_profile,
                         N2_face_profile)
    _project_wave_envelope!(S.B, G, options, L, workspace)
    _diagnose_flow!(S, G, options, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    N2_face_profile=N2_face_profile,
                    compute_w=false, use_wave_feedback=true)
    return S
end

"""
Advection terms `nq`, `nB` and refraction term `rB` of the prepared stage
state `S`, and its wave amplitude A for the dispersion term (unless
`amplitude` is false, for an A that is already current).
"""
function _stage_terms!(nq, nB, rB, S, G, options, plans, a, L, workspace,
                       N2_profile, N2_face_profile; amplitude::Bool=true)
    if _linear(options)
        fill!(nq, zero(eltype(nq)))
    else
        convol_waqg_q!(nq, S.u, S.v, S.q, G, plans; Lmask=L, workspace)
    end
    return _wave_terms!(nB, rB, S, G, options, plans, a, L, workspace,
                        N2_profile, N2_face_profile; amplitude)
end

"""
Divergence-form advection of the spectral field `χ` by the velocity
`(-∂ᵧψ, ∂ₓψ)` of `psi`, written to `nχ` (which may be `χ`). `psi` is
projected as `_diagnose_flow!` projects a stage's ψ, so this is the
advection by a diagnosed stage's `u` and `v` (`product` as in
`convol_waqg_B!`, or `_real_product` as in `convol_waqg_q!`), but the
velocity is formed in scratch rather than read from `u` and `v`. The field's
transform and the divergence of the fluxes are those of
`Nonlinear._convol_advect!`.
"""
function _advect_by_psi!(nχ, psi, χ, G, plans, options, L, product, workspace)
    with_scratch(workspace) do
        mask = Nonlinear._nonlinear_mask(L, G, nχ)
        field = Nonlinear._dealiased_physical(χ, mask, plans, workspace)
        # ∂ₓψ = v and ∂ᵧψ = -u, in physical space. The destination is free
        # until the divergence is written to it, so it holds ∂ₓψ unless it
        # is also an input.
        ψx = nχ === χ || nχ === psi ? scratch_like(workspace, χ) : nχ
        ψy = scratch_like(workspace, χ)
        velocity_mask = _linear(options) ? KeepAllModes() : _stage_mask(L, psi)
        launch!(Nonlinear._gradient_kernel!, ψx, size(ψx), ψx, ψy, psi,
                G.kx, G.ky, velocity_mask)
        fft_backward!(ψx, ψx, plans)
        fft_backward!(ψy, ψy, plans)
        kernel_map!((ψyᵢ, χᵢ) -> product(-real(ψyᵢ), χᵢ), ψy, ψy, field)
        kernel_map!((ψxᵢ, χᵢ) -> product(real(ψxᵢ), χᵢ), field, ψx, field)
        Nonlinear._flux_divergence!(nχ, ψy, field, G, plans, mask)
    end
    return nχ
end

"""
Wave advection `nB` of the state `S`, by `S.u` and `S.v`, or with
`from_psi` by the velocity of `S.psi` (see `_advect_by_psi!`).
"""
function _wave_advection!(nB, S, G, options, plans, L, workspace;
                          from_psi::Bool=false)
    # B is a complex physical field represented by its complex Fourier
    # transform. Both YBJ formulations therefore use complex advection and
    # refraction kernels; only the B -> A diagnostic relation differs.
    if _linear(options)
        fill!(nB, zero(eltype(nB)))
    elseif from_psi
        _advect_by_psi!(nB, S.psi, S.B, G, plans, options, L, *, workspace)
    else
        convol_waqg_B!(nB, S.u, S.v, S.B, G, plans; Lmask=L, workspace)
    end
    return nB
end

"""Wave refraction `rB` of the state `S`; passive waves have none."""
function _wave_refraction!(rB, S, G, options, plans, L, workspace)
    if _passive_wave(options)
        fill!(rB, zero(eltype(rB)))
    else
        refraction_waqg_B!(rB, S.B, S.psi, G, plans; Lmask=L, workspace)
    end
    return rB
end

"""
The amplitude `A`, and `C = A_z`, of the envelope `B` by the formulation's
relation: B = L⁺A for YBJ⁺ and passive waves, and B = LA for YBJ, whose
depth mean makes the tendency with advection `nB` and refraction `rB`
solvable.
"""
function _wave_amplitude!(A, C, B, nB, rB, G, options, a, L, workspace,
                          N2_profile, N2_face_profile)
    if _ybj_plus(options)
        Elliptic._invert_B_to_A!(A, C, B, G, a, workspace)
    else
        # The cutoff is required by nonlinear products, but linear
        # normal-YBJ recovery is a one-column operation and is valid at
        # every resolved mode.
        sigma = compute_sigma(options.f, G, nB, rB; Lmask=L, workspace)
        normal_N² = N2_face_profile === nothing ? N2_profile : N2_face_profile
        compute_A!(A, C, B, sigma, G; f=options.f,
                   Lmask=_linear(options) ? nothing : L, workspace,
                   N2_profile=normal_N²)
    end
    return A, C
end

"""
Recover `S.A`, and `S.C = A_z`, from `S.B` by the formulation's relation (see
`_wave_amplitude!`); ψ, u, v, and w are only read. The YBJ recovery uses the
waves' advection by `S.u` and `S.v`, or with `from_psi` by the velocity of
`S.psi`, formed in scratch. The advection and refraction are formed in `S.A`
and `S.C` themselves, which the recovery then overwrites, so it borrows no
buffer of its own. `S` needs only the fields named here, so it may be a
named tuple of them.
"""
function _recover_wave_amplitude!(S, G, options, plans, a, L, workspace,
                                  N2_profile, N2_face_profile;
                                  from_psi::Bool=false)
    if _ybj_plus(options)
        Elliptic._invert_B_to_A!(S.A, S.C, S.B, G, a, workspace)
        return S
    end
    nB = _wave_advection!(S.A, S, G, options, plans, L, workspace; from_psi)
    rB = _wave_refraction!(S.C, S, G, options, plans, L, workspace)
    _wave_amplitude!(S.A, S.C, S.B, nB, rB, G, options, a, L, workspace,
                     N2_profile, N2_face_profile)
    return S
end

"""
Wave advection `nB` and refraction `rB` of the prepared state `S`, and its
amplitude A for the dispersion term (unless `amplitude` is false).
"""
function _wave_terms!(nB, rB, S, G, options, plans, a, L, workspace,
                      N2_profile, N2_face_profile; amplitude::Bool=true)
    _wave_advection!(nB, S, G, options, plans, L, workspace)
    _wave_refraction!(rB, S, G, options, plans, L, workspace)
    amplitude && !_no_dispersion(options) &&
        _wave_amplitude!(S.A, S.C, S.B, nB, rB, G, options, a, L, workspace,
                         N2_profile, N2_face_profile)
    return nB, rB
end

"""Inviscid wave tendency, written over the wave advection term in `Bt`."""
@kernel function _wave_tendency_kernel!(Bt, @Const(rB), @Const(A), @Const(kx),
                                        @Const(ky), mask, alpha_disp, linear)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        nB = Bt[k, i, j]
        amplitude = A === nothing ? zero(nB) : A[k, i, j]
        _, tB = _etdrk2_tendencies(zero(nB), nB, rB[k, i, j], amplitude,
            Bt, Bt, kx, k, i, j, kx, ky, mask, alpha_disp,
            zero(alpha_disp), zero(alpha_disp), 1, false, true, linear, A === nothing,
            false)
        Bt[k, i, j] = tB
    end
end

"""
Inviscid tendency `∂ₜB` (advection, refraction, dispersion) of the state `S`
from its ψ and B alone: the velocity and the amplitude are formed in
scratch, so `S` is only read.
"""
function _wave_tendency!(Bt, S, G, options, plans, a, L, workspace,
                         N2_profile, N2_face_profile)
    # The advection comes first, while it can borrow the most scratch.
    _wave_advection!(Bt, S, G, options, plans, L, workspace; from_psi=true)
    with_scratch(workspace) do
        rB = _wave_refraction!(scratch_like(workspace, S.B), S, G, options,
                               plans, L, workspace)
        A = nothing
        if !_no_dispersion(options)
            A = scratch_like(workspace, S.A)
            _wave_amplitude!(A, scratch_like(workspace, S.C), S.B, Bt, rB, G,
                             options, a, L, workspace, N2_profile,
                             N2_face_profile)
        end
        alpha_disp, _, _, _, _, _, linear, _ = _tendency_parameters(options, G)
        launch!(_wave_tendency_kernel!, Bt, size(Bt), Bt, rB, A, G.kx, G.ky,
                _stage_mask(L, Bt), alpha_disp, linear)
    end
    return Bt
end

"""
    _wave_pv_derivative!(D, S, G, options, plans, a, L; kwargs...)

Material derivative of the wave PV following the mean flow,
`D = ∂ₜqʷ + J(ψ, qʷ)`, at the state `S`, from its ψ and B alone (`S` is only
read). It forces the mean-flow vertical velocity in coupled runs: the
conserved PV is `q_g + qʷ`, so `D_t q_g = -D_t qʷ`. `∂ₜqʷ` uses the inviscid
wave tendency, restricted as a step restricts B.
"""
function _wave_pv_derivative!(D, S::ModelFields, G::RuntimeGeometry,
                              options::ETDModelOptions, plans, a, L;
                              workspace=nothing, N2_profile=nothing,
                              N2_face_profile=nothing)
    with_scratch(workspace) do
        Bt = scratch_like(workspace, S.B)
        _wave_tendency!(Bt, S, G, options, plans, a, L, workspace,
                        N2_profile, N2_face_profile)
        # A step removes the depth mean that refraction gives B (see
        # `_project_wave_envelope!`), so B changes only by the rest of Bt.
        _project_wave_envelope!(Bt, G, options, L, workspace)

        # ∂ₜqʷ by polarization of the quadratic qʷ(B), which is exact for any
        # ε; scaling ε B_t to the size of B keeps the round-off small.
        B_scale = maximum(abs, S.B)
        t_scale = maximum(abs, Bt)
        ε = iszero(B_scale) || iszero(t_scale) ? one(real(eltype(Bt))) :
            convert(real(eltype(Bt)), B_scale / t_scale)
        kernel_map!((b, bt) -> b + ε * bt, D, S.B, Bt)
        compute_qw_complex!(D, D, G, plans; f=options.f, Lmask=L, workspace)
        kernel_map!((b, bt) -> b - ε * bt, Bt, S.B, Bt)
        compute_qw_complex!(Bt, Bt, G, plans; f=options.f, Lmask=L, workspace)
        kernel_map!((plus, minus) -> (plus - minus) / (2ε), D, D, Bt)

        # J(ψ, qʷ), in the divergence form of the model's advection.
        compute_qw_complex!(Bt, S.B, G, plans; f=options.f, Lmask=L, workspace)
        _advect_by_psi!(Bt, S.psi, Bt, G, plans, options, L,
                        Nonlinear._real_product, workspace)
        kernel_map!(+, D, D, Bt)
    end
    return D
end

"""
The tendencies of the state `S` that a step starts from. `start` says what
`S` holds (see `_advance_etdrk2!`): with `:new`, B is projected and ψ, u, v,
and A are diagnosed; with `:projected`, B is as a step left it, so only the
diagnostics are recomputed; with `:diagnosed`, everything is as a step left
it. `amplitude` says whether A is recovered as well; it is not when A is
still the one the last step left, whatever became of ψ, u, and v.
"""
function _compute_etdrk2_rhs!(rhsq, rhsB, S::ModelFields, G::RuntimeGeometry,
                              options::ETDModelOptions, plans;
                              a, dealias_mask=nothing, workspace=nothing,
                              N2_profile=nothing,
                              N2_face_profile=nothing,
                              timestep_workspace=nothing, start::Symbol=:new,
                              amplitude::Bool=start !== :diagnosed)
    L = isnothing(dealias_mask) ? KeepAllModes() : dealias_mask
    if start === :new
        _prepare_stage!(S, G, options, plans, a, L, workspace, N2_profile,
                        N2_face_profile)
    elseif start === :projected
        _diagnose_flow!(S, G, options, plans, a, L;
                        workspace=workspace, N2_profile=N2_profile,
                        N2_face_profile=N2_face_profile,
                        compute_w=false, use_wave_feedback=true)
    end
    with_scratch(workspace) do
        # The advection terms go straight into the tendencies, which the
        # assembly kernel then completes.
        rBk = scratch_like(workspace, S.B)
        _stage_terms!(rhsq, rhsB, rBk, S, G, options, plans, a, L, workspace,
                      N2_profile, N2_face_profile; amplitude)
        launch!(_etdrk2_rhs_kernel!, rhsq, size(rhsq),
                rhsq, rhsB, S.q, rBk, S.A, S.psi, a, G.kx, G.ky,
                _stage_mask(L, rhsq), _tendency_parameters(options, G)...)
    end
    return rhsq, rhsB
end

function _finalize_etdrk2_state!(S::ModelFields, G::RuntimeGeometry,
                                 options::ETDModelOptions, plans, a, L;
                                 workspace=nothing, N2_profile=nothing,
                                 N2_face_profile=nothing,
                                 timestep_workspace=nothing)
    _project_wave_envelope!(S.B, G, options, L, workspace)
    restore_full_fixed_diagnostics = _fixed_flow(options) && !_linear(options)
    # The vertical velocity does not enter the dynamics. It is diagnosed on
    # demand (`compute_velocities!`, output, particles), which in coupled runs
    # needs the wave tendency, so a step leaves w = 0.
    _diagnose_flow!(S, G, options, plans, a, L;
                    workspace=workspace, N2_profile=N2_profile,
                    N2_face_profile=N2_face_profile,
                    compute_w=false, use_wave_feedback=true)

    # Passive waves (which `_ybj_plus` includes) do not need A, but it is
    # kept as `set!`, output, and the energies define it: B = L⁺A. The YBJ
    # recovery uses the stage velocity just diagnosed.
    _recover_wave_amplitude!(S, G, options, plans, a, L, workspace,
                             N2_profile, N2_face_profile)

    # Nonlinear fixed-flow stages use a projected prescribed ψ so both
    # operands of each quadratic product are alias-safe. The public final
    # diagnostics, however, must correspond to the restored prescribed ψ.
    if restore_full_fixed_diagnostics
        # Without w, no scalar N² is needed (see `_diagnose_flow!`).
        compute_velocities!(S, G; plans, f=options.f, compute_w=false,
            N2_profile, N2_face_profile, workspace, dealias_mask=L)
    end
    return S
end

@kernel function _etdrk2_first_stage_kernel!(q_stage, B_stage, @Const(qn), @Const(Bn),
                                             rhsq0, @Const(rhsB0),
                                             @Const(Eq), @Const(hphi1q), @Const(hphi2q),
                                             @Const(EB), @Const(hphi1B),
                                             @Const(psi), @Const(a), @Const(kx), @Const(ky),
                                             mask, fixed_flow, linear, relax, Δt, Δz⁻², nz)
    k, i, j = @index(Global, NTuple)
    @inbounds if fixed_flow
        q_stage[k, i, j] = qn[k, i, j]
    elseif keep_mode(mask, i, j) || linear
        if relax
            # The explicit part also holds λqʷ, qʷ = q - (balanced PV of ψ),
            # applied with hφ₁λ = 1 - E (finite where λΔt overflows).
            kₕ² = kx[i]^2 + ky[j]^2
            qʷ = qn[k, i, j] - _balanced_pv(psi, a, k, i, j, kₕ², Δz⁻², nz)
            c₁ = 1 - Eq[i, j]
            c₂ = 1 - hphi1q[i, j] / Δt
            n₀ = rhsq0[k, i, j]
            q_stage[k, i, j] = Eq[i, j] * qn[k, i, j] + hphi1q[i, j] * n₀ + c₁ * qʷ
            # Keep what the second stage needs of the start of the step:
            # (hφ₁ - hφ₂)N₀ + (hφ₁ - hφ₂)λqʷ₀, with ψ₀ about to be replaced.
            rhsq0[k, i, j] = (hphi1q[i, j] - hphi2q[i, j]) * n₀ + (c₁ - c₂) * qʷ
        else
            q_stage[k, i, j] = Eq[i, j] * qn[k, i, j] + hphi1q[i, j] * rhsq0[k, i, j]
        end
    else
        q_stage[k, i, j] = zero(eltype(q_stage))
    end
    @inbounds if keep_mode(mask, i, j) || linear
        B_stage[k, i, j] = EB[i, j] * Bn[k, i, j] + hphi1B[i, j] * rhsB0[k, i, j]
    else
        B_stage[k, i, j] = zero(eltype(B_stage))
    end
end

"""
Second stage, fused with its tendencies: from the stage terms (`nq`, `nB`,
`rB`, `A`, and the stage `q` for vertical diffusion), update `q` and `B` in
place. `q` and `B` hold the start-of-step values on entry.
"""
@kernel function _etdrk2_second_stage_kernel!(q, B, @Const(q_stage), @Const(nq),
                                              @Const(nB), @Const(rB), @Const(A),
                                              @Const(rhsq0), @Const(rhsB0),
                                              @Const(psi), @Const(a),
                                              @Const(kx), @Const(ky),
                                              @Const(Eq), @Const(hphi1q), @Const(hphi2q),
                                              @Const(EB), @Const(hphi1B), @Const(hphi2B),
                                              Δt, mask, alpha_disp, νz, Δz⁻², nz, diffuse,
                                              fixed_flow, linear, no_dispersion, relax)
    k, i, j = @index(Global, NTuple)
    @inbounds rhsq1, rhsB1 = _etdrk2_tendencies(nq[k, i, j], nB[k, i, j],
        rB[k, i, j], A[k, i, j], q_stage, psi, a, k, i, j, kx, ky, mask,
        alpha_disp, νz, Δz⁻², nz, diffuse, fixed_flow, linear, no_dispersion, relax)
    # A fixed flow keeps its q.
    @inbounds if !fixed_flow
        if (keep_mode(mask, i, j) || linear) && relax
            # rhsq0 holds the first stage's share (see the first-stage kernel);
            # the stage's λqʷ₁ enters with hφ₂λ = 1 - hφ₁/Δt.
            kₕ² = kx[i]^2 + ky[j]^2
            qʷ₁ = q_stage[k, i, j] - _balanced_pv(psi, a, k, i, j, kₕ², Δz⁻², nz)
            c₂ = 1 - hphi1q[i, j] / Δt
            q[k, i, j] = Eq[i, j] * q[k, i, j] + rhsq0[k, i, j] +
                         hphi2q[i, j] * rhsq1 + c₂ * qʷ₁
        elseif keep_mode(mask, i, j) || linear
            q[k, i, j] = Eq[i, j] * q[k, i, j] +
                         hphi1q[i, j] * rhsq0[k, i, j] +
                         hphi2q[i, j] * (rhsq1 - rhsq0[k, i, j])
        else
            q[k, i, j] = zero(eltype(q))
        end
    end
    @inbounds if keep_mode(mask, i, j) || linear
        B[k, i, j] = EB[i, j] * B[k, i, j] +
                     hphi1B[i, j] * rhsB0[k, i, j] +
                     hphi2B[i, j] * (rhsB1 - rhsB0[k, i, j])
    else
        B[k, i, j] = zero(eltype(B))
    end
end

"""
Internal ETD-RK2 step of the fields `S`, updated in place.

`start` says what `S` holds when the step begins. A step ends by projecting B
(see `_project_wave_envelope!`) and diagnosing ψ, u, v, A, and C from it, so a
step that follows another can start from that state: `:diagnosed` when
nothing has changed `S` since (the diagnosis is then not repeated), and
`:projected` when only the diagnostics may have been overwritten (they are
diagnosed again, to the same values, from q and B). `:new`, for any other
state, projects B and diagnoses it. The projection is not exactly idempotent
in floating point, so the first two differ from `:new` in the last bits.
"""
function _advance_etdrk2!(S::ModelFields, G::RuntimeGeometry,
                          options::ETDModelOptions, plans;
                          Δt::Real, a, dealias_mask=nothing, workspace=nothing,
                          N2_profile=nothing, N2_face_profile=nothing,
                          timestep_workspace=nothing, start::Symbol=:new)
    # A nonlinear fixed flow's stages advect with the velocity of its
    # projected ψ, but a step ends with the velocity of the full ψ, so the
    # velocity is diagnosed again. The amplitude the step left is the stage's
    # own (it was recovered before the full velocity was restored) and stays.
    amplitude = start !== :diagnosed
    start === :diagnosed && _fixed_flow(options) && !_linear(options) &&
        (start = :projected)
    L = isnothing(dealias_mask) ? KeepAllModes() : dealias_mask
    storage = timestep_workspace === nothing ?
              ExponentialRungeKutta2Workspace(S; G) : timestep_workspace
    etd = storage.etd
    fill_etd_table!(etd, S, G, options, Δt)

    _compute_etdrk2_rhs!(storage.rhsq0, storage.rhsB0, S, G, options, plans;
                         a=a, dealias_mask=L, workspace=workspace,
                         N2_profile=N2_profile,
                         N2_face_profile=N2_face_profile, start, amplitude)

    # The first-stage state has its own q and B; its diagnostics are computed
    # in the diagnostic arrays of S, which the final diagnosis overwrites.
    stage = _stage_fields(S, storage)
    mask = _stage_mask(L, S.q)
    T = real(eltype(S.q))
    launch!(_etdrk2_first_stage_kernel!, stage.q, size(stage.q),
            stage.q, stage.B, S.q, S.B, storage.rhsq0, storage.rhsB0,
            etd.Eq, etd.hphi1q, etd.hphi2q, etd.EB, etd.hphi1B,
            S.psi, a, G.kx, G.ky, mask, _fixed_flow(options), _linear(options),
            _relaxes_to_wave_pv(options), T(Δt), 1 / vertical_spacing(G)^2, G.nz)

    # Second-stage terms go to scratch and straight into the update, so the
    # second-stage tendencies never need storage of their own.
    _prepare_stage!(stage, G, options, plans, a, L, workspace, N2_profile,
                    N2_face_profile)
    with_scratch(workspace) do
        nq = scratch_like(workspace, S.q)
        nB = scratch_like(workspace, S.B)
        rB = scratch_like(workspace, S.B)
        _stage_terms!(nq, nB, rB, stage, G, options, plans, a, L, workspace,
                      N2_profile, N2_face_profile)
        launch!(_etdrk2_second_stage_kernel!, S.q, size(S.q),
                S.q, S.B, stage.q, nq, nB, rB, stage.A,
                storage.rhsq0, storage.rhsB0, S.psi, a, G.kx, G.ky,
                etd.Eq, etd.hphi1q, etd.hphi2q, etd.EB, etd.hphi1B, etd.hphi2B,
                T(Δt), mask, _tendency_parameters(options, G)...)
    end

    _finalize_etdrk2_state!(S, G, options, plans, a, L;
                            workspace=workspace, N2_profile=N2_profile,
                            N2_face_profile=N2_face_profile)
    return S
end
