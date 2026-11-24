"""
Precomputed vertical coefficients used by model operators. Both N² profiles
are bounded below by the floor of `a_ell_from_N2`.
"""
struct OperatorCoefficients{T}
    "Pointwise N² sampled at cell centers (`grid.z`)."
    N²::Vector{T}
    "N² sampled at each cell's upper face (`grid.z_faces[2:end]`)."
    N²_face::Vector{T}
    "Flux coefficient f²/N² sampled at the same upper faces as `N²_face`."
    a_ell::Vector{T}
end

"""
    ModelRuntime

Ephemeral execution resources associated with a model: its architecture,
kernel-facing geometry, transform plans, scratch workspace, operator
coefficients, and dealiasing mask.

`coefficients` holds the host profiles (written to output files and returned
to users); `device_coefficients` holds the same profiles stored on the
architecture, which is what the kernels read. On the CPU both refer to the
same vectors.
"""
mutable struct ModelRuntime{A, G, P, W, C, K, L}
    architecture::A
    geometry::G
    plans::P
    workspace::W
    coefficients::C
    device_coefficients::K
    dealias_mask::L
    finalized::Bool
    # What the model's fields hold of the last time step (see `_step!`).
    step_state::Symbol
end

_coriolis_frequency(coriolis::FPlane) = coriolis.f
_stratification_N²(stratification::ConstantStratification) = stratification.N²
_runtime_profile(stratification::ConstantStratification) =
    ConstantN(sqrt(float(stratification.N²)))
_runtime_profile(stratification::StratificationProfile) = stratification

"""Construct execution resources for `grid` on the grid's architecture."""
function build_runtime(grid::RectilinearGrid, physics::ModelPhysics,
    numerics::ModelNumerics; verbose::Bool=true)

    architecture = grid.architecture
    T = eltype(grid)
    nx, ny, nz = grid.size
    geometry = RuntimeGeometry(grid)
    plans = Plans(architecture, nz, nx, ny, Complex{T})
    workspace = ModelWorkspace(architecture, (nz, nx, ny); T)

    # The host profiles (written to output files) are Float64; the kernels
    # read copies in the grid's precision. Every operator divides by the
    # same bounded N² (see `_bounded_N2`), the centres taking the faces' floor.
    profile = _runtime_profile(physics.stratification)
    _warn_uncovered_depths(profile, grid)
    N²_face = _bounded_N2(
        Float64.(_compute_stratification_face_profile(profile, grid)))
    N² = max.(Float64.(compute_stratification_profile(profile, grid)),
              _N2_floor(N²_face, Float64))
    a_ell = a_ell_from_N2(N²_face, physics.coriolis)
    coefficients = OperatorCoefficients(N², N²_face, a_ell)
    in_precision(profile) = eltype(profile) === T ? profile : T.(profile)
    device_coefficients = (
        N²=on_architecture(architecture, in_precision(N²)),
        N²_face=on_architecture(architecture, in_precision(N²_face)),
        a_ell=on_architecture(architecture, in_precision(a_ell)),
    )
    mask = on_architecture(architecture, dealias_mask(grid))

    if verbose
        @info "QGYBJModel runtime initialized" architecture size=grid.size precision=T
    end
    return ModelRuntime(architecture, geometry, plans, workspace, coefficients,
                        device_coefficients, mask, false, :new)
end

"""Allocate model fields on the runtime's architecture, in the grid's precision."""
function allocate_fields(grid::RectilinearGrid, runtime::ModelRuntime;
    T::Type{<:AbstractFloat}=eltype(grid))

    runtime.finalized && error("cannot allocate fields from a finalized runtime")
    grid.size == (runtime.geometry.nx,
                  runtime.geometry.ny,
                  runtime.geometry.nz) ||
        throw(ArgumentError("runtime geometry does not match the public grid"))
    nx, ny, nz = grid.size
    return ModelFields(runtime.architecture, T, (nz, nx, ny))
end

"""Release runtime-owned scratch storage; repeated calls are safe."""
function finalize_runtime!(runtime::ModelRuntime)
    runtime.finalized && return runtime
    empty!(runtime.workspace.pool.buffers)
    runtime.workspace.pool.taken = 0
    runtime.finalized = true
    return runtime
end

architecture(runtime::ModelRuntime) = runtime.architecture
architecture(model::QGYBJModel) = model.runtime.architecture

#=
What a step can reuse of the one before it. A step leaves B projected and ψ,
u, v, A, and C diagnosed from it, and records that as `:diagnosed`. Whatever
then writes to the model's fields says so here, so that the next step of a
run never trusts a field that is no longer the step's: an operator that
overwrites diagnostics leaves `:projected` (q and B are still the step's),
and anything that sets q, B, or a prescribed ψ leaves `:new`.
=#
const _STEP_STATES = (:new, :projected, :diagnosed)
_step_state_rank(state::Symbol) = findfirst(==(state), _STEP_STATES)

"""Record that the model's diagnostic fields (ψ, u, v, w, A, C) were overwritten."""
function _diagnostics_changed!(model::QGYBJModel)
    runtime = model.runtime
    runtime.step_state === :diagnosed && (runtime.step_state = :projected)
    return model
end

"""Record that the model's state (q, B, or a prescribed ψ) was set."""
_state_changed!(model::QGYBJModel) = (model.runtime.step_state = :new; model)

"""
Whether ψ, u, v, A, and C are as the model's last step left them, as far as
the model's own operators and setters know. A caller's direct write to
`model.fields` is not recorded, so this is asked only inside `run!`, between
its steps, where nothing else can write to them; everything else rebuilds
what it needs from q and B.
"""
_fields_as_stepped(model::QGYBJModel) = model.runtime.step_state === :diagnosed

"""
The streamfunction that a step diagnoses for `fields`: the balanced flow of
their q (less the wave PV, with feedback), in a buffer borrowed from the
model's workspace, or a fixed flow's prescribed ψ itself. `fields` is only
read. Call it inside `with_scratch`.
"""
function _diagnosed_psi(fields::ModelFields, context, options::ETDModelOptions)
    _fixed_flow(options) && return fields.psi
    workspace = context.workspace
    return _invert_total_q_to_psi!(scratch_like(workspace, fields.psi), fields,
        context.grid, options, context.plans, context.a, context.mask;
        workspace, project_nonlinear_state=true)
end

# Model-owned physical operator entry points use global geometry from the model
# and kernel-facing resources from its runtime.
function _operator_context(model::QGYBJModel)
    runtime = model.runtime
    runtime.finalized && error("cannot use operators after model finalization")
    coefficients = runtime.device_coefficients
    return (
        fields=model.fields,
        grid=runtime.geometry,
        plans=runtime.plans,
        workspace=runtime.workspace,
        mask=runtime.dealias_mask,
        f=model.physics.coriolis.f,
        N2=coefficients.N²,
        N2_face=coefficients.N²_face,
        a=coefficients.a_ell,
    )
end

"""Constant N² used when an operator falls back from its profile (host value)."""
_reference_N²(model::QGYBJModel) = first(model.runtime.coefficients.N²)

"""
    invert_q_to_psi!(model)

Invert the model's potential vorticity `q` for its balanced streamfunction ψ,
in place (`model.fields.psi`). With `WaveMeanFeedback` the wave PV qʷ is
removed first, so ψ is the flow of `q - qʷ`. A `FixedFlow` model has no such
inversion: its prescribed ψ is independent of q, and this is an error.
"""
function Elliptic.invert_q_to_psi!(model::QGYBJModel)
    context = _operator_context(model)
    options = ETDModelOptions(model.physics, model.numerics)
    _fixed_flow(options) && throw(ArgumentError(
        "a FixedFlow model's streamfunction is prescribed, not inverted from " *
        "q; set it with set!(model; ψ=...)"))
    _diagnostics_changed!(model)
    _invert_total_q_to_psi!(context.fields, context.grid, options,
        context.plans, context.a, context.mask;
        workspace=context.workspace, check_compatibility=true)
    return model
end

"""
    invert_B_to_A!(model)

Recover the wave amplitude A, and C = ∂A/∂z, from the envelope B, in place,
by the relation of the model's formulation: B = L⁺A for `YBJPlus()` and
`PassiveWave()`, and the solvability-constrained B = LA for `YBJ()`, whose
recovery also uses the velocity of the balanced flow, which is taken from q
as a step diagnoses it (from a `FixedFlow` model's prescribed ψ), not from
whatever `model.fields.psi` holds. Only A and C change.
"""
function Elliptic.invert_B_to_A!(model::QGYBJModel)
    _refresh_wave_diagnostics!(model)
    return model
end

function Elliptic.invert_helmholtz!(destination, rhs, model::QGYBJModel; kwargs...)
    context = _operator_context(model)
    invert_helmholtz!(destination, rhs, context.grid;
        workspace=context.workspace, kwargs...)
    return destination
end

"""
Wave-PV forcing of the mean-flow vertical velocity for a model with wave
feedback, or `nothing` (see `compute_vertical_velocity!`). It reads only ψ
and B, so it is the same whatever u, v, and A hold, and leaves them as they
are.
"""
function _wave_pv_forcing(model::QGYBJModel, context)
    options = ETDModelOptions(model.physics, model.numerics)
    _wave_feedback_enabled(options) || return nothing
    return D -> _wave_pv_derivative!(D, context.fields, context.grid, options,
        context.plans, context.a, context.mask; workspace=context.workspace,
        N2_profile=context.N2, N2_face_profile=context.N2_face)
end

"""
    compute_velocities!(model; compute_w=true)

Diagnose the balanced (Lagrangian-mean) velocity from the model's
streamfunction into `model.fields.u` and `v`, and, with `compute_w`, its
vertical velocity `w` from the QG omega equation (with the wave-PV forcing
when the model has `WaveMeanFeedback`); otherwise `w = 0`. Uses the model's
own coefficients, stratification profile, and dealiasing.
"""
function Operators.compute_velocities!(model::QGYBJModel;
    compute_w::Bool=true, use_ybj_w::Bool=false)
    Operators._reject_use_ybj_w(use_ybj_w)
    context = _operator_context(model)
    _diagnostics_changed!(model)
    compute_velocities!(context.fields, context.grid;
        plans=context.plans, f=context.f, N2=_reference_N²(model),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, workspace=context.workspace,
        dealias_mask=context.mask, pv_forcing=_wave_pv_forcing(model, context))
    return model
end

"""
    compute_vertical_velocity!(model)

Diagnose the balanced flow's vertical velocity `model.fields.w` from the QG
omega equation, leaving `u` and `v` as they are (see
[`compute_velocities!`](@ref)). The wave-PV forcing of a model with
`WaveMeanFeedback` uses the velocity of ψ, whatever `u` and `v` hold.
"""
function Operators.compute_vertical_velocity!(model::QGYBJModel)
    _diagnostics_changed!(model)
    return _compute_vertical_velocity!(model)
end

"""
`compute_vertical_velocity!(model)` without the record that a diagnostic
field changed: w is the one field that no step, energy, or particle velocity
reads, so the output of a run fills it between two steps and the next step
still reuses the others.
"""
function _compute_vertical_velocity!(model::QGYBJModel)
    context = _operator_context(model)
    compute_vertical_velocity!(context.fields, context.grid, context.plans;
        f=context.f, N2=_reference_N²(model), N2_profile=context.N2,
        N2_face_profile=context.N2_face,
        workspace=context.workspace, dealias_mask=context.mask,
        pv_forcing=_wave_pv_forcing(model, context))
    return model
end


"""
    compute_ybj_vertical_velocity!(model; t=nothing, skip_inversion=false)

The vertical velocity of the waves at time `t` (at phase zero without `t`),
from continuity with the wave velocity `e^{-ift} LA` (Asselin & Young 2019,
eq. 2.10), into `model.fields.w`. A is first recovered from B by the model's
formulation (see [`invert_B_to_A!`](@ref)), unless `skip_inversion` reuses
the current A. The flow fields ψ, u, and v do not change.
"""
function Operators.compute_ybj_vertical_velocity!(model::QGYBJModel;
    skip_inversion::Bool=false, t=nothing)
    context = _operator_context(model)
    _diagnostics_changed!(model)
    # A is recovered by the formulation's own relation: B = L⁺A for YBJ⁺ and
    # passive waves, the solvability-constrained B = LA for YBJ.
    skip_inversion || _refresh_wave_diagnostics!(model)
    compute_ybj_vertical_velocity!(context.fields, context.grid, context.plans;
        f=context.f, N2=_reference_N²(model), N2_profile=context.N2,
        N2_face_profile=context.N2_face,
        workspace=context.workspace, skip_inversion=true, t)
    return model
end

"""
    compute_total_velocities!(model; t, compute_w=true, include_wave_velocity=true)

The velocity at a fixed point into `model.fields.u`, `v`, and `w`: the
balanced (Lagrangian-mean) velocity, less the waves' Stokes drift, plus the
wave velocity `e^{-ift} LA` at time `t`. With `include_wave_velocity=false`
it is the Eulerian-mean velocity, and `t` is not needed.
"""
function Operators.compute_total_velocities!(model::QGYBJModel; t=nothing,
    compute_w::Bool=true, use_ybj_w::Bool=false,
    include_wave_velocity::Bool=true)
    context = _operator_context(model)
    _diagnostics_changed!(model)
    compute_total_velocities!(context.fields, context.grid; t,
        plans=context.plans, f=context.f, N2=_reference_N²(model),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, use_ybj_w, include_wave_velocity,
        formulation=model.physics.formulation,
        workspace=context.workspace, dealias_mask=context.mask,
        pv_forcing=_wave_pv_forcing(model, context))
    return model
end

"""
    compute_wave_velocities!(model; t, compute_w=true, include_wave_velocity=true)

Change the velocity already in `model.fields` by the waves: subtract their
Stokes drift and, with `include_wave_velocity`, add the wave velocity at time
`t`. Called directly after [`compute_velocities!`](@ref), it gives the same
fields as [`compute_total_velocities!`](@ref); called after anything else, it
adds to whatever is there.
"""
function Operators.compute_wave_velocities!(model::QGYBJModel; t=nothing,
    compute_w::Bool=true, include_wave_velocity::Bool=true)
    context = _operator_context(model)
    _diagnostics_changed!(model)
    compute_wave_velocities!(context.fields, context.grid; t,
        plans=context.plans, f=context.f, N2=_reference_N²(model),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, include_wave_velocity,
        formulation=model.physics.formulation, workspace=context.workspace)
    return model
end

"""Advance `model` by one model-owned exponential Runge-Kutta step."""
step!(model::QGYBJModel, timestepper::ExponentialRungeKutta2) =
    _step!(model, timestepper, :new)

"""
One step of `model`, which starts from the state that `start` describes (see
`_advance_etdrk2!`), or from less when the model's operators or setters have
written to its fields since its last step (see `_diagnostics_changed!`):
`run!` passes `:diagnosed` for a step that follows another of its steps, and
the step then reuses whatever the model still holds of that one. A direct
write to `model.fields` is not recorded, so only a caller that rules those
out, as `run!` does between its own steps, may pass more than `:new`.
"""
function _step!(model::QGYBJModel, timestepper::ExponentialRungeKutta2,
                start::Symbol)
    Δt = timestepper.Δt
    isfinite(Δt) && Δt > zero(Δt) ||
        throw(ArgumentError("Δt must be finite and positive (got $Δt)"))

    start in _STEP_STATES ||
        throw(ArgumentError("start must be :new, :projected, or :diagnosed"))
    runtime = model.runtime
    _step_state_rank(runtime.step_state) < _step_state_rank(start) &&
        (start = runtime.step_state)
    context = _operator_context(model)
    # A step that fails leaves fields no later step may reuse.
    runtime.step_state = :new
    timestep_workspace = timestepper.workspace
    owner_changed = timestepper.workspace_owner !== nothing &&
                    timestepper.workspace_owner !== model.runtime
    if timestep_workspace === nothing || owner_changed
        timestep_workspace = ExponentialRungeKutta2Workspace(
            model.fields, context.plans; G=context.grid)
    elseif !(timestep_workspace isa ExponentialRungeKutta2Workspace)
        throw(ArgumentError("timestepper workspace has an incompatible type"))
    elseif !_etdrk2_workspace_matches(timestep_workspace, model.fields) ||
           !_etdrk2_workspace_belongs_to(timestep_workspace, context.grid)
        timestep_workspace = ExponentialRungeKutta2Workspace(
            model.fields, context.plans; G=context.grid)
    end
    timestepper.workspace = timestep_workspace
    timestep_workspace isa ExponentialRungeKutta2Workspace ||
        throw(ArgumentError("timestepper workspace has an incompatible type"))
    timestepper.workspace_owner = model.runtime

    options = ETDModelOptions(model.physics, model.numerics)
    _advance_etdrk2!(model.fields, context.grid, options, context.plans;
        Δt,
        a=context.a,
        dealias_mask=context.mask,
        workspace=context.workspace,
        N2_profile=context.N2,
        N2_face_profile=context.N2_face,
        timestep_workspace=timestep_workspace, start)
    runtime.step_state = :diagnosed
    return model
end
