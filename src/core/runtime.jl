"""Precomputed vertical coefficients used by model operators."""
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
    # read copies in the grid's precision.
    profile = _runtime_profile(physics.stratification)
    _warn_uncovered_depths(profile, grid)
    N² = Float64.(compute_stratification_profile(profile, grid))
    N²_face = Float64.(_compute_stratification_face_profile(profile, grid))
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
                        device_coefficients, mask, false)
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
recovery also uses the balanced flow's velocity.
"""
function Elliptic.invert_B_to_A!(model::QGYBJModel)
    context = _operator_context(model)
    if model.physics.formulation isa YBJ
        # The YBJ recovery diagnoses the velocity it needs, which clears w;
        # keep the caller's w, since the flow itself does not change.
        w = copy(model.fields.w)
        _refresh_wave_diagnostics!(model)
        copyto!(model.fields.w, w)
    else
        invert_B_to_A!(context.fields, context.grid, context.a;
            workspace=context.workspace)
    end
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
feedback, or `nothing` (see `compute_vertical_velocity!`).
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
[`compute_velocities!`](@ref)).
"""
function Operators.compute_vertical_velocity!(model::QGYBJModel)
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
the current A.
"""
function Operators.compute_ybj_vertical_velocity!(model::QGYBJModel;
    skip_inversion::Bool=false, t=nothing)
    context = _operator_context(model)
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
    compute_wave_velocities!(context.fields, context.grid; t,
        plans=context.plans, f=context.f, N2=_reference_N²(model),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, include_wave_velocity,
        formulation=model.physics.formulation, workspace=context.workspace)
    return model
end

"""Advance `model` by one model-owned exponential Runge-Kutta step."""
function step!(model::QGYBJModel, timestepper::ExponentialRungeKutta2)
    Δt = timestepper.Δt
    isfinite(Δt) && Δt > zero(Δt) ||
        throw(ArgumentError("Δt must be finite and positive (got $Δt)"))

    context = _operator_context(model)
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
        timestep_workspace=timestep_workspace)
    return model
end
