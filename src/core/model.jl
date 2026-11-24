"""Focused physical choices owned by a `QGYBJModel`."""
struct ModelPhysics{C, S, F, B, W}
    coriolis::C
    stratification::S
    flow::F
    feedback::B
    formulation::W
end

"""Focused numerical choices owned by a `QGYBJModel`."""
struct ModelNumerics{C, V, D, N, P}
    closure::C
    vertical_diffusion::V
    dissipation::D
    dynamics::N
    dispersion::P
end

"""
    QGYBJModel

Composition-first owner of immutable geometry, typed physics and numerics,
model fields, and ephemeral runtime resources.
"""
mutable struct QGYBJModel{G, F, P, N, R}
    grid::G
    fields::F
    physics::P
    numerics::N
    runtime::R
    # `nothing` until particles are initialized, then a ParticleTracker that
    # holds a back-reference to this model. Typing the field concretely would
    # make the two types mutually recursive, so it stays untyped; it is read
    # once per step, never inside a kernel loop.
    particles::Any
end

_coriolis_component(coriolis::AbstractCoriolis) = coriolis
_coriolis_component(f::Real) = FPlane(f)

_stratification_component(stratification::AbstractStratification) = stratification
_stratification_component(stratification::StratificationProfile) = stratification
_stratification_component(N²::Real) = ConstantStratification(N²)

_flow_component(flow::FlowEvolution) = flow
_flow_component(flow::Bool) = flow ? FixedFlow() : EvolvingFlow()
_flow_component(flow::Symbol) = flow === :fixed ? FixedFlow() :
                                flow === :evolving ? EvolvingFlow() :
                                throw(ArgumentError("flow must be :fixed, :evolving, or a FlowEvolution"))

_feedback_component(feedback::FeedbackMode) = feedback
_feedback_component(feedback::Bool) = feedback ? WaveMeanFeedback() : NoFeedback()
function _feedback_component(feedback::Symbol)
    feedback in (:none, :off) && return NoFeedback()
    feedback in (:wave_mean, :on) && return WaveMeanFeedback()
    feedback === :no_wave_feedback && return NoWaveFeedback()
    throw(ArgumentError("feedback must be :none, :wave_mean, :no_wave_feedback, or a FeedbackMode"))
end

function _formulation_component(formulation, ybj_plus)
    formulation isa WaveFormulation && return formulation
    formulation === :ybj_plus && return YBJPlus()
    formulation === :ybj && return YBJ()
    formulation === :passive && return PassiveWave()
    formulation === nothing ||
        throw(ArgumentError("formulation must be :ybj_plus, :ybj, :passive, or a WaveFormulation"))
    return ybj_plus ? YBJPlus() : YBJ()
end

_dissipation_component(mode::DissipationMode) = mode
_dissipation_component(inviscid::Bool) = inviscid ? Inviscid() : Dissipative()
_dynamics_component(mode::DynamicsMode) = mode
_dynamics_component(linear::Bool) = linear ? LinearDynamics() : NonlinearDynamics()
_dispersion_component(mode::DispersionMode) = mode
_dispersion_component(no_dispersion::Bool) = no_dispersion ? NoDispersion() : Dispersive()

"""
`closure` on `grid`: a `GridScaleHyperdiffusivity` takes the largest kₕ² that
the grid's dealiasing keeps; other closures are their own coefficients.
"""
_closure_on_grid(closure::AbstractClosure, grid::RectilinearGrid) = closure
_closure_on_grid(closure::GridScaleHyperdiffusivity, grid::RectilinearGrid) =
    GridScaleHyperdiffusivity(closure.order, closure.efold_steps,
                              maximum(grid.kh2[dealias_mask(grid)]))

"""
    QGYBJModel(; grid, coriolis=FPlane(f=1e-4),
               stratification=ConstantStratification(N²=1e-5), closure,
               vertical_diffusion=VerticalDiffusivity(), flow=EvolvingFlow(),
               feedback=NoFeedback(), formulation=YBJPlus(), inviscid=false,
               linear=false, no_dispersion=false, verbose=true)

Build a model on the architecture of `grid` (`CPU()` or `GPU()`), in the
precision of the grid. Public symbols and booleans are converted to typed
components before transform plans or fields are allocated:

- `coriolis`: an `FPlane`, or the Coriolis parameter f itself.
- `stratification`: a `ConstantStratification`, a [`StratificationProfile`](@ref),
  or the constant N² itself.
- `closure`: horizontal hyperdiffusion. The default,
  [`GridScaleHyperdiffusivity`](@ref)`()`, damps the smallest scale the grid
  keeps by one e-fold every 10 steps, in any units.
- `vertical_diffusion`: a `VerticalDiffusivity`, or its coefficient.
- `flow`: `EvolvingFlow()` (or `:evolving`), or `FixedFlow()` (`:fixed`,
  `true`) for a prescribed streamfunction.
- `feedback`: `NoFeedback()` (`:none`), `WaveMeanFeedback()` (`:wave_mean`),
  or `NoWaveFeedback()`.
- `formulation`: `YBJPlus()` (`:ybj_plus`), `YBJ()` (`:ybj`, or
  `ybj_plus=false`), or `PassiveWave()` (`:passive`).
- `inviscid`, `linear`, `no_dispersion`: `true` (or `Inviscid()`,
  `LinearDynamics()`, `NoDispersion()`) removes dissipation, the nonlinear
  terms, or wave dispersion.
- `verbose`: report the architecture and size.
"""
function QGYBJModel(; grid::RectilinearGrid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure::Union{Nothing, AbstractClosure}=nothing,
    vertical_diffusion=VerticalDiffusivity(),
    flow=:evolving,
    feedback=:none,
    formulation=nothing,
    ybj_plus::Bool=true,
    inviscid=false,
    linear=false,
    no_dispersion=false,
    verbose::Bool=true)

    coriolis_component = _coriolis_component(coriolis)
    stratification_component = _stratification_component(stratification)
    flow_component = _flow_component(flow)
    feedback_component = _feedback_component(feedback)
    formulation_component = _formulation_component(formulation, ybj_plus)
    vertical_component = vertical_diffusion isa VerticalDiffusivity ?
                         vertical_diffusion : VerticalDiffusivity(vertical_diffusion)
    flow_component isa FixedFlow && feedback_component isa WaveMeanFeedback &&
        @warn "feedback=WaveMeanFeedback() has no effect with flow=FixedFlow(): " *
              "the prescribed flow does not respond to the waves, so the model " *
              "runs without wave feedback. Use flow=EvolvingFlow() for two-way " *
              "coupling."

    physics = ModelPhysics(coriolis_component, stratification_component,
                           flow_component, feedback_component,
                           formulation_component)
    closure_component = _closure_on_grid(
        closure === nothing ? GridScaleHyperdiffusivity() : closure, grid)
    numerics = ModelNumerics(closure_component, vertical_component,
                             _dissipation_component(inviscid),
                             _dynamics_component(linear),
                             _dispersion_component(no_dispersion))

    runtime = build_runtime(grid, physics, numerics; verbose)
    fields = allocate_fields(grid, runtime)
    return QGYBJModel(grid, fields, physics, numerics, runtime, nothing)
end

"""
    finalize_model!(model)

Release the scratch buffers of `model`'s workspace and mark the model
finalized: it cannot be used afterwards. Repeated calls are safe.
"""
finalize_model!(model::QGYBJModel) = (finalize_runtime!(model.runtime); model)

function Base.show(io::IO, model::QGYBJModel)
    print(io, "QGYBJModel(grid=$(model.grid.size), " *
              "architecture=$(model.runtime.architecture))")
end
