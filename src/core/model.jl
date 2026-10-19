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
Warn when the default closure barely damps the smallest scale the grid keeps.
Its coefficients come from nondimensional setups: with dimensional
wavenumbers (kₕ ~ 10⁻⁴ m⁻¹) they damp nothing, and a nonlinear run is
effectively inviscid. The damping is measured against the inertial period,
the one time scale every model has.
"""
function _warn_inert_default_closure(grid::RectilinearGrid,
                                     closure::HorizontalHyperdiffusivity, f::Real)
    kept = dealias_mask(grid)
    k² = maximum(grid.kh2[kept])
    iszero(k²) && return nothing
    rate = max(Nonlinear._hyperdiffusion_exponent(closure.flow, k², 1.0),
               Nonlinear._hyperdiffusion_exponent(closure.wave, k², 1.0))
    rate < 1e-6 * abs(f) || return nothing
    periods = abs(f) / (2π * rate)
    @warn "The default closure, HorizontalHyperdiffusivity(), barely damps this " *
          "grid: the smallest scale it keeps, at kₕ = $(round(sqrt(k²); sigdigits=3)), " *
          "decays by e in $(round(periods; sigdigits=3)) inertial periods. Its " *
          "coefficients suit nondimensional setups. For a dimensional one, pass a " *
          "closure with coefficients from compute_hyperdiff_coeff, or " *
          "inviscid=Inviscid() to run without horizontal damping deliberately." maxlog=1
    return nothing
end

"""
    QGYBJModel(; grid, coriolis, stratification, closure, ...)

Build a model on the architecture of `grid` (`CPU()` or `GPU()`). Public
symbols and booleans are converted to typed components before transform plans
or fields are allocated.

`closure` defaults to `HorizontalHyperdiffusivity()`, whose coefficients suit
nondimensional setups; with `verbose`, a nonlinear, dissipative model warns
when they barely damp its grid, as they do for dimensional wavenumbers.
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
    closure_component = closure === nothing ? HorizontalHyperdiffusivity() : closure
    numerics = ModelNumerics(closure_component, vertical_component,
                             _dissipation_component(inviscid),
                             _dynamics_component(linear),
                             _dispersion_component(no_dispersion))
    closure === nothing && verbose && numerics.dissipation isa Dissipative &&
        numerics.dynamics isa NonlinearDynamics &&
        _warn_inert_default_closure(grid, closure_component, coriolis_component.f)

    runtime = build_runtime(grid, physics, numerics; verbose)
    fields = allocate_fields(grid, runtime)
    return QGYBJModel(grid, fields, physics, numerics, runtime, nothing)
end

"""Release resources owned by `model`; repeated calls are safe."""
finalize_model!(model::QGYBJModel) = (finalize_runtime!(model.runtime); model)

function Base.show(io::IO, model::QGYBJModel)
    print(io, "QGYBJModel(grid=$(model.grid.size), " *
              "architecture=$(model.runtime.architecture))")
end
