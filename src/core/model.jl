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
    QGYBJModel(; grid, coriolis, stratification, closure, ...)

Build a model transactionally. Public symbols and booleans are converted to
typed components before MPI or FFT resources are allocated.
"""
function QGYBJModel(; grid::RectilinearGrid,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure::AbstractClosure=HorizontalHyperdiffusivity(),
    vertical_diffusion=VerticalDiffusivity(),
    flow=:evolving,
    feedback=:none,
    formulation=nothing,
    ybj_plus::Bool=true,
    inviscid=false,
    linear=false,
    no_dispersion=false,
    topology=nothing,
    parallel_io::Bool=false,
    verbose::Bool=true)

    coriolis_component = _coriolis_component(coriolis)
    stratification_component = _stratification_component(stratification)
    flow_component = _flow_component(flow)
    feedback_component = _feedback_component(feedback)
    formulation_component = _formulation_component(formulation, ybj_plus)
    vertical_component = vertical_diffusion isa VerticalDiffusivity ?
                         vertical_diffusion : VerticalDiffusivity(vertical_diffusion)

    physics = ModelPhysics(coriolis_component, stratification_component,
                           flow_component, feedback_component,
                           formulation_component)
    numerics = ModelNumerics(closure, vertical_component,
                             _dissipation_component(inviscid),
                             _dynamics_component(linear),
                             _dispersion_component(no_dispersion))

    runtime = nothing
    try
        runtime = build_runtime(grid, physics, numerics;
                                topology, parallel_io, verbose)
        fields = allocate_fields(grid, runtime)
        return QGYBJModel(grid, fields, physics, numerics, runtime)
    catch
        runtime === nothing || finalize_runtime!(runtime; synchronize=false)
        rethrow()
    end
end

"""Release resources owned by `model`; repeated calls are safe."""
finalize_model!(model::QGYBJModel) = (finalize_runtime!(model.runtime); model)

is_root(model::QGYBJModel) = model.runtime.mpi.is_root
nprocs(model::QGYBJModel) = model.runtime.mpi.nprocs

function Base.show(io::IO, model::QGYBJModel)
    print(io, "QGYBJModel(grid=$(model.grid.size), " *
              "ranks=$(model.runtime.mpi.nprocs))")
end
