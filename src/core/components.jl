# Focused, immutable configuration objects used by the composition-first API.

abstract type AbstractCoriolis end
abstract type AbstractStratification end
abstract type AbstractClosure end
abstract type FlowEvolution end
abstract type FeedbackMode end
abstract type WaveFormulation end
abstract type DissipationMode end
abstract type DynamicsMode end
abstract type DispersionMode end
abstract type AbstractSchedule end

"""Constant Coriolis parameter for an f-plane model."""
struct FPlane{T} <: AbstractCoriolis
    f::T
end

FPlane(; f::Real) = FPlane(f)

function FPlane(f::Real)
    value = float(f)
    isfinite(value) || throw(ArgumentError("f must be finite (got $f)"))
    !iszero(value) || throw(ArgumentError("f must be non-zero"))
    return FPlane{typeof(value)}(value)
end

"""Constant buoyancy frequency squared `N²` in s⁻²."""
struct ConstantStratification{T} <: AbstractStratification
    N²::T
end

ConstantStratification(; N²::Real) = ConstantStratification(N²)

function ConstantStratification(N²::Real)
    value = float(N²)
    isfinite(value) || throw(ArgumentError("N² must be finite (got $N²)"))
    value > zero(value) || throw(ArgumentError("N² must be positive (got $N²)"))
    return ConstantStratification{typeof(value)}(value)
end

"""Hold the balanced flow fixed while the wave field evolves."""
struct FixedFlow <: FlowEvolution end

"""Evolve the balanced flow together with the wave field."""
struct EvolvingFlow <: FlowEvolution end

"""Disable all wave–mean-flow coupling."""
struct NoFeedback <: FeedbackMode end

"""Enable bidirectional wave–mean-flow coupling."""
struct WaveMeanFeedback <: FeedbackMode end

"""Let the flow affect waves without wave feedback on potential vorticity."""
struct NoWaveFeedback <: FeedbackMode end

"""Use the elliptically regularized YBJ⁺ wave formulation."""
struct YBJPlus <: WaveFormulation end

"""Use the original Young–Ben Jelloul wave formulation."""
struct YBJ <: WaveFormulation end

"""Advect the wave envelope as a passive field."""
struct PassiveWave <: WaveFormulation end

"""Apply the configured horizontal and vertical dissipative closures."""
struct Dissipative <: DissipationMode end

"""Disable all dissipative terms."""
struct Inviscid <: DissipationMode end

"""Retain nonlinear advection and wave–flow interactions."""
struct NonlinearDynamics <: DynamicsMode end

"""Disable nonlinear advection terms."""
struct LinearDynamics <: DynamicsMode end

"""Retain wave dispersion."""
struct Dispersive <: DispersionMode end

"""Disable wave dispersion."""
struct NoDispersion <: DispersionMode end

"""Vertical diffusivity for balanced potential vorticity."""
struct VerticalDiffusivity{T}
    coefficient::T
end

function VerticalDiffusivity(coefficient::Real)
    value = float(coefficient)
    isfinite(value) || throw(ArgumentError("vertical diffusivity must be finite"))
    value >= 0 || throw(ArgumentError("vertical diffusivity must be non-negative"))
    return VerticalDiffusivity{typeof(value)}(value)
end

VerticalDiffusivity(; coefficient::Real=0) = VerticalDiffusivity(coefficient)

"""Horizontal hyperdiffusion coefficients for the balanced flow and waves."""
struct HorizontalHyperdiffusivity{T} <: AbstractClosure
    flow::T
    flow2::T
    flow_laplacian_order::Int
    flow_laplacian_order2::Int
    waves::T
    waves2::T
    wave_laplacian_order::Int
    wave_laplacian_order2::Int
end

function HorizontalHyperdiffusivity(; flow::Real=0.01, flow2::Real=10.0,
    flow_laplacian_order::Int=2, flow_laplacian_order2::Int=6,
    waves::Real=0.0, waves2::Real=10.0,
    wave_laplacian_order::Int=2, wave_laplacian_order2::Int=6)

    coefficients = float.((flow, flow2, waves, waves2))
    all(isfinite, coefficients) ||
        throw(ArgumentError("hyperdiffusion coefficients must be finite"))
    all(>=(0), coefficients) ||
        throw(ArgumentError("hyperdiffusion coefficients must be non-negative"))

    orders = (flow_laplacian_order, flow_laplacian_order2,
              wave_laplacian_order, wave_laplacian_order2)
    all(>(0), orders) || throw(ArgumentError("Laplacian orders must be positive"))

    T = promote_type(map(typeof, coefficients)...)
    return HorizontalHyperdiffusivity{T}(
        T(flow), T(flow2), flow_laplacian_order, flow_laplacian_order2,
        T(waves), T(waves2), wave_laplacian_order, wave_laplacian_order2)
end

"""Horizontally uniform, surface-confined wave initial condition."""
struct SurfaceWave{T}
    amplitude::T
    scale::T
    profile::Symbol
end

function SurfaceWave(; amplitude::Real, scale::Real, profile::Symbol=:gaussian)
    values = float.((amplitude, scale))
    all(isfinite, values) || throw(ArgumentError("wave amplitude and scale must be finite"))
    values[2] > 0 || throw(ArgumentError("wave scale must be positive (got $scale)"))
    profile in (:gaussian, :exponential) ||
        throw(ArgumentError("profile must be :gaussian or :exponential"))

    T = promote_type(map(typeof, values)...)
    return SurfaceWave{T}(T(amplitude), T(scale), profile)
end

"""Schedule output or diagnostics at a time interval measured in seconds."""
struct TimeInterval{T} <: AbstractSchedule
    interval::T
end

function TimeInterval(interval::Real)
    value = float(interval)
    isfinite(value) || throw(ArgumentError("interval must be finite (got $interval)"))
    value > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
    return TimeInterval{typeof(value)}(value)
end

"""Schedule output or diagnostics every `interval` model iterations."""
struct IterationInterval <: AbstractSchedule
    interval::Int
end

function IterationInterval(interval::Integer)
    interval > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
    return IterationInterval(Int(interval))
end

"""
    NetCDFOutput(; path="output", schedule=nothing, fields=(:ψ, :waves), velocities=false)

Declarative NetCDF output settings. `schedule` may be a [`TimeInterval`](@ref)
or [`IterationInterval`](@ref).
"""
struct NetCDFOutput{S}
    path::String
    schedule::S
    fields::Tuple
    velocities::Bool
end

function NetCDFOutput(; path::AbstractString="output", schedule=nothing,
    fields=(:ψ, :waves), velocities::Bool=false)

    schedule === nothing || schedule isa AbstractSchedule ||
        throw(ArgumentError("schedule must be a TimeInterval or IterationInterval"))
    isempty(path) && throw(ArgumentError("output path cannot be empty"))
    output_fields = fields isa Symbol ? (fields,) : Tuple(fields)
    return NetCDFOutput(String(path), schedule, output_fields, velocities)
end
