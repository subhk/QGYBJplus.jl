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

"""Omit wave potential-vorticity feedback from the balanced-flow inversion."""
struct NoFeedback <: FeedbackMode end

"""Enable bidirectional wave–mean-flow coupling."""
struct WaveMeanFeedback <: FeedbackMode end

"""Express one-way flow-to-wave coupling without wave PV feedback."""
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

"""
    FlowHyperdiffusivity(; coefficient, order=4)

Horizontal damping for the balanced flow only. `order` is the total derivative
order and must be positive and even. The wave field is not damped by this
closure.
"""
struct FlowHyperdiffusivity{T, N} <: AbstractClosure
    coefficients::NTuple{N, T}
    orders::NTuple{N, Int}

    function FlowHyperdiffusivity{T, N}(
        coefficients::NTuple{N, T},
        orders::NTuple{N, Int},
    ) where {T, N}
        return new{T, N}(coefficients, orders)
    end
end

function _hyperdiffusivity_terms(field::Symbol, coefficients::Tuple, orders::Tuple)
    isempty(coefficients) && throw(ArgumentError(
        "$field hyperdiffusivity requires at least one term"))
    length(coefficients) == length(orders) || throw(ArgumentError(
        "$field hyperdiffusivity coefficients and orders must have equal lengths"))
    all(coefficient -> coefficient isa Real, coefficients) || throw(ArgumentError(
        "$field hyperdiffusivity coefficients must be real"))
    all(order -> order isa Integer, orders) || throw(ArgumentError(
        "$field hyperdiffusivity orders must be integers"))

    values = float.(coefficients)
    all(value -> isfinite(value) && value >= 0, values) || throw(ArgumentError(
        "$field hyperdiffusivity coefficients must be finite and non-negative"))

    derivative_orders = Int.(orders)
    all(order -> order > 0 && iseven(order), derivative_orders) ||
        throw(ArgumentError(
            "$field hyperdiffusivity orders must be positive and even"))

    T = promote_type(map(typeof, values)...)
    N = length(values)
    return ntuple(index -> T(values[index]), N), derivative_orders
end

function FlowHyperdiffusivity(coefficients::Tuple, orders::Tuple)
    values, derivative_orders =
        _hyperdiffusivity_terms(:flow, coefficients, orders)
    T = typeof(first(values))
    N = length(values)
    return FlowHyperdiffusivity{T, N}(values, derivative_orders)
end

FlowHyperdiffusivity(coefficient::Real; order::Int=4) =
    FlowHyperdiffusivity((coefficient,), (order,))

FlowHyperdiffusivity(; coefficient::Real, order::Int=4) =
    FlowHyperdiffusivity(coefficient; order)

"""
    WaveHyperdiffusivity(; coefficient, order=4)

Horizontal damping for the wave field only. `order` is the total derivative
order and must be positive and even. The balanced flow is not damped by this
closure.
"""
struct WaveHyperdiffusivity{T, N} <: AbstractClosure
    coefficients::NTuple{N, T}
    orders::NTuple{N, Int}

    function WaveHyperdiffusivity{T, N}(
        coefficients::NTuple{N, T},
        orders::NTuple{N, Int},
    ) where {T, N}
        return new{T, N}(coefficients, orders)
    end
end

function WaveHyperdiffusivity(coefficients::Tuple, orders::Tuple)
    values, derivative_orders =
        _hyperdiffusivity_terms(:wave, coefficients, orders)
    T = typeof(first(values))
    N = length(values)
    return WaveHyperdiffusivity{T, N}(values, derivative_orders)
end

WaveHyperdiffusivity(coefficient::Real; order::Int=4) =
    WaveHyperdiffusivity((coefficient,), (order,))

WaveHyperdiffusivity(; coefficient::Real, order::Int=4) =
    WaveHyperdiffusivity(coefficient; order)

"""
    HorizontalHyperdiffusivity(; flow, wave)

Horizontal damping composed from explicit balanced-flow and wave components.
The no-argument constructor retains the model's standard fourth- and
twelfth-order damping terms.
"""
struct HorizontalHyperdiffusivity{F<:FlowHyperdiffusivity,
                                  W<:WaveHyperdiffusivity} <: AbstractClosure
    flow::F
    wave::W
end

function HorizontalHyperdiffusivity(;
    flow::FlowHyperdiffusivity=FlowHyperdiffusivity(
        (0.01, 10.0), (4, 12)),
    wave::WaveHyperdiffusivity=WaveHyperdiffusivity(
        (0.0, 10.0), (4, 12)))

    return HorizontalHyperdiffusivity(flow, wave)
end

"""Horizontally uniform, surface-confined wave initial condition."""
struct SurfaceWave{T}
    amplitude::T
    scale::T
    profile::Symbol
end

"""Deterministic random streamfunction spectrum for model initialization."""
struct RandomStreamfunction{T}
    amplitude::T
    spectral_slope::T
    seed::Int
end

"""A three-dimensional field supplied directly to [`set!`](@ref)."""
struct FieldArray{A<:AbstractArray}
    values::A
    space::Symbol
    layout::Symbol
end

function FieldArray(values::AbstractArray;
    space::Symbol=:physical, layout::Symbol=:zxy)

    ndims(values) == 3 ||
        throw(ArgumentError("field arrays must be three-dimensional"))
    space in (:physical, :spectral) ||
        throw(ArgumentError("field space must be :physical or :spectral"))
    layout in (:zxy, :xyz) ||
        throw(ArgumentError("field layout must be :zxy or :xyz"))
    return FieldArray(values, space, layout)
end

"""A NetCDF-backed three-dimensional field supplied to [`set!`](@ref)."""
struct FieldFile
    path::String
    variable::String
    space::Symbol
    layout::Symbol
end

function FieldFile(path::AbstractString, variable::AbstractString;
    space::Symbol=:physical, layout::Symbol=:xyz)

    isempty(path) && throw(ArgumentError("field-file path cannot be empty"))
    isempty(variable) && throw(ArgumentError("field-file variable cannot be empty"))
    space in (:physical, :spectral) ||
        throw(ArgumentError("field space must be :physical or :spectral"))
    layout in (:zxy, :xyz) ||
        throw(ArgumentError("field layout must be :zxy or :xyz"))
    return FieldFile(String(path), String(variable), space, layout)
end


function RandomStreamfunction(; amplitude::Real=1.0,
    spectral_slope::Real=-3.0, seed::Integer=0)
    values = float.((amplitude, spectral_slope))
    all(isfinite, values) ||
        throw(ArgumentError("random streamfunction parameters must be finite"))
    values[1] >= 0 || throw(ArgumentError("amplitude must be non-negative"))
    T = promote_type(map(typeof, values)...)
    return RandomStreamfunction{T}(T(amplitude), T(spectral_slope), Int(seed))
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

"""
    EnergyDiagnosticsOutput(; path="output/diagnostic", schedule)

Configuration for simulation-owned energy time-series files. `path` is the
diagnostic directory itself, and `schedule` may be time- or iteration-based.
"""
struct EnergyDiagnosticsOutput{S<:AbstractSchedule}
    path::String
    schedule::S
end

function EnergyDiagnosticsOutput(; path::AbstractString="output/diagnostic",
    schedule::AbstractSchedule=IterationInterval(1))

    isempty(path) && throw(ArgumentError("diagnostic path cannot be empty"))
    return EnergyDiagnosticsOutput(String(path), schedule)
end
