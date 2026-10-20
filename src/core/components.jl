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

The no-argument constructor, also the model's default closure, keeps the
fourth- and twelfth-order terms of nondimensional setups: coefficients
`(0.01, 10.0)` for the flow and `(0.0, 10.0)` for the waves. With dimensional
wavenumbers (kₕ ~ 10⁻⁴ m⁻¹) they damp nothing, so a dimensional run should
give its own coefficients, for instance from [`compute_hyperdiff_coeff`](@ref):

```julia
ν = compute_hyperdiff_coeff(dx=grid.dx, dy=grid.dy, dt=Δt, order=8)
closure = HorizontalHyperdiffusivity(flow=FlowHyperdiffusivity(coefficient=ν, order=8),
                                     wave=WaveHyperdiffusivity(coefficient=ν, order=8))
```
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

"""
    SurfaceWave(; amplitude, scale, profile=:gaussian)

Horizontally uniform, surface-confined wave initial condition for
[`set!`](@ref): `amplitude` times `exp(-d²/scale²)` (`profile=:gaussian`) or
`exp(-d/scale)` (`:exponential`), with `d` the depth of each cell's upper face
below the top of the domain, its rigid lid (so the top cell has the full
amplitude). Its depth mean is removed when it is set, since the wave velocity
has no barotropic part.
"""
struct SurfaceWave{T}
    amplitude::T
    scale::T
    profile::Symbol
end

"""
    RandomStreamfunction(; amplitude, spectral_slope=-3, seed=0, profile=nothing)

A random balanced flow for [`set!`](@ref): a streamfunction whose horizontal
velocity has root-mean-square `amplitude` over the domain (m/s in a
dimensional setup). Every horizontal Fourier mode within the two-thirds
dealiasing radius has a random phase and a variance proportional to
`|k|^spectral_slope`, with `|k|` the mode index. The field is the same at
every level, or proportional to `profile(z)` at the level `z` when a profile
function is given. The same `seed` gives the same field on every
architecture, up to rounding in a GPU's math library.
"""
struct RandomStreamfunction{T, P}
    amplitude::T
    spectral_slope::T
    seed::Int
    profile::P
end

"""
    FieldArray(values; space=:physical, layout=:zxy, x=nothing, y=nothing, z=nothing)

A three-dimensional field supplied directly to [`set!`](@ref), on the model's
grid or, when the coordinate vectors `x`, `y`, and `z` of its points are
given (in the grid's coordinates, z positive upward), on any rectilinear
grid, from which it is interpolated onto the model's. `layout` is the order
of the array's dimensions, `:zxy` or `:xyz`. Its values must be finite.
"""
struct FieldArray{A<:AbstractArray, C}
    values::A
    space::Symbol
    layout::Symbol
    coordinates::C
end

function FieldArray(values::AbstractArray;
    space::Symbol=:physical, layout::Symbol=:zxy,
    x=nothing, y=nothing, z=nothing)

    ndims(values) == 3 ||
        throw(ArgumentError("field arrays must be three-dimensional"))
    space in (:physical, :spectral) ||
        throw(ArgumentError("field space must be :physical or :spectral"))
    layout in (:zxy, :xyz) ||
        throw(ArgumentError("field layout must be :zxy or :xyz"))
    given = count(!isnothing, (x, y, z))
    given in (0, 3) ||
        throw(ArgumentError("give all three coordinates x, y, and z, or none"))
    given == 0 && return FieldArray(values, space, layout, nothing)

    space === :physical || throw(ArgumentError(
        "only physical fields can be interpolated; give spectral fields on the model's grid"))
    coordinates = (x=_real_values(x, "x", "the FieldArray"),
                   y=_real_values(y, "y", "the FieldArray"),
                   z=_real_values(z, "z", "the FieldArray"))
    xyz_size = layout === :xyz ? size(values) : size(values)[[2, 3, 1]]
    xyz_size == map(length, Tuple(coordinates)) || throw(DimensionMismatch(
        "the FieldArray has size $xyz_size along x, y, and z, but its " *
        "coordinates have lengths $(map(length, Tuple(coordinates)))"))
    return FieldArray(values, space, layout, coordinates)
end

"""
    FieldFile(path, variable; space=:physical, layout=:xyz, group=nothing)

A three-dimensional field supplied to [`set!`](@ref), read from a NetCDF file,
or from a JLD2 file when `path` ends in `.jld2`. `group` names the NetCDF
group, or the JLD2 group, NamedTuple, or Dict, that holds `variable`.

The dimensions of a NetCDF variable are put in order by their names when
they name the axes (`x`, `y`, and `z` or `depth`, or common alternatives such
as `lon`, `lat`, and `lev`); otherwise, and for JLD2 arrays, `layout` gives
their order (`:xyz` or `:zxy`). Extra dimensions of length one, such as a
single time, are dropped. Missing (fill) and non-finite values are errors.

When the same place holds coordinate vectors `x`, `y`, and `z` (or `depth`),
a field on another grid is interpolated onto the model's. `z` is in the
grid's coordinates, positive upward, and `depth` is measured down from the
top of the grid, its lid; a `z` with no point below the lid, and some above
it, is read as depth, as in earlier versions.
"""
struct FieldFile
    path::String
    variable::String
    space::Symbol
    layout::Symbol
    group::Union{Nothing, String}
end

function FieldFile(path::AbstractString, variable::AbstractString;
    space::Symbol=:physical, layout::Symbol=:xyz,
    group::Union{Nothing, AbstractString}=nothing)

    isempty(path) && throw(ArgumentError("field-file path cannot be empty"))
    isempty(variable) && throw(ArgumentError("field-file variable cannot be empty"))
    space in (:physical, :spectral) ||
        throw(ArgumentError("field space must be :physical or :spectral"))
    layout in (:zxy, :xyz) ||
        throw(ArgumentError("field layout must be :zxy or :xyz"))
    return FieldFile(String(path), String(variable), space, layout,
                     group === nothing ? nothing : String(group))
end


function RandomStreamfunction(; amplitude::Real,
    spectral_slope::Real=-3.0, seed::Integer=0, profile=nothing)
    values = float.((amplitude, spectral_slope))
    all(isfinite, values) ||
        throw(ArgumentError("random streamfunction parameters must be finite"))
    values[1] >= 0 || throw(ArgumentError("amplitude must be non-negative"))
    profile === nothing || profile isa Function ||
        throw(ArgumentError("profile must be a function of z, or nothing"))
    T = promote_type(map(typeof, values)...)
    return RandomStreamfunction{T, typeof(profile)}(
        T(amplitude), T(spectral_slope), Int(seed), profile)
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

"""
    TimeInterval(interval)

Schedule output or diagnostics every `interval` of model time (seconds in a
dimensional setup). The scheduled times are the multiples of `interval` from
t = 0, so a restored run keeps the times of the original one; each is written
at the first step that reaches it. The interval is kept in `Float64`, like
the simulation clock.
"""
struct TimeInterval{T<:AbstractFloat} <: AbstractSchedule
    interval::T

    function TimeInterval{T}(interval::Real) where T<:AbstractFloat
        value = T(interval)
        isfinite(value) || throw(ArgumentError("interval must be finite (got $interval)"))
        value > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
        return new{T}(value)
    end
end

TimeInterval(interval::Real) = TimeInterval{Float64}(interval)

"""
    IterationInterval(interval)

Schedule output or diagnostics every `interval` model iterations.
"""
struct IterationInterval <: AbstractSchedule
    interval::Int

    function IterationInterval(interval::Integer)
        interval > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
        return new(Int(interval))
    end
end

"""
    _next_deadline(time, interval)

The first multiple of `interval` after `time`, up to rounding. Every
`TimeInterval` schedule lies on this grid of multiples from t = 0, so a
restored run keeps the original output times, and a deadline computed with
one multiplication does not drift as a running sum would.
"""
function _next_deadline(time, interval)
    tolerance = 8eps(max(abs(time), abs(interval), one(time)))
    return (floor((time + tolerance) / interval) + 1) * interval
end

"""
    NetCDFOutput(; path="output", schedule=nothing, fields=(:ψ, :waves), velocities=false, overwrite=false)

Declarative NetCDF output settings: one state file `stateNNNN.nc` in `path` per
write. `schedule` may be a [`TimeInterval`](@ref) or
[`IterationInterval`](@ref); with `schedule=nothing` only the initial and
final states are written.

Every file holds the prognostic `q` and `B` needed for a restart (and ψ for a
`FixedFlow` model). `fields` adds physical-space fields: `:ψ` (or `:psi`,
`:q`, `:flow`) for the streamfunction, `:waves` (or `:wave`, `:B`, `:A`) for
the wave amplitude `A` and velocity `LA`, and `:velocities` for u, v, and w,
which `velocities=true` also adds.

A run never silently replaces state files already in `path`. A simulation
restored with [`restore!`](@ref) continues the numbering after the files up
to its restart iteration. Any other existing files raise an error unless
`overwrite=true`, which removes them: all state files for a run that starts
at iteration 0, and the files after the restart iteration for a restored run.
"""
struct NetCDFOutput{S}
    path::String
    schedule::S
    fields::Tuple
    velocities::Bool
    overwrite::Bool
end

function NetCDFOutput(; path::AbstractString="output", schedule=nothing,
    fields=(:ψ, :waves), velocities::Bool=false, overwrite::Bool=false)

    schedule === nothing || schedule isa AbstractSchedule ||
        throw(ArgumentError("schedule must be a TimeInterval or IterationInterval"))
    isempty(path) && throw(ArgumentError("output path cannot be empty"))
    output_fields = fields isa Symbol ? (fields,) : Tuple(fields)
    unknown = filter(field -> !(field in _OUTPUT_FIELDS), output_fields)
    isempty(unknown) || throw(ArgumentError(
        "unknown output field(s) $(join(repr.(unknown), ", ")); choose from " *
        ":ψ (or :psi, :q, :flow), :waves (or :wave, :B, :A), and :velocities"))
    return NetCDFOutput(String(path), schedule, output_fields, velocities,
                        overwrite)
end

"""Field names that [`NetCDFOutput`](@ref) accepts."""
const _OUTPUT_FIELDS = (:ψ, :psi, :q, :flow, :waves, :wave, :B, :A, :velocities)

"""
    EnergyDiagnosticsOutput(; path="output/diagnostic", schedule, overwrite=false)

Configuration for simulation-owned energy time-series files. `path` is the
diagnostic directory itself, and `schedule` may be time- or iteration-based.
The files are rewritten as the run goes, at most every few seconds of wall
time, and once more when it ends.

A restored simulation appends to the series already in `path`. Any other
existing series raises an error unless `overwrite=true`, which replaces it
(for a restored run, only the records after the restart time).
"""
struct EnergyDiagnosticsOutput{S<:AbstractSchedule}
    path::String
    schedule::S
    overwrite::Bool
end

function EnergyDiagnosticsOutput(; path::AbstractString="output/diagnostic",
    schedule::AbstractSchedule=IterationInterval(1), overwrite::Bool=false)

    isempty(path) && throw(ArgumentError("diagnostic path cannot be empty"))
    return EnergyDiagnosticsOutput(String(path), schedule, overwrite)
end
