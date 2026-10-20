"""
Lagrangian particle tracking for QG-YBJ⁺ simulations.

Particles move horizontally, each at its own fixed depth, with the velocity
selected by `wave_motion` (see `ParticleConfig`): the Lagrangian-mean velocity
of the balanced flow (`:averaged`, the default), or the near-inertial wave
velocity added to the Lagrangian mean less the Stokes drift that the wave
orbits make up (`:instantaneous`).

Particle positions, the velocity that advects them, and the advection itself
live on the model's architecture (CPU or GPU): a particle step copies the
model's horizontal velocity within that architecture and advances every
particle in one kernel. Positions reach the host only when they are written
or recorded.

A particle step covers `[t, t + Δt]` and is taken after the model has been
advanced over the same interval, as `run!` does. The tracker keeps the
velocity at the end of the previous step as the velocity at the start of this
one, and every integration stage interpolates linearly in time between the
two. RK2 and RK4 trajectories are therefore second-order accurate in time,
like the model's ETD-RK2 flow.

Vertically, the velocity columns are mirrored at the lid and the bottom, where
the horizontal velocity has zero vertical derivative, so the interpolation is
smooth at every depth. Horizontal boundaries are periodic.

A positive `particle_advec_time` delays the release of the particles. It is
compared with the simulation time passed as `current_time`.
"""

module UnifiedParticleAdvection

using KernelAbstractions: @kernel, @index, @Const

# Bind names from parent module (QGYBJplus) without using/import
const _PARENT = Base.parentmodule(@__MODULE__)
const RuntimeGeometry = _PARENT.RuntimeGeometry
const ModelFields = _PARENT.ModelFields
const plan_transforms! = _PARENT.plan_transforms!
const compute_total_velocities! = _PARENT.compute_total_velocities!
const compute_velocities! = _PARENT.compute_velocities!
const compute_wave_velocities! = _PARENT.compute_wave_velocities!
const YBJPlus = _PARENT.YBJPlus
const AbstractArchitecture = _PARENT.AbstractArchitecture
const CPU = _PARENT.CPU
const launch! = _PARENT.launch!
const on_architecture = _PARENT.on_architecture
const zeros_on = _PARENT.zeros_on
const _next_deadline = _PARENT._next_deadline

"""Validate the particle wave-motion options."""
function _check_wave_motion(use_ybj_w::Bool, wave_motion::Symbol)
    use_ybj_w && throw(ArgumentError(
        "use_ybj_w was removed: use wave_motion=:instantaneous for trajectories " *
        "that include the wave velocity, or the default wave_motion=:averaged"))
    wave_motion in (:averaged, :instantaneous) || throw(ArgumentError(
        "wave_motion must be :averaged or :instantaneous (got :$wave_motion)"))
    return nothing
end

"""Particles move with the horizontal velocity: vertical advection was removed."""
function _check_horizontal_advection(use_3d_advection::Bool)
    use_3d_advection && throw(ArgumentError(
        "use_3d_advection was removed: particles move with the horizontal " *
        "velocity and keep their depth"))
    return nothing
end

export ParticleConfig, ParticleState, ParticleTracker,
       initialize_particles!,
       advect_particles!, interpolate_velocity_at_position,
       # Advanced interpolation methods
       InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
       interpolate_velocity_advanced,
       # 3D particle distributions
       ParticleConfig3D, ParticleDistribution,
       initialize_particles_3d!, UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM,
       # Simplified particle initialization
       particles_in_box, particles_in_circle, particles_in_grid_3d, particles_in_layers,
       particles_random_3d, particles_custom

include("interpolation_schemes.jl")
using .InterpolationSchemes

"""Validate the integration and interpolation methods of a particle configuration."""
function _check_particle_methods(integration_method::Symbol,
                                 interpolation_method::InterpolationMethod)
    integration_method in (:euler, :rk2, :rk4) || throw(ArgumentError(
        "integration_method must be :euler, :rk2, or :rk4 (got :$integration_method)"))
    interpolation_method == ADAPTIVE &&
        @warn "interpolation_method = ADAPTIVE is deprecated and now uses TRICUBIC"
    return nothing
end

# NOTE: particle_config.jl is included AFTER ParticleConfig struct definition below
# because it needs to access ParticleConfig from this module

"""
    ParticleConfig{T}(; x_min=0, x_max, y_min=0, y_max, z_level, kwargs...)

Configure a regular horizontal particle grid at `z_level`, with positions
stored in precision `T`. `nx_particles` and `ny_particles` default to 10.
Particles move with the horizontal velocity and keep their depth.

`particle_advec_time` delays the release until the simulation time reaches
it; once released, particles keep moving in later runs. It and
`save_interval` are times, kept in `Float64` whatever `T` is.

`integration_method` selects the trajectory integrator: `:euler`, `:rk2`
(default), or `:rk4`. Each integration stage interpolates the velocity
linearly in time between the start and the end of the particle step, so in an
evolving flow `:rk2` and `:rk4` are second-order accurate in time and `:euler`
first-order. `:rk4` is still more accurate per step where trajectories curve
strongly.

`interpolation_method` selects the spatial interpolation: `TRILINEAR`
(O(h²)), `TRICUBIC` (default; Catmull-Rom, O(h³)), or `QUINTIC` (degree-5
Lagrange, O(h⁶)). `ADAPTIVE` is deprecated and now uses `TRICUBIC`.

A finite `save_interval` makes the tracker keep an in-memory history of the
positions, up to `max_save_points` records, for
`write_particle_trajectories`: when a run starts and then at the multiples of
`save_interval` (every step for 0). The default, `Inf`, keeps none:
simulations write trajectories with a `ParticleOutputManager`.

`wave_motion` selects how particles feel the near-inertial waves:

- `:averaged` (default): phase-averaged trajectories, advected by the
  Lagrangian-mean velocity of the balanced flow, `∇⊥ψ`. As ψ is the
  Lagrangian-mean streamfunction (Xie & Vanneste 2015; Wagner & Young 2016),
  this already contains the waves' Stokes drift.
- `:instantaneous`: trajectories that follow the inertial oscillation, advected
  by the wave velocity `e^{-ift} LA`, with the phase evaluated at each
  integration stage, plus the Lagrangian mean minus the Stokes drift of
  horizontal orbits. The orbits make up that part of the drift themselves,
  so over an inertial period the particles drift with the Lagrangian mean, as
  with `:averaged`; the part of the Stokes drift from vertical excursions,
  which particles that keep their depth do not make, stays in the mean. The
  particle time step must resolve the inertial period (`f Δt ≲ 0.2`).

`use_ybj_w` and `use_3d_advection` were removed and must stay `false`.
"""
Base.@kwdef struct ParticleConfig{T<:AbstractFloat}
    # Spatial domain for particle initialization (x_max, y_max, z_level are required)
    x_min::T = 0.0
    x_max::T
    y_min::T = 0.0
    y_max::T
    z_level::T

    # Number of particles
    nx_particles::Int = 10
    ny_particles::Int = 10

    # Physics options
    use_ybj_w::Bool = false           # Removed (must stay false); see wave_motion
    wave_motion::Symbol = :averaged   # :averaged or :instantaneous
    use_3d_advection::Bool = false    # Removed (must stay false): particles move horizontally

    # Advection timing control (times are Float64, like the simulation clock)
    particle_advec_time::Float64 = 0.0  # Start advecting particles at this time (0.0 = from beginning)

    # Integration method
    integration_method::Symbol = :rk2  # :euler, :rk2, :rk4

    # Interpolation method
    interpolation_method::InterpolationMethod = TRICUBIC  # TRILINEAR, TRICUBIC, QUINTIC

    # Horizontal boundary conditions
    periodic_x::Bool = true
    periodic_y::Bool = true

    # In-memory trajectory history (see the docstring)
    save_interval::Float64 = Inf     # History interval: Inf = no history, 0 = every step
    max_save_points::Int = 1000      # Most history records kept (per file when splitting)
    auto_split_files::Bool = false   # Automatically create new files when max_save_points is reached

    function ParticleConfig{T}(x_min, x_max, y_min, y_max, z_level,
                               nx_particles, ny_particles,
                               use_ybj_w, wave_motion, use_3d_advection,
                               particle_advec_time, integration_method,
                               interpolation_method, periodic_x, periodic_y,
                               save_interval, max_save_points,
                               auto_split_files) where T<:AbstractFloat
        all(isfinite, (x_min, x_max, y_min, y_max, z_level)) ||
            throw(ArgumentError("particle bounds and z_level must be finite"))
        x_max >= x_min || throw(ArgumentError("x_max must not be below x_min"))
        y_max >= y_min || throw(ArgumentError("y_max must not be below y_min"))
        nx_particles > 0 && ny_particles > 0 ||
            throw(ArgumentError("nx_particles and ny_particles must be positive"))
        _check_particle_times(particle_advec_time, save_interval, max_save_points)
        return new{T}(x_min, x_max, y_min, y_max, z_level,
                      nx_particles, ny_particles,
                      use_ybj_w, wave_motion, use_3d_advection,
                      particle_advec_time, integration_method,
                      interpolation_method, periodic_x, periodic_y,
                      save_interval, max_save_points, auto_split_files)
    end
end

"""Validate the release time and history settings of a particle configuration."""
function _check_particle_times(particle_advec_time, save_interval, max_save_points)
    isfinite(particle_advec_time) ||
        throw(ArgumentError("particle_advec_time must be finite"))
    save_interval >= 0 || throw(ArgumentError(
        "save_interval must be non-negative: 0 records every step, Inf none " *
        "(got $save_interval)"))
    max_save_points > 0 ||
        throw(ArgumentError("max_save_points must be positive (got $max_save_points)"))
    return nothing
end

"""
    particles_in_box([T=Float64], z_level;
                     x_min=0, x_max, y_min=0, y_max, nx=10, ny=10, kwargs...)

Create an `nx` by `ny` particle grid at the coordinate `z_level`, with
positions stored in precision `T`. Keep the default `Float64` for long or
slow trajectories in large domains: a `Float32` position 1000 km from the
origin is rounded to about 6 cm, so steps of a few cm are rounded away.

```julia
config = particles_in_box(
    -500.0;
    x_min=first(model.grid.x_faces), x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces), y_max=last(model.grid.y_faces),
    nx=20, ny=20,
)
```
"""
function particles_in_box(::Type{T}, z_level::Real;
                          x_max::Real, y_max::Real,  # REQUIRED
                          x_min::Real=0.0, y_min::Real=0.0,
                          nx::Int=10, ny::Int=10,
                          kwargs...) where T<:AbstractFloat
    return ParticleConfig{T}(;
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        z_level=T(z_level), nx_particles=nx, ny_particles=ny,
        kwargs...)
end

function particles_in_box(z_level::Real;
                          x_max::Real, y_max::Real,  # REQUIRED
                          x_min::Real=0.0, y_min::Real=0.0,
                          nx::Int=10, ny::Int=10,
                          kwargs...)
    return particles_in_box(Float64, z_level;
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        nx=nx, ny=ny, kwargs...)
end

# Include 3D particle configuration AFTER ParticleConfig is defined
# (particle_config.jl needs to access ParticleConfig from this module)
include("particle_config.jl")
using .EnhancedParticleConfig

"""
Particle state: positions, global IDs, and the horizontal velocity at the
particles' time, with the in-memory trajectory history. Positions and
velocities live on the model's architecture (copy them with `Array` to index
them on the host when the model runs on a GPU); the IDs and the history are
host arrays.

Positions and velocities have the particles' precision `T`, but times are
`Float64`: in single precision, times of a few months are already rounded to
seconds, which would distort the time interpolation of the velocity.
"""
mutable struct ParticleState{T<:AbstractFloat}
    # Current state
    x::AbstractVector{T}
    y::AbstractVector{T}
    z::AbstractVector{T}
    id::Vector{Int}
    u::AbstractVector{T}
    v::AbstractVector{T}
    time::Float64
    np::Int

    # Trajectory history
    x_history::Vector{Vector{T}}
    y_history::Vector{Vector{T}}
    z_history::Vector{Vector{T}}
    id_history::Vector{Vector{Int}}
    time_history::Vector{Float64}

    function ParticleState{T}(np::Int) where T
        ids = collect(1:np)
        new{T}(
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            ids,
            Vector{T}(undef, np), Vector{T}(undef, np),
            0.0, np,
            Vector{T}[], Vector{T}[], Vector{T}[], Vector{Int}[], Float64[]
        )
    end
end

"""
The horizontal velocity that advects particles, at one time, on the model's
architecture in the particles' precision: the mean velocity (the Lagrangian
mean for `wave_motion = :averaged`; for `:instantaneous`, the Lagrangian mean
less the Stokes drift of horizontal orbits) and, for `:instantaneous` only,
the amplitude of the wave velocity, which at
time t is u = c LA_r + s LA_i and v = c LA_i - s LA_r, with c = cos(f t) and
s = sin(f t). Unused arrays are empty. `valid` is false until the snapshot
has been filled. The time is `Float64`, like the particles' time.
"""
mutable struct VelocitySnapshot{T<:AbstractFloat, A<:AbstractArray{T, 3}}
    u::A
    v::A
    LA_r::A
    LA_i::A
    time::Float64
    valid::Bool
end

function VelocitySnapshot{T}(architecture::AbstractArchitecture,
                             dims::Dims{3}=(0, 0, 0)) where T
    return VelocitySnapshot(zeros_on(architecture, T, dims...),
                            zeros_on(architecture, T, dims...),
                            zeros_on(architecture, T, 0, 0, 0),
                            zeros_on(architecture, T, 0, 0, 0), 0.0, false)
end

"""Size the arrays of `snapshot` for `dims`, the wave amplitude only if `waves`."""
function _resize_snapshot!(snapshot::VelocitySnapshot{T}, dims, waves::Bool,
                           architecture) where T
    if size(snapshot.u) != dims
        snapshot.u = zeros_on(architecture, T, dims...)
        snapshot.v = zeros_on(architecture, T, dims...)
    end
    if waves && size(snapshot.LA_r) != dims
        snapshot.LA_r = zeros_on(architecture, T, dims...)
        snapshot.LA_i = zeros_on(architecture, T, dims...)
    end
    return snapshot
end

"""
Particle tracker: configuration, particle state, and the velocity that
advects the particles, on the model's architecture.
"""
mutable struct ParticleTracker{T<:AbstractFloat, P, A<:AbstractArchitecture,
                               S<:VelocitySnapshot{T}}
    config::ParticleConfig{T}
    particles::ParticleState{T}
    # Back-reference to the owning QGYBJModel. Deliberately untyped: the model
    # stores this tracker, so a concrete parameter here would be circular.
    # Read once per advection call, never inside a per-particle loop.
    model::Any
    architecture::A

    # RuntimeGeometry information. The vertical extent is the grid's own,
    # z_bottom to z_top, which need not end at z = 0.
    nx::Int; ny::Int; nz::Int
    Lx::T; Ly::T; Lz::T
    x0::T; y0::T
    z_bottom::T; z_top::T
    dx::T; dy::T; dz::T

    # Velocity at the end of the current particle step (`current`) and at its
    # start (`previous`, kept from the end of the preceding step). Integration
    # stages interpolate linearly in time between the two.
    current::S
    previous::S
    # `:instantaneous` only: the inertial frequency of the wave phase.
    inertial_frequency::Float64

    # Whether the particles have been released (they then keep moving, even
    # when a later run's clock starts before `particle_advec_time`), and
    # whether their time has advanced since they were initialized.
    released::Bool
    advanced::Bool

    # Transform plans (for velocity computation)
    plans::P

    # In-memory history: records saved, the last one's time, the next
    # nominal deadline, and whether max_save_points has been reached.
    save_counter::Int
    last_save_time::Float64
    next_history_time::Float64
    history_full::Bool

    # File splitting for large simulations
    output_file_sequence::Int          # Current file sequence number (0, 1, 2, ...)
    base_output_filename::String       # Base filename for automatic file splitting
    auto_file_splitting::Bool          # Enable automatic file splitting when max_save_points reached

    function ParticleTracker{T}(config::ParticleConfig{T}, grid::RuntimeGeometry;
                                plans=nothing, model=nothing) where T
        _check_wave_motion(config.use_ybj_w, config.wave_motion)
        _check_horizontal_advection(config.use_3d_advection)
        _check_particle_methods(config.integration_method,
                                config.interpolation_method)
        np = config.nx_particles * config.ny_particles
        particles = ParticleState{T}(np)

        transform_plans = plans === nothing ? plan_transforms!(grid) : plans
        architecture = grid.architecture
        current = VelocitySnapshot{T}(architecture, (grid.nz, grid.nx, grid.ny))
        previous = VelocitySnapshot{T}(architecture)

        new{T, typeof(transform_plans), typeof(architecture), typeof(current)}(
            config, particles, model, architecture,
            grid.nx, grid.ny, grid.nz,
            grid.Lx, grid.Ly, grid.Lz,
            grid.x0, grid.y0,
            first(grid.z_faces), last(grid.z_faces),
            grid.Lx/grid.nx, grid.Ly/grid.ny, grid.dz,
            current, previous, 0.0,
            false, false,
            transform_plans,
            0, 0.0, 0.0, false,
            0, "", false  # output_file_sequence, base_output_filename, auto_file_splitting
        )
    end
end

ParticleTracker(config::ParticleConfig{T}, grid::RuntimeGeometry; kwargs...) where T =
    ParticleTracker{T}(config, grid; kwargs...)

@inline function _periodic_offset(value::T, origin::T, extent::T) where T
    offset = mod(value - origin, extent)
    return offset >= extent ? zero(T) : offset
end

"""Move the particle positions and velocities to the tracker's architecture."""
function _place_on_architecture!(tracker::ParticleTracker)
    particles = tracker.particles
    architecture = tracker.architecture
    particles.x = on_architecture(architecture, particles.x)
    particles.y = on_architecture(architecture, particles.y)
    particles.z = on_architecture(architecture, particles.z)
    particles.u = on_architecture(architecture, particles.u)
    particles.v = on_architecture(architecture, particles.v)
    return tracker
end

"""Give the particle positions and velocities host storage, to fill them in place."""
function _host_particle_storage!(particles::ParticleState{T}) where T
    particles.x isa Vector{T} || (particles.x = Vector{T}(Array(particles.x)))
    particles.y isa Vector{T} || (particles.y = Vector{T}(Array(particles.y)))
    particles.z isa Vector{T} || (particles.z = Vector{T}(Array(particles.z)))
    particles.u isa Vector{T} || (particles.u = Vector{T}(Array(particles.u)))
    particles.v isa Vector{T} || (particles.v = Vector{T}(Array(particles.v)))
    return particles
end

"""Host storage for `np` particles."""
function _host_particle_storage!(particles::ParticleState, np::Int)
    _host_particle_storage!(particles)
    foreach(array -> resize!(array, np),
            (particles.x, particles.y, particles.z, particles.id,
             particles.u, particles.v))
    particles.np = np
    return particles
end

"""
    initialize_particles!(tracker, config)

Place the particles of `config` (a `ParticleConfig` or `ParticleConfig3D`) on
`tracker`, which adopts the configuration. Particles from an earlier
initialization are replaced, with their history, release, and velocity.
Every particle must lie within the grid's vertical extent: `z` is a
coordinate, negative below a lid at z = 0, not a depth.
"""
function initialize_particles!(tracker::ParticleTracker{T},
                              config::ParticleConfig{T}) where T
    _adopt_config!(tracker, config)
    _host_particle_storage!(tracker.particles,
                            config.nx_particles * config.ny_particles)
    initialize_particles_serial!(tracker, config)
    return _finish_initialization!(tracker)
end

function initialize_particles!(tracker::ParticleTracker{T},
                              config::ParticleConfig3D{T}) where T
    _adopt_config!(tracker, EnhancedParticleConfig.convert_to_basic_config(config))
    initialize_particles_3d!(tracker, config)
    return _finish_initialization!(tracker)
end

"""Use `config` for later steps, and forget the particles of an earlier one."""
function _adopt_config!(tracker::ParticleTracker{T}, config::ParticleConfig{T}) where T
    _check_wave_motion(config.use_ybj_w, config.wave_motion)
    _check_horizontal_advection(config.use_3d_advection)
    # The tracker's constructor has checked (and warned about) its own config.
    config === tracker.config ||
        _check_particle_methods(config.integration_method, config.interpolation_method)
    tracker.config = config
    particles = tracker.particles
    foreach(empty!, (particles.x_history, particles.y_history, particles.z_history,
                     particles.id_history, particles.time_history))
    tracker.save_counter = 0
    tracker.last_save_time = 0.0
    tracker.history_full = false
    tracker.released = false
    tracker.advanced = false
    tracker.current.valid = false
    tracker.previous.valid = false
    return tracker
end

function _finish_initialization!(tracker::ParticleTracker)
    _check_particle_positions(tracker)
    _place_on_architecture!(tracker)
    _start_history!(tracker)
    return tracker
end

"""
Error unless every particle lies within the grid's vertical extent, and
within its horizontal extent along directions that are not periodic.
"""
function _check_particle_positions(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    config = tracker.config
    z_bottom, z_top = tracker.z_bottom, tracker.z_top
    tolerance = 4eps(T) * max(abs(z_bottom), abs(z_top), tracker.Lz)
    inside(value, low, high) =
        low - 4eps(T) * max(abs(low), abs(high)) <= value <=
        high + 4eps(T) * max(abs(low), abs(high))
    for (x, y, z) in zip(particles.x, particles.y, particles.z)
        isfinite(x) && isfinite(y) && isfinite(z) ||
            throw(ArgumentError("particle positions must be finite"))
        z_bottom - tolerance <= z <= z_top + tolerance || throw(ArgumentError(
            "a particle at z = $z lies outside the grid's vertical extent " *
            "[$z_bottom, $z_top]; z is a coordinate, not a depth"))
        config.periodic_x || inside(x, tracker.x0, tracker.x0 + tracker.Lx) ||
            throw(ArgumentError("a particle at x = $x lies outside the grid " *
                                "[$(tracker.x0), $(tracker.x0 + tracker.Lx)]"))
        config.periodic_y || inside(y, tracker.y0, tracker.y0 + tracker.Ly) ||
            throw(ArgumentError("a particle at y = $y lies outside the grid " *
                                "[$(tracker.y0), $(tracker.y0 + tracker.Ly)]"))
    end
    return nothing
end

"""Record the initial positions when the configuration keeps a history."""
function _start_history!(tracker::ParticleTracker)
    tracker.history_full = false
    tracker.next_history_time = tracker.particles.time
    should_save_particles(tracker) && save_particle_state!(tracker)
    return tracker
end

"""
    _align_particle_time!(tracker, time)

Put the particles on the time axis of a run that starts at `time`. Between
runs the particles keep their own time, but a simulation restored with
`restore!` starts at its restart time and a new simulation at zero.
Particles not advanced since they were initialized take the start time, their
initial history record included. Particles that have moved continue on the
new time axis: their history records the positions at the start again, under
the new time, and the inertial phase of `:instantaneous` trajectories
restarts there.
"""
function _align_particle_time!(tracker::ParticleTracker, time)
    particles = tracker.particles
    time = Float64(time)
    old = particles.time
    old == time && return tracker
    history = particles.time_history
    if !tracker.advanced
        replace!(history, old => time)
    elseif isfinite(tracker.config.save_interval) ||
           tracker.config.wave_motion === :instantaneous
        @warn "The particles were at t = $old and continue from t = $time, the " *
              "start of this run" *
              (tracker.config.wave_motion === :instantaneous ?
               ", where the inertial phase of their wave velocity restarts" : "") *
              ". To continue the original time axis, restore the last state " *
              "file with restore!(simulation, path)." maxlog=1
    end
    particles.time = time
    tracker.current.valid = false
    tracker.previous.valid = false
    if !isempty(history) && last(history) == time
        _advance_history_deadline!(tracker)
    else
        tracker.next_history_time = time
        should_save_particles(tracker) && save_particle_state!(tracker)
    end
    return tracker
end

"""
Initialize particles for serial execution.
"""
function initialize_particles_serial!(tracker::ParticleTracker{T},
                                     config::ParticleConfig{T}) where T
    particles = tracker.particles

    # Create uniform grid of particles
    x_range = range(config.x_min, config.x_max, length=config.nx_particles+1)[1:end-1]
    y_range = range(config.y_min, config.y_max, length=config.ny_particles+1)[1:end-1]

    idx = 1
    for y in y_range, x in x_range
        particles.x[idx] = x
        particles.y[idx] = y
        particles.z[idx] = config.z_level
        particles.id[idx] = idx
        particles.u[idx] = 0.0
        particles.v[idx] = 0.0
        idx += 1
    end

    particles.time = 0.0
    particles.np = length(particles.x)

    return tracker
end

"""
    advect_particles!(tracker, state, grid, dt, current_time=nothing; f, N2, N2_profile, N2_face_profile, formulation, workspace)

Advance the particles from `current_time` (default: the particles' own time)
to `current_time + dt`. `state` is taken as the fluid state at the end of
that interval, so call this after advancing the model over it, as `run!`
does. The velocity at the start of the interval is the one kept from the
previous call, if that call ended at `current_time`; integration stages
interpolate linearly in time between the two. Without it (on the first call,
or after a gap), the end-of-interval velocity is used for the whole step.

Particles stay in place until the simulation time reaches
`particle_advec_time`, and keep moving once released.

# Keywords
The velocity diagnostic's coefficients default to those of the model that
owns the tracker (a tracker built with `ParticleTracker(config, model)` or
`initialize_particles!(model, config)`); for a tracker without a model they
default to `f = 1`, `N2 = 1`, and `YBJPlus()`.
- `f`, `N2`: Physical coefficients used by the velocity diagnostic.
- `N2_profile`: Optional cell-center N²(z) profile for nonuniform stratification.
- `N2_face_profile`: Optional upper-face N²(z) profile, used by the Stokes
  drift of `:instantaneous` trajectories.
- `formulation`: `YBJPlus()` or `YBJ()`, which fixes how the wave velocity LA
  follows from B.
- `workspace`: Optional model workspace for the velocity diagnostic's scratch.

See `ParticleConfig` for the two trajectory models, `wave_motion = :averaged`
(default) and `:instantaneous`.
"""
function advect_particles!(tracker::ParticleTracker{T},
                          state::ModelFields, grid::RuntimeGeometry, dt::Real,
                          current_time=nothing; kwargs...) where T
    velocity_options = _velocity_options(tracker; kwargs...)
    _place_on_architecture!(tracker)
    particles = tracker.particles
    t0 = current_time === nothing ? particles.time : Float64(current_time)
    t1 = t0 + Float64(dt)
    particles.time = t0
    tracker.advanced = true

    if !_released(tracker, t0)
        # Not yet released. If the next step releases the particles, keep this
        # end-of-step velocity as the start of that step.
        _released(tracker, t1) &&
            update_velocity_fields!(tracker, state, grid; time=t1,
                                    velocity_options...)
        particles.time = t1
        should_save_particles(tracker) && save_particle_state!(tracker)
        return tracker
    end
    tracker.released = true

    update_velocity_fields!(tracker, state, grid; time=t1, velocity_options...)
    # The previous snapshot is the start of this step only if it was taken at
    # t0; otherwise the end-of-step velocity is used for the whole step.
    previous = tracker.previous
    previous.valid = previous.valid && _same_time(previous.time, t0, dt)
    f = velocity_options.f
    if tracker.config.wave_motion === :instantaneous && abs(f) * dt > 0.5
        @warn "Particle time step does not resolve the inertial oscillation " *
              "(f Δt = $(abs(f) * dt)); instantaneous trajectories need f Δt ≲ 0.2." maxlog=1
    end

    _advance_particles!(tracker, t0, dt)
    particles.time = t1
    should_save_particles(tracker) && save_particle_state!(tracker)
    return tracker
end

"""
Whether the particles move at time `t`: once released, or from
`particle_advec_time` on (up to the rounding of a time `t₀ + nΔt`).
"""
function _released(tracker::ParticleTracker, t)
    tracker.released && return true
    release_time = tracker.config.particle_advec_time
    return t >= release_time - 8eps(max(abs(t), abs(release_time), 1.0))
end

"""
The velocity diagnostic's coefficients given as keywords, with those of the
tracker's model as defaults, or `f = 1`, `N2 = 1`, and `YBJPlus()` for a
tracker without a model.
"""
function _velocity_options(tracker::ParticleTracker; f=nothing, N2=nothing,
                           N2_profile=nothing, N2_face_profile=nothing,
                           formulation=nothing, workspace=nothing)
    defaults = tracker.model === nothing ?
        (; f=1, N2=1, N2_profile=nothing, N2_face_profile=nothing,
           formulation=YBJPlus(), workspace=nothing) :
        _PARENT._particle_velocity_options(tracker.model)
    given = (; f, N2, N2_profile, N2_face_profile, formulation, workspace)
    return map((value, default) -> value === nothing ? default : value,
               given, defaults)
end

"""
Move every particle from `t0` to `t0 + dt` with the tracker's velocity, apply
the horizontal boundary conditions, and store each particle's velocity at its
new position and time, in one kernel on the tracker's architecture.
"""
function _advance_particles!(tracker::ParticleTracker{T}, t0, dt) where T
    particles = tracker.particles
    integrator = Val(tracker.config.integration_method)
    launch!(_advect_particles_kernel!, particles.x, (particles.np,),
            particles.x, particles.y, particles.z, particles.u, particles.v,
            _velocity_arguments(tracker, t0), integrator, T(dt))
    return tracker
end

"""Store each particle's velocity at its position and the time `t`."""
function _store_particle_velocities!(tracker::ParticleTracker{T}, t) where T
    _place_on_architecture!(tracker)
    particles = tracker.particles
    launch!(_particle_velocity_kernel!, particles.u, (particles.np,),
            particles.u, particles.v, particles.x, particles.y, particles.z,
            _velocity_arguments(tracker, t))
    return tracker
end

"""
Whether the snapshot time `t` is the step start `t0`, up to the rounding of
the particle time step, which a caller may give in the particles' precision.
"""
@inline _same_time(t, t0, dt) =
    abs(t - t0) <= max(dt / 100, 4 * eps(Float64) * abs(t0))

"""
    update_velocity_fields!(tracker, state, grid; time=tracker.particles.time, f, N2, N2_profile, N2_face_profile, formulation, workspace)

Store the horizontal velocity of `state` as the tracker's velocity at `time`.
The snapshot it replaces is kept as the previous one, for interpolation in
time.

- `wave_motion = :averaged`: the Lagrangian-mean velocity of the balanced flow,
  `∇⊥ψ`.
- `wave_motion = :instantaneous`: the Lagrangian mean minus the Stokes drift
  of horizontal orbits, and the amplitude of the wave velocity, whose inertial
  phase is applied at each integration stage.

The velocity is diagnosed and kept on the model's architecture. When the
tracker belongs to a model and `state` is that model's fields, the model's
operators are used. The keywords are those of [`advect_particles!`](@ref),
with the same defaults.
"""
function update_velocity_fields!(tracker::ParticleTracker{T},
                                state::ModelFields, grid::RuntimeGeometry;
                                time=tracker.particles.time, kwargs...) where T
    (; f, N2, N2_profile, N2_face_profile, formulation, workspace) =
        _velocity_options(tracker; kwargs...)
    config = tracker.config
    model = tracker.model
    if model !== nothing && state === getproperty(model, :fields)
        compute_velocities!(model; compute_w=false)
    else
        compute_velocities!(state, grid; plans=tracker.plans, f, N2,
                            compute_w=false, N2_profile, N2_face_profile,
                            workspace)
    end
    instantaneous = config.wave_motion === :instantaneous

    # The current snapshot becomes the previous one; refill the other.
    tracker.current, tracker.previous = tracker.previous, tracker.current
    snapshot = tracker.current
    snapshot.valid = false
    _resize_snapshot!(snapshot, size(state.u), instantaneous, tracker.architecture)
    snapshot.u .= state.u
    snapshot.v .= state.v
    if instantaneous
        # The Lagrangian mean minus the Stokes drift of horizontal orbits:
        # particles that keep their depth make up that part of the drift
        # themselves, but not the part from vertical excursions. The drift is
        # taken from the snapshot, so the model's velocity stays the
        # Lagrangian mean.
        _PARENT.Operators._compute_wave_velocities!(state, grid;
            plans=tracker.plans, f, N2, compute_w=false,
            include_wave_velocity=false, N2_profile, N2_face_profile,
            formulation, workspace, vertical_excursions=false,
            u=snapshot.u, v=snapshot.v)
        _PARENT.Operators.wave_velocity_amplitude!(snapshot.LA_r, snapshot.LA_i,
            state, grid; plans=tracker.plans, formulation, workspace)
        tracker.inertial_frequency = Float64(f)
    end
    snapshot.time = Float64(time)
    snapshot.valid = true
    return tracker
end

"""
    interpolate_velocity_at_position(x, y, z, tracker, t=tracker.particles.time)

The horizontal velocity `(u, v)` that advects a particle at `(x, y, z)` at
time `t`, with the configured interpolation. Within the current particle step
the velocity is interpolated linearly in time between its start and its end.
With `wave_motion = :instantaneous` the wave velocity is added at the inertial
phase of `t`. The tracker must hold a velocity, recorded by a run or by
[`advect_particles!`](@ref).
"""
function interpolate_velocity_at_position(x::T, y::T, z::T,
                                        tracker::ParticleTracker{T},
                                        t=tracker.particles.time) where T
    tracker.current.valid || throw(ArgumentError(
        "the tracker holds no velocity yet: run a simulation or advance the " *
        "particles with advect_particles! first"))
    velocity = _velocity_arguments(tracker, t)
    tracker.architecture isa CPU && return _particle_velocity(x, y, z, zero(T), velocity)
    # Interpolate on the device; only the two values come back.
    result = zeros_on(tracker.architecture, T, 2)
    launch!(_point_velocity_kernel!, result, (1,), result, x, y, z, velocity)
    u, v = Array(result)
    return u, v
end

#=
Kernel-side velocity. `_velocity_arguments` packs the tracker's two snapshots
and the interpolation parameters into a named tuple of device arrays and
isbits values, which every particle kernel receives.

Kernels work in the particles' precision with the time τ elapsed since a
reference time t_ref, never with the time itself: in single precision a time
of a few months is rounded to seconds. The weights of the two snapshots and
the inertial phase at t_ref are formed on the host in Float64.
=#

"""
The tracker's velocity as kernel arguments, for times `t_ref + τ`: the fields
of its two snapshots, the weight `θ₀ + θ_rate τ` of the current one, the
inertial phase `phase₀ + f τ` of the waves, and the interpolation and domain
parameters.
"""
function _velocity_arguments(tracker::ParticleTracker{T}, t_ref) where T
    config = tracker.config
    waves = config.wave_motion === :instantaneous
    fields(snapshot) = waves ? (snapshot.u, snapshot.v, snapshot.LA_r, snapshot.LA_i) :
                               (snapshot.u, snapshot.v)
    current, previous = tracker.current, tracker.previous
    t = Float64(t_ref)
    # Linear in time between the two snapshots; the current one alone when
    # there is no valid start-of-step snapshot.
    span = current.time - previous.time
    blend = previous.valid && span > 0
    θ₀ = blend ? (t - previous.time) / span : 1.0
    θ_rate = blend ? 1 / span : 0.0
    f = tracker.inertial_frequency
    phase₀ = waves ? mod(f * t, 2π) : 0.0
    z_min, z_max = tracker.z_bottom, tracker.z_top
    return (; now=fields(current), before=fields(previous),
            θ₀=T(θ₀), θ_rate=T(θ_rate), f=T(f), phase₀=T(phase₀),
            waves=Val(waves), stencil=_stencil(config.interpolation_method),
            grid_info=(dx=tracker.dx, dy=tracker.dy, dz=tracker.dz,
                       Lx=tracker.Lx, Ly=tracker.Ly, Lz=tracker.Lz,
                       z_min=z_min, z_max=z_max, z0=z_min + tracker.dz / 2),
            boundaries=(periodic_x=config.periodic_x, periodic_y=config.periodic_y),
            x0=tracker.x0, y0=tracker.y0)
end

"""Stencil width of an interpolation method; `ADAPTIVE` is `TRICUBIC`."""
_stencil(method::InterpolationMethod) =
    method == TRILINEAR ? Val(2) : method == QUINTIC ? Val(6) : Val(4)

"""
Weight of the end-of-step velocity at the time τ after the reference time:
linear in time between the two snapshots, and 1 when there is no valid
start-of-step snapshot.
"""
@inline _current_weight(velocity, τ::T) where T =
    clamp(velocity.θ₀ + velocity.θ_rate * τ, zero(T), one(T))

"""
Horizontal velocity of one snapshot's `fields` at a position, with the wave
phase of the time τ after the reference time. Horizontal velocities have zero
vertical derivative at the lid and the bottom, so every column is mirrored
evenly there.
"""
@inline function _snapshot_velocity(fields::NTuple{N, Any}, x, y, z, τ,
                                    velocity) where N
    values = InterpolationSchemes._interpolate_stencil(velocity.stencil, fields,
        ntuple(_ -> 1, Val(N)), x - velocity.x0, y - velocity.y0, z,
        velocity.grid_info, velocity.boundaries)
    return _horizontal_velocity(velocity.waves, values, τ, velocity)
end

@inline _horizontal_velocity(::Val{false}, (u, v), τ, velocity) = (u, v)
@inline function _horizontal_velocity(::Val{true}, (u, v, LA_r, LA_i), τ, velocity)
    # Wave velocity at time t: u + iv = e^{-ift} LA.
    s, c = sincos(velocity.phase₀ + velocity.f * τ)
    return (u + c * LA_r + s * LA_i, v + c * LA_i - s * LA_r)
end

"""
The horizontal velocity that advects a particle at `(x, y, z)` at the time τ
after the reference time of `velocity`.
"""
@inline function _particle_velocity(x, y, z, τ, velocity)
    θ = _current_weight(velocity, τ)
    θ == 0 && return _snapshot_velocity(velocity.before, x, y, z, τ, velocity)
    uc, vc = _snapshot_velocity(velocity.now, x, y, z, τ, velocity)
    θ == 1 && return (uc, vc)
    up, vp = _snapshot_velocity(velocity.before, x, y, z, τ, velocity)
    return (θ * uc + (1 - θ) * up, θ * vc + (1 - θ) * vp)
end

# Integration methods: one step from (x, y) at the time t0 after the reference
# time (zero for a particle step), at the fixed depth z.

@inline function _integrate(::Val{:euler}, x, y, z, t0, dt, velocity)
    u, v = _particle_velocity(x, y, z, t0, velocity)
    return x + dt * u, y + dt * v
end

@inline function _integrate(::Val{:rk2}, x0, y0, z, t0, dt, velocity)
    u1, v1 = _particle_velocity(x0, y0, z, t0, velocity)
    # Midpoint
    u2, v2 = _particle_velocity(x0 + dt / 2 * u1, y0 + dt / 2 * v1, z,
                                t0 + dt / 2, velocity)
    return x0 + dt * u2, y0 + dt * v2
end

@inline function _integrate(::Val{:rk4}, x0, y0, z, t0, dt, velocity)
    u1, v1 = _particle_velocity(x0, y0, z, t0, velocity)
    u2, v2 = _particle_velocity(x0 + dt / 2 * u1, y0 + dt / 2 * v1, z,
                                t0 + dt / 2, velocity)
    u3, v3 = _particle_velocity(x0 + dt / 2 * u2, y0 + dt / 2 * v2, z,
                                t0 + dt / 2, velocity)
    u4, v4 = _particle_velocity(x0 + dt * u3, y0 + dt * v3, z, t0 + dt, velocity)
    return (x0 + dt * (u1 + 2*u2 + 2*u3 + u4) / 6,
            y0 + dt * (v1 + 2*v2 + 2*v3 + v4) / 6)
end

"""Wrap a position into a periodic domain, or clamp it into a bounded one."""
@inline function _horizontal_boundaries(x, y, velocity)
    x0, y0 = velocity.x0, velocity.y0
    Lx, Ly = velocity.grid_info.Lx, velocity.grid_info.Ly
    x = velocity.boundaries.periodic_x ? x0 + _periodic_offset(x, x0, Lx) :
                                         clamp(x, x0, x0 + Lx)
    y = velocity.boundaries.periodic_y ? y0 + _periodic_offset(y, y0, Ly) :
                                         clamp(y, y0, y0 + Ly)
    return x, y
end

# Each work item moves one particle over the step [t_ref, t_ref + dt] of
# `velocity`; z is read only, since particles keep their depth.
@kernel function _advect_particles_kernel!(x, y, @Const(z), u, v, velocity,
                                           integrator, dt)
    p = @index(Global, Linear)
    @inbounds begin
        zp = z[p]
        xp, yp = _integrate(integrator, x[p], y[p], zp, zero(dt), dt, velocity)
        xp, yp = _horizontal_boundaries(xp, yp, velocity)
        x[p] = xp
        y[p] = yp
        u[p], v[p] = _particle_velocity(xp, yp, zp, dt, velocity)
    end
end

# The velocity at the reference time of `velocity`.
@kernel function _particle_velocity_kernel!(u, v, @Const(x), @Const(y), @Const(z),
                                            velocity)
    p = @index(Global, Linear)
    @inbounds u[p], v[p] = _particle_velocity(x[p], y[p], z[p], zero(eltype(u)),
                                              velocity)
end

@kernel function _point_velocity_kernel!(result, x, y, z, velocity)
    u, v = _particle_velocity(x, y, z, zero(x), velocity)
    @inbounds result[1] = u
    @inbounds result[2] = v
end

"""
Save current particle state to trajectory history.

If auto_split_files is enabled and max_save_points is reached, automatically
creates a new file and resets the trajectory history to continue saving.
"""
function save_particle_state!(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    
    # Check if we've reached max_save_points
    if length(particles.time_history) >= tracker.config.max_save_points
        if tracker.config.auto_split_files && !isempty(tracker.base_output_filename)
            # Save current trajectory segment to file
            split_and_save_trajectory_segment!(tracker)
            
            # Reset trajectory history for next segment
            empty!(particles.x_history)
            empty!(particles.y_history) 
            empty!(particles.z_history)
            empty!(particles.id_history)
            empty!(particles.time_history)
            
            tracker.output_file_sequence += 1
        else
            if !tracker.history_full
                tracker.history_full = true
                @warn "Particle history reached max_save_points = " *
                      "$(tracker.config.max_save_points) at t = $(particles.time); " *
                      "later positions are not recorded. Raise max_save_points, " *
                      "call enable_auto_file_splitting!, or write trajectories " *
                      "with a ParticleOutputManager."
            end
            return tracker
        end
    end
    
    # Save current state to history
    push!(particles.x_history, Array(particles.x))
    push!(particles.y_history, Array(particles.y))
    push!(particles.z_history, Array(particles.z))
    push!(particles.id_history, copy(particles.id))
    push!(particles.time_history, particles.time)
    
    tracker.save_counter += 1
    tracker.last_save_time = particles.time
    _advance_history_deadline!(tracker)
    return tracker
end

"""
Set the next history deadline to the first multiple of `save_interval` after
the current time, so that late records do not delay later ones.
"""
function _advance_history_deadline!(tracker::ParticleTracker)
    interval = tracker.config.save_interval
    (isfinite(interval) && interval > 0) || return tracker
    tracker.next_history_time = _next_deadline(tracker.particles.time, interval)
    return tracker
end

"""
    split_and_save_trajectory_segment!(tracker)

Save current trajectory history to a sequentially numbered file and prepare for next segment.
Used internally by save_particle_state! when auto_split_files is enabled.
"""
function split_and_save_trajectory_segment!(tracker::ParticleTracker{T}) where T
    if isempty(tracker.particles.time_history)
        return tracker
    end
    
    # Create filename with sequence number
    base_name = tracker.base_output_filename
    if tracker.output_file_sequence == 0
        filename = "$(base_name).nc"
    else
        filename = "$(base_name)_part$(tracker.output_file_sequence).nc"
    end
    
    # Calculate time range for this segment
    start_time = tracker.particles.time_history[1]
    end_time = tracker.particles.time_history[end]
    n_points = length(tracker.particles.time_history)
    
    println("Auto-splitting: Saving trajectory segment to $filename")
    println("  Time range: $(round(start_time, digits=4)) - $(round(end_time, digits=4))")
    println("  Points: $n_points / $(tracker.config.max_save_points) (max)")
    
    # Create metadata for this segment
    metadata = Dict(
        "segment_number" => tracker.output_file_sequence,
        "start_time" => start_time,
        "end_time" => end_time,
        "points_in_segment" => n_points,
        "max_points_per_file" => tracker.config.max_save_points,
        "auto_split_enabled" => true
    )
    
    # ParticleIO is included after this module, so resolve its exported writer
    # from the parent at call time instead of capturing an undefined binding.
    writer = getfield(_PARENT, :write_particle_trajectories)
    writer(filename, tracker; metadata=metadata)
    println("  ✅ Successfully saved segment $(tracker.output_file_sequence)")
    
    return tracker
end

"""
    should_save_particles(tracker)

Whether the tracker's in-memory history is due a record: never for
`save_interval = Inf` (the default), every step for 0, and otherwise when a
run starts and at the multiples of `save_interval`, which late records do
not shift.
"""
function should_save_particles(tracker::ParticleTracker)
    interval = tracker.config.save_interval
    isfinite(interval) || return false
    interval > 0 || return true
    time = tracker.particles.time
    tolerance = 8eps(max(abs(time), interval, 1.0))
    return time >= tracker.next_history_time - tolerance
end

"""
    enable_auto_file_splitting!(tracker, base_filename; max_points_per_file=1000)

Enable automatic file splitting for long particle trajectories.

When enabled, the tracker will automatically create new files when max_save_points
is reached, allowing unlimited trajectory length with manageable file sizes.
The tracker must keep a history: its configuration needs a finite
`save_interval`.

Parameters:
- tracker: ParticleTracker instance
- base_filename: Base name for output files (without .nc extension)
- max_points_per_file: Maximum trajectory points per file (default: 1000)

Files created:
- "base_filename.nc" (first segment)
- "base_filename_part1.nc" (second segment)
- "base_filename_part2.nc" (third segment), etc.

Example:
```julia
using QGYBJplus.UnifiedParticleAdvection: enable_auto_file_splitting!,
                                          finalize_trajectory_files!

config = particles_in_box(-500.0; x_max=Lx, y_max=Ly, save_interval=600.0)
initialize_particles!(simulation, config)
enable_auto_file_splitting!(simulation.model.particles, "long_simulation";
                            max_points_per_file=500)

run!(simulation)   # writes long_simulation.nc, long_simulation_part1.nc, ...

# Write the final segment:
finalize_trajectory_files!(simulation.model.particles)
```
"""
function enable_auto_file_splitting!(tracker::ParticleTracker{T}, base_filename::String;
                                     max_points_per_file::Int=1000) where T
    max_points_per_file > 0 ||
        throw(ArgumentError("max_points_per_file must be positive"))
    isfinite(tracker.config.save_interval) || throw(ArgumentError(
        "file splitting needs a tracker that keeps a history: give its " *
        "configuration a finite save_interval"))

    # Update tracker configuration
    tracker.base_output_filename = base_filename
    tracker.auto_file_splitting = true
    tracker.output_file_sequence = 0

    # Create new config with updated max_save_points and auto_split_files
    # Use keyword arguments as ParticleConfig uses @kwdef
    old_config = tracker.config
    new_config = ParticleConfig{T}(
        use_ybj_w = old_config.use_ybj_w,
        x_min = old_config.x_min,
        x_max = old_config.x_max,
        y_min = old_config.y_min,
        y_max = old_config.y_max,
        z_level = old_config.z_level,
        nx_particles = old_config.nx_particles,
        ny_particles = old_config.ny_particles,
        wave_motion = old_config.wave_motion,
        use_3d_advection = old_config.use_3d_advection,
        particle_advec_time = old_config.particle_advec_time,
        integration_method = old_config.integration_method,
        interpolation_method = old_config.interpolation_method,
        periodic_x = old_config.periodic_x,
        periodic_y = old_config.periodic_y,
        save_interval = old_config.save_interval,
        max_save_points = max_points_per_file,
        auto_split_files = true
    )

    tracker.config = new_config

    println("✅ Auto file splitting enabled:")
    println("  Base filename: $base_filename")
    println("  Max points per file: $max_points_per_file")
    println("  Files will be created: $(base_filename).nc, $(base_filename)_part1.nc, ...")

    return tracker
end

"""
    finalize_trajectory_files!(tracker; final_metadata=Dict())

Save final trajectory segment for auto-splitting trackers.

Call this at the end of long simulations to ensure the final trajectory 
segment is saved to file.
"""
function finalize_trajectory_files!(tracker::ParticleTracker; final_metadata::Dict=Dict())
    if !tracker.config.auto_split_files || isempty(tracker.particles.time_history)
        return tracker
    end
    
    println("Finalizing trajectory files...")
    
    # Save final segment
    split_and_save_trajectory_segment!(tracker)
    
    total_files = tracker.output_file_sequence + 1
    total_points = tracker.save_counter
    
    println("✅ Trajectory finalization complete:")
    println("  Total files created: $total_files")
    println("  Total trajectory points: $total_points")
    println("  Average points per file: $(round(total_points/total_files, digits=1))")
    
    return tracker
end

end # module UnifiedParticleAdvection

using .UnifiedParticleAdvection
