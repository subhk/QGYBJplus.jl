"""Mutable simulation clock owned by a [`Simulation`](@ref)."""
mutable struct Clock{T}
    time::T
    iteration::Int
end

Clock(::Type{T}=Float64) where T = Clock(zero(T), 0)

"""Lifecycle states for a [`Simulation`](@ref)."""
@enum SimulationState begin
    Ready
    Running
    Stopped
    Failed
    Finalized
end

"""Output and reporting choices owned by a simulation."""
mutable struct SimulationRunOptions{T}
    output_dir::String
    save_interval::Union{Nothing, T}
    diagnostics_interval::Int
    verbose::Bool
    save_psi::Bool
    save_waves::Bool
    save_velocities::Bool
    # Output/diagnostic specifications: `false`, a NetCDFOutput /
    # EnergyDiagnosticsOutput, or a schedule. Untyped because a user may change
    # the specification between runs; read once per output decision.
    output
    diagnostics
end

default_run_options(::Type{T}) where T = SimulationRunOptions{T}(
    "output", nothing, 10, true, true, true, false, false, false)

Base.copy(options::SimulationRunOptions{T}) where T = SimulationRunOptions{T}(
    (getfield(options, name) for name in fieldnames(SimulationRunOptions))...)

"""
    Simulation(model; Δt=1.0, stop_time=nothing, stop_iteration=nothing,
               output=false, diagnostics=false, particle_output=nothing,
               verbose=true)

Lifecycle owner for a model run. The simulation composes a model, clock,
typed timestepper, stop criteria, scheduling/output configuration, and state.

- `Δt`: the ETD-RK2 time step. The clock keeps time in `Float64` as
  `t₀ + nΔt`, whatever the type of `Δt`.
- `stop_time` or `stop_iteration` (not both): when [`run!`](@ref) stops; with
  neither, after 1000 iterations.
- `output`: a [`NetCDFOutput`](@ref), or `false` for none.
- `diagnostics`: an [`EnergyDiagnosticsOutput`](@ref), a schedule (an
  `IterationInterval`, `TimeInterval`, or integer number of iterations) for
  energy files in the `diagnostic/` subdirectory of the output's path (or of
  `output/`), or `false` for none.
- `particle_output`: a [`ParticleOutputManager`](@ref) for the particles
  installed on the model, or `false`/`nothing`. A manager serves one run.
- `verbose`: log the progress every `diagnostics_interval` iterations (10 by
  default; see `run!`).

A simulation runs once: after `run!` it is `Stopped`, and a new simulation
around the same model continues from its fields.
"""
mutable struct Simulation{M, T, I, R}
    model::M
    clock::Clock{T}
    timestepper::I
    stop_time::Union{Nothing, T}
    stop_iteration::Union{Nothing, Int}
    run_options::R
    # Managers are created lazily by run!, once the output specification and
    # clock type are known, so these mutable fields cannot carry a concrete
    # parameter. Touched once per scheduled write, never in a kernel.
    output_manager::Any
    diagnostics_manager::Any
    particle_output_manager::Any
    state::SimulationState
end
