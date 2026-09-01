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

"""
    Simulation

Lifecycle owner for a model run. The simulation composes a model, clock,
typed timestepper, stop criteria, scheduling/output configuration, and state.
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
