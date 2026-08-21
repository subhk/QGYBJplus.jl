"""Mutable simulation clock owned by a [`Simulation`](@ref)."""
mutable struct Clock{T}
    time::T
    iteration::Int
end

Clock(::Type{T}=Float64) where T = Clock(zero(T), 0)

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
    output
end

default_run_options(::Type{T}) where T = SimulationRunOptions{T}(
    "output", nothing, 10, true, true, true, false, false)

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
    state::SimulationState
end

