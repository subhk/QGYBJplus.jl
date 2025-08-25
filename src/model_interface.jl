"""
High-level model interface for QG-YBJ simulations.

This module provides the main user interface for setting up and running QG-YBJ simulations
with the configuration system, including time stepping with output management.
"""

using Printf
using ..QGYBJ: QGParams, Grid, State, setup_model, default_params
using ..QGYBJ: plan_transforms!, init_grid, init_state
using ..QGYBJ: first_projection_step!, leapfrog_step!
using ..QGYBJ: invert_q_to_psi!, compute_velocities!

# Import our new modules
include("config.jl")
include("netcdf_io.jl")
include("initialization.jl")
include("stratification.jl")
include("parallel_interface.jl")

"""
    QGYBJSimulation{T}

Main simulation object containing all model components.
"""
mutable struct QGYBJSimulation{T}
    # Configuration
    config::ModelConfig{T}
    
    # Model components
    params::QGParams{T}
    grid::Grid
    state::State
    state_old::State  # For leapfrog
    
    # Transform plans
    plans
    
    # Output management
    output_manager
    
    # Parallel configuration
    parallel_config::ParallelConfig
    
    # Stratification
    stratification_profile
    N2_profile::Vector{T}
    
    # Time stepping
    current_time::T
    time_step::Int
    
    # Diagnostics
    diagnostics::Dict{String, Any}
end

"""
    setup_simulation(config::ModelConfig; use_mpi::Bool=false)

Set up a complete QG-YBJ simulation from configuration.
"""
function setup_simulation(config::ModelConfig{T}; use_mpi::Bool=false) where T
    @info "Setting up QG-YBJ simulation"
    
    # Initialize parallel environment
    parallel_config = if use_mpi
        setup_parallel_environment()
    else
        ParallelConfig(use_mpi=false)
    end
    
    # Print parallel info
    if parallel_config.use_mpi
        import MPI
        rank = MPI.Comm_rank(parallel_config.comm)
        nprocs = MPI.Comm_size(parallel_config.comm)
        if rank == 0
            @info "Running with MPI: $nprocs processes"
        end
    else
        @info "Running in serial mode"
    end
    
    # Validate configuration (only on rank 0 to avoid spam)
    should_print = !parallel_config.use_mpi || (MPI.Comm_rank(parallel_config.comm) == 0)
    
    if should_print
        errors, warnings = validate_config(config)
        
        if !isempty(errors)
            error("Configuration errors:\n" * join(errors, "\n"))
        end
        
        if !isempty(warnings)
            @warn "Configuration warnings:\n" * join(warnings, "\n")
        end
        
        # Print configuration summary
        print_config_summary(config)
    end
    
    # Create QGParams from configuration
    params = QGParams{T}(;
        nx = config.domain.nx,
        ny = config.domain.ny,
        nz = config.domain.nz,
        Lx = config.domain.Lx,
        Ly = config.domain.Ly,
        dt = config.dt,
        nt = ceil(Int, config.total_time / config.dt),
        f0 = config.f0,
        nu_h = config.nu_h,
        nu_v = config.nu_v,
        linear_vert_structure = 0,
        stratification = config.stratification.type,
        W2F = T(1e-6),  # Default wave-to-flow energy ratio
        gamma = T(1e-3),  # Robert-Asselin filter
        nuh1 = T(0.01), nuh2 = T(10.0), ilap1 = 2, ilap2 = 6,
        nuh1w = T(0.0), nuh2w = T(10.0), ilap1w = 2, ilap2w = 6,
        nuz = T(0.0),
        inviscid = config.inviscid,
        linear = config.linear,
        no_dispersion = config.no_dispersion,
        passive_scalar = config.passive_scalar,
        ybj_plus = config.ybj_plus,
        no_feedback = config.no_wave_feedback || config.no_feedback,  # Handle both flags
        fixed_flow = config.fixed_mean_flow,
        no_wave_feedback = config.no_wave_feedback,
        # Skewed Gaussian parameters (from config)
        N02_sg = config.stratification.N02_sg,
        N12_sg = config.stratification.N12_sg,
        sigma_sg = config.stratification.sigma_sg,
        z0_sg = config.stratification.z0_sg,
        alpha_sg = config.stratification.alpha_sg
    )
    
    # Initialize grid and state with parallel support
    if parallel_config.use_mpi && should_print
        @info "Initializing parallel grid and state"
    elseif should_print
        @info "Initializing grid and state"
    end
    
    if parallel_config.use_mpi
        grid = init_parallel_grid(params, parallel_config)
        state = init_parallel_state(grid, parallel_config)
        state_old = init_parallel_state(grid, parallel_config)
        plans = plan_transforms!(grid, parallel_config)
    else
        grid = init_grid(params)
        state = init_state(grid)
        state_old = init_state(grid)
        plans = plan_transforms!(grid)
    end
    
    # Set up stratification
    @info "Setting up stratification profile"
    stratification_profile = create_stratification_profile(config.stratification)
    N2_profile = compute_stratification_profile(stratification_profile, grid)
    
    # Validate stratification
    strat_errors, strat_warnings = validate_stratification(N2_profile)
    if !isempty(strat_errors)
        error("Stratification errors:\n" * join(strat_errors, "\n"))
    end
    if !isempty(strat_warnings)
        @warn "Stratification warnings:\n" * join(strat_warnings, "\n")
    end
    
    # Initialize fields
    if should_print
        @info "Initializing model fields"
    end
    
    if parallel_config.use_mpi
        parallel_initialize_fields!(state, grid, plans, config, parallel_config)
    else
        initialize_from_config(config, grid, state, plans)
    end
    
    # Check initial conditions (only on rank 0 to avoid spam)
    ic_diagnostics = if should_print
        check_initial_conditions(state, grid, plans)
    else
        Dict("skipped" => "parallel rank > 0")
    end
    
    # Set up output management with parallel support
    if should_print
        @info "Setting up output management"
    end
    
    # Create unified output manager that handles both serial and parallel I/O
    output_manager = OutputManager(config.output, params, parallel_config)
    
    # Initialize diagnostics
    diagnostics = Dict{String, Any}(
        "initial_conditions" => ic_diagnostics
    )
    
    simulation = QGYBJSimulation{T}(
        config,
        params,
        grid,
        state,
        state_old,
        plans,
        output_manager,
        parallel_config,
        stratification_profile,
        N2_profile,
        T(0),  # current_time
        0,     # time_step
        diagnostics
    )
    
    @info "Simulation setup complete"
    return simulation
end

"""
    run_simulation!(sim::QGYBJSimulation; progress_callback=nothing)

Run the complete simulation with output management.
"""
function run_simulation!(sim::QGYBJSimulation{T}; progress_callback=nothing) where T
    @info "Starting QG-YBJ simulation"
    @info "Total time: $(sim.config.total_time), Time step: $(sim.params.dt), Steps: $(sim.params.nt)"
    
    # Write initial output
    if should_output_psi(sim.output_manager, sim.current_time)
        write_state_file(sim.output_manager, sim.state, sim.grid, sim.plans, 
                        sim.current_time; params=sim.params)
    end
    
    # Copy initial state to old state for first step
    sim.state_old.psi .= sim.state.psi
    sim.state_old.B .= sim.state.B
    
    # First step using projection method
    @info "Performing first projection step"
    first_projection_step!(sim.state, sim.state_old, sim.grid, sim.params, sim.plans)
    
    sim.time_step = 1
    sim.current_time = sim.params.dt
    
    # Output after first step if needed
    check_and_output!(sim)
    
    # Main time stepping loop  
    @info "Starting main time integration loop"
    
    for step in 2:sim.params.nt
        # Leapfrog time step
        leapfrog_step!(sim.state, sim.state_old, sim.grid, sim.params, sim.plans)
        
        sim.time_step = step
        sim.current_time = step * sim.params.dt
        
        # Output if needed
        check_and_output!(sim)
        
        # Compute diagnostics periodically
        if should_output_diagnostics(sim.output_manager, sim.current_time)
            compute_and_output_diagnostics!(sim)
        end
        
        # Progress callback
        if !isnothing(progress_callback)
            progress_callback(sim)
        end
        
        # Check for early termination conditions
        if check_termination_conditions(sim)
            @warn "Simulation terminated early at t=$(sim.current_time)"
            break
        end
        
        # Progress reporting
        if step % max(1, sim.params.nt ÷ 20) == 0
            progress = 100 * step / sim.params.nt
            @info @sprintf("Progress: %.1f%% (t=%.3f)", progress, sim.current_time)
        end
    end
    
    @info "Simulation completed successfully"
    @info "Final time: $(sim.current_time)"
    
    return sim
end

"""
    check_and_output!(sim::QGYBJSimulation)

Check if output is needed and write files.
"""
function check_and_output!(sim::QGYBJSimulation)
    # State output - unified interface handles both serial and parallel
    if should_output_psi(sim.output_manager, sim.current_time) || 
       should_output_waves(sim.output_manager, sim.current_time)
        
        write_state_file(sim.output_manager, sim.state, sim.grid, sim.plans,
                        sim.current_time, sim.parallel_config; params=sim.params)
    end
end

"""
    compute_and_output_diagnostics!(sim::QGYBJSimulation)

Compute diagnostic quantities and write to file.
"""
function compute_and_output_diagnostics!(sim::QGYBJSimulation{T}) where T
    diagnostics = Dict{String, Any}()
    
    # Energy diagnostics
    diagnostics["kinetic_energy"] = compute_kinetic_energy(sim.state, sim.grid, sim.plans)
    diagnostics["potential_energy"] = compute_potential_energy(sim.state, sim.grid, sim.plans, sim.N2_profile)
    diagnostics["wave_energy"] = compute_wave_energy(sim.state, sim.grid, sim.plans)
    
    # Domain-integrated quantities
    diagnostics["total_enstrophy"] = compute_enstrophy(sim.state, sim.grid, sim.plans)
    
    # Extrema
    psir = similar(sim.state.psi, Float64)
    fft_backward!(psir, sim.state.psi, sim.plans)
    diagnostics["psi_min"] = minimum(psir)
    diagnostics["psi_max"] = maximum(psir)
    diagnostics["psi_rms"] = sqrt(sum(abs2, psir) / length(psir))
    
    # Wave field extrema
    Br = similar(sim.state.B, Float64)
    fft_backward!(Br, real.(sim.state.B), sim.plans)
    diagnostics["wave_min"] = minimum(Br)
    diagnostics["wave_max"] = maximum(Br)
    diagnostics["wave_rms"] = sqrt(sum(abs2, Br) / length(Br))
    
    # Store in simulation object
    sim.diagnostics["step_$(sim.time_step)"] = diagnostics
    
    # Write to file
    write_diagnostics_file(sim.output_manager, diagnostics, sim.current_time)
end

"""
    compute_kinetic_energy(state::State, grid::Grid, plans)

Compute domain-integrated kinetic energy.
"""
function compute_kinetic_energy(state::State, grid::Grid, plans)
    # This is a simplified version - would need proper velocity computation
    psir = similar(state.psi, Float64)
    fft_backward!(psir, state.psi, plans)
    
    # Rough estimate from stream function
    KE = 0.5 * sum(abs2, psir) / (grid.nx * grid.ny * grid.nz)
    return KE
end

"""
    compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector)

Compute domain-integrated potential energy.
"""
function compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector{T}) where T
    # Simplified calculation
    psir = similar(state.psi, Float64)
    fft_backward!(psir, state.psi, plans)
    
    PE = T(0)
    for k in 1:grid.nz
        PE += N2_profile[k] * sum(abs2, view(psir, :, :, k))
    end
    PE *= 0.5 / (grid.nx * grid.ny * grid.nz)
    
    return PE
end

"""
    compute_wave_energy(state::State, grid::Grid, plans)

Compute wave energy.
"""
function compute_wave_energy(state::State, grid::Grid, plans)
    Br = similar(state.B, Float64)
    fft_backward!(Br, real.(state.B), plans)
    
    WE = 0.5 * sum(abs2, Br) / (grid.nx * grid.ny * grid.nz)
    return WE
end

"""
    compute_enstrophy(state::State, grid::Grid, plans)

Compute domain-integrated enstrophy.
"""
function compute_enstrophy(state::State, grid::Grid, plans)
    # This would require proper vorticity calculation
    # For now, simplified estimate
    psir = similar(state.psi, Float64)
    fft_backward!(psir, state.psi, plans)
    
    enstrophy = sum(abs2, psir) / (grid.nx * grid.ny * grid.nz)
    return enstrophy
end

"""
    check_termination_conditions(sim::QGYBJSimulation)

Check for early termination conditions (blow-up, etc.).
"""
function check_termination_conditions(sim::QGYBJSimulation{T}) where T
    # Check for NaNs or Infs
    if any(x -> !isfinite(x), sim.state.psi) || any(x -> !isfinite(x), sim.state.B)
        @error "NaN or Inf detected in solution"
        return true
    end
    
    # Check for blow-up (very large values)
    psir = similar(sim.state.psi, Float64)
    fft_backward!(psir, sim.state.psi, sim.plans)
    
    if maximum(abs, psir) > 1e10
        @error "Solution appears to be blowing up (max |psi| = $(maximum(abs, psir)))"
        return true
    end
    
    return false
end

"""
    create_simple_config(; kwargs...)

Create a simple configuration for quick testing.
"""
function create_simple_config(; kwargs...)
    # Default simple configuration
    domain = create_domain_config(nx=64, ny=64, nz=32, Lx=4π, Ly=4π, Lz=2π)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,
        wave_amplitude=0.01
    )
    output = create_output_config(
        output_dir="./output_simple",
        psi_interval=1.0,
        wave_interval=1.0
    )
    
    config = create_model_config(domain, stratification, initial_conditions, output;
                                dt=1e-3, total_time=10.0, kwargs...)
    
    return config
end

"""
    run_simple_simulation(; kwargs...)

Run a simple simulation with default parameters.
"""
function run_simple_simulation(; kwargs...)
    config = create_simple_config(; kwargs...)
    sim = setup_simulation(config)
    run_simulation!(sim)
    return sim
end

# Convenience function for backward compatibility with existing code
"""
    setup_model_with_config(config::ModelConfig)

Set up model components from configuration (for compatibility).
"""
function setup_model_with_config(config::ModelConfig{T}) where T
    params = QGParams{T}(;
        nx = config.domain.nx,
        ny = config.domain.ny,
        nz = config.domain.nz,
        Lx = config.domain.Lx,
        Ly = config.domain.Ly,
        dt = config.dt,
        nt = ceil(Int, config.total_time / config.dt),
        f0 = config.f0
    )
    
    grid = init_grid(params)
    state = init_state(params)
    plans = plan_transforms!(grid)
    
    return params, grid, state, plans
end