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
using ..QGYBJ: local_to_global

# Energy diagnostics module for separate file output
using ..QGYBJ.EnergyDiagnostics: EnergyDiagnosticsManager, should_output, record_energies!
using ..QGYBJ.EnergyDiagnostics: write_all_energy_files!, finalize!

# Note: config.jl, netcdf_io.jl, initialization.jl, stratification.jl, parallel_interface.jl
# are included in QGYBJ.jl before this file to avoid duplicate includes

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

    # Energy diagnostics manager (saves to diagnostic/ folder)
    energy_diagnostics_manager::EnergyDiagnosticsManager{T}
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
        try
            M = Base.require(:MPI)
            rank = M.Comm_rank(parallel_config.comm)
            nprocs = M.Comm_size(parallel_config.comm)
            if rank == 0; @info "Running with MPI: $nprocs processes"; end
        catch
            @info "Running with MPI (rank unknown)"
        end
    else
        @info "Running in serial mode"
    end
    
    # Validate configuration (only on rank 0 to avoid spam)
    should_print = !parallel_config.use_mpi || begin
        try
            M = Base.require(:MPI); M.Comm_rank(parallel_config.comm) == 0
        catch
            true
        end
    end
    
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
        f₀ = config.f0,
        νₕ = config.nu_h,
        νᵥ = config.nu_v,
        linear_vert_structure = 0,
        stratification = config.stratification.type,
        W2F = T(1e-6),  # Default wave-to-flow energy ratio
        N² = T(1.0),    # Default buoyancy frequency squared
        γ = T(1e-3),    # Robert-Asselin filter
        νₕ₁ = T(0.01), νₕ₂ = T(10.0), ilap1 = 2, ilap2 = 6,
        νₕ₁ʷ = T(0.0), νₕ₂ʷ = T(10.0), ilap1w = 2, ilap2w = 6,
        νz = T(0.0),
        inviscid = config.inviscid,
        linear = config.linear,
        no_dispersion = config.no_dispersion,
        passive_scalar = config.passive_scalar,
        ybj_plus = config.ybj_plus,
        no_feedback = config.no_wave_feedback || config.no_feedback,  # Handle both flags
        fixed_flow = config.fixed_mean_flow,
        no_wave_feedback = config.no_wave_feedback,
        # Skewed Gaussian parameters (from config)
        N₀²_sg = config.stratification.N02_sg,
        N₁²_sg = config.stratification.N12_sg,
        σ_sg = config.stratification.sigma_sg,
        z₀_sg = config.stratification.z0_sg,
        α_sg = config.stratification.alpha_sg
        # Optional vertical profiles not provided via config yet; leave as nothing
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
    
    # Derive density-like vertical profiles from N² and populate params
    rho_ut_prof, rho_st_prof = derive_density_profiles(params, grid; N2_profile=N2_profile)
    params = QGParams{T}(;
        (name => getfield(params, name) for name in fieldnames(typeof(params)) if !(name in (:ρ_ut_profile, :ρ_st_profile, :b_ell_profile)))...,
        ρ_ut_profile = rho_ut_prof,
        ρ_st_profile = rho_st_prof,
        b_ell_profile = nothing,
    )

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

    # Create energy diagnostics manager for separate energy files in diagnostic/ folder
    energy_diag_manager = EnergyDiagnosticsManager(
        config.output.output_dir;
        output_interval=config.output.diagnostics_interval
    )
    @info "Energy diagnostics will be saved to: $(energy_diag_manager.diagnostic_dir)"

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
        diagnostics,
        energy_diag_manager  # Energy diagnostics manager
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
    
    # Finalize energy diagnostics - write all energy files to diagnostic/ folder
    finalize!(sim.energy_diagnostics_manager)

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
    compute_kinetic_energy(state::State, grid::Grid, plans; Ar2::Real=1.0)

Compute domain-integrated kinetic energy following the Fortran diag_zentrum routine.

The kinetic energy is computed as:
    KE = (1/2) ∑_{kx,ky,z} (|u|² + |v|² + Ar² |w|²)

with proper dealiasing correction (subtract 0.5 × value at kx=ky=0).

# Arguments
- `state::State`: Current model state with psi (streamfunction)
- `grid::Grid`: Grid structure with wavenumbers
- `plans`: FFT plans
- `Ar2::Real`: Aspect ratio squared (default 1.0)

# Returns
Domain-integrated kinetic energy (scalar).
"""
function compute_kinetic_energy(state::State, grid::Grid, plans; Ar2::Real=1.0)
    T = eltype(real(state.psi[1]))
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny

    # Compute velocities from streamfunction in spectral space
    # u = -∂ψ/∂y = -i*ky*ψ, v = ∂ψ/∂x = i*kx*ψ
    psi_arr = parent(state.psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    KE = T(0)

    for k in 1:nz_local
        KE_level = T(0)
        KE_zero_mode = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Get wavenumbers
            i_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(i_local, 1, grid) : i_local
            j_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(j_local, 2, grid) : j_local

            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]

            psi_k = psi_arr[i_local, j_local, k]

            # u = -i*ky*psi, v = i*kx*psi
            u_k = -im * ky_val * psi_k
            v_k = im * kx_val * psi_k

            # |u|² + |v|² = ky²|ψ|² + kx²|ψ|² = kh²|ψ|²
            energy_mode = abs2(u_k) + abs2(v_k)
            KE_level += energy_mode

            # Track zero mode for dealiasing correction
            if kx_val == 0 && ky_val == 0
                KE_zero_mode = energy_mode
            end
        end

        # Dealiasing correction: subtract 0.5 × zero mode
        KE_level -= T(0.5) * KE_zero_mode
        KE += KE_level
    end

    # Normalize by grid size
    KE *= T(0.5) / (nx * ny * nz)
    return KE
end

"""
    compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector; a_ell::Real=1.0)

Compute domain-integrated potential energy following the Fortran diag_zentrum routine.

The potential energy is computed from buoyancy as:
    PE = (1/2) ∑_{kx,ky,z} |b|² × (a_ell × r_1/r_2)

where b = ∂ψ/∂z / r_1 (thermal wind balance) and a_ell = f²/N².

# Arguments
- `state::State`: Current model state with psi (streamfunction)
- `grid::Grid`: Grid structure
- `plans`: FFT plans
- `N2_profile::Vector`: Buoyancy frequency squared N²(z)
- `a_ell::Real`: Elliptic coefficient f²/N² (default 1.0)

# Returns
Domain-integrated potential energy (scalar).
"""
function compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector{T}; a_ell::Real=T(1.0)) where T
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny
    dz = nz > 1 ? (grid.z[2] - grid.z[1]) : T(1.0)

    psi_arr = parent(state.psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    PE = T(0)

    for k in 1:nz_local
        PE_level = T(0)
        PE_zero_mode = T(0)

        # r_1 = 1.0 (Boussinesq), r_2 = N²
        r_1 = T(1.0)
        r_2 = N2_profile[min(k, length(N2_profile))]
        coeff = a_ell * r_1 / max(r_2, eps(T))

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Compute buoyancy b = ∂ψ/∂z / r_1 using finite differences
            if k < nz_local
                psi_up = psi_arr[i_local, j_local, k+1]
                psi_curr = psi_arr[i_local, j_local, k]
                b_k = (psi_up - psi_curr) / (r_1 * dz)
            else
                # At top boundary, use one-sided difference or set to zero (Neumann BC)
                b_k = complex(T(0))
            end

            energy_mode = abs2(b_k) * coeff
            PE_level += energy_mode

            # Track zero mode
            i_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(i_local, 1, grid) : i_local
            j_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(j_local, 2, grid) : j_local
            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]

            if kx_val == 0 && ky_val == 0
                PE_zero_mode = energy_mode
            end
        end

        # Dealiasing correction
        PE_level -= T(0.5) * PE_zero_mode
        PE += PE_level
    end

    PE *= T(0.5) / (nx * ny * nz)
    return PE
end

"""
    compute_wave_energy(state::State, grid::Grid, plans)

Compute wave energy from the YBJ+ wave envelope B.

Wave energy is computed as:
    WE = (1/2) ∑_{kx,ky,z} |B|²

with proper dealiasing correction.

# Arguments
- `state::State`: Current model state with B (wave envelope)
- `grid::Grid`: Grid structure
- `plans`: FFT plans

# Returns
Domain-integrated wave energy (scalar).
"""
function compute_wave_energy(state::State, grid::Grid, plans)
    T = eltype(real(state.B[1]))
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny

    B_arr = parent(state.B)
    nx_local, ny_local, nz_local = size(B_arr)

    WE = T(0)

    for k in 1:nz_local
        WE_level = T(0)
        WE_zero_mode = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            B_k = B_arr[i_local, j_local, k]
            energy_mode = abs2(B_k)
            WE_level += energy_mode

            # Track zero mode
            i_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(i_local, 1, grid) : i_local
            j_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(j_local, 2, grid) : j_local
            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]

            if kx_val == 0 && ky_val == 0
                WE_zero_mode = energy_mode
            end
        end

        # Dealiasing correction
        WE_level -= T(0.5) * WE_zero_mode
        WE += WE_level
    end

    WE *= T(0.5) / (nx * ny * nz)
    return WE
end

"""
    compute_enstrophy(state::State, grid::Grid, plans)

Compute domain-integrated enstrophy (mean squared vorticity).

Enstrophy is computed as:
    Z = (1/2) ∑_{kx,ky,z} |ζ|²

where ζ = ∇²ψ = -kh²ψ is the relative vorticity (in spectral space).

# Arguments
- `state::State`: Current model state with psi (streamfunction)
- `grid::Grid`: Grid structure with wavenumbers
- `plans`: FFT plans

# Returns
Domain-integrated enstrophy (scalar).
"""
function compute_enstrophy(state::State, grid::Grid, plans)
    T = eltype(real(state.psi[1]))
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny

    psi_arr = parent(state.psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    Z = T(0)

    for k in 1:nz_local
        Z_level = T(0)
        Z_zero_mode = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Get wavenumbers
            i_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(i_local, 1, grid) : i_local
            j_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(j_local, 2, grid) : j_local

            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]
            kh2 = kx_val^2 + ky_val^2

            psi_k = psi_arr[i_local, j_local, k]

            # Relative vorticity ζ = ∇²ψ = -kh²ψ (in spectral space)
            zeta_k = -kh2 * psi_k

            enstrophy_mode = abs2(zeta_k)
            Z_level += enstrophy_mode

            # Track zero mode
            if kx_val == 0 && ky_val == 0
                Z_zero_mode = enstrophy_mode
            end
        end

        # Dealiasing correction
        Z_level -= T(0.5) * Z_zero_mode
        Z += Z_level
    end

    Z *= T(0.5) / (nx * ny * nz)
    return Z
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
