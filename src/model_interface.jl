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
using ..QGYBJ: a_ell_ut, dealias_mask
using ..QGYBJ: OutputManager, write_state_file, OutputConfig, ParallelConfig

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
    # For constant_N stratification, use N0² from config; otherwise use default
    N²_value = if config.stratification.type == :constant_N
        T(config.stratification.N0^2)  # N² = N0²
    else
        T(1.0)  # Default for non-constant stratification (profile-based)
    end

    params = QGParams{T}(;
        nx = config.domain.nx,
        ny = config.domain.ny,
        nz = config.domain.nz,
        Lx = config.domain.Lx,
        Ly = config.domain.Ly,
        Lz = config.domain.Lz,  # Was missing!
        dt = config.dt,
        nt = ceil(Int, config.total_time / config.dt),
        f₀ = config.f0,
        νₕ = config.nu_h,
        νᵥ = config.nu_v,
        linear_vert_structure = 0,
        stratification = config.stratification.type,
        W2F = T(1e-6),  # Default wave-to-flow energy ratio
        N² = N²_value,  # From config.stratification.N0 for constant_N
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

Energy diagnostics are saved to separate files in the diagnostic/ folder:
- wave_KE.nc: Wave kinetic energy
- wave_PE.nc: Wave potential energy
- wave_CE.nc: Wave correction energy (YBJ+)
- mean_flow_KE.nc: Mean flow kinetic energy
- mean_flow_PE.nc: Mean flow potential energy
- total_energy.nc: Summary file with all energies
"""
function compute_and_output_diagnostics!(sim::QGYBJSimulation{T}) where T
    diagnostics = Dict{String, Any}()

    # Mean flow energy diagnostics
    mean_flow_KE = compute_kinetic_energy(sim.state, sim.grid, sim.plans)
    mean_flow_PE = compute_potential_energy(sim.state, sim.grid, sim.plans, sim.N2_profile)

    # Wave energy diagnostics (detailed: WKE, WPE, WCE)
    wave_KE, wave_PE, wave_CE = compute_detailed_wave_energy(sim.state, sim.grid, sim.params)

    # Record to energy diagnostics manager (will be written to diagnostic/ folder)
    record_energies!(
        sim.energy_diagnostics_manager,
        sim.current_time,
        wave_KE,
        wave_PE,
        wave_CE,
        mean_flow_KE,
        mean_flow_PE
    )

    # Store in diagnostics dict for backward compatibility
    diagnostics["mean_flow_kinetic_energy"] = mean_flow_KE
    diagnostics["mean_flow_potential_energy"] = mean_flow_PE
    diagnostics["wave_kinetic_energy"] = wave_KE
    diagnostics["wave_potential_energy"] = wave_PE
    diagnostics["wave_correction_energy"] = wave_CE
    diagnostics["total_wave_energy"] = wave_KE + wave_PE + wave_CE
    diagnostics["total_mean_flow_energy"] = mean_flow_KE + mean_flow_PE
    diagnostics["total_energy"] = mean_flow_KE + mean_flow_PE + wave_KE + wave_PE + wave_CE

    # Legacy names for backward compatibility
    diagnostics["kinetic_energy"] = mean_flow_KE
    diagnostics["potential_energy"] = mean_flow_PE
    diagnostics["wave_energy"] = wave_KE + wave_PE + wave_CE

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

    # Write to legacy diagnostics file (for backward compatibility)
    write_diagnostics_file(sim.output_manager, diagnostics, sim.current_time)
end

"""
    compute_detailed_wave_energy(state::State, grid::Grid, params::QGParams) -> (WKE, WPE, WCE)

Compute detailed wave energy components following the Fortran wave_energy routine:
- WKE: Wave kinetic energy from |B|²
- WPE: Wave potential energy from |C|² where C = ∂A/∂z
- WCE: Wave correction energy from |A|² (YBJ+ higher-order term)

Matches the Fortran `wave_energy` subroutine in QG_YBJp/diagnostics.f90.
"""
function compute_detailed_wave_energy(state::State, grid::Grid, params::QGParams{T}) where T
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny
    a_ell = params.f₀^2 / params.N²  # Elliptic coefficient

    # Get arrays
    B_arr = parent(state.B)
    A_arr = parent(state.A)
    C_arr = parent(state.C)

    nx_local, ny_local, nz_local = size(B_arr)

    WKE = T(0)
    WPE = T(0)
    WCE = T(0)

    for k in 1:nz_local
        WKE_level = T(0)
        WPE_level = T(0)
        WCE_level = T(0)
        WKE_zero = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Get wavenumbers
            i_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(i_local, 1, grid) : i_local
            j_global = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing ?
                       local_to_global(j_local, 2, grid) : j_local

            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]
            kh2 = kx_val^2 + ky_val^2

            B_k = B_arr[i_local, j_local, k]
            A_k = A_arr[i_local, j_local, k]
            C_k = C_arr[i_local, j_local, k]

            # Split B into real/imag parts for proper energy calculation
            BR = real(B_k)
            BI = imag(B_k)

            # WKE: |BR|² + |BI|² (wave kinetic energy)
            wke_contrib = BR^2 + BI^2
            WKE_level += wke_contrib

            # WPE: (0.5/a_ell) × kh² × |C|² (wave potential energy)
            # C = ∂A/∂z, represents vertical wave structure
            CR = real(C_k)
            CI = imag(C_k)
            wpe_contrib = (T(0.5) / max(a_ell, eps(T))) * kh2 * (CR^2 + CI^2)
            WPE_level += wpe_contrib

            # WCE: (1/8) × (1/a_ell²) × kh⁴ × |A|² (wave correction energy, YBJ+)
            AR = real(A_k)
            AI = imag(A_k)
            wce_contrib = (T(1)/T(8)) * (T(1)/(a_ell^2 + eps(T))) * kh2^2 * (AR^2 + AI^2)
            WCE_level += wce_contrib

            # Track zero mode for dealiasing correction
            if kx_val == 0 && ky_val == 0
                WKE_zero = wke_contrib
            end
        end

        # Dealiasing correction for WKE
        WKE_level -= T(0.5) * WKE_zero

        WKE += WKE_level
        WPE += WPE_level
        WCE += WCE_level
    end

    # Normalize by grid size
    norm_factor = T(0.5) / (nx * ny * nz)
    WKE *= norm_factor
    WPE *= norm_factor
    WCE *= norm_factor

    return WKE, WPE, WCE
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

#= ============================================================================
                    SIMPLIFIED SIMULATION RUNNER
============================================================================ =#

"""
    run_simulation!(S, G, par, plans; kwargs...)

Run a complete QG-YBJ+ simulation with automatic time-stepping, output, and diagnostics.

This is the recommended high-level interface that handles all the details of:
- Leapfrog time integration with Robert-Asselin filter
- State array rotation (no manual management needed)
- Periodic output to NetCDF files
- Progress reporting and diagnostics

# Arguments
- `S::State`: Initial state (will be modified in-place)
- `G::Grid`: Spatial grid
- `par::QGParams`: Model parameters (includes dt, nt)
- `plans`: FFT plans

# Keyword Arguments
- `output_config::OutputConfig`: Output configuration (required for saving)
- `mpi_config=nothing`: MPI configuration for parallel runs
- `parallel_config=nothing`: Parallel I/O configuration
- `workspace=nothing`: Pre-allocated workspace arrays
- `print_progress::Bool=true`: Print progress to stdout
- `progress_interval::Int=0`: Steps between progress updates (0 = auto, based on nt)

# Returns
- Final `State` at the end of the simulation

# Example
```julia
# Setup
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, dt=100.0, nt=5000)
G, S, plans, a = setup_model(par)

# Initialize flow and waves
initialize_dipole!(S, G, par)
initialize_surface_waves!(S, G, par)

# Configure output
output_config = OutputConfig(
    output_dir = "output",
    psi_interval = 2π / par.f₀,     # Save every inertial period
    wave_interval = 2π / par.f₀
)

# Run simulation - all time-stepping handled automatically
S_final = run_simulation!(S, G, par, plans; output_config=output_config)
```

See also: [`OutputConfig`](@ref), [`leapfrog_step!`](@ref)
"""
function run_simulation!(S::State, G::Grid, par::QGParams, plans;
                         output_config::Union{OutputConfig,Nothing}=nothing,
                         mpi_config=nothing,
                         parallel_config=nothing,
                         workspace=nothing,
                         print_progress::Bool=true,
                         progress_interval::Int=0)

    # Determine if running in MPI mode
    is_mpi = mpi_config !== nothing
    is_root = !is_mpi || mpi_config.is_root

    # Setup workspace if not provided
    # For MPI mode, init_mpi_workspace provides pre-allocated arrays for transposes
    # For serial mode, workspace=nothing is fine (arrays allocated on demand)
    if workspace === nothing && is_mpi
        workspace = init_mpi_workspace(G, mpi_config)
    end

    # Setup parallel config if not provided (for I/O)
    if parallel_config === nothing && is_mpi
        parallel_config = ParallelConfig(
            use_mpi = true,
            comm = mpi_config.comm,
            parallel_io = false
        )
    elseif parallel_config === nothing
        parallel_config = ParallelConfig(use_mpi = false)
    end

    # Compute coefficients
    a_ell = a_ell_ut(par, G)
    L_mask = dealias_mask(G)

    # Initial velocity computation
    compute_velocities!(S, G; plans=plans, params=par, workspace=workspace)

    # Create output manager if config provided
    output_manager = nothing
    if output_config !== nothing
        output_manager = OutputManager(output_config, par, parallel_config)

        # Save initial state
        if is_root && print_progress
            println("Saving initial state...")
        end
        write_state_file(output_manager, S, G, plans, 0.0, parallel_config; params=par)
    end

    # First projection step (Forward Euler to initialize leapfrog)
    first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace)

    # Setup leapfrog states
    Sn = deepcopy(S)
    Snm1 = deepcopy(S)
    Snp1 = deepcopy(S)

    # Determine progress interval
    nt = par.nt
    dt = par.dt
    if progress_interval <= 0
        progress_interval = max(1, nt ÷ 20)  # ~20 progress updates
    end

    # Compute save intervals in steps
    psi_save_steps = output_config !== nothing && output_config.psi_interval > 0 ?
                     max(1, round(Int, output_config.psi_interval / dt)) : 0
    wave_save_steps = output_config !== nothing && output_config.wave_interval > 0 ?
                      max(1, round(Int, output_config.wave_interval / dt)) : 0
    save_steps = psi_save_steps > 0 ? psi_save_steps : wave_save_steps

    # Print header
    if is_root && print_progress
        println("\n" * "="^60)
        println("Starting time integration...")
        println("  Steps: $nt, dt: $dt")
        if save_steps > 0
            println("  Saving every $save_steps steps")
        end
        println("="^60 * "\n")
    end

    # Time integration loop
    for step in 1:nt
        # Leapfrog step
        leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                       a=a_ell, dealias_mask=L_mask, workspace=workspace)

        # Rotate states: (n-1) ← (n) ← (n+1) ← (n-1)
        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        current_time = step * dt

        # Progress output
        if is_root && print_progress && step % progress_interval == 0
            progress_pct = round(100 * step / nt, digits=1)
            @printf("  Step %d/%d (%.1f%%) - t = %.2e\n", step, nt, progress_pct, current_time)
        end

        # Save state
        if output_manager !== nothing && save_steps > 0 && step % save_steps == 0
            write_state_file(output_manager, Sn, G, plans, current_time, parallel_config; params=par)
        end
    end

    # Copy final state back to S
    copyto!(parent(S.psi), parent(Sn.psi))
    copyto!(parent(S.q), parent(Sn.q))
    copyto!(parent(S.B), parent(Sn.B))
    copyto!(parent(S.A), parent(Sn.A))
    copyto!(parent(S.u), parent(Sn.u))
    copyto!(parent(S.v), parent(Sn.v))
    copyto!(parent(S.w), parent(Sn.w))

    # Print completion message
    if is_root && print_progress
        println("\n" * "="^60)
        println("Simulation complete!")
        if output_manager !== nothing
            println("Output saved to: $(output_config.output_dir)/")
        end
        println("="^60)
    end

    return S
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
    create_simple_config(; Lx, Ly, Lz, kwargs...)

Create a simple configuration for quick testing.

# Arguments
- `Lx, Ly, Lz`: Domain size in meters (REQUIRED - no defaults)
- `kwargs...`: Additional parameters passed to sub-configs

# Example
```julia
config = create_simple_config(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km
```
"""
function create_simple_config(; Lx::Real, Ly::Real, Lz::Real, kwargs...)
    # Default simple configuration
    domain = create_domain_config(nx=64, ny=64, nz=32, Lx=Lx, Ly=Ly, Lz=Lz)
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
        Lz = config.domain.Lz,  # Was missing!
        dt = config.dt,
        nt = ceil(Int, config.total_time / config.dt),
        f₀ = config.f0  # Note: QGParams uses f₀, ModelConfig uses f0
    )

    grid = init_grid(params)
    state = init_state(grid)  # Fixed: init_state takes Grid, not QGParams
    plans = plan_transforms!(grid)

    return params, grid, state, plans
end
