"""
High-level model interface for QG-YBJ simulations.

This module provides the main user interface for setting up and running QG-YBJ simulations
with the configuration system, including time stepping with output management.
"""

using Printf
using ..QGYBJplus: QGParams, Grid, State, setup_model, default_params
using ..QGYBJplus: plan_transforms!, init_grid, init_state, fft_backward!
using ..QGYBJplus: first_projection_step!, leapfrog_step!
using ..QGYBJplus: IMEXWorkspace, init_imex_workspace, imex_cn_step!
using ..QGYBJplus: invert_q_to_psi!, invert_B_to_A!, compute_velocities!
using ..QGYBJplus: local_to_global
using ..QGYBJplus: transpose_to_z_pencil!, local_to_global_z, allocate_z_pencil
using ..QGYBJplus: a_ell_ut, dealias_mask
using ..QGYBJplus: OutputManager, write_state_file, OutputConfig, MPIConfig
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
import PencilArrays: PencilArray

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

# Energy diagnostics module for separate file output
using ..QGYBJplus.EnergyDiagnostics: EnergyDiagnosticsManager, should_output, record_energies!
using ..QGYBJplus.EnergyDiagnostics: write_all_energy_files!, finalize!

# Global energy diagnostics (MPI-aware)
using ..QGYBJplus.Diagnostics: flow_kinetic_energy_global, wave_energy_global


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
    
    # MPI configuration
    parallel_config::MPIConfig
    
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
    setup_simulation(config::ModelConfig; topology=nothing)

Set up a complete QG-YBJ simulation from configuration.

MPI is automatically initialized if not already done.

# Arguments
- `config::ModelConfig`: Model configuration
- `topology`: Optional MPI process topology (px, py). Auto-computed if not provided.
"""
function setup_simulation(config::ModelConfig{T}; topology=nothing) where T
    @info "Setting up QG-YBJ simulation"

    # Initialize MPI environment (always required)
    parallel_config = setup_mpi_environment(; topology=topology)

    # Print parallel info (only rank 0)
    if parallel_config.is_root
        @info "Running with MPI: $(parallel_config.nprocs) processes, topology=$(parallel_config.topology)"
    end

    # Validate configuration (only on rank 0 to avoid spam)
    should_print = parallel_config.is_root
    
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
        Lz = config.domain.Lz,
        x0 = config.domain.x0,
        y0 = config.domain.y0,

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
        νₕ₁ = T(config.nu_h1), νₕ₂ = T(config.nu_h2), 
        
        ilap1 = config.ilap1, ilap2 = config.ilap2,
        νₕ₁ʷ = T(config.nu_h1_wave), νₕ₂ʷ = T(config.nu_h2_wave), 
        ilap1w = config.ilap1_wave, ilap2w = config.ilap2_wave,
        νz = T(config.nu_v),  # Vertical diffusion coefficient for q

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
    
    # Initialize grid and state with MPI
    if should_print
        @info "Initializing MPI grid and state"
    end

    grid = init_mpi_grid(params, parallel_config)
    plans = plan_mpi_transforms(grid, parallel_config)
    state = init_mpi_state(grid, plans, parallel_config)
    state_old = init_mpi_state(grid, plans, parallel_config)
    
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
    
    # Initialize fields (MPI-aware)
    parallel_initialize_fields!(state, grid, plans, config, parallel_config; params=params, N2_profile=N2_profile)
    
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

Uses leapfrog time stepping with Robert-Asselin filter, which requires
three time levels (n-1, n, n+1). The simulation struct only stores two
states, so a third is allocated internally for the integration.
"""
function run_simulation!(sim::QGYBJSimulation{T}; progress_callback=nothing) where T
    @info "Starting QG-YBJ simulation"
    @info "Total time: $(sim.config.total_time), Time step: $(sim.params.dt), Steps: $(sim.params.nt)"

    # Compute required coefficients
    # Use custom N2_profile for elliptic operators if provided, otherwise use params-derived profile
    # This ensures consistency between the stratification used in elliptic inversions (q→ψ, B→A)
    # and vertical velocity calculations. Previously, custom N2_profiles were only used for
    # vertical velocity, causing inconsistent physics in non-constant stratification runs.
    if sim.N2_profile !== nothing && !isempty(sim.N2_profile)
        a_ell = a_ell_from_N2(sim.N2_profile, sim.params)
        @info "Using custom N² profile for elliptic operators ($(length(sim.N2_profile)) levels)"
    else
        a_ell = a_ell_ut(sim.params, sim.grid)
    end
    L_mask = dealias_mask(sim.grid)

    # Write initial output
    write_psi = should_output_psi(sim.output_manager, sim.current_time)
    write_waves = should_output_waves(sim.output_manager, sim.current_time)
    if write_psi || write_waves
        write_state_file(sim.output_manager, sim.state, sim.grid, sim.plans,
                        sim.current_time; params=sim.params,
                        write_psi=write_psi, write_waves=write_waves)
    end

    # Leapfrog requires 3 time levels: Snm1 (n-1), Sn (n), Snp1 (n+1)
    # sim.state will be used as the "current" state for output
    # We allocate working states for the time integration

    # Save initial state as Snm1 (time 0)
    # Use copy_state instead of deepcopy to preserve PencilArray topology
    Snm1 = copy_state(sim.state)

    # First projection step (Forward Euler to initialize leapfrog)
    # Advances sim.state from time 0 to time dt
    @info "Performing first projection step"
    first_projection_step!(sim.state, sim.grid, sim.params, sim.plans;
                          a=a_ell, dealias_mask=L_mask, N2_profile=sim.N2_profile)

    sim.time_step = 1
    sim.current_time = sim.params.dt

    # Sn = state at time dt (after first projection step)
    Sn = copy_state(sim.state)
    # Snp1 will hold state at time 2*dt (computed by first leapfrog step)
    Snp1 = copy_state(sim.state)

    # Output after first step if needed
    check_and_output!(sim)

    # Main time stepping loop
    @info "Starting main time integration loop"

    for step in 2:sim.params.nt
        # Leapfrog time step: compute Snp1 from Sn and Snm1
        leapfrog_step!(Snp1, Sn, Snm1, sim.grid, sim.params, sim.plans;
                      a=a_ell, dealias_mask=L_mask, N2_profile=sim.N2_profile)

        # Rotate states: (n-1) ← (n) ← (n+1) ← (n-1)
        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        # Update sim.state to point to current state (for output/diagnostics)
        # Copy the current state (Sn after rotation) to sim.state
        sim.state.q .= Sn.q
        sim.state.psi .= Sn.psi
        sim.state.B .= Sn.B
        sim.state.A .= Sn.A
        sim.state.C .= Sn.C
        sim.state.u .= Sn.u
        sim.state.v .= Sn.v
        sim.state.w .= Sn.w

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
    write_psi = should_output_psi(sim.output_manager, sim.current_time)
    write_waves = should_output_waves(sim.output_manager, sim.current_time)
    if write_psi || write_waves
        write_state_file(sim.output_manager, sim.state, sim.grid, sim.plans,
                        sim.current_time, sim.parallel_config; params=sim.params,
                        write_psi=write_psi, write_waves=write_waves)
    end
end

# MPI reduction functions are now defined in parallel_mpi.jl
# Re-export for backward compatibility
const reduce_if_mpi = reduce_sum_if_mpi

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

In MPI mode, energies are reduced across all processes to get global totals.
"""
function compute_and_output_diagnostics!(sim::QGYBJSimulation{T}) where T
    diagnostics = Dict{String, Any}()

    # Compute f² for potential energy scaling
    # PE = (1/2) ∫ (ψ_z)² × (f²/N²) dV, so a_ell should be f²
    # (the function already divides by N² internally via r_2)
    f_squared = sim.params.f₀^2

    # Mean flow energy diagnostics (local)
    mean_flow_KE_local = compute_kinetic_energy(sim.state, sim.grid, sim.plans)
    mean_flow_PE_local = compute_potential_energy(sim.state, sim.grid, sim.plans, sim.N2_profile; a_ell=f_squared)

    # Wave energy diagnostics (local, detailed: WKE, WPE, WCE)
    wave_KE_local, wave_PE_local, wave_CE_local = compute_detailed_wave_energy(
        sim.state, sim.grid, sim.params; N2_profile=sim.N2_profile
    )

    # Reduce across MPI processes if running in parallel
    mean_flow_KE = reduce_if_mpi(mean_flow_KE_local, sim.parallel_config)
    mean_flow_PE = reduce_if_mpi(mean_flow_PE_local, sim.parallel_config)

    wave_KE = reduce_if_mpi(wave_KE_local, sim.parallel_config)
    wave_PE = reduce_if_mpi(wave_PE_local, sim.parallel_config)
    wave_CE = reduce_if_mpi(wave_CE_local, sim.parallel_config)

    # Record energies only on root to avoid concurrent NetCDF writes.
    if sim.parallel_config === nothing || sim.parallel_config.is_root
        record_energies!(
            sim.energy_diagnostics_manager,
            sim.current_time,
            wave_KE,
            wave_PE,
            wave_CE,
            mean_flow_KE,
            mean_flow_PE
        )
    end

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

    # Domain-integrated quantities (with MPI reduction)
    enstrophy_local = compute_enstrophy(sim.state, sim.grid, sim.plans)
    diagnostics["total_enstrophy"] = reduce_sum_if_mpi(enstrophy_local, sim.parallel_config)

    # Extrema (with MPI reduction for global min/max)
    # Note: fft_backward! returns complex arrays, extract real part for diagnostics
    psir_complex = _allocate_fft_dst(sim.state.psi, sim.plans)
    fft_backward!(psir_complex, sim.state.psi, sim.plans)
    psir = real.(parent(psir_complex))
    diagnostics["psi_min"] = reduce_min_if_mpi(minimum(psir), sim.parallel_config)
    diagnostics["psi_max"] = reduce_max_if_mpi(maximum(psir), sim.parallel_config)

    # RMS needs sum reduction, then divide by global size
    psi_sum_sq  = reduce_sum_if_mpi(sum(abs2, psir), sim.parallel_config)
    global_size = reduce_sum_if_mpi(length(psir), sim.parallel_config)
    diagnostics["psi_rms"] = sqrt(psi_sum_sq / global_size)

    # Wave field extrema (with MPI reduction)
    # Transform full complex B to physical space, then extract real part
    Br_complex = _allocate_fft_dst(sim.state.B, sim.plans)
    fft_backward!(Br_complex, sim.state.B, sim.plans)
    Br = real.(parent(Br_complex))
    diagnostics["wave_min"] = reduce_min_if_mpi(minimum(Br), sim.parallel_config)
    diagnostics["wave_max"] = reduce_max_if_mpi(maximum(Br), sim.parallel_config)

    # Wave RMS with global reduction
    wave_sum_sq      = reduce_sum_if_mpi(sum(abs2, Br), sim.parallel_config)
    wave_global_size = reduce_sum_if_mpi(length(Br), sim.parallel_config)
    diagnostics["wave_rms"] = sqrt(wave_sum_sq / wave_global_size)

    # Store in simulation object
    sim.diagnostics["step_$(sim.time_step)"] = diagnostics

    # Write to legacy diagnostics file on root only (serial NetCDF).
    if sim.parallel_config === nothing || sim.parallel_config.is_root
        write_diagnostics_file(sim.output_manager, diagnostics, sim.current_time)
    else
        # Keep counters/timestamps consistent across ranks.
        sim.output_manager.diagnostics_counter += 1
        sim.output_manager.last_diagnostics_output = sim.current_time
    end
end

"""
    compute_detailed_wave_energy(state::State, grid::Grid, params::QGParams; N2_profile=nothing) -> (WKE, WPE, WCE)

Compute detailed wave energy components per YBJ+ paper:
- WKE: Wave kinetic energy = (1/2)|LA|² per equation (4.7)
       where LA = ∂_z(a(z) × C) using the L operator from equation (1.3)
       with a(z) = f²/N² and C = ∂A/∂z
- WPE: Wave potential energy from |C|² where C = ∂A/∂z
- WCE: Wave correction energy from |A|² (YBJ+ higher-order term)

# Keyword Arguments
- `N2_profile`: Optional N²(z) profile for variable stratification. When provided,
  uses `a_ell(z) = f₀²/N²(z)` per vertical level.
"""
function compute_detailed_wave_energy(state::State, grid::Grid, params::QGParams{T}; N2_profile=nothing) where T
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny
    # Guard against N² ≈ 0 to avoid division by zero
    N2_safe = max(params.N², eps(T))
    a_ell_const = params.f₀^2 / N2_safe  # Elliptic coefficient (constant stratification)
    use_profile = N2_profile !== nothing && length(N2_profile) == nz

    # Build a_ell profile
    a_ell_arr = if use_profile
        [params.f₀^2 / max(N2_profile[k], eps(T)) for k in 1:nz]
    else
        fill(a_ell_const, nz)
    end

    # Get arrays
    B_arr = parent(state.B)
    A_arr = parent(state.A)
    C_arr = parent(state.C)

    nz_local, nx_local, ny_local = size(B_arr)

    # Grid spacing for vertical derivative
    Δz = nz > 1 ? abs(grid.z[2] - grid.z[1]) : T(1)

    WKE = T(0)
    WPE = T(0)
    WCE = T(0)

    # For WKE, compute LA = ∂_z(a × C) using the L operator from equation (1.3)
    # Loop over (i,j) modes first, then compute LA across z levels
    for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, state.B)
        j_global = local_to_global(j_local, 3, state.B)

        kx_val = grid.kx[min(i_global, length(grid.kx))]
        ky_val = grid.ky[min(j_global, length(grid.ky))]
        kh2 = kx_val^2 + ky_val^2

        for k in 1:nz_local
            k_global = local_to_global(k, 1, state.B)
            a_ell = a_ell_arr[min(k_global, nz)]

            C_k = C_arr[k, i_local, j_local]
            A_k = A_arr[k, i_local, j_local]
            CR = real(C_k)
            CI = imag(C_k)
            AR = real(A_k)
            AI = imag(A_k)

            # Compute LA = ∂_z(a × C) using finite differences
            # C[k] is at interface z = k*Δz, a[k] is at z = (k-1)*Δz
            # So (a×C) at interface k uses a[k+1] (both at z = k*Δz)
            # LA[k] = (a[k+1] × C[k] - a[k] × C[k-1]) / Δz
            if nz == 1
                LA_r = T(0)
                LA_i = T(0)
            elseif k_global == 1
                # Bottom boundary: C[0] = 0 (Neumann BC), so only upper flux
                # Interface above cell 1 is at z = Δz, uses a[2]
                a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
                LA_r = a_ell_kp1 * CR / Δz
                LA_i = a_ell_kp1 * CI / Δz
            elseif k_global == nz
                # Top boundary: C[nz] = 0 (Neumann BC), so only lower flux
                # Interface below cell nz is at z = (nz-1)*Δz, uses a[nz]
                CR_km1 = real(C_arr[k-1, i_local, j_local])
                CI_km1 = imag(C_arr[k-1, i_local, j_local])
                LA_r = -a_ell * CR_km1 / Δz
                LA_i = -a_ell * CI_km1 / Δz
            else
                # Interior: LA[k] = (a[k+1]*C[k] - a[k]*C[k-1]) / Δz
                a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
                CR_km1 = real(C_arr[k-1, i_local, j_local])
                CI_km1 = imag(C_arr[k-1, i_local, j_local])
                LA_r = (a_ell_kp1 * CR - a_ell * CR_km1) / Δz
                LA_i = (a_ell_kp1 * CI - a_ell * CI_km1) / Δz
            end

            # WKE contribution (factor of 0.5 in norm_factor)
            WKE += LA_r^2 + LA_i^2

            # WPE: (0.5/a_ell) × kh² × |C|²
            WPE += (T(0.5) / max(a_ell, eps(T))) * kh2 * (CR^2 + CI^2)

            # WCE: (1/8) × (1/a_ell²) × kh⁴ × |A|²
            WCE += (T(1)/T(8)) * (T(1) / max(a_ell^2, eps(T))) * kh2^2 * (AR^2 + AI^2)
        end
    end

    # Dealiasing correction for WKE at kh=0 mode
    # Uses same corrected indexing as main WKE loop
    wke_zero = T(0)
    for k in 1:nz_local
        k_global = local_to_global(k, 1, state.B)
        a_ell = a_ell_arr[min(k_global, nz)]
        CR = real(C_arr[k, 1, 1])
        CI = imag(C_arr[k, 1, 1])

        if nz == 1
            LA_r = T(0)
            LA_i = T(0)
        elseif k_global == 1
            a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
            LA_r = a_ell_kp1 * CR / Δz
            LA_i = a_ell_kp1 * CI / Δz
        elseif k_global == nz
            CR_km1 = real(C_arr[k-1, 1, 1])
            CI_km1 = imag(C_arr[k-1, 1, 1])
            LA_r = -a_ell * CR_km1 / Δz
            LA_i = -a_ell * CI_km1 / Δz
        else
            a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
            CR_km1 = real(C_arr[k-1, 1, 1])
            CI_km1 = imag(C_arr[k-1, 1, 1])
            LA_r = (a_ell_kp1 * CR - a_ell * CR_km1) / Δz
            LA_i = (a_ell_kp1 * CI - a_ell * CI_km1) / Δz
        end
        wke_zero += LA_r^2 + LA_i^2
    end
    WKE -= T(0.5) * wke_zero

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
    nz_local, nx_local, ny_local = size(psi_arr)

    KE = T(0)

    for k in 1:nz_local
        KE_level = T(0)
        KE_zero_mode = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Get wavenumbers
            i_global = local_to_global(i_local, 2, state.psi)
            j_global = local_to_global(j_local, 3, state.psi)

            kx_val = grid.kx[min(i_global, length(grid.kx))]

            ky_val = grid.ky[min(j_global, length(grid.ky))]

            psi_k = psi_arr[k, i_local, j_local]

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
    compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector; a_ell::Real=1.0, workspace=nothing)

Compute domain-integrated potential energy following the Fortran diag_zentrum routine.

The potential energy is computed as:
    PE = (1/2) ∑_{kx,ky,z} |ψ_z|² × (a_ell / N²(z))

For dimensionally correct QG available potential energy, use `a_ell = f₀²`:
    PE = (1/2) ∑_{kx,ky,z} |ψ_z|² × (f₀² / N²(z))

This matches the standard QG APE formula: PE = (1/2) ∫ (f²/N²) (∂ψ/∂z)² dV

# Arguments
- `state::State`: Current model state with psi (streamfunction)
- `grid::Grid`: Grid structure
- `plans`: FFT plans
- `N2_profile::Vector`: Buoyancy frequency squared N²(z)
- `a_ell::Real`: Scaling factor (default 1.0). For physical PE, use `f₀²` (Coriolis squared).
- `workspace`: Optional MPI workspace with `psi_z` for z-pencil operations.

# Returns
Domain-integrated potential energy (scalar).

# Example
```julia
# Dimensionless PE (a_ell=1, default)
PE_nondim = compute_potential_energy(state, grid, plans, N2_profile)

# Physical PE with f² scaling
PE_phys = compute_potential_energy(state, grid, plans, N2_profile; a_ell=params.f₀^2)
```
"""
function compute_potential_energy(state::State, grid::Grid, plans, N2_profile::Vector{T}; a_ell::Real=T(1.0), workspace=nothing) where T
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny
    dz = nz > 1 ? (grid.z[2] - grid.z[1]) : T(1.0)

    use_transpose = hasfield(typeof(grid), :decomp) && grid.decomp !== nothing &&
                    hasfield(typeof(grid.decomp), :pencil_z)

    psi_field = state.psi
    if use_transpose
        psi_z = workspace !== nothing && hasfield(typeof(workspace), :psi_z) ?
                workspace.psi_z : allocate_z_pencil(grid, eltype(state.psi))
        transpose_to_z_pencil!(psi_z, state.psi, grid)
        psi_field = psi_z
    end

    psi_arr = parent(psi_field)
    nz_local, nx_local, ny_local = size(psi_arr)

    PE = T(0)

    for k in 1:nz_local
        PE_level = T(0)
        PE_zero_mode = T(0)

        # Use global z-index for N2_profile lookup in 2D decomposition
        k_global = use_transpose ? local_to_global_z(k, 1, grid) : local_to_global(k, 1, psi_field)

        # r_1 = 1.0 (Boussinesq), r_2 = N²
        r_1 = T(1.0)
        r_2 = N2_profile[min(k_global, length(N2_profile))]
        coeff = a_ell * r_1 / max(r_2, eps(T))

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Compute buoyancy b = ∂ψ/∂z / r_1 using finite differences
            if k < nz_local
                psi_up = psi_arr[k+1, i_local, j_local]
                psi_curr = psi_arr[k, i_local, j_local]
                b_k = (psi_up - psi_curr) / (r_1 * dz)
            else
                # At top boundary, use one-sided difference or set to zero (Neumann BC)
                b_k = complex(T(0))
            end

            energy_mode = abs2(b_k) * coeff
            PE_level += energy_mode

            # Track zero mode
            i_global = use_transpose ?
                       local_to_global_z(i_local, 2, grid) :
                       local_to_global(i_local, 2, psi_field)

            j_global = use_transpose ?
                       local_to_global_z(j_local, 3, grid) :
                       local_to_global(j_local, 3, psi_field)

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
    compute_wave_energy(state::State, grid::Grid, plans; params=nothing)

Compute wave kinetic energy per YBJ+ equation (4.7).

Wave kinetic energy is computed as:
    WKE = (1/2) ∑_{kx,ky,z} |LA|²

where LA = ∂_z(a(z) × C) using the L operator from equation (1.3),
with a(z) = f²/N² and C = ∂A/∂z.

# Arguments
- `state::State`: Current model state with B, A, and C fields
- `grid::Grid`: Grid structure with wavenumbers
- `plans`: FFT plans
- `params`: Optional QGParams for stratification (uses constant N² if not provided)

# Returns
Domain-integrated wave kinetic energy (scalar).
"""
function compute_wave_energy(state::State, grid::Grid, plans; params=nothing)
    T = eltype(real(state.B[1]))
    nz = grid.nz
    nx = grid.nx
    ny = grid.ny

    # Get elliptic coefficient a = f²/N²
    a_ell_const = if params !== nothing
        params.f₀^2 / max(params.N², eps(T))
    else
        T(1)  # Default to 1 if no params
    end
    a_ell_arr = fill(a_ell_const, nz)

    C_arr = parent(state.C)
    nz_local, nx_local, ny_local = size(C_arr)

    # Grid spacing for vertical derivative
    Δz = nz > 1 ? abs(grid.z[2] - grid.z[1]) : T(1)

    WE = T(0)

    # Compute LA = ∂_z(a × C) for each mode
    # Compute LA = ∂_z(a × C) for each mode
    # C[k] is at interface z = k*Δz, a[k] is at z = (k-1)*Δz
    # So (a×C) at interface k uses a[k+1] (both at z = k*Δz)
    for j_local in 1:ny_local, i_local in 1:nx_local
        for k in 1:nz_local
            k_global = local_to_global(k, 1, state.C)
            a_ell = a_ell_arr[min(k_global, nz)]
            CR = real(C_arr[k, i_local, j_local])
            CI = imag(C_arr[k, i_local, j_local])

            # Compute LA = ∂_z(a × C) using finite differences
            # LA[k] = (a[k+1] × C[k] - a[k] × C[k-1]) / Δz
            if nz == 1
                LA_r = T(0)
                LA_i = T(0)
            elseif k_global == 1
                # Bottom: C[0] = 0 (Neumann BC), interface above uses a[2]
                a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
                LA_r = a_ell_kp1 * CR / Δz
                LA_i = a_ell_kp1 * CI / Δz
            elseif k_global == nz
                # Top: C[nz] = 0 (Neumann BC), interface below uses a[nz]
                CR_km1 = real(C_arr[k-1, i_local, j_local])
                CI_km1 = imag(C_arr[k-1, i_local, j_local])
                LA_r = -a_ell * CR_km1 / Δz
                LA_i = -a_ell * CI_km1 / Δz
            else
                # Interior: LA[k] = (a[k+1]*C[k] - a[k]*C[k-1]) / Δz
                a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
                CR_km1 = real(C_arr[k-1, i_local, j_local])
                CI_km1 = imag(C_arr[k-1, i_local, j_local])
                LA_r = (a_ell_kp1 * CR - a_ell * CR_km1) / Δz
                LA_i = (a_ell_kp1 * CI - a_ell * CI_km1) / Δz
            end

            WE += LA_r^2 + LA_i^2
        end
    end

    # Dealiasing correction for kh=0 mode (same corrected indexing)
    wke_zero = T(0)
    for k in 1:nz_local
        k_global = local_to_global(k, 1, state.C)
        a_ell = a_ell_arr[min(k_global, nz)]
        CR = real(C_arr[k, 1, 1])
        CI = imag(C_arr[k, 1, 1])

        if nz == 1
            LA_r = T(0)
            LA_i = T(0)
        elseif k_global == 1
            a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
            LA_r = a_ell_kp1 * CR / Δz
            LA_i = a_ell_kp1 * CI / Δz
        elseif k_global == nz
            CR_km1 = real(C_arr[k-1, 1, 1])
            CI_km1 = imag(C_arr[k-1, 1, 1])
            LA_r = -a_ell * CR_km1 / Δz
            LA_i = -a_ell * CI_km1 / Δz
        else
            a_ell_kp1 = a_ell_arr[min(k_global + 1, nz)]
            CR_km1 = real(C_arr[k-1, 1, 1])
            CI_km1 = imag(C_arr[k-1, 1, 1])
            LA_r = (a_ell_kp1 * CR - a_ell * CR_km1) / Δz
            LA_i = (a_ell_kp1 * CI - a_ell * CI_km1) / Δz
        end
        wke_zero += LA_r^2 + LA_i^2
    end
    WE -= T(0.5) * wke_zero

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
    nz_local, nx_local, ny_local = size(psi_arr)

    Z = T(0)

    for k in 1:nz_local
        Z_level = T(0)
        Z_zero_mode = T(0)

        for j_local in 1:ny_local, i_local in 1:nx_local
            # Get wavenumbers
            i_global = local_to_global(i_local, 2, state.psi)
            j_global = local_to_global(j_local, 3, state.psi)

            kx_val = grid.kx[min(i_global, length(grid.kx))]
            ky_val = grid.ky[min(j_global, length(grid.ky))]

            kh2 = kx_val^2 + ky_val^2

            psi_k = psi_arr[k, i_local, j_local]

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
- `N2_profile=nothing`: Optional N²(z) profile for consistent stratification physics.
  When provided, this profile is used for:
  - Elliptic coefficient a_ell = f²/N²(z) in q↔ψ and B↔A inversions
  - Vertical velocity computation (omega equation and YBJ w)
  When `nothing`, uses constant N² from `par.N²`.
- `print_progress::Bool=true`: Print progress to stdout
- `progress_interval::Int=0`: Steps between progress updates (0 = auto, based on nt)
- `timestepper::Symbol=:leapfrog`: Time-stepping method. Options:
  - `:leapfrog`: Leapfrog with Robert-Asselin filter (default, explicit, CFL-limited)
  - `:imex_cn`: IMEX Crank-Nicolson (implicit dispersion, allows larger dt)

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

# Or use IMEX for larger timesteps
S_final = run_simulation!(S, G, par, plans;
                          output_config=output_config,
                          timestepper=:imex_cn)
```

See also: `OutputConfig`, `leapfrog_step!`, `imex_cn_step!`
"""
function run_simulation!(S::State, G::Grid, par::QGParams, plans;
                         output_config::Union{OutputConfig,Nothing}=nothing,
                         mpi_config=nothing,
                         parallel_config=nothing,
                         workspace=nothing,
                         N2_profile=nothing,
                         print_progress::Bool=true,
                         progress_interval::Int=0,
                         diagnostics_interval::Int=0,
                         timestepper::Symbol=:leapfrog)

    # Setup parallel config if not provided (for I/O)
    # MPI is required, so use the mpi_config directly
    if parallel_config === nothing && mpi_config !== nothing
        parallel_config = mpi_config
    elseif parallel_config === nothing
        # If no config provided at all, setup MPI environment
        parallel_config = setup_mpi_environment()
    end

    if G.decomp !== nothing && mpi_config === nothing
        mpi_config = parallel_config
    elseif G.decomp === nothing
        mpi_config = nothing
    end

    # Determine if running in MPI mode
    is_mpi = mpi_config !== nothing
    is_root = !is_mpi || mpi_config.is_root

    # Setup workspace if not provided
    # For MPI mode, init_mpi_workspace provides pre-allocated arrays for transposes
    # For serial mode, workspace=nothing is fine (arrays allocated on demand)
    if workspace === nothing && is_mpi
        workspace = init_mpi_workspace(G, mpi_config)
    end

    # Compute coefficients
    # Use N2_profile for elliptic coefficient if provided, ensuring consistent physics
    # across q↔ψ inversions, B↔A inversions, and vertical velocity calculations
    if N2_profile !== nothing && length(N2_profile) == G.nz
        a_ell = a_ell_from_N2(N2_profile, par)
    else
        a_ell = a_ell_ut(par, G)
        # Warn if stratification is non-constant but no profile provided
        if hasfield(typeof(par), :stratification) && par.stratification != :constant_N
            @warn "run_simulation!: par.stratification=$(par.stratification) but no N2_profile provided. " *
                  "Using constant N² from par.N² for elliptic inversions. For consistent physics, " *
                  "pass N2_profile or use QGYBJSimulation API." maxlog=1
        end
    end
    L_mask = dealias_mask(G)

    # Initial velocity computation
    # Skip omega equation for IMEX-CN (not needed and expensive)
    skip_w = timestepper == :imex_cn
    compute_velocities!(S, G; plans=plans, params=par, workspace=workspace, N2_profile=N2_profile, compute_w=!skip_w)

    # Initialize A from B via elliptic inversion: A = (L⁺)⁻¹·B
    # This is needed before the first diagnostics printout (step 0)
    # For YBJ+, A is the wave amplitude and B = L⁺·A
    if par.ybj_plus
        invert_B_to_A!(S, G, par, a_ell; workspace=workspace)
    end

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

    # Determine progress interval
    nt = par.nt
    dt = par.dt
    if progress_interval <= 0
        progress_interval = max(1, nt ÷ 20)  # ~20 progress updates
    end

    # Determine diagnostics interval (defaults to progress_interval if not set)
    if diagnostics_interval <= 0
        diagnostics_interval = progress_interval
    end

    # Compute save intervals in steps
    psi_save_steps = output_config !== nothing && output_config.save_psi && output_config.psi_interval > 0 ?
                     max(1, round(Int, output_config.psi_interval / dt)) : 0
    wave_save_steps = output_config !== nothing && output_config.save_waves && output_config.wave_interval > 0 ?
                      max(1, round(Int, output_config.wave_interval / dt)) : 0
    # Print header
    timestepper_name = timestepper == :imex_cn ? "IMEX Crank-Nicolson" : "Leapfrog"
    if is_root && print_progress
        println("\n" * "="^60)
        println("Starting time integration ($timestepper_name)...")
        println("  Steps: $nt, dt: $dt")
        if psi_save_steps > 0 || wave_save_steps > 0
            if psi_save_steps > 0 && wave_save_steps > 0
                println("  Saving ψ every $psi_save_steps steps, waves every $wave_save_steps steps")
            elseif psi_save_steps > 0
                println("  Saving ψ every $psi_save_steps steps")
            else
                println("  Saving waves every $wave_save_steps steps")
            end
        end
        println("  Diagnostics every $diagnostics_interval steps")
        println("="^60)
        # Print diagnostics header
        @printf("\n%8s  %10s  %12s  %12s  %12s\n", "Step", "Time", "max|u|", "max|B|", "max|A|")
        println("-"^60)
    end

    # Allocate temporary arrays for physical space diagnostics
    # These are used to transform B and A from spectral to physical space
    # Must use _allocate_fft_dst for correct pencil allocation in MPI case
    B_phys = _allocate_fft_dst(S.B, plans)
    A_phys = _allocate_fft_dst(S.A, plans)

    report_step = function(state::State, step::Int, current_time)
        if diagnostics_interval > 0 && step % diagnostics_interval == 0
            max_u = reduce_max_if_mpi(maximum(abs, parent(state.u)), mpi_config)
            max_v = reduce_max_if_mpi(maximum(abs, parent(state.v)), mpi_config)
            max_vel = max(max_u, max_v)

            fft_backward!(B_phys, state.B, plans)
            fft_backward!(A_phys, state.A, plans)
            max_B = reduce_max_if_mpi(maximum(abs, parent(B_phys)), mpi_config)
            max_A = reduce_max_if_mpi(maximum(abs, parent(A_phys)), mpi_config)

            if is_root && print_progress
                @printf("%8d  %10.2e  %12.4e  %12.4e  %12.4e\n",
                        step, current_time, max_vel, max_B, max_A)
            end
        end

        if output_manager !== nothing
            write_psi = psi_save_steps > 0 && step % psi_save_steps == 0
            write_waves = wave_save_steps > 0 && step % wave_save_steps == 0
            if write_psi || write_waves
                write_state_file(output_manager, state, G, plans, current_time, parallel_config;
                                 params=par, write_psi=write_psi, write_waves=write_waves)
            end
        end
    end

    # Print initial diagnostics (step 0)
    # Note: All processes must participate in MPI reductions, so compute on all
    if is_mpi || print_progress
        # Velocities u, v are already in physical space
        max_u = reduce_max_if_mpi(maximum(abs, parent(S.u)), mpi_config)
        max_v = reduce_max_if_mpi(maximum(abs, parent(S.v)), mpi_config)
        max_vel = max(max_u, max_v)

        # Transform B and A to physical space for meaningful diagnostics
        fft_backward!(B_phys, S.B, plans)
        fft_backward!(A_phys, S.A, plans)
        max_B = reduce_max_if_mpi(maximum(abs, parent(B_phys)), mpi_config)
        max_A = reduce_max_if_mpi(maximum(abs, parent(A_phys)), mpi_config)

        if is_root && print_progress
            @printf("%8d  %10.2e  %12.4e  %12.4e  %12.4e\n",
                    0, 0.0, max_vel, max_B, max_A)
        end
    end

    # Branch based on timestepper
    if timestepper == :leapfrog
        # ==================== LEAPFROG TIME STEPPING ====================
        # Setup leapfrog states - save initial state BEFORE advancing
        # Snm1 = state at time 0 (initial condition)
        # Use copy_state instead of deepcopy to preserve PencilArray topology
        Snm1 = copy_state(S)

        # First projection step (Forward Euler to initialize leapfrog)
        # Advances S from time 0 to time dt
        first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace, N2_profile=N2_profile)

        # Sn = state at time dt (after first projection step)
        Sn = copy_state(S)
        # Snp1 will hold state at time 2*dt (computed by first leapfrog step)
        Snp1 = copy_state(S)

        # Time integration loop
        for step in 1:nt
            # Leapfrog step
            leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                           a=a_ell, dealias_mask=L_mask, workspace=workspace, N2_profile=N2_profile)

            # Rotate states: (n-1) ← (n) ← (n+1) ← (n-1)
            Snm1, Sn, Snp1 = Sn, Snp1, Snm1

            current_time = step * dt

            report_step(Sn, step, current_time)
        end

        # Copy final state back to S
        copyto!(parent(S.psi), parent(Sn.psi))
        copyto!(parent(S.q), parent(Sn.q))
        copyto!(parent(S.B), parent(Sn.B))
        copyto!(parent(S.A), parent(Sn.A))
        copyto!(parent(S.C), parent(Sn.C))
        copyto!(parent(S.u), parent(Sn.u))
        copyto!(parent(S.v), parent(Sn.v))
        copyto!(parent(S.w), parent(Sn.w))

    elseif timestepper == :imex_cn
        # ==================== IMEX CRANK-NICOLSON ====================
        # IMEX only needs two time levels: Sn (current) and Snp1 (next)
        # Dispersion is treated implicitly for unconditional stability

        # Initialize IMEX workspace
        imex_ws = init_imex_workspace(S, G)

        # Setup states
        Sn = copy_state(S)
        Snp1 = copy_state(S)

        for step in 1:nt
            imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws;
                          a=a_ell, dealias_mask=L_mask, workspace=workspace, N2_profile=N2_profile)

            Sn, Snp1 = Snp1, Sn

            current_time = step * dt
            report_step(Sn, step, current_time)
        end

        # Copy final state back to S
        copyto!(parent(S.psi), parent(Sn.psi))
        copyto!(parent(S.q), parent(Sn.q))
        copyto!(parent(S.B), parent(Sn.B))
        copyto!(parent(S.A), parent(Sn.A))
        copyto!(parent(S.C), parent(Sn.C))
        copyto!(parent(S.u), parent(Sn.u))
        copyto!(parent(S.v), parent(Sn.v))
        copyto!(parent(S.w), parent(Sn.w))

    else
        error("Unknown timestepper: $timestepper. Valid options: :leapfrog, :imex_cn")
    end

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
    psi_arr = parent(sim.state.psi)
    B_arr = parent(sim.state.B)
    local_bad = any(x -> !isfinite(x), psi_arr) || any(x -> !isfinite(x), B_arr)
    bad_count = reduce_sum_if_mpi(local_bad ? 1 : 0, sim.parallel_config)
    if bad_count > 0
        if sim.parallel_config === nothing || sim.parallel_config.is_root
            @error "NaN or Inf detected in solution"
        end
        return true
    end

    # Check for blow-up (very large values)
    # Note: fft_backward! returns complex arrays
    psir_complex = _allocate_fft_dst(sim.state.psi, sim.plans)
    fft_backward!(psir_complex, sim.state.psi, sim.plans)
    psir = real.(parent(psir_complex))

    local_max = maximum(abs, psir)
    global_max = reduce_max_if_mpi(local_max, sim.parallel_config)
    if global_max > 1e10
        if sim.parallel_config === nothing || sim.parallel_config.is_root
            @error "Solution appears to be blowing up (max |psi| = $global_max)"
        end
        return true
    end

    return false
end

"""
    create_simple_config(; Lx, Ly, Lz, kwargs...)

Create a simple configuration for quick testing.

# Arguments
- `Lx, Ly, Lz`: Domain size in meters (REQUIRED - no defaults)
- `nx, ny, nz`: Grid resolution (default: 64, 64, 32)
- `dt`: Time step (default: 1e-3)
- `total_time`: Total simulation time (default: 10.0)
- `output_interval`: Time between outputs for all fields (default: 1.0)
- `output_dir`: Output directory (default: "./output_simple")
- `kwargs...`: Additional ModelConfig parameters (f0, inviscid, linear, etc.)

# Example
```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # 500km × 500km × 4km
    nx=64, ny=64, nz=32,
    output_interval=0.5,
    output_dir="./my_run"
)
```
"""
function create_simple_config(;
        Lx::Real, Ly::Real, Lz::Real,
        nx::Int=64, ny::Int=64, nz::Int=32,
        dt::Real=1e-3, total_time::Real=10.0,
        output_interval::Real=1.0,
        output_dir::String="./output_simple",
        kwargs...
    )
    # Create sub-configurations with user-provided values
    domain = create_domain_config(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,
        wave_amplitude=0.01
    )
    output = create_output_config(
        output_dir=output_dir,
        psi_interval=output_interval,
        wave_interval=output_interval,
        diagnostics_interval=output_interval
    )

    config = create_model_config(domain, stratification, initial_conditions, output;
                                dt=dt, total_time=total_time, kwargs...)

    return config
end

"""
    run_simple_simulation(config::ModelConfig)
    run_simple_simulation(; kwargs...)

Run a simulation with the given configuration or create one from kwargs.

# Examples
```julia
# With pre-built config
config = create_simple_config(Lx=500e3, Ly=500e3, Lz=4000.0)
result = run_simple_simulation(config)

# With kwargs (creates config internally)
result = run_simple_simulation(Lx=500e3, Ly=500e3, Lz=4000.0, dt=0.001)
```
"""
function run_simple_simulation(config::ModelConfig)
    sim = setup_simulation(config)
    run_simulation!(sim)
    return sim
end

function run_simple_simulation(; kwargs...)
    config = create_simple_config(; kwargs...)
    return run_simple_simulation(config)
end

# Convenience function for backward compatibility with existing code
"""
    setup_model_with_config(config::ModelConfig)

Set up model components from configuration (for compatibility).

Maps all fields from ModelConfig to QGParams, including:
- Domain parameters (nx, ny, nz, Lx, Ly, Lz)
- Time stepping (dt, nt)
- Physical parameters (f₀, N²)
- Viscosity (νₕ, νᵥ, νz)
- Hyperdiffusion (νₕ₁, νₕ₂, ilap1, ilap2, etc.)
- Physics switches (inviscid, linear, ybj_plus, etc.)
- Stratification parameters (for skewed_gaussian)
"""
function setup_model_with_config(config::ModelConfig{T}) where T
    # Extract stratification parameters
    strat = config.stratification
    strat_type = strat.type

    # Get N² value based on stratification type
    N2_value = if strat_type == :constant_N
        strat.N0^2  # N² = N0²
    else
        T(1.0)  # For skewed_gaussian, N² profile varies spatially
    end

    # Compute number of time steps
    nt = ceil(Int, config.total_time / config.dt)

    # Build QGParams with all required fields from ModelConfig
    params = QGParams{T}(;
        # Domain
        nx = config.domain.nx,
        ny = config.domain.ny,
        nz = config.domain.nz,

        Lx = config.domain.Lx,
        Ly = config.domain.Ly,
        Lz = config.domain.Lz,
        x0 = config.domain.x0,
        y0 = config.domain.y0,

        # Time stepping
        dt = config.dt,
        nt = nt,

        # Physical parameters
        f₀ = config.f0,
        N² = N2_value,
        W2F = T(0.01),  # Default (deprecated parameter)
        γ = T(1e-3),    # Robert-Asselin filter coefficient

        # Legacy viscosity (use hyperdiffusion instead)
        νₕ = config.nu_h,
        νᵥ = config.nu_v,

        # Hyperdiffusion for mean flow (from config)
        νₕ₁ = T(config.nu_h1),
        νₕ₂ = T(config.nu_h2),

        ilap1 = config.ilap1,
        ilap2 = config.ilap2,

        # Hyperdiffusion for waves (from config)
        νₕ₁ʷ = T(config.nu_h1_wave),
        νₕ₂ʷ = T(config.nu_h2_wave),

        ilap1w = config.ilap1_wave,
        ilap2w = config.ilap2_wave,

        # Vertical diffusion (from config)
        νz = T(config.nu_v),

        # Flags
        linear_vert_structure = 0,
        stratification = strat_type,
        inviscid = config.inviscid,
        linear = config.linear,

        no_dispersion = config.no_dispersion,
        passive_scalar = config.passive_scalar,
        ybj_plus = config.ybj_plus,

        no_feedback = config.no_feedback,
        fixed_flow = config.fixed_mean_flow,
        no_wave_feedback = config.no_wave_feedback,

        # Skewed Gaussian stratification parameters
        N₀²_sg = strat.N02_sg,
        N₁²_sg = strat.N12_sg,
        σ_sg = strat.sigma_sg,
        z₀_sg = strat.z0_sg,
        α_sg = strat.alpha_sg,
    )

    grid = init_grid(params)
    state = init_state(grid)
    plans = plan_transforms!(grid)

    return params, grid, state, plans
end
