"""
Demonstration of the new QG-YBJ user interface.

This example shows how to use the configuration system to set up and run
QG-YBJ simulations with various initial conditions, stratification profiles,
and output options.
"""

using QGYBJ

function demo_basic_simulation()
    println("=== Basic QG-YBJ Simulation ===")
    
    # Create domain configuration
    domain = create_domain_config(
        nx=64, ny=64, nz=32,           # Grid resolution
        Lx=4π, Ly=4π, Lz=2π,          # Domain size
        dom_x_m=314159.0,              # Physical domain: ~314 km
        dom_y_m=314159.0,              # Physical domain: ~314 km  
        dom_z_m=4000.0                 # Physical domain: 4 km depth
    )
    
    # Create stratification configuration
    stratification = create_stratification_config(
        :constant_N,
        N0=1.0  # Constant buoyancy frequency
    )
    
    # Create initial condition configuration
    initial_conditions = create_initial_condition_config(
        psi_type=:random,              # Random stream function
        wave_type=:random,             # Random wave field
        psi_amplitude=0.1,             # Stream function amplitude
        wave_amplitude=0.01,           # Wave amplitude
        random_seed=1234               # For reproducibility
    )
    
    # Create output configuration
    output = create_output_config(
        output_dir="./demo_basic",
        psi_interval=1.0,              # Save every 1 time unit
        wave_interval=1.0,
        diagnostics_interval=0.5,      # Diagnostics every 0.5 time units
        state_file_pattern="state%04d.nc"  # Files: state0001.nc, state0002.nc, ...
    )
    
    # Create complete model configuration
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        f0=1.0,                        # Coriolis parameter
        dt=1e-3,                       # Time step
        total_time=5.0,                # Total simulation time
        linear=false,                  # Include nonlinear terms
        inviscid=true,                 # No viscosity
        ybj_plus=true,                 # Use YBJ+ formulation
        no_feedback=false              # Include wave-mean flow feedback
    )
    
    # Set up and run simulation
    sim = setup_simulation(config)
    run_simulation!(sim)
    
    println("Basic simulation completed!")
    return sim
end

function demo_stratified_simulation()
    println("\n=== Stratified QG-YBJ Simulation with Tropopause ===")
    
    # Domain with higher vertical resolution for stratification
    domain = create_domain_config(nx=128, ny=128, nz=64, Lx=6π, Ly=6π, Lz=2π)
    
    # Pycnocline-like stratification
    stratification = create_stratification_config(
        :tanh_profile,
        N_upper=0.01,                   # Weak stratification in upper ocean
        N_lower=0.04,                  # Strong stratification in deep ocean
        z_pycno=0.6,                    # Pycnocline at 60% of domain depth
        width=0.05                     # Sharp transition
    )
    
    # Initialize from analytical expressions
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,          # Analytical stream function
        wave_type=:analytical,         # Analytical wave field
        psi_amplitude=0.2,
        wave_amplitude=0.005
    )
    
    # More frequent output for detailed analysis
    output = create_output_config(
        output_dir="./demo_stratified",
        psi_interval=0.5,
        wave_interval=0.5,
        diagnostics_interval=0.1,
        save_velocities=true,          # Also save velocity fields
        save_vorticity=true            # Save vorticity
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=5e-4,                       # Smaller time step for stability
        total_time=10.0
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim)
    
    println("Stratified simulation completed!")
    return sim
end

function demo_file_based_initialization()
    println("\n=== File-Based Initialization Demo ===")
    
    # First, create some initial condition files
    create_sample_initial_condition_files()
    
    domain = create_domain_config(nx=64, ny=64, nz=32)
    
    # Skewed Gaussian stratification (from Fortran test case)
    stratification = create_stratification_config(:skewed_gaussian)
    
    # Initialize from NetCDF files
    initial_conditions = create_initial_condition_config(
        psi_type=:from_file,
        psi_filename="psi_initial_demo.nc",
        wave_type=:from_file,
        wave_filename="wave_initial_demo.nc"
    )
    
    output = create_output_config(
        output_dir="./demo_file_init",
        psi_interval=2.0,
        wave_interval=2.0
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=8.0
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim)
    
    println("File-based initialization demo completed!")
    return sim
end

function create_sample_initial_condition_files()
    println("Creating sample initial condition files...")
    
    # This would create sample NetCDF files for demonstration
    # In practice, these would come from observations, other models, etc.
    
    # Create a simple grid for the demo files
    nx, ny, nz = 64, 64, 32
    
    # Simple analytical fields
    x = LinRange(0, 2π, nx)
    y = LinRange(0, 2π, ny)
    z = LinRange(0, 2π, nz)
    
    # Stream function: sum of Rossby waves
    psi_data = zeros(nx, ny, nz)
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y), (k, zk) in enumerate(z)
        psi_data[i,j,k] = 0.1 * (sin(2*xi) * cos(yj) * cos(zk) + 
                                0.5 * cos(xi) * sin(2*yj) * sin(zk))
    end
    
    # Wave field: localized wave packets
    LAr_data = zeros(nx, ny, nz)
    LAi_data = zeros(nx, ny, nz)
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y), (k, zk) in enumerate(z)
        envelope = exp(-((zk - π)^2) / (2 * 0.5^2))
        LAr_data[i,j,k] = 0.01 * sin(4*xi + zk) * cos(2*yj) * envelope
        LAi_data[i,j,k] = 0.001 * cos(4*xi + zk) * sin(2*yj) * envelope
    end
    
    # Write to NetCDF files (simplified - would use proper NetCDF writing)
    # This is pseudocode for the concept:
    println("  Writing psi_initial_demo.nc and wave_initial_demo.nc")
    println("  (In real implementation, would use NCDatasets.jl)")
    
    # In a real implementation:
    # write_initial_condition_file("psi_initial_demo.nc", x, y, z, psi_data)
    # write_initial_condition_file("wave_initial_demo.nc", x, y, z, LAr_data, LAi_data)
end

function demo_parameter_sweep()
    println("\n=== Parameter Sweep Demo ===")
    
    # Run simulations with different time steps
    dt_values = [5e-4, 1e-3, 2e-3]
    
    for (i, dt_val) in enumerate(dt_values)
        
        domain = create_domain_config(nx=32, ny=32, nz=16)  # Smaller for speed
        stratification = create_stratification_config(:constant_N, N0=1.0)
        initial_conditions = create_initial_condition_config(
            psi_type=:random, 
            wave_type=:random,
            random_seed=1234  # Same initial conditions for comparison
        )
        output = create_output_config(
            output_dir="./demo_sweep_dt_$(Int(round(1e6*dt_val)))",
            psi_interval=1.0,
            wave_interval=1.0
        )
        
        config = create_model_config(
            domain, stratification, initial_conditions, output,
            dt=dt_val,
            total_time=3.0  # Shorter runs for parameter sweep
        )
        
        sim = setup_simulation(config)
        run_simulation!(sim)
    end
    
    println("Parameter sweep completed!")
end

function demo_simple_interface()
    println("\n=== Simple Interface Demo ===")
    
    # The simplest possible way to run a simulation
    sim = run_simple_simulation(
        nx=64, ny=64, nz=32,
        total_time=5.0,
        output_dir="./demo_simple"
    )
    
    println("Simple simulation completed!")
    return sim
end

function demo_custom_stratification()
    println("\n=== Custom Stratification Demo ===")
    
    # Example of creating and using custom stratification profiles
    T = Float64
    
    # Two-layer stratification
    profiles = create_standard_profiles()
    
    domain = create_domain_config(nx=64, ny=64, nz=48)
    
    # Use the two-layer profile
    stratification = create_stratification_config(:constant_N, N0=1.0)  # Will be overridden
    
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,
        wave_type=:zero  # No waves for this demo
    )
    
    output = create_output_config(
        output_dir="./demo_custom_strat",
        psi_interval=1.0,
        wave_interval=1.0,
        save_diagnostics=true
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=6.0
    )
    
    sim = setup_simulation(config)
    
    # Override with two-layer profile
    sim.stratification_profile = profiles[:two_layer]
    sim.N2_profile = compute_stratification_profile(sim.stratification_profile, sim.grid)
    
    run_simulation!(sim)
    
    println("Custom stratification demo completed!")
    return sim
end

# Main demonstration function
function main()
    println("QG-YBJ Model User Interface Demonstration")
    println("=========================================")
    
    # Run all demonstrations
    try
        # Basic simulation
        sim1 = demo_basic_simulation()
        
        # Stratified case  
        sim2 = demo_stratified_simulation()
        
        # Simple interface
        sim3 = demo_simple_interface()
        
        # Custom stratification
        sim4 = demo_custom_stratification()
        
        # File-based initialization (commented out since it requires NetCDF files)
        # sim5 = demo_file_based_initialization()
        
        # Parameter sweep (commented out to save time)
        # demo_parameter_sweep()
        
        println("\n=== All Demonstrations Completed Successfully! ===")
        println("\nOutput files have been created in various demo_* directories")
        println("Each contains state files in NetCDF format with the pattern state0001.nc, state0002.nc, ...")
        
    catch e
        println("Error during demonstration: $e")
        rethrow(e)
    end
end

# Run the demonstration if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end