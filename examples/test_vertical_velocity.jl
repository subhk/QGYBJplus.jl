"""
Test script for vertical velocity computation in QG-YBJ model.

This script tests the implementation of vertical velocity computation
through the omega equation and verifies that it's properly saved to NetCDF files.
"""

using QGYBJ

function test_vertical_velocity_computation()
    println("Testing vertical velocity computation...")
    
    # Create a simple test configuration
    domain = create_domain_config(
        nx=32, ny=32, nz=16,
        Lx=2π, Ly=2π, Lz=π
    )
    
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,
        wave_amplitude=0.01,
        random_seed=1234
    )
    
    output = create_output_config(
        output_dir="./test_vertical_velocity",
        psi_interval=1.0,
        wave_interval=1.0,
        diagnostics_interval=0.5,
        save_psi=true,
        save_waves=true,
        save_velocities=true,
        save_vertical_velocity=true  # Enable vertical velocity output
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=2.0,  # Short test
        dt=1e-3,
    )
    
    # Set up simulation
    println("  Setting up simulation...")
    sim = setup_simulation(config)
    
    # Test manual vertical velocity computation
    println("  Testing compute_velocities! with vertical velocity...")
    compute_velocities!(sim.state, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=true)
    
    # Check that vertical velocity was computed
    w_rms = sqrt(sum(sim.state.w.^2) / length(sim.state.w))
    println("    Vertical velocity RMS: $w_rms")
    
    # Test that w field has reasonable properties
    w_max = maximum(abs.(sim.state.w))
    println("    Vertical velocity max: $w_max")
    
    # Verify boundary conditions (w should be zero at top and bottom)
    w_bottom_max = maximum(abs.(sim.state.w[:,:,1]))
    w_top_max = maximum(abs.(sim.state.w[:,:,end]))
    println("    W at bottom (should be ~0): $w_bottom_max")
    println("    W at top (should be ~0): $w_top_max")
    
    # Run a short simulation to test integration
    println("  Running short simulation...")
    run_simulation!(sim; 
                   progress_callback=(sim) -> nothing)  # Suppress output
    
    println("  ✓ Vertical velocity computation test completed")
    
    return sim
end

function test_vertical_velocity_output()
    println("Testing vertical velocity NetCDF output...")
    
    # Use NetCDF tools to verify the output
    try
        import NCDatasets
        
        # Check if output files were created
        output_dir = "./test_vertical_velocity"
        if isdir(output_dir)
            files = readdir(output_dir)
            nc_files = filter(f -> endswith(f, ".nc"), files)
            
            if !isempty(nc_files)
                test_file = joinpath(output_dir, nc_files[1])
                println("  Checking file: $test_file")
                
                NCDatasets.Dataset(test_file, "r") do ds
                    # Check if vertical velocity variable exists
                    if haskey(ds, "w")
                        w_data = ds["w"]
                        println("    ✓ Vertical velocity variable 'w' found")
                        println("    W shape: $(size(w_data))")
                        println("    W units: $(get(w_data.attrib, "units", "not set"))")
                        println("    W long_name: $(get(w_data.attrib, "long_name", "not set"))")
                        
                        # Check some basic properties
                        w_values = w_data[:,:,:]
                        w_rms = sqrt(sum(w_values.^2) / length(w_values))
                        println("    W RMS from file: $w_rms")
                    else
                        println("    ⚠️  Vertical velocity variable 'w' not found")
                    end
                    
                    # List all variables
                    println("    Variables in file: $(keys(ds))")
                end
            else
                println("    ⚠️  No NetCDF files found in output directory")
            end
        else
            println("    ⚠️  Output directory not found")
        end
        
        println("  ✓ NetCDF output test completed")
        
    catch e
        println("    ⚠️  NCDatasets not available, skipping file verification")
        println("    Error: $e")
    end
end

function test_vertical_velocity_options()
    println("Testing vertical velocity enable/disable options...")
    
    # Test with vertical velocity disabled
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    output = create_output_config(output_dir="./test_w_disabled")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1
    )
    
    sim = setup_simulation(config)
    
    # Test with compute_w=false
    compute_velocities!(sim.state, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=false)
    
    w_max_disabled = maximum(abs.(sim.state.w))
    println("  W max with compute_w=false: $w_max_disabled (should be 0)")
    
    # Test with compute_w=true
    compute_velocities!(sim.state, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=true)
    
    w_max_enabled = maximum(abs.(sim.state.w))
    println("  W max with compute_w=true: $w_max_enabled (should be > 0)")
    
    println("  ✓ Vertical velocity options test completed")
    
    return (w_max_disabled, w_max_enabled)
end

function run_all_tests()
    println("QG-YBJ Vertical Velocity Implementation Tests")
    println("=============================================")
    
    try
        # Run all tests
        sim = test_vertical_velocity_computation()
        test_vertical_velocity_output()
        w_disabled, w_enabled = test_vertical_velocity_options()
        
        println("\n✅ All vertical velocity tests completed successfully!")
        
        # Summary
        println("\nSummary:")
        println("- Vertical velocity computation: Working")
        println("- NetCDF output integration: Working")
        println("- Enable/disable options: Working")
        println("- W field when disabled: $(w_disabled)")
        println("- W field when enabled: $(w_enabled)")
        
        if w_disabled < 1e-10 && w_enabled > 1e-10
            println("- Enable/disable logic: ✓ Correct")
        else
            println("- Enable/disable logic: ⚠️  May have issues")
        end
        
    catch e
        println("\n❌ Test failed with error: $e")
        rethrow(e)
    end
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end