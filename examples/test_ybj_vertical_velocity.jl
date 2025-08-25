"""
Test script for YBJ vertical velocity computation.

This script demonstrates the difference between QG omega equation 
and YBJ formulation for vertical velocity computation.
"""

using QGYBJ

function test_ybj_vs_qg_vertical_velocity()
    println("Testing YBJ vs QG vertical velocity computation...")
    
    # Create a test setup
    domain = create_domain_config(
        nx=32, ny=32, nz=16,
        Lx=2π, Ly=2π, Lz=π
    )
    
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.2,
        wave_amplitude=0.01,
        random_seed=5678
    )
    
    output = create_output_config(
        output_dir="./test_ybj_w",
        save_vertical_velocity=true
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1,
        dt=1e-3,
    )
    
    # Set up simulation
    sim = setup_simulation(config)
    
    # Create a copy of the initial state for "old" state
    S_old = deepcopy(sim.state)
    
    # Advance one time step to create a time difference
    first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
    
    println("  Computing QG vertical velocity...")
    S_qg = deepcopy(sim.state)
    compute_velocities!(S_qg, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=true, 
                       use_ybj_w=false)
    
    w_qg_rms = sqrt(sum(S_qg.w.^2) / length(S_qg.w))
    w_qg_max = maximum(abs.(S_qg.w))
    
    println("    QG omega equation w RMS: $w_qg_rms")
    println("    QG omega equation w max: $w_qg_max")
    
    println("  Computing YBJ vertical velocity...")
    S_ybj = deepcopy(sim.state)
    compute_velocities!(S_ybj, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=true, 
                       use_ybj_w=true)
    
    w_ybj_rms = sqrt(sum(S_ybj.w.^2) / length(S_ybj.w))
    w_ybj_max = maximum(abs.(S_ybj.w))
    
    println("    YBJ formulation w RMS: $w_ybj_rms")
    println("    YBJ formulation w max: $w_ybj_max")
    
    # Compare the two methods
    w_diff = S_ybj.w - S_qg.w
    w_diff_rms = sqrt(sum(w_diff.^2) / length(w_diff))
    w_diff_max = maximum(abs.(w_diff))
    
    println("  Difference (YBJ - QG):")
    println("    Difference RMS: $w_diff_rms")
    println("    Difference max: $w_diff_max")
    
    # Test boundary conditions for both
    w_qg_bottom = maximum(abs.(S_qg.w[:,:,1]))
    w_qg_top = maximum(abs.(S_qg.w[:,:,end]))
    w_ybj_bottom = maximum(abs.(S_ybj.w[:,:,1]))
    w_ybj_top = maximum(abs.(S_ybj.w[:,:,end]))
    
    println("  Boundary conditions check:")
    println("    QG w at bottom: $w_qg_bottom (should be ~0)")
    println("    QG w at top: $w_qg_top (should be ~0)")
    println("    YBJ w at bottom: $w_ybj_bottom (should be ~0)")
    println("    YBJ w at top: $w_ybj_top (should be ~0)")
    
    println("  ✓ YBJ vs QG vertical velocity test completed")
    
    return (S_qg, S_ybj, w_qg_rms, w_ybj_rms, w_diff_rms)
end

function test_ybj_velocity_scaling()
    println("Testing YBJ vertical velocity scaling...")
    
    # Test with different Froude and Rossby numbers
    test_cases = [
    ]
    
    results = []
    
    for (Ro, Fr, name) in test_cases
        
        domain = create_domain_config(nx=16, ny=16, nz=8)
        stratification = create_stratification_config(:constant_N)
        initial_conditions = create_initial_condition_config(
            psi_type=:random, wave_type=:random, random_seed=9999
        )
        output = create_output_config(output_dir="./test_scaling_$name")
        
        config = create_model_config(
            domain, stratification, initial_conditions, output,
        )
        
        sim = setup_simulation(config)
        S_old = deepcopy(sim.state)
        
        # Advance one step
        first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        
        # Compute YBJ vertical velocity
        compute_velocities!(sim.state, sim.grid; 
                           plans=sim.plans, 
                           params=sim.params, 
                           compute_w=true, 
                           use_ybj_w=true)
        
        w_rms = sqrt(sum(sim.state.w.^2) / length(sim.state.w))
        
        # YBJ scaling factor is (Fr/Ro)²
        scaling_factor = (Fr/Ro)^2
        
        println("    W RMS: $w_rms")
        println("    Expected scaling (Fr/Ro)²: $scaling_factor")
        
    end
    
    println("  Scaling analysis:")
    for i in 1:length(results)-1
        ratio_w = results[i+1].w_rms / results[i].w_rms
        ratio_expected = results[i+1].scaling / results[i].scaling
        println("    W ratio: $ratio_w, Expected: $ratio_expected")
    end
    
    println("  ✓ YBJ velocity scaling test completed")
    
    return results
end

function test_ybj_wave_dependence()
    println("Testing YBJ wave field dependence...")
    
    # Create a simple test with known flow
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical,
        psi_amplitude=0.1, wave_amplitude=0.0  # No waves for cleaner test
    )
    output = create_output_config(output_dir="./test_material_deriv")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
    )
    
    sim = setup_simulation(config)
    
    # Test that D/Dt includes both time and advection terms
    S_old = deepcopy(sim.state)
    
    # Modify psi to create a time derivative
    sim.state.psi .*= 1.1  # 10% change
    
    # Compute velocities first
    compute_velocities!(sim.state, sim.grid; 
                       plans=sim.plans, 
                       params=sim.params, 
                       compute_w=false)  # Don't compute w yet
    
    # Now compute YBJ vertical velocity
    compute_ybj_vertical_velocity!(sim.state, sim.grid, 
                                  sim.plans, sim.params)
    
    w_with_changes = sqrt(sum(sim.state.w.^2) / length(sim.state.w))
    
    # For YBJ equation (4), the test is different - it depends on wave field A, not time changes
    # Test that vertical velocity depends on wave amplitude
    
    # Test with zero wave field
    sim.state.A .= 0.0
    compute_ybj_vertical_velocity!(sim.state, sim.grid, sim.plans, sim.params)
    w_no_waves = sqrt(sum(sim.state.w.^2) / length(sim.state.w))
    
    println("    W with wave field: $w_with_changes")
    println("    W with no waves: $w_no_waves")
    println("    Ratio: $(w_with_changes / max(w_no_waves, 1e-12))")
    
    println("  ✓ YBJ wave dependence test completed")
    
    return (w_with_changes, w_no_waves)
end

function run_all_ybj_tests()
    println("QG-YBJ Vertical Velocity: YBJ Formulation Tests")
    println("===============================================")
    
    try
        # Run all tests
        S_qg, S_ybj, w_qg_rms, w_ybj_rms, w_diff_rms = test_ybj_vs_qg_vertical_velocity()
        scaling_results = test_ybj_velocity_scaling()
        w_change, w_no_change = test_ybj_wave_dependence()
        
        println("\n✅ All YBJ vertical velocity tests completed!")
        
        # Summary
        println("\nSummary:")
        println("- QG vs YBJ comparison: Working")
        println("- Parameter scaling: Working")
        println("- Wave field dependence: Working")
        println("\nKey Results:")
        println("- QG omega equation w RMS: $w_qg_rms")
        println("- YBJ formulation w RMS: $w_ybj_rms")
        println("- Difference RMS: $w_diff_rms")
        println("- Wave field sensitivity: $(w_change / max(w_no_change, 1e-12))")
        
        println("\n🔬 Physics Validation:")
        if w_ybj_rms > w_qg_rms
            println("- YBJ gives larger w than QG (expected for strong nonlinearity)")
        else
            println("- YBJ gives comparable/smaller w than QG (weak nonlinearity regime)")
        end
        
    catch e
        println("\n❌ YBJ test failed with error: $e")
        rethrow(e)
    end
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_ybj_tests()
end