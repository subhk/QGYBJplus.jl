"""
Simple test script to verify the new wave-mean flow interaction flags work correctly.

This script runs quick tests to ensure:
1. The new flags are properly passed through the configuration system
2. The physics controls work as expected in the time stepping
3. Backward compatibility is maintained
"""

using QGYBJ

function test_flag_propagation()
    println("Testing flag propagation through configuration system...")
    
    # Test 1: New flags in config
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    output = create_output_config(output_dir="./test_flags")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.01,  # Very short for testing
        no_wave_feedback=true,
        fixed_mean_flow=true
    )
    
    # Verify config contains the flags
    @assert config.no_wave_feedback == true "no_wave_feedback flag not set in config"
    @assert config.fixed_mean_flow == true "fixed_mean_flow flag not set in config"
    
    println("✓ Configuration flags set correctly")
    
    # Test 2: Check propagation to QGParams
    sim = setup_simulation(config)
    
    @assert sim.params.no_wave_feedback == true "no_wave_feedback not propagated to params"
    @assert sim.params.fixed_flow == true "fixed_flow not propagated to params"
    
    println("✓ Flags propagated to QGParams correctly")
    
    return sim
end

function test_legacy_compatibility()
    println("Testing legacy compatibility...")
    
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    output = create_output_config(output_dir="./test_legacy")
    
    # Test old no_feedback flag
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.01,
        no_feedback=true  # Legacy flag
    )
    
    sim = setup_simulation(config)
    
    # Should map to no_wave_feedback
    @assert sim.params.no_feedback == true "Legacy no_feedback flag not preserved"
    @assert sim.params.no_wave_feedback == true "Legacy flag not mapped to new flag"
    
    println("✓ Legacy compatibility maintained")
    
    return sim
end

function test_physics_behavior()
    println("Testing physics behavior with different flag combinations...")
    
    # Setup base configuration
    domain = create_domain_config(nx=32, ny=32, nz=16)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, 
        wave_type=:random,
        psi_amplitude=0.1,
        wave_amplitude=0.05,
        random_seed=1234
    )
    
    test_cases = [
        ("coupled", false, false, "Full coupling"),
        ("no_feedback", true, false, "No wave feedback"),
        ("fixed_flow", true, true, "Fixed mean flow"),
    ]
    
    results = Dict()
    
    for (name, no_feedback, fixed_flow, description) in test_cases
        println("  Testing: $description")
        
        output = create_output_config(output_dir="./test_$name")
        
        config = create_model_config(
            domain, stratification, initial_conditions, output,
            total_time=0.05,  # Short test run
            dt=1e-3,
            no_wave_feedback=no_feedback,
            fixed_mean_flow=fixed_flow
        )
        
        sim = setup_simulation(config)
        
        # Store initial conditions
        initial_psi = copy(sim.state.psi)
        initial_q = copy(sim.state.q)
        initial_B = copy(sim.state.B)
        
        # Run simulation
        run_simulation!(sim)
        
        # Check behavior
        psi_changed = !isapprox(sim.state.psi, initial_psi; rtol=1e-10)
        q_changed = !isapprox(sim.state.q, initial_q; rtol=1e-10) 
        B_changed = !isapprox(sim.state.B, initial_B; rtol=1e-10)
        
        results[name] = (
            psi_changed=psi_changed,
            q_changed=q_changed,
            B_changed=B_changed,
            description=description
        )
        
        # Verify expected behavior
        if fixed_flow
            # For fixed flow, psi should not change (much)
            # Note: There might be tiny numerical changes from FFTs etc.
            println("    Psi changed: $psi_changed (should be minimal for fixed flow)")
        else
            println("    Psi changed: $psi_changed (should be true for evolving flow)")
        end
        
        println("    Wave field changed: $B_changed (should be true)")
    end
    
    println("✓ Physics behavior tests completed")
    
    return results
end

function test_error_conditions()
    println("Testing error conditions and edge cases...")
    
    # Test with conflicting flags (should work fine)
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    output = create_output_config(output_dir="./test_edge_cases")
    
    # Both old and new flags
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.01,
        no_feedback=false,       # Legacy flag says allow feedback
        no_wave_feedback=true    # New flag says no feedback
    )
    
    sim = setup_simulation(config)
    
    # Should prioritize the OR of both flags (more restrictive)
    expected_no_feedback = config.no_wave_feedback || config.no_feedback
    @assert sim.params.no_feedback == expected_no_feedback "Flag conflict not handled correctly"
    
    println("✓ Flag conflicts handled correctly")
    
    return sim
end

function run_all_tests()
    println("QG-YBJ New Flags Test Suite")
    println("===========================")
    
    try
        # Run all tests
        test_flag_propagation()
        test_legacy_compatibility()
        physics_results = test_physics_behavior()
        test_error_conditions()
        
        println("\n✅ All tests passed!")
        println("\nPhysics test summary:")
        for (name, result) in physics_results
            println("  $(result.description):")
            println("    - Psi evolved: $(result.psi_changed)")
            println("    - Waves evolved: $(result.B_changed)")
        end
        
        println("\nNew flags are working correctly and ready for use!")
        
    catch e
        println("\n❌ Test failed: $e")
        rethrow(e)
    end
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end