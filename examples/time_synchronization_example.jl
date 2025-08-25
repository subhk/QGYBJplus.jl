"""
Example demonstrating proper time synchronization between particles and simulation.

This example shows the difference between:
1. Using simulation time for particle advection (CORRECT)
2. Using particle internal time only (INCORRECT - can cause desynchronization)

Key points:
- Particles should be synchronized with the actual simulation time
- particle_advec_time is compared against simulation time, not particle time
- Pass current_time to advect_particles! for proper synchronization
"""

using QGYBJ

function time_synchronization_example()
    println("Time Synchronization Example")
    println("============================")
    
    # Simple simulation setup
    domain = create_domain_config(nx=16, ny=16, nz=4)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, wave_type=:random,
        psi_amplitude=0.1, wave_amplitude=0.02
    )
    output = create_output_config(output_dir="./sync_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=1.0, dt=0.1
    )
    
    sim = setup_simulation(config)
    
    println("Testing time synchronization with particle_advec_time=0.3")
    
    # Create particle configuration with delayed start
    particle_config = create_particle_config(
        nx_particles=2, ny_particles=2,
        particle_advec_time=0.3,  # Start at t=0.3
        integration_method=:euler
    )
    
    # Method 1: CORRECT - Pass simulation time
    tracker_correct = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_correct, particle_config)
    
    # Method 2: INCORRECT - Don't pass simulation time (for comparison)
    tracker_incorrect = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_incorrect, particle_config)
    
    println("\\nRunning simulation with both methods...")
    
    steps = Int(config.total_time / config.dt)
    
    for step in 1:steps
        simulation_time = step * config.dt
        
        println("\\nStep $step - Simulation time: $(simulation_time)")
        
        # Method 1: CORRECT - Pass current simulation time
        advect_particles!(tracker_correct, sim.state, sim.grid, config.dt, simulation_time)
        
        # Method 2: INCORRECT - No simulation time (uses internal time only)
        advect_particles!(tracker_incorrect, sim.state, sim.grid, config.dt)
        
        # Check timing behavior
        correct_should_move = simulation_time >= particle_config.particle_advec_time
        incorrect_should_move = tracker_incorrect.particles.time >= particle_config.particle_advec_time
        
        println("  Simulation time: $(simulation_time)")
        println("  Correct tracker time: $(tracker_correct.particles.time)")
        println("  Incorrect tracker time: $(tracker_incorrect.particles.time)")
        println("  Should particles move (simulation time): $(correct_should_move)")
        println("  Should particles move (internal time): $(incorrect_should_move)")
        
        if correct_should_move && !incorrect_should_move
            println("  ⚠️  DESYNCHRONIZATION! Methods disagree on whether particles should move!")
        elseif correct_should_move == incorrect_should_move
            if correct_should_move
                println("  ✓ Both methods agree: particles should move")
            else
                println("  ✓ Both methods agree: particles should remain stationary")
            end
        end
    end
    
    println("\\n" * "="*60)
    println("SUMMARY: Time Synchronization")
    println("="*60)
    
    println("\\nCORRECT METHOD (with simulation time):")
    println("  • Pass current_time to advect_particles!()")
    println("  • particle_advec_time compared against simulation time")
    println("  • Particles synchronized with fluid evolution")
    println("  • Example: advect_particles!(tracker, state, grid, dt, current_time)")
    
    println("\\nINCORRECT METHOD (internal time only):")
    println("  • Don't pass current_time to advect_particles!()")
    println("  • particle_advec_time compared against particle internal time")
    println("  • Risk of desynchronization with fluid simulation")
    println("  • Example: advect_particles!(tracker, state, grid, dt)  # NO current_time")
    
    println("\\n✅ Always use the CORRECT method for proper synchronization!")
    
    # Demonstrate timing difference
    final_sim_time = config.total_time
    println("\\nFinal verification:")
    println("  Final simulation time: $(final_sim_time)")
    println("  Correct tracker time: $(tracker_correct.particles.time)")
    println("  Incorrect tracker time: $(tracker_incorrect.particles.time)")
    
    if abs(tracker_correct.particles.time - final_sim_time) < 1e-10
        println("  ✓ Correct method: Perfect synchronization!")
    else
        println("  ⚠️  Correct method: Time mismatch")
    end
    
    if abs(tracker_incorrect.particles.time - final_sim_time) > 1e-10
        println("  ⚠️  Incorrect method: Desynchronized as expected")
    else
        println("  ? Incorrect method: Accidentally synchronized")
    end
    
    return (tracker_correct, tracker_incorrect)
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    time_synchronization_example()
end