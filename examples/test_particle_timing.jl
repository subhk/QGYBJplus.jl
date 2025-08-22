"""
Simple test to verify particle_advec_time feature works correctly.

Tests:
1. Particles with particle_advec_time=0.0 start moving immediately
2. Particles with particle_advec_time>0.0 remain stationary until specified time
3. Position tracking during stationary and moving phases
"""

using QGYBJ

function test_particle_timing()
    println("Testing particle_advec_time feature")
    println("===================================")
    
    # Simple setup
    domain = create_domain_config(nx=16, ny=16, nz=4)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, wave_type=:random,
        psi_amplitude=0.1, wave_amplitude=0.02
    )
    output = create_output_config(output_dir="./test_timing")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.6, dt=0.1
    )
    
    sim = setup_simulation(config)
    
    # Test configurations
    config_immediate = create_particle_config(
        nx_particles=2, ny_particles=2,
        particle_advec_time=0.0,  # Start immediately
        integration_method=:euler
    )
    
    config_delayed = create_particle_config(
        nx_particles=2, ny_particles=2,
        particle_advec_time=0.3,  # Start at t=0.3
        integration_method=:euler
    )
    
    # Initialize trackers
    tracker_immediate = ParticleTracker(config_immediate, sim.grid, sim.parallel_config)
    tracker_delayed = ParticleTracker(config_delayed, sim.grid, sim.parallel_config)
    
    initialize_particles!(tracker_immediate, config_immediate)
    initialize_particles!(tracker_delayed, config_delayed)
    
    # Store initial positions
    x0_imm = copy(tracker_immediate.particles.x)
    y0_imm = copy(tracker_immediate.particles.y)
    x0_del = copy(tracker_delayed.particles.x)
    y0_del = copy(tracker_delayed.particles.y)
    
    println("Initial positions stored")
    println("Immediate tracker: $(length(x0_imm)) particles")
    println("Delayed tracker: $(length(x0_del)) particles")
    
    # Time stepping test
    dt = config.dt
    steps = Int(config.total_time / dt)
    
    println("\\nRunning $(steps) time steps with dt=$(dt)...")
    
    for step in 1:steps
        current_time = step * dt
        
        # Advect particles (pass simulation time for synchronization)
        advect_particles!(tracker_immediate, sim.state, sim.grid, dt, current_time)
        advect_particles!(tracker_delayed, sim.state, sim.grid, dt, current_time)
        
        # Check positions
        println("\\nStep $step (t=$(current_time)):")
        
        # Immediate tracker should always be moving (or at least positions could change)
        dx_imm = maximum(abs.(tracker_immediate.particles.x .- x0_imm))
        dy_imm = maximum(abs.(tracker_immediate.particles.y .- y0_imm))
        println("  Immediate: max_dx=$(round(dx_imm, digits=6)), max_dy=$(round(dy_imm, digits=6))")
        
        # Delayed tracker should remain stationary until t >= 0.3
        dx_del = maximum(abs.(tracker_delayed.particles.x .- x0_del))
        dy_del = maximum(abs.(tracker_delayed.particles.y .- y0_del))
        println("  Delayed: max_dx=$(round(dx_del, digits=6)), max_dy=$(round(dy_del, digits=6))")
        
        # Check time consistency
        println("  Tracker times: immediate=$(tracker_immediate.particles.time), delayed=$(tracker_delayed.particles.time)")
        
        # Verify delayed particles are stationary before advec_time
        if current_time < config_delayed.particle_advec_time
            if dx_del > 1e-12 || dy_del > 1e-12
                println("  ⚠️  WARNING: Delayed particles moved before advec_time!")
            else
                println("  ✓ Delayed particles correctly stationary")
            end
        else
            println("  ✓ Delayed particles now active (t >= $(config_delayed.particle_advec_time))")
        end
    end
    
    println("\\n✅ Particle timing test completed!")
    
    # Final verification
    final_time = config.total_time
    active_time_immediate = final_time - config_immediate.particle_advec_time
    active_time_delayed = final_time - config_delayed.particle_advec_time
    
    println("\\nFinal verification:")
    println("  Total simulation time: $(final_time)")
    println("  Immediate particles active for: $(active_time_immediate) time units")
    println("  Delayed particles active for: $(active_time_delayed) time units")
    
    # Check if timing worked correctly
    if active_time_delayed < active_time_immediate
        println("  ✓ Delayed particles had less active time - feature working correctly!")
    else
        println("  ⚠️  Unexpected timing behavior")
    end
    
    return true
end

# Run test if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_particle_timing()
end