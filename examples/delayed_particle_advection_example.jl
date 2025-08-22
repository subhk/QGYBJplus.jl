"""
Example demonstrating delayed particle advection in QG-YBJ simulations.

This example shows how to control when particles start moving using the 
particle_advec_time parameter:

1. Particles can remain stationary initially (particle_advec_time > 0)
2. Flow field develops before particles start moving
3. Comparison between immediate vs delayed particle release
4. Useful for studying transient features or established flow patterns

The particle_advec_time feature allows users to:
- Let the flow field spin up before releasing particles
- Study particle transport in established vs developing flows
- Compare different release timing strategies
- Simulate realistic oceanographic scenarios (e.g., dye release experiments)
"""

using QGYBJ

function delayed_particle_advection_example()
    println("QG-YBJ Delayed Particle Advection Example")
    println("=========================================")
    
    # Simulation setup
    domain = create_domain_config(nx=64, ny=64, nz=16, Lx=2π, Ly=2π, Lz=π)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.2,     # Strong initial conditions
        wave_amplitude=0.05,
        random_seed=123
    )
    
    output = create_output_config(
        output_dir="./delayed_particle_test",
        save_vertical_velocity=true
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=2.0,        # Longer simulation to see delayed effect
        dt=2e-3,
        Ro=0.1,
        Fr=0.1
    )
    
    sim = setup_simulation(config)
    
    println("Setting up particle configurations with different release times...")
    
    # Configuration 1: Immediate release (traditional approach)
    config_immediate = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        particle_advec_time=0.0,     # Start immediately
        use_ybj_w=false,
        use_3d_advection=true,
        integration_method=:rk4,
        save_interval=0.05
    )
    
    # Configuration 2: Delayed release - let flow develop first
    config_delayed_early = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        particle_advec_time=0.5,     # Wait 0.5 time units
        use_ybj_w=false,
        use_3d_advection=true,
        integration_method=:rk4,
        save_interval=0.05
    )
    
    # Configuration 3: Late release - well-developed flow
    config_delayed_late = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        particle_advec_time=1.0,     # Wait 1.0 time units
        use_ybj_w=false,
        use_3d_advection=true,
        integration_method=:rk4,
        save_interval=0.05
    )
    
    # Initialize particle trackers
    println("Initializing particle trackers...")
    tracker_immediate = ParticleTracker(config_immediate, sim.grid, sim.parallel_config)
    tracker_delayed_early = ParticleTracker(config_delayed_early, sim.grid, sim.parallel_config)
    tracker_delayed_late = ParticleTracker(config_delayed_late, sim.grid, sim.parallel_config)
    
    initialize_particles!(tracker_immediate, config_immediate)
    initialize_particles!(tracker_delayed_early, config_delayed_early)
    initialize_particles!(tracker_delayed_late, config_delayed_late)
    
    println("  Immediate release: $(tracker_immediate.particles.np) particles (start at t=0.0)")
    println("  Early delayed: $(tracker_delayed_early.particles.np) particles (start at t=0.5)")
    println("  Late delayed: $(tracker_delayed_late.particles.np) particles (start at t=1.0)")
    
    # Store initial positions for comparison
    x0_immediate = copy(tracker_immediate.particles.x)
    y0_immediate = copy(tracker_immediate.particles.y)
    x0_delayed_early = copy(tracker_delayed_early.particles.x)
    y0_delayed_early = copy(tracker_delayed_early.particles.y)
    x0_delayed_late = copy(tracker_delayed_late.particles.x)
    y0_delayed_late = copy(tracker_delayed_late.particles.y)
    
    println("\\nRunning simulation with delayed particle releases...")
    
    # Simulation parameters
    total_steps = Int(config.total_time / config.dt)
    output_interval = 50  # Print every 50 steps
    
    for step in 1:total_steps
        # Advance fluid simulation one step
        # (This would be the actual time-stepping routine)
        current_time = step * config.dt
        
        # Update particle positions for all trackers (pass simulation time for synchronization)
        advect_particles!(tracker_immediate, sim.state, sim.grid, config.dt, current_time)
        advect_particles!(tracker_delayed_early, sim.state, sim.grid, config.dt, current_time)
        advect_particles!(tracker_delayed_late, sim.state, sim.grid, config.dt, current_time)
        
        # Print status periodically
        if step % output_interval == 0
            println(\"  Step $step (t=$(round(current_time, digits=2))): \")
            
            # Check which trackers are actively advecting
            immediate_moving = current_time >= config_immediate.particle_advec_time
            early_moving = current_time >= config_delayed_early.particle_advec_time
            late_moving = current_time >= config_delayed_late.particle_advec_time
            
            status_immediate = immediate_moving ? \"MOVING\" : \"stationary\"
            status_early = early_moving ? \"MOVING\" : \"stationary\"
            status_late = late_moving ? \"MOVING\" : \"stationary\"
            
            println(\"    Immediate particles: $status_immediate\")
            println(\"    Early delayed particles: $status_early\")
            println(\"    Late delayed particles: $status_late\")
            
            # Compute displacement from initial positions
            if tracker_immediate.particles.np > 0
                dx_imm = sqrt(sum((tracker_immediate.particles.x .- x0_immediate).^2 + 
                                 (tracker_immediate.particles.y .- y0_immediate).^2) / tracker_immediate.particles.np)
                println(\"    Mean displacement (immediate): $(round(dx_imm, digits=4))\")
            end
            
            if tracker_delayed_early.particles.np > 0 && early_moving
                dx_early = sqrt(sum((tracker_delayed_early.particles.x .- x0_delayed_early).^2 + 
                                   (tracker_delayed_early.particles.y .- y0_delayed_early).^2) / tracker_delayed_early.particles.np)
                println(\"    Mean displacement (early delayed): $(round(dx_early, digits=4))\")
            end
            
            if tracker_delayed_late.particles.np > 0 && late_moving
                dx_late = sqrt(sum((tracker_delayed_late.particles.x .- x0_delayed_late).^2 + 
                                  (tracker_delayed_late.particles.y .- y0_delayed_late).^2) / tracker_delayed_late.particles.np)
                println(\"    Mean displacement (late delayed): $(round(dx_late, digits=4))\")
            end
        end
    end
    
    println(\"\\n✅ Delayed particle advection example completed!\")
    
    # Final analysis
    final_time = config.total_time
    println(\"\\nFinal Analysis (t=$(final_time)):\")
    
    # Calculate final displacements
    if tracker_immediate.particles.np > 0
        final_disp_imm = sqrt(sum((tracker_immediate.particles.x .- x0_immediate).^2 + 
                                 (tracker_immediate.particles.y .- y0_immediate).^2) / tracker_immediate.particles.np)
        active_time_imm = final_time - config_immediate.particle_advec_time
        println(\"  Immediate release:\")
        println(\"    Active advection time: $(active_time_imm) time units\")
        println(\"    Final mean displacement: $(round(final_disp_imm, digits=4))\")
    end
    
    if tracker_delayed_early.particles.np > 0
        final_disp_early = sqrt(sum((tracker_delayed_early.particles.x .- x0_delayed_early).^2 + 
                                   (tracker_delayed_early.particles.y .- y0_delayed_early).^2) / tracker_delayed_early.particles.np)
        active_time_early = final_time - config_delayed_early.particle_advec_time
        println(\"  Early delayed release:\")
        println(\"    Active advection time: $(active_time_early) time units\")
        println(\"    Final mean displacement: $(round(final_disp_early, digits=4))\")
    end
    
    if tracker_delayed_late.particles.np > 0
        final_disp_late = sqrt(sum((tracker_delayed_late.particles.x .- x0_delayed_late).^2 + 
                                  (tracker_delayed_late.particles.y .- y0_delayed_late).^2) / tracker_delayed_late.particles.np)
        active_time_late = final_time - config_delayed_late.particle_advec_time
        println(\"  Late delayed release:\")
        println(\"    Active advection time: $(active_time_late) time units\")
        println(\"    Final mean displacement: $(round(final_disp_late, digits=4))\")
    end
    
    println(\"\\nKey insights:\")
    println(\"  • Particles released later experience different flow conditions\")
    println(\"  • Early flow development vs established patterns affect transport\")
    println(\"  • particle_advec_time enables studying flow evolution effects\")
    println(\"  • Useful for realistic oceanographic release scenarios\")
    
    return (tracker_immediate, tracker_delayed_early, tracker_delayed_late)
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    delayed_particle_advection_example()
end