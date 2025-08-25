"""
Example demonstrating Euler timestep for particle advection: x = x + dt*u

This example shows that particles are advected using the simple Euler formula:
- x_new = x_old + dt * u
- y_new = y_old + dt * v  
- z_new = z_old + dt * w

where (u,v,w) is the interpolated total velocity at the particle position.
"""

using QGYBJ

function euler_timestep_example()
    println("Euler Timestep Example: x = x + dt*u")
    println("====================================")
    
    # Simple simulation setup
    domain = create_domain_config(nx=16, ny=16, nz=4)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, wave_type=:random,
        psi_amplitude=0.1, wave_amplitude=0.02
    )
    output = create_output_config(output_dir="./euler_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1, dt=0.01
    )
    
    sim = setup_simulation(config)
    
    # Create particle with explicit Euler integration
    particle_config = create_particle_config(
        x_min=π/2, x_max=π/2,  # Single particle
        y_min=π/2, y_max=π/2, 
        z_level=π/4,
        nx_particles=1, ny_particles=1,
        integration_method=:euler,  # Explicit Euler method
        use_ybj_w=true,
        use_3d_advection=true
    )
    
    tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, particle_config)
    
    println("Testing Euler integration: x = x + dt*u")
    println("Initial particle position: ($(tracker.particles.x[1]), $(tracker.particles.y[1]), $(tracker.particles.z[1]))")
    
    # Run simulation and demonstrate Euler timestep
    nsteps = 5
    
    for step in 1:nsteps
        current_time = step * config.dt
        
        # Store position before advection
        x_old = tracker.particles.x[1]
        y_old = tracker.particles.y[1] 
        z_old = tracker.particles.z[1]
        
        # Advance the flow
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        # Get velocity before advection (this is what will be used)
        update_velocity_fields!(tracker, sim.state, sim.grid)
        u, v, w = interpolate_velocity_at_position(x_old, y_old, z_old, tracker)
        
        # Advect particle using Euler method: x = x + dt*u
        advect_particles!(tracker, sim.state, sim.grid, config.dt, current_time)
        
        # Get new positions
        x_new = tracker.particles.x[1]
        y_new = tracker.particles.y[1]
        z_new = tracker.particles.z[1]
        
        # Verify Euler formula was applied correctly
        expected_x = x_old + config.dt * u
        expected_y = y_old + config.dt * v
        expected_z = z_old + config.dt * w
        
        println("\\nStep $step (t=$(current_time)):")
        println("  Old position: ($(round(x_old,digits=6)), $(round(y_old,digits=6)), $(round(z_old,digits=6)))")
        println("  Velocity:     ($(round(u,digits=6)), $(round(v,digits=6)), $(round(w,digits=6)))")
        println("  dt = $(config.dt)")
        println("  Expected:     ($(round(expected_x,digits=6)), $(round(expected_y,digits=6)), $(round(expected_z,digits=6)))")
        println("  Actual:       ($(round(x_new,digits=6)), $(round(y_new,digits=6)), $(round(z_new,digits=6)))")
        
        # Check if Euler formula was applied correctly
        error_x = abs(x_new - expected_x)
        error_y = abs(y_new - expected_y)
        error_z = abs(z_new - expected_z)
        
        if error_x < 1e-14 && error_y < 1e-14 && error_z < 1e-14
            println("  ✓ Perfect Euler timestep: x = x + dt*u")
        else
            println("  ⚠ Error in Euler timestep:")
            println("    Error: ($(error_x), $(error_y), $(error_z))")
        end
    end
    
    println("\\n" * "="*50)
    println("EULER INTEGRATION VERIFICATION")
    println("="*50)
    
    println("\\n✅ Particle advection uses exact Euler formula:")
    println("   x_new = x_old + dt * u")
    println("   y_new = y_old + dt * v")
    println("   z_new = z_old + dt * w")
    
    println("\\n📚 Integration method options:")
    println("   - :euler  -> Simple Euler: x = x + dt*u")
    println("   - :rk2    -> 2nd-order Runge-Kutta")  
    println("   - :rk4    -> 4th-order Runge-Kutta")
    
    println("\\n⚙️  Default is now :euler for all particles")
    println("   Set integration_method=:euler in create_particle_config()")
    
    return tracker
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    euler_timestep_example()
end