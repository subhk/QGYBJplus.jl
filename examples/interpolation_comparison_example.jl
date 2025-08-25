"""
Interpolation Method Comparison Example for QG-YBJ Particle Advection.

This example demonstrates the accuracy and performance differences between
different interpolation methods: trilinear, tricubic, and adaptive schemes.
"""

using QGYBJ

function interpolation_comparison_example()
    println("QG-YBJ Interpolation Methods Comparison")
    println("=======================================")
    
    # Create a test simulation with a known analytical solution
    domain = create_domain_config(
        nx=32, ny=32, nz=16,
        Lx=2π, Ly=2π, Lz=π
    )
    
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    # Use analytical initial conditions for known velocity field
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,
        wave_type=:analytical,
        psi_amplitude=0.2,
        wave_amplitude=0.05
    )
    
    output = create_output_config(
        output_dir="./interpolation_test",
        save_vertical_velocity=true
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.2,  # Short time for accuracy testing
        dt=1e-3,
    )
    
    sim = setup_simulation(config)
    
    println("Testing interpolation methods with $(domain.nx)×$(domain.ny)×$(domain.nz) grid")
    
    # Test different interpolation methods
    interpolation_methods = [
        (TRILINEAR, "Trilinear (O(h²))"),
        (TRICUBIC, "Tricubic (O(h⁴))"),
        (ADAPTIVE, "Adaptive")
    ]
    
    results = []
    
    for (method, method_name) in interpolation_methods
        println("\\nTesting $method_name interpolation...")
        
        # Create particle configuration for this method
        particle_config = create_particle_config(
            x_min=π/2, x_max=3π/2,
            y_min=π/2, y_max=3π/2,
            z_level=π/2,
            nx_particles=6, ny_particles=6,
            use_ybj_w=true,
            use_3d_advection=true,
            integration_method=:rk4,
            interpolation_method=method  # Key difference!
        )
        
        # Initialize tracker
        tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
        initialize_particles!(tracker, particle_config)
        
        println("  Initialized $(tracker.particles.np) particles")
        println("  Interpolation method: $method_name")
        
        # Record initial positions for later analysis
        x0 = copy(tracker.particles.x)
        y0 = copy(tracker.particles.y)
        z0 = copy(tracker.particles.z)
        
        # Time the interpolation performance
        start_time = time()
        
        # Run a few timesteps to test accuracy
        nsteps = 100
        for step in 1:nsteps
            current_time = step * sim.config.dt
            
            if step == 1
                first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
            else
                leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
            end
            
            advect_particles!(tracker, sim.state, sim.grid, sim.config.dt, current_time)
        end
        
        elapsed_time = time() - start_time
        
        # Compute final displacement statistics
        dx = tracker.particles.x - x0
        dy = tracker.particles.y - y0
        dz = tracker.particles.z - z0
        
        total_displacement = sqrt.(dx.^2 + dy.^2 + dz.^2)
        mean_displacement = mean(total_displacement)
        max_displacement = maximum(total_displacement)
        displacement_std = std(total_displacement)
        
        # Compute velocity statistics at final positions
        final_velocities = []
        for i in 1:tracker.particles.np
            u, v, w = interpolate_velocity_at_position(
                tracker.particles.x[i], 
                tracker.particles.y[i], 
                tracker.particles.z[i], 
                tracker
            )
            push!(final_velocities, sqrt(u^2 + v^2 + w^2))
        end
        
        mean_velocity = mean(final_velocities)
        velocity_std = std(final_velocities)
        
        result = (
            method = method,
            method_name = method_name,
            mean_displacement = mean_displacement,
            max_displacement = max_displacement,
            displacement_std = displacement_std,
            mean_velocity = mean_velocity,
            velocity_std = velocity_std,
            elapsed_time = elapsed_time,
            time_per_step = elapsed_time / nsteps
        )
        
        push!(results, result)
        
        println("  Results:")
        println("    Mean displacement: $(round(mean_displacement, digits=6))")
        println("    Max displacement: $(round(max_displacement, digits=6))")
        println("    Displacement std: $(round(displacement_std, digits=6))")
        println("    Mean final velocity: $(round(mean_velocity, digits=6))")
        println("    Elapsed time: $(round(elapsed_time, digits=3)) seconds")
        println("    Time per step: $(round(1000*elapsed_time/nsteps, digits=2)) ms")
        
        # Save trajectories for this method
        write_particle_trajectories("trajectories_$(lowercase(string(method))).nc", 
                                   tracker, metadata=Dict("interpolation_method" => method_name))
    end
    
    # Performance and accuracy comparison
    println("\\n" * "="^50)
    println("INTERPOLATION METHOD COMPARISON")
    println("="^50)
    
    # Reference (trilinear) results
    ref_result = results[1]
    
    println("\\nACCURACY COMPARISON (relative to trilinear):")
    println("Method\\t\\t\\tMean Disp\\tMax Disp\\tVel Std")
    println("-"^60)
    
    for result in results
        rel_mean = result.mean_displacement / ref_result.mean_displacement
        rel_max = result.max_displacement / ref_result.max_displacement  
        rel_vel_std = result.velocity_std / ref_result.velocity_std
        
        println("$(result.method_name)\\t$(round(rel_mean, digits=3))\\t\\t$(round(rel_max, digits=3))\\t\\t$(round(rel_vel_std, digits=3))")
    end
    
    println("\\nPERFORMACE COMPARISON:")
    println("Method\\t\\t\\tTime/Step (ms)\\tRelative Cost")
    println("-"^60)
    
    for result in results
        rel_time = result.time_per_step / ref_result.time_per_step
        println("$(result.method_name)\\t$(round(1000*result.time_per_step, digits=2))\\t\\t$(round(rel_time, digits=2))x")
    end
    
    # Theoretical error analysis
    println("\\nTHEORETICAL ERROR SCALING:")
    h = π / domain.nx  # Typical grid spacing
    println("Grid spacing h = $(round(h, digits=4))")
    
    for (method, method_name) in interpolation_methods
        error_est = interpolation_error_estimate(method, h)
        println("$method_name: O(h^n) ≈ $(round(error_est, digits=6))")
    end
    
    # Recommendations
    println("\\n" * "="^50) 
    println("RECOMMENDATIONS")
    println("="^50)
    
    tricubic_result = results[2]
    adaptive_result = results[3]
    
    println("\\n🎯 ACCURACY:")
    if tricubic_result.displacement_std < ref_result.displacement_std * 0.9
        println("  ✅ Tricubic provides improved accuracy")
    else
        println("  ⚠️  Tricubic accuracy similar to trilinear for this case")
    end
    
    println("\\n⚡ PERFORMANCE:")
    if tricubic_result.time_per_step < ref_result.time_per_step * 2.0
        println("  ✅ Tricubic cost increase is acceptable (<2x)")
    else
        println("  ⚠️  Tricubic has significant cost increase (>2x)")
    end
    
    println("\\n🧠 ADAPTIVE:")
    if adaptive_result.displacement_std < tricubic_result.displacement_std && 
       adaptive_result.time_per_step < tricubic_result.time_per_step
        println("  ✅ Adaptive method provides best accuracy/performance balance")
    else
        println("  ⚠️  Adaptive method needs tuning for this problem")
    end
    
    println("\\n✅ Interpolation comparison completed!")
    println("\\nOutput files:")
    for (method, _) in interpolation_methods
        println("  - trajectories_$(lowercase(string(method))).nc")
    end
    
    return results
end

function simple_interpolation_test()
    println("Simple Interpolation Accuracy Test")
    println("==================================")
    
    # Quick test with smaller grid
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical
    )
    output = create_output_config(output_dir="./simple_interp_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.05, dt=1e-3
    )
    
    sim = setup_simulation(config)
    
    # Test tricubic vs trilinear with single particle
    test_methods = [TRILINEAR, TRICUBIC]
    
    for method in test_methods
        println("\\nTesting $(method) interpolation...")
        
        particle_config = create_particle_config(
            x_min=π, x_max=π, y_min=π, y_max=π, z_level=π/2,
            nx_particles=1, ny_particles=1,
            interpolation_method=method,
            use_ybj_w=true
        )
        
        tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
        initialize_particles!(tracker, particle_config)
        
        println("  Initial position: ($(tracker.particles.x[1]), $(tracker.particles.y[1]), $(tracker.particles.z[1]))")
        
        # Single timestep
        current_time = sim.config.dt
        first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        advect_particles!(tracker, sim.state, sim.grid, sim.config.dt, current_time)
        
        println("  Final position: ($(tracker.particles.x[1]), $(tracker.particles.y[1]), $(tracker.particles.z[1]))")
        println("  Final velocity: ($(tracker.particles.u[1]), $(tracker.particles.v[1]), $(tracker.particles.w[1]))")
    end
    
    println("\\n✅ Simple interpolation test completed!")
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Uncomment the test you want to run:
    
    # Full comparison (recommended)
    interpolation_comparison_example()
    
    # Or simple quick test
    # simple_interpolation_test()
end