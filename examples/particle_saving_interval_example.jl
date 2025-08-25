"""
Example demonstrating particle saving rate control with time intervals.

This example shows how to control how frequently particle positions are saved 
to trajectory history using the save_interval parameter:

- save_interval: Time interval between trajectory saves (e.g., 0.1 = save every 0.1 time units)
- max_save_points: Maximum number of trajectory points to store (prevents memory overflow)

Key features:
- Particles are advected at every timestep (using dt)  
- Particle positions are only saved at save_interval intervals
- This reduces memory usage for long simulations
- Independent control of simulation timestep vs output frequency
"""

using QGYBJ

function particle_saving_interval_example()
    println("Particle Saving Rate Control Example")
    println("====================================")
    
    # Simulation setup
    domain = create_domain_config(nx=32, ny=32, nz=8)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, wave_type=:random,
        psi_amplitude=0.2, wave_amplitude=0.05,
        random_seed=123
    )
    output = create_output_config(output_dir="./saving_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=1.0,    # 1 second total
        dt=0.001           # Small timestep: 1ms
    )
    
    sim = setup_simulation(config)
    
    println("Simulation setup:")
    println("  Total time: $(config.total_time) s")
    println("  Timestep dt: $(config.dt) s") 
    println("  Total steps: $(Int(config.total_time/config.dt))")
    
    # Test different saving intervals
    saving_configs = [
        (0.01, "High frequency - save every 0.01s"),
        (0.05, "Medium frequency - save every 0.05s"), 
        (0.1,  "Low frequency - save every 0.1s"),
        (0.2,  "Very low frequency - save every 0.2s")
    ]
    
    trackers = []
    
    for (save_interval, description) in saving_configs
        println("\\n" * "="*50)
        println("Testing: $description")
        println("="*50)
        
        # Create particle configuration with specific save_interval
        particle_config = create_particle_config(
            x_min=π/2, x_max=3π/2,
            y_min=π/2, y_max=3π/2,
            z_level=π/2,
            nx_particles=4, ny_particles=4,
            save_interval=save_interval,     # KEY PARAMETER!
            max_save_points=1000,           # Prevent memory overflow
            use_ybj_w=true,
            integration_method=:euler
        )
        
        tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
        initialize_particles!(tracker, particle_config)
        
        println("  Particles: $(tracker.particles.np)")
        println("  Save interval: $(save_interval) s")
        println("  Expected saves: $(Int(ceil(config.total_time/save_interval)))")
        
        # Reset simulation state
        sim_copy = setup_simulation(config)
        
        # Run simulation with this tracker
        nsteps = Int(config.total_time / config.dt)
        save_count = 0
        
        for step in 1:nsteps
            current_time = step * config.dt
            
            # Advance fluid
            if step == 1
                first_projection_step!(sim_copy.state, sim_copy.grid, sim_copy.params, sim_copy.plans)
            else
                leapfrog_step!(sim_copy.state, sim_copy.grid, sim_copy.params, sim_copy.plans)
            end
            
            # Count saves before advection
            old_history_length = length(tracker.particles.x_history)
            
            # Advect particles (may trigger saving)
            advect_particles!(tracker, sim_copy.state, sim_copy.grid, config.dt, current_time)
            
            # Count if a save occurred
            new_history_length = length(tracker.particles.x_history)
            if new_history_length > old_history_length
                save_count += 1
                println("    Save #$save_count at t=$(current_time)")
            end
            
            # Progress indicator
            if step % 100 == 0
                println("    Step $step/$nsteps (t=$(current_time))")
            end
        end
        
        final_history_length = length(tracker.particles.x_history)
        println("  \\n  RESULTS:")
        println("    Total saves: $save_count")
        println("    History length: $final_history_length")
        println("    Memory per particle: $(final_history_length * 3 * 8) bytes (x,y,z positions)")
        println("    Total memory: $(tracker.particles.np * final_history_length * 3 * 8) bytes")
        
        # Save trajectories with descriptive filename
        interval_str = replace(string(save_interval), "." => "p")
        filename = "trajectories_interval_$(interval_str)s.nc"
        write_particle_trajectories(filename, tracker,
                                   metadata=Dict("save_interval" => save_interval,
                                               "description" => description))
        println("    Saved: $filename")
        
        push!(trackers, (tracker, save_interval, description, save_count, final_history_length))
    end
    
    # Summary comparison
    println("\\n" * "="*60)
    println("SAVING INTERVAL COMPARISON")
    println("="*60)
    
    println("\\nInterval\\tSaves\\tMemory/particle\\tTotal Memory")
    println("-"*50)
    
    for (tracker, save_interval, desc, save_count, hist_len) in trackers
        mem_per_particle = hist_len * 3 * 8  # 3 coords × 8 bytes
        total_mem = tracker.particles.np * mem_per_particle
        mem_str = format_bytes(total_mem)
        
        println("$(save_interval)s\\t\\t$(save_count)\\t$(format_bytes(mem_per_particle))\\t\\t$(mem_str)")
    end
    
    println("\\n" * "="*60)
    println("KEY BENEFITS")
    println("="*60)
    
    println("\\n🔄 SIMULATION vs SAVING:")
    println("  • Particles advected every dt = $(config.dt)s ($(Int(1/config.dt)) Hz)")
    println("  • Positions saved every save_interval (user controlled)")
    println("  • Simulation accuracy: controlled by dt")
    println("  • Output size: controlled by save_interval")
    
    println("\\n💾 MEMORY MANAGEMENT:")
    println("  • Larger save_interval = less memory usage")
    println("  • max_save_points prevents memory overflow")
    println("  • Good for long simulations with many particles")
    
    println("\\n⚙️ USAGE:")
    println("  particle_config = create_particle_config(")
    println("      save_interval=0.1,      # Save every 0.1 time units")
    println("      max_save_points=1000,   # Limit trajectory length")
    println("      ...)")
    
    println("\\n🔗 RELATED FEATURES:")
    println("  • Auto file splitting: Set auto_split_files=true to create new files")
    println("    when max_save_points is reached (unlimited trajectory length)")
    println("  • Z-level separation: Use write_particle_trajectories_by_zlevel()")
    println("    to save different depths to separate files")
    println("  • See auto_file_splitting_example.jl and multilevel_particle_example.jl")
    
    println("\\n✅ Particle saving interval feature demonstrated!")
    
    return trackers
end

function format_bytes(bytes::Int)
    if bytes < 1024
        return "$(bytes)B"
    elseif bytes < 1024^2
        return "$(round(bytes/1024, digits=1))KB" 
    elseif bytes < 1024^3
        return "$(round(bytes/1024^2, digits=1))MB"
    else
        return "$(round(bytes/1024^3, digits=1))GB"
    end
end

function simple_saving_test()
    println("Simple Particle Saving Test")
    println("===========================")
    
    # Quick test with different save intervals
    domain = create_domain_config(nx=16, ny=16, nz=4)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical
    )
    output = create_output_config(output_dir="./simple_save_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.2, dt=0.01
    )
    
    sim = setup_simulation(config)
    
    # Test with two different save rates
    for save_interval in [0.02, 0.1]
        println("\\nTesting save_interval = $save_interval")
        
        particle_config = create_particle_config(
            x_min=π, x_max=π, y_min=π, y_max=π, z_level=π/2,
            nx_particles=1, ny_particles=1,
            save_interval=save_interval
        )
        
        tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
        initialize_particles!(tracker, particle_config)
        
        # Run simulation
        nsteps = Int(config.total_time / config.dt)
        for step in 1:nsteps
            current_time = step * config.dt
            
            if step == 1
                first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
            else
                leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
            end
            
            advect_particles!(tracker, sim.state, sim.grid, config.dt, current_time)
        end
        
        println("  Trajectory length: $(length(tracker.particles.x_history)) points")
        println("  Expected: $(Int(ceil(config.total_time/save_interval))) points")
    end
    
    println("\\n✅ Simple saving test completed!")
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    # Uncomment the test you want to run:
    
    # Full demonstration (recommended)
    particle_saving_interval_example()
    
    # Or simple quick test
    # simple_saving_test()
end