"""
Example demonstrating automatic file splitting for long particle trajectories.

This example shows how to use the automatic file splitting feature to manage 
large trajectory datasets by splitting them into smaller, manageable files.

Key features:
- Automatic file creation when max_save_points is reached
- Sequential file naming: simulation.nc, simulation_part1.nc, simulation_part2.nc, etc.
- Unlimited trajectory length with controlled file sizes
- Perfect for very long simulations or high-frequency data collection

Use cases:
- Long-term oceanographic simulations (months/years)
- High-frequency particle tracking (small save_interval)
- Memory management for large particle ensembles
- Automated post-processing workflows
"""

using QGYBJ

function auto_file_splitting_example()
    println("Automatic File Splitting Example")
    println("================================")
    
    # Setup for a relatively long simulation to demonstrate file splitting
    domain = create_domain_config(nx=32, ny=32, nz=8)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    initial_conditions = create_initial_condition_config(
        psi_type=:random, wave_type=:random,
        psi_amplitude=0.2, wave_amplitude=0.05,
        random_seed=789
    )
    output = create_output_config(output_dir="./auto_split_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=2.0,        # Longer simulation to generate more data
        dt=0.005               # Moderate timestep
    )
    
    sim = setup_simulation(config)
    
    println("Simulation setup:")
    println("  Total time: $(config.total_time)")
    println("  Timestep: $(config.dt)")
    println("  Total steps: $(Int(config.total_time/config.dt))")
    
    # Method 1: Enable auto-splitting with convenience function (RECOMMENDED)
    println("\\n1. Method 1: Using enable_auto_file_splitting!")
    println("="*50)
    
    particle_config_auto = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        save_interval=0.05,    # Frequent saving to demonstrate splitting
        integration_method=:euler,
        use_ybj_w=true
    )
    
    tracker_auto = ParticleTracker(particle_config_auto, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_auto, particle_config_auto)
    
    # Enable automatic file splitting with small files for demonstration
    enable_auto_file_splitting!(tracker_auto, "auto_split_demo", max_points_per_file=25)
    
    # Method 2: Configure auto-splitting directly in ParticleConfig (ALTERNATIVE)  
    println("\\n2. Method 2: Direct configuration in ParticleConfig")
    println("="*50)
    
    particle_config_direct = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        save_interval=0.05,
        max_save_points=30,     # Small file size for demo
        auto_split_files=true,  # Enable auto-splitting directly
        integration_method=:euler,
        use_ybj_w=true
    )
    
    tracker_direct = ParticleTracker(particle_config_direct, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_direct, particle_config_direct)
    
    # Set base filename for direct method
    tracker_direct.base_output_filename = "direct_split_demo"
    tracker_direct.auto_file_splitting = true
    
    println("Auto-splitting configurations:")
    println("  Method 1 - Max points per file: 25")
    println("  Method 2 - Max points per file: 30")
    println("  Save interval: $(particle_config_auto.save_interval)")
    
    # Method 3: Traditional approach (no auto-splitting) for comparison
    println("\\n3. Method 3: Traditional approach (no auto-splitting)")
    println("="*50)
    
    particle_config_traditional = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=6, ny_particles=6,
        save_interval=0.05,
        max_save_points=50,     # Will stop saving after 50 points
        auto_split_files=false, # Traditional behavior
        integration_method=:euler,
        use_ybj_w=true
    )
    
    tracker_traditional = ParticleTracker(particle_config_traditional, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_traditional, particle_config_traditional)
    
    # Run simulation with all three methods
    println("\\n4. Running simulation with all three approaches")
    println("="*50)
    
    nsteps = Int(config.total_time / config.dt)
    println("Running $nsteps steps...")
    
    # Progress tracking
    file_creation_times = Dict()
    
    for step in 1:nsteps
        current_time = step * config.dt
        
        # Advance fluid simulation
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        # Track trajectory lengths before advection
        auto_len_before = length(tracker_auto.particles.time_history)
        direct_len_before = length(tracker_direct.particles.time_history)
        trad_len_before = length(tracker_traditional.particles.time_history)
        
        # Advect all trackers
        advect_particles!(tracker_auto, sim.state, sim.grid, config.dt, current_time)
        advect_particles!(tracker_direct, sim.state, sim.grid, config.dt, current_time)
        advect_particles!(tracker_traditional, sim.state, sim.grid, config.dt, current_time)
        
        # Check if files were created (history was reset)
        auto_len_after = length(tracker_auto.particles.time_history)
        direct_len_after = length(tracker_direct.particles.time_history)
        
        if auto_len_after < auto_len_before
            file_creation_times["auto_$(tracker_auto.output_file_sequence-1)"] = current_time
            println("  📁 Auto method: Created file segment at t=$(current_time)")
        end
        
        if direct_len_after < direct_len_before  
            file_creation_times["direct_$(tracker_direct.output_file_sequence-1)"] = current_time
            println("  📁 Direct method: Created file segment at t=$(current_time)")
        end
        
        # Progress indicator
        if step % 100 == 0
            println("    Step $step/$nsteps (t=$(current_time))")
            println("      Auto: $(length(tracker_auto.particles.time_history)) points, seq=$(tracker_auto.output_file_sequence)")
            println("      Direct: $(length(tracker_direct.particles.time_history)) points, seq=$(tracker_direct.output_file_sequence)")  
            println("      Traditional: $(length(tracker_traditional.particles.time_history)) points")
        end
    end
    
    # Finalize auto-splitting trackers
    println("\\n5. Finalizing trajectory files")
    println("="*50)
    
    println("Finalizing auto method...")
    finalize_trajectory_files!(tracker_auto)
    
    println("Finalizing direct method...")
    finalize_trajectory_files!(tracker_direct)
    
    # Save traditional method (single file)
    println("Saving traditional method...")
    write_particle_trajectories("traditional_demo.nc", tracker_traditional,
                               metadata=Dict("method" => "traditional", 
                                           "stopped_at_max_points" => true,
                                           "max_save_points" => particle_config_traditional.max_save_points))
    
    # Analysis and comparison
    println("\\n6. File splitting analysis")
    println("="*50)
    
    # Count files created
    auto_files = tracker_auto.output_file_sequence + 1
    direct_files = tracker_direct.output_file_sequence + 1
    
    println("Files created:")
    println("  Auto method: $auto_files files")
    println("  Direct method: $direct_files files")
    println("  Traditional: 1 file")
    
    println("\\nFile creation timeline:")
    for (file_id, time) in sort(collect(file_creation_times), by=x->x[2])
        println("  $file_id: t=$(round(time, digits=3))")
    end
    
    println("\\nTrajectory points analysis:")
    auto_total = tracker_auto.save_counter
    direct_total = tracker_direct.save_counter
    trad_total = length(tracker_traditional.particles.time_history)
    
    println("  Auto method: $auto_total total points")
    println("  Direct method: $direct_total total points")
    println("  Traditional: $trad_total points (stopped at max_save_points)")
    
    # List generated files
    println("\\n" * "="*60)
    println("GENERATED FILES")
    println("="*60)
    
    println("\\nAuto method files:")
    for i in 0:(auto_files-1)
        if i == 0
            println("  auto_split_demo.nc")
        else
            println("  auto_split_demo_part$(i).nc")
        end
    end
    
    println("\\nDirect method files:")
    for i in 0:(direct_files-1)
        if i == 0
            println("  direct_split_demo.nc")
        else
            println("  direct_split_demo_part$(i).nc")
        end
    end
    
    println("\\nTraditional method:")
    println("  traditional_demo.nc")
    
    println("\\n" * "="*60)
    println("KEY BENEFITS")
    println("="*60)
    
    println("\\n✅ AUTOMATIC FILE SPLITTING:")
    println("  • Unlimited trajectory length with manageable file sizes")
    println("  • Sequential file naming for easy post-processing")
    println("  • Automatic metadata tracking (time ranges, segment numbers)")
    println("  • Memory efficient - old trajectory data is written to disk")
    
    println("\\n📊 COMPARED TO TRADITIONAL APPROACH:")
    println("  • Traditional: Stops at max_save_points, loses data")
    println("  • Auto-splitting: Continues indefinitely, saves all data")
    println("  • File sizes remain bounded and manageable")
    
    println("\\n⚙️  USAGE RECOMMENDATIONS:")
    println("  • For long simulations: max_points_per_file = 500-2000")
    println("  • For high-frequency data: smaller max_points_per_file")
    println("  • Use enable_auto_file_splitting!() for simplicity")
    println("  • Always call finalize_trajectory_files!() at end")
    
    println("\\n🌊 OCEANOGRAPHIC APPLICATIONS:")
    println("  • Multi-year climate simulations")
    println("  • High-resolution Lagrangian studies")
    println("  • Large particle ensembles (oil spills, larvae, etc.)")
    println("  • Automated data processing pipelines")
    
    return (tracker_auto, tracker_direct, tracker_traditional)
end

function simple_file_splitting_test()
    println("Simple Auto File Splitting Test")
    println("===============================")
    
    # Quick test with very small file sizes
    domain = create_domain_config(nx=16, ny=16, nz=4)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical
    )
    output = create_output_config(output_dir="./simple_split")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.5, dt=0.01
    )
    
    sim = setup_simulation(config)
    
    # Create tracker with very small max_save_points for quick testing
    particle_config = create_particle_config(
        x_min=π, x_max=π, y_min=π, y_max=π, z_level=π/2,
        nx_particles=1, ny_particles=1,
        save_interval=0.02,    # Save frequently 
        integration_method=:euler
    )
    
    tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, particle_config)
    
    # Enable auto-splitting with tiny files for demo
    enable_auto_file_splitting!(tracker, "simple_test", max_points_per_file=5)
    
    println("Running simulation with max 5 points per file...")
    
    nsteps = Int(config.total_time / config.dt)
    for step in 1:nsteps
        current_time = step * config.dt
        
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        len_before = length(tracker.particles.time_history)
        advect_particles!(tracker, sim.state, sim.grid, config.dt, current_time)
        len_after = length(tracker.particles.time_history)
        
        if len_after < len_before
            println("  📁 File created at t=$(current_time), sequence=$(tracker.output_file_sequence-1)")
        end
    end
    
    # Finalize
    finalize_trajectory_files!(tracker)
    
    total_files = tracker.output_file_sequence + 1
    println("\\n✅ Simple test completed!")
    println("Created $total_files files with ~5 points each")
    
    return tracker
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    # Uncomment the example you want to run:
    
    # Full comprehensive example (recommended)
    auto_file_splitting_example()
    
    # Or simple quick test
    # simple_file_splitting_test()
end