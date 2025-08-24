"""
Example demonstrating particle advection at multiple z-levels with separate file saving.

This example shows how to:
1. Initialize particles at multiple z-levels (depths) 
2. Advect all particles together in one simulation
3. Save each z-level to separate NetCDF files for independent analysis
4. Compare particle behavior at different depths

Key features:
- Multiple z-level particle initialization
- Unified advection of all particles
- Automatic z-level file separation: particles_z0.785.nc, particles_z1.571.nc, etc.
- Perfect for studying depth-dependent ocean dynamics
"""

using QGYBJ

function multilevel_particle_example()
    println("Multi-Level Particle Advection Example")
    println("=====================================")
    
    # Simulation setup with realistic stratification
    domain = create_domain_config(
        nx=48, ny=48, nz=16, 
        Lx=2π, Ly=2π, Lz=π
    )
    
    # Strong stratification to see depth effects
    stratification = create_stratification_config(:constant_N, N0=2.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random, 
        psi_amplitude=0.3,     # Strong QG flow
        wave_amplitude=0.1,    # Significant wave activity
        random_seed=456
    )
    
    output = create_output_config(output_dir="./multilevel_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.5,        # Moderate simulation time
        dt=1e-3
    )
    
    sim = setup_simulation(config)
    
    println("Domain: $(domain.nx)×$(domain.ny)×$(domain.nz)")
    println("Stratification: N₀ = $(stratification.N0)")
    
    # Method 1: Using layered distribution (RECOMMENDED)
    println("\\n1. Method 1: Using layered particle distribution")
    println("="*50)
    
    # Define specific z-levels (depths) for particles
    z_levels = [π/6, π/3, π/2, 2π/3, 5π/6]  # 5 different depths
    
    println("Particle z-levels:")
    for (i, z) in enumerate(z_levels)
        println("  Level $i: z = $(round(z, digits=3)) ($(round(100*z/π, digits=1))% of domain height)")
    end
    
    # Create layered particle distribution
    layered_config = create_layered_distribution(
        π/4, 7π/4,    # x-range: most of domain
        π/4, 7π/4,    # y-range: most of domain  
        z_levels,     # specific z-levels
        6, 6          # 6x6 particles per level = 36 particles per level
    )
    
    # Enhance config with simulation parameters
    layered_config = ParticleConfig(
        layered_config.x_min, layered_config.x_max,
        layered_config.y_min, layered_config.y_max,
        layered_config.z_min, layered_config.z_max,
        layered_config.nx_particles, layered_config.ny_particles, layered_config.nz_particles,
        z_level=π/2,  # Default (not used for layered)
        distribution_type=layered_config.distribution_type,
        z_levels=layered_config.z_levels,
        particles_per_level=layered_config.particles_per_level,
        
        # Advection settings
        use_ybj_w=true,
        use_3d_advection=true,
        integration_method=:euler,
        
        # I/O settings for z-level analysis
        save_interval=0.02,    # Frequent saving for detailed analysis
        max_save_points=100
    )
    
    # Initialize layered tracker
    layered_tracker = ParticleTracker(layered_config, sim.grid, sim.parallel_config)
    initialize_particles!(layered_tracker, layered_config)
    
    total_particles = layered_tracker.particles.np
    println("\\nLayered distribution:")
    println("  Total particles: $total_particles")
    println("  Expected: $(length(z_levels) * 6 * 6) = $(length(z_levels)) levels × 36 particles/level")
    println("  Particles per level: $(layered_config.particles_per_level)")
    
    # Method 2: Multiple single-level configurations (ALTERNATIVE)
    println("\\n2. Method 2: Multiple single-level particle configurations")
    println("="*50)
    
    single_trackers = []
    single_configs = []
    
    for (i, z_level) in enumerate(z_levels)
        config_single = create_particle_config(
            x_min=π/4, x_max=7π/4,
            y_min=π/4, y_max=7π/4,
            z_level=z_level,
            nx_particles=6, ny_particles=6,
            use_ybj_w=true,
            use_3d_advection=true,
            integration_method=:euler,
            save_interval=0.02,
            max_save_points=100
        )
        
        tracker_single = ParticleTracker(config_single, sim.grid, sim.parallel_config)
        initialize_particles!(tracker_single, config_single)
        
        push!(single_trackers, tracker_single)
        push!(single_configs, config_single)
        
        println("  Level $i: z=$(round(z_level,digits=3)), particles=$(tracker_single.particles.np)")
    end
    
    # Run simulation with layered particles
    println("\\n3. Running simulation with layered particles")
    println("="*50)
    
    nsteps = Int(config.total_time / config.dt)
    println("Running $nsteps steps (dt=$(config.dt), total_time=$(config.total_time))")
    
    for step in 1:nsteps
        current_time = step * config.dt
        
        # Advance fluid simulation
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        # Advect layered particles (all levels together)
        advect_particles!(layered_tracker, sim.state, sim.grid, config.dt, current_time)
        
        # Advect single-level particles (for comparison)
        for tracker in single_trackers
            advect_particles!(tracker, sim.state, sim.grid, config.dt, current_time)
        end
        
        # Progress indicator
        if step % 100 == 0
            println("  Step $step/$nsteps (t=$(current_time))")
        end
    end
    
    # Save trajectories using z-level file separation
    println("\\n4. Saving trajectories with z-level file separation")
    println("="*50)
    
    # Method 1: Automatic z-level separation from layered tracker
    println("Method 1: Automatic z-level file separation")
    layered_files = write_particle_trajectories_by_zlevel("particles_layered", layered_tracker;
                                                         metadata=Dict("method" => "layered_distribution",
                                                                      "total_levels" => length(z_levels)))
    
    println("\\nGenerated layered files:")
    for (z_level, filename) in sort(collect(layered_files))
        println("  z=$(round(z_level, digits=3)): $filename")
    end
    
    # Method 2: Manual saving of single-level trackers  
    println("\\nMethod 2: Manual single-level file saving")
    single_files = String[]
    for (i, (tracker, z_level)) in enumerate(zip(single_trackers, z_levels))
        z_str = replace(string(round(z_level, digits=3)), "." => "p")
        filename = "particles_single_z$(z_str).nc"
        write_particle_trajectories(filename, tracker,
                                   metadata=Dict("method" => "single_level",
                                               "z_level" => z_level,
                                               "level_index" => i))
        push!(single_files, filename)
        println("  z=$(round(z_level, digits=3)): $filename")
    end
    
    # Analysis and comparison
    println("\\n5. Multi-level analysis")
    println("="*50)
    
    # Analyze dispersion at each level
    println("\\nDispersion analysis by z-level:")
    println("Level\\tZ-depth\\tParticles\\tMean_displacement\\tMax_displacement")
    println("-"*70)
    
    # Get initial positions for layered particles (use first history entry)
    if !isempty(layered_tracker.particles.x_history)
        initial_x = layered_tracker.particles.x_history[1]
        initial_y = layered_tracker.particles.y_history[1] 
        initial_z = layered_tracker.particles.z_history[1]
        
        # Current positions
        final_x = layered_tracker.particles.x
        final_y = layered_tracker.particles.y
        final_z = layered_tracker.particles.z
        
        # Group particles by z-level for analysis
        z_tolerance = 1e-6
        for (level, target_z) in enumerate(z_levels)
            # Find particles at this z-level
            level_indices = Int[]
            for (i, z) in enumerate(initial_z)
                if abs(z - target_z) <= z_tolerance
                    push!(level_indices, i)
                end
            end
            
            if !isempty(level_indices)
                # Compute displacements for this level
                dx = final_x[level_indices] - initial_x[level_indices]
                dy = final_y[level_indices] - initial_y[level_indices]
                dz = final_z[level_indices] - initial_z[level_indices]
                
                displacements = sqrt.(dx.^2 + dy.^2 + dz.^2)
                mean_disp = mean(displacements)
                max_disp = maximum(displacements)
                
                println("$level\\t$(round(target_z,digits=3))\\t$(length(level_indices))\\t\\t$(round(mean_disp,digits=4))\\t\\t\\t$(round(max_disp,digits=4))")
            end
        end
    end
    
    println("\\n" * "="*60)
    println("SUMMARY: Multi-Level Particle Analysis")
    println("="*60)
    
    println("\\n✅ KEY FEATURES DEMONSTRATED:")
    println("  • Multiple z-level particle initialization")
    println("  • Unified advection of all particles")
    println("  • Automatic z-level file separation")
    println("  • Depth-dependent dispersion analysis")
    
    println("\\n📁 OUTPUT FILES:")
    println("  Layered method (automatic separation):")
    for (z_level, filename) in sort(collect(layered_files))
        println("    $filename")
    end
    println("  Single method (manual files):")
    for filename in single_files
        println("    $filename")
    end
    
    println("\\n⚙️  USAGE PATTERNS:")
    println("  # Method 1: Layered distribution (recommended)")
    println("  config = create_layered_distribution(x_min, x_max, y_min, y_max, z_levels, nx, ny)")
    println("  tracker = ParticleTracker(config, grid, parallel_config)")
    println("  # Run simulation...")
    println("  files = write_particle_trajectories_by_zlevel(\"particles\", tracker)")
    
    println("\\n  # Method 2: Multiple single-level configs")
    println("  for z in z_levels")
    println("      config = create_particle_config(z_level=z, ...)")
    println("      # Run and save separately")
    println("  end")
    
    println("\\n🌊 OCEANOGRAPHIC APPLICATIONS:")
    println("  • Mixed layer vs thermocline particle behavior")
    println("  • Depth-dependent horizontal dispersion") 
    println("  • Vertical mixing analysis")
    println("  • Multi-layer current interactions")
    
    return (layered_tracker, single_trackers, layered_files, single_files)
end

function simple_multilevel_test()
    println("Simple Multi-Level Particle Test")
    println("===============================")
    
    # Quick test setup
    domain = create_domain_config(nx=16, ny=16, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical
    )
    output = create_output_config(output_dir="./simple_multilevel")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1, dt=0.01
    )
    
    sim = setup_simulation(config)
    
    # Test with 3 z-levels
    z_levels = [π/4, π/2, 3π/4]
    
    println("Testing $(length(z_levels)) z-levels: $(round.(z_levels, digits=3))")
    
    # Create layered distribution
    layered_config = create_layered_distribution(
        0.0, 2π, 0.0, 2π, z_levels, 3, 3  # 3x3 particles per level = 9 per level
    )
    
    # Convert to full ParticleConfig
    config_full = ParticleConfig(
        layered_config.x_min, layered_config.x_max,
        layered_config.y_min, layered_config.y_max, 
        layered_config.z_min, layered_config.z_max,
        layered_config.nx_particles, layered_config.ny_particles, layered_config.nz_particles,
        z_level=π/2,
        distribution_type=layered_config.distribution_type,
        z_levels=layered_config.z_levels,
        particles_per_level=layered_config.particles_per_level,
        save_interval=0.02
    )
    
    tracker = ParticleTracker(config_full, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, config_full)
    
    println("Initialized $(tracker.particles.np) particles")
    
    # Run short simulation
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
    
    # Test z-level file separation
    println("\\nTesting z-level file separation...")
    files = write_particle_trajectories_by_zlevel("simple_test", tracker)
    
    println("\\n✅ Simple multi-level test completed!")
    println("Created $(length(files)) z-level files:")
    for (z, filename) in sort(collect(files))
        println("  z=$(round(z, digits=3)): $filename")
    end
    
    return tracker, files
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    # Uncomment the example you want to run:
    
    # Full comprehensive example (recommended)
    multilevel_particle_example()
    
    # Or simple quick test
    # simple_multilevel_test()
end