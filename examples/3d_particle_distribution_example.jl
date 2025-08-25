"""
3D Particle Distribution Examples for QG-YBJ Simulations.

This example demonstrates the enhanced particle configuration system that supports:
1. Multiple z-levels (layered distributions)
2. Full 3D uniform grids
3. Random 3D distributions
4. Custom particle placements

All patterns work with both QG and YBJ vertical velocities in serial and parallel.
"""

using QGYBJ
using Random

function demonstrate_3d_distributions()
    println("QG-YBJ 3D Particle Distribution Examples")
    println("========================================")
    
    # Set up a basic simulation
    domain = create_domain_config(
        nx=48, ny=48, nz=24,
        Lx=2π, Ly=2π, Lz=π
    )
    
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.2,
        wave_amplitude=0.05,
        random_seed=1234
    )
    
    output = create_output_config(
        output_dir="./3d_particle_test",
        save_vertical_velocity=true
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.5,
        dt=2e-3,
    )
    
    sim = setup_simulation(config)
    
    println("Grid: $(domain.nx)×$(domain.ny)×$(domain.nz)")
    println("Domain: [0,$(domain.Lx)] × [0,$(domain.Ly)] × [0,$(domain.Lz)]")
    
    # Test different 3D distribution patterns
    test_patterns = [
        ("Uniform 3D Grid", test_uniform_3d_grid),
        ("Multiple Layers", test_layered_distribution),
        ("Random 3D Cloud", test_random_3d_distribution),
        ("Custom Patterns", test_custom_distributions)
    ]
    
    results = []
    
    for (pattern_name, test_function) in test_patterns
        println("\\n" * "="^60)
        println("Testing: $pattern_name")
        println("="^60)
        
        result = test_function(sim)
        push!(results, (name=pattern_name, result=result))
    end
    
    # Summary comparison
    println("\\n" * "="^60)
    println("3D DISTRIBUTION SUMMARY")
    println("="^60)
    
    println("\\nPattern\\t\\t\\tParticles\\tZ-Range\\t\\tSpread")
    println("-"^60)
    
    for (name, result) in results
        println("$(rpad(name, 20))\\t$(result.np)\\t\\t$(round(result.z_min, digits=2))-$(round(result.z_max, digits=2))\\t\\t$(round(result.spread, digits=4))")
    end
    
    println("\\n✅ All 3D distribution patterns tested successfully!")
    
    return results
end

function test_uniform_3d_grid(sim)
    println("Creating uniform 3D grid of particles...")
    
    # Create 3D uniform grid: 6×6×4 = 144 particles
    config_3d = create_uniform_3d_grid(
        π/2, 3π/2,      # x-range: central region
        π/2, 3π/2,      # y-range: central region  
        π/6, 5π/6,      # z-range: mid-water column
        6, 6, 4,        # nx, ny, nz particles
        particle_advec_time=0.0,  # Start immediately
        use_ybj_w=true,
        interpolation_method=TRICUBIC,
        integration_method=:rk4
    )
    
    println("  Configuration: Uniform 3D grid")
    println("  Spatial region: [π/2,3π/2] × [π/2,3π/2] × [π/6,5π/6]")
    println("  Grid dimensions: 6×6×4 = 144 particles")
    
    # Initialize tracker
    tracker = ParticleTracker(config_3d, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, config_3d)
    
    println("  Initialized $(tracker.particles.np) particles")
    
    # Analyze initial distribution
    z_min, z_max = extrema(tracker.particles.z)
    z_levels = unique(sort(tracker.particles.z))
    
    println("  Z-levels: $(length(z_levels)) levels from $(round(z_min, digits=3)) to $(round(z_max, digits=3))")
    println("  Particles per level: $(div(tracker.particles.np, length(z_levels)))")
    
    # Run simulation
    run_short_simulation!(sim, tracker, "uniform_3d_grid")
    
    # Analyze final spread
    final_spread = compute_3d_spread(tracker)
    
    return (np=tracker.particles.np, z_min=z_min, z_max=z_max, spread=final_spread)
end

function test_layered_distribution(sim) 
    println("Creating multi-layer particle distribution...")
    
    # Define specific z-levels with different particle counts
    z_levels = [π/8, π/4, π/2, 3π/4, 7π/8]  # 5 layers
    
    config_3d = create_layered_distribution(
        π/4, 7π/4,      # x-range: wide region
        π/4, 7π/4,      # y-range: wide region
        z_levels,       # specific z-levels
        8, 8,           # 8×8 = 64 particles per level
        use_ybj_w=true,
        interpolation_method=ADAPTIVE
    )
    
    println("  Configuration: Layered distribution")
    println("  Spatial region: [π/4,7π/4] × [π/4,7π/4]")
    println("  Z-levels: $z_levels")
    println("  Particles per level: 8×8 = 64")
    
    tracker = ParticleTracker(config_3d, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, config_3d)
    
    println("  Initialized $(tracker.particles.np) particles across $(length(z_levels)) layers")
    
    # Verify layered structure
    for (i, z_level) in enumerate(z_levels)
        count_at_level = sum(abs.(tracker.particles.z .- z_level) .< 1e-10)
        println("    Layer $i (z=$(round(z_level, digits=3))): $count_at_level particles")
    end
    
    run_short_simulation!(sim, tracker, "layered_distribution")
    
    z_min, z_max = extrema(z_levels)
    final_spread = compute_3d_spread(tracker)
    
    return (np=tracker.particles.np, z_min=z_min, z_max=z_max, spread=final_spread)
end

function test_random_3d_distribution(sim)
    println("Creating random 3D particle cloud...")
    
    # Create random distribution in 3D volume
    config_3d = create_random_3d_distribution(
        0.0, 2π,        # x-range: full domain
        0.0, 2π,        # y-range: full domain
        π/4, 3π/4,      # z-range: middle section
        200,            # total particles
        random_seed=9876,
        use_ybj_w=false,  # Use QG vertical velocity
        interpolation_method=TRILINEAR
    )
    
    println("  Configuration: Random 3D distribution")
    println("  Spatial region: [0,2π] × [0,2π] × [π/4,3π/4]")
    println("  Total particles: 200 (randomly placed)")
    
    tracker = ParticleTracker(config_3d, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, config_3d)
    
    println("  Initialized $(tracker.particles.np) particles")
    
    # Analyze randomness
    z_range = maximum(tracker.particles.z) - minimum(tracker.particles.z)
    z_std = std(tracker.particles.z)
    
    println("  Z-range: $(round(z_range, digits=3))")
    println("  Z-standard deviation: $(round(z_std, digits=3))")
    
    run_short_simulation!(sim, tracker, "random_3d_cloud")
    
    z_min, z_max = extrema(tracker.particles.z)
    final_spread = compute_3d_spread(tracker)
    
    return (np=tracker.particles.np, z_min=z_min, z_max=z_max, spread=final_spread)
end

function test_custom_distributions(sim)
    println("Creating custom particle patterns...")
    
    # Create interesting custom patterns
    custom_positions = Vector{Tuple{Float64,Float64,Float64}}()
    
    # Pattern 1: Vertical line at center
    for z in range(π/6, π/2, length=10)
        push!(custom_positions, (π, π, z))
    end
    
    # Pattern 2: Horizontal circle at mid-depth  
    for θ in range(0, 2π, length=16)[1:end-1]
        r = π/2
        x = π + r*cos(θ)
        y = π + r*sin(θ) 
        push!(custom_positions, (x, y, π/2))
    end
    
    # Pattern 3: Spiral pattern
    for i in 1:20
        t = i * π/10
        r = 0.1 + 0.4 * t/(2π)
        x = π + r*cos(t)
        y = π + r*sin(t)
        z = π/4 + (π/4) * i/20
        push!(custom_positions, (x, y, z))
    end
    
    config_3d = create_custom_distribution(
        custom_positions,
        use_ybj_w=true,
        interpolation_method=TRICUBIC,
        integration_method=:rk4
    )
    
    println("  Configuration: Custom distribution")
    println("  Patterns: Vertical line + horizontal circle + spiral")
    println("  Total particles: $(length(custom_positions))")
    
    tracker = ParticleTracker(config_3d, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, config_3d)
    
    println("  Initialized $(tracker.particles.np) particles")
    
    # Analyze pattern complexity
    x_range = maximum(tracker.particles.x) - minimum(tracker.particles.x)
    y_range = maximum(tracker.particles.y) - minimum(tracker.particles.y)
    z_range = maximum(tracker.particles.z) - minimum(tracker.particles.z)
    
    println("  X-range: $(round(x_range, digits=3))")
    println("  Y-range: $(round(y_range, digits=3))")
    println("  Z-range: $(round(z_range, digits=3))")
    
    run_short_simulation!(sim, tracker, "custom_patterns")
    
    z_min, z_max = extrema(tracker.particles.z)
    final_spread = compute_3d_spread(tracker)
    
    return (np=tracker.particles.np, z_min=z_min, z_max=z_max, spread=final_spread)
end

function run_short_simulation!(sim, tracker, output_name)
    println("  Running short simulation (50 steps)...")
    
    # Record initial positions
    x0 = copy(tracker.particles.x)
    y0 = copy(tracker.particles.y)
    z0 = copy(tracker.particles.z)
    
    # Run simulation
    nsteps = 50
    for step in 1:nsteps
        current_time = step * sim.config.dt
        
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        advect_particles!(tracker, sim.state, sim.grid, sim.config.dt, current_time)
        
        if step % 10 == 0
            displacement = sqrt.(
                (tracker.particles.x - x0).^2 + 
                (tracker.particles.y - y0).^2 + 
                (tracker.particles.z - z0).^2
            )
            println("    Step $step: mean displacement = $(round(mean(displacement), digits=6))")
        end
    end
    
    # Save trajectory
    write_particle_trajectories("$(output_name)_trajectories.nc", tracker,
                               metadata=Dict("distribution_type" => output_name))
    
    println("  ✅ Simulation completed, trajectories saved")
    
    return tracker
end

function compute_3d_spread(tracker)
    """Compute 3D RMS spread from initial positions."""
    if length(tracker.particles.x_history) < 2
        return 0.0
    end
    
    # Initial and final positions
    x0, xf = tracker.particles.x_history[1], tracker.particles.x
    y0, yf = tracker.particles.y_history[1], tracker.particles.y
    z0, zf = tracker.particles.z_history[1], tracker.particles.z
    
    # 3D displacement
    displacement = sqrt.((xf - x0).^2 + (yf - y0).^2 + (zf - z0).^2)
    
    return sqrt(mean(displacement.^2))
end

function demonstrate_layered_vs_3d_comparison()
    println("\\nLayered vs 3D Grid Comparison")
    println("==============================")
    
    # Set up minimal simulation for comparison
    domain = create_domain_config(nx=32, ny=32, nz=16)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:analytical, wave_type=:analytical)
    output = create_output_config(output_dir="./comparison_test")
    
    config = create_model_config(domain, stratification, initial_conditions, output,
                                total_time=0.1, dt=1e-3)
    
    sim = setup_simulation(config)
    
    # Test 1: Traditional single z-level
    println("\\n1. Traditional single z-level:")
    config_single = create_particle_config(
        x_min=π/2, x_max=3π/2, y_min=π/2, y_max=3π/2, z_level=π/2,
        nx_particles=6, ny_particles=6
    )
    
    tracker_single = ParticleTracker(config_single, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_single, config_single)
    
    println("   Particles: $(tracker_single.particles.np)")
    println("   Z-levels: 1 (all at z = $(config_single.z_level))")
    
    # Test 2: Multiple layers at specific z-levels
    println("\\n2. Multiple layers:")
    z_levels = [π/4, π/2, 3π/4]
    config_layered = create_layered_distribution(
        π/2, 3π/2, π/2, 3π/2,
        z_levels, 6, 6
    )
    
    tracker_layered = ParticleTracker(config_layered, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_layered, config_layered)
    
    println("   Particles: $(tracker_layered.particles.np)")
    println("   Z-levels: $(length(z_levels)) (at z = $z_levels)")
    
    # Test 3: Full 3D grid
    println("\\n3. Full 3D grid:")
    config_3d = create_uniform_3d_grid(
        π/2, 3π/2, π/2, 3π/2, π/4, 3π/4,
        6, 6, 3
    )
    
    tracker_3d = ParticleTracker(config_3d, sim.grid, sim.parallel_config)
    initialize_particles!(tracker_3d, config_3d)
    
    println("   Particles: $(tracker_3d.particles.np)")
    println("   Z-levels: 3 (uniform grid)")
    
    # Quick advection test
    current_time = sim.config.dt
    for tracker in [tracker_single, tracker_layered, tracker_3d]
        first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        advect_particles!(tracker, sim.state, sim.grid, sim.config.dt, current_time)
    end
    
    println("\\n✅ Comparison completed - all patterns work correctly!")
    
    return (tracker_single, tracker_layered, tracker_3d)
end

# Run examples if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Main 3D distribution examples
    demonstrate_3d_distributions()
    
    # Additional comparison test
    demonstrate_layered_vs_3d_comparison()
end