"""
Comprehensive example of unified particle advection in QG-YBJ simulation.

This example demonstrates the unified particle advection system that automatically
handles both serial and parallel execution:

1. Setting up particles on a horizontal region at constant z-level
2. Advecting particles using total velocity (QG + YBJ) 
3. Comparing QG vs YBJ vertical velocity effects on particles
4. Saving particle trajectories to NetCDF files
5. Automatic MPI detection and domain decomposition
6. Particle migration between MPI domains
"""

using QGYBJ

function particle_advection_example()
    println("QG-YBJ Unified Particle Advection Example") 
    println("==========================================")
    
    # Detect execution mode
    try
        import MPI
        if MPI.Initialized()
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)
            println("Running in PARALLEL mode: rank $rank of $nprocs")
        else
            println("Running in SERIAL mode")
        end
    catch
        println("Running in SERIAL mode (MPI not available)")
    end
    
    # 1. Create model configuration
    println("Setting up QG-YBJ simulation...")
    
    domain = create_domain_config(
        nx=64, ny=64, nz=16,
        Lx=2π, Ly=2π, Lz=π
    )
    
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.3,
        wave_amplitude=0.05,
        random_seed=1234
    )
    
    output = create_output_config(
        output_dir="./particle_example",
        save_vertical_velocity=true,
        output_interval=10
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=1.0,
        dt=2e-3,
        Ro=0.1,
        Fr=0.1
    )
    
    # Set up simulation
    sim = setup_simulation(config)
    
    # 2. Configure particles for comparison tests
    println("Setting up particle configurations...")
    
    # Test region: central part of domain at mid-depth
    z_level = π/2  # Mid-depth
    
    # Particle configuration with QG vertical velocity
    particle_config_qg = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=z_level,
        nx_particles=8, ny_particles=8,
        use_ybj_w=false,      # Use QG omega equation
        use_3d_advection=true,
        integration_method=:rk4
    )
    
    # Particle configuration with YBJ vertical velocity
    particle_config_ybj = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=z_level,
        nx_particles=8, ny_particles=8,
        use_ybj_w=true,       # Use YBJ formulation
        use_3d_advection=true,
        integration_method=:rk4
    )
    
    # 2D advection for comparison
    particle_config_2d = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=z_level,
        nx_particles=8, ny_particles=8,
        use_ybj_w=false,
        use_3d_advection=false,  # Pure 2D advection (w=0)
        integration_method=:rk4
    )
    
    # 3. Initialize unified particle trackers (automatically handles serial/parallel)
    println("Initializing unified particle trackers...")
    
    tracker_qg = ParticleTracker(particle_config_qg, sim.grid)
    tracker_ybj = ParticleTracker(particle_config_ybj, sim.grid)
    tracker_2d = ParticleTracker(particle_config_2d, sim.grid)
    
    initialize_particles!(tracker_qg, particle_config_qg)
    initialize_particles!(tracker_ybj, particle_config_ybj)
    initialize_particles!(tracker_2d, particle_config_2d)
    
    # Report local particle counts (in parallel, each rank reports its local count)
    println("  QG particles (local): $(tracker_qg.particles.np)")
    println("  YBJ particles (local): $(tracker_ybj.particles.np)")
    println("  2D particles (local): $(tracker_2d.particles.np)")
    
    if tracker_qg.is_parallel
        println("  Running in parallel mode with $(tracker_qg.nprocs) processes")
        println("  This is rank $(tracker_qg.rank)")
    else
        println("  Running in serial mode")
    end
    
    # 4. Create output files
    create_particle_output_file("particles_qg.nc", tracker_qg)
    create_particle_output_file("particles_ybj.nc", tracker_ybj)
    create_particle_output_file("particles_2d.nc", tracker_2d)
    
    # 5. Run simulation with particle advection
    println("Running simulation with particle advection...")
    
    nsteps = Int(sim.config.total_time / sim.config.dt)
    output_every = 10
    
    for step in 1:nsteps
        current_time = step * sim.config.dt
        
        # Advance the flow (time step the QG-YBJ model)
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        # Unified particle advection (automatically handles velocities and parallel migration)
        
        # Advect QG particles (automatically computes QG vertical velocity)
        advect_particles!(tracker_qg, sim.state, sim.grid, sim.config.dt)
        
        # Advect YBJ particles (automatically computes YBJ vertical velocity)
        advect_particles!(tracker_ybj, sim.state, sim.grid, sim.config.dt)
        
        # Advect 2D particles (automatically sets w=0 for 2D case)
        advect_particles!(tracker_2d, sim.state, sim.grid, sim.config.dt)
        
        # Output particle positions
        if step % output_every == 0
            println("  Step $step/$nsteps (t=$(current_time))")
            
            # Save particle snapshots
            write_particle_snapshot("snapshot_qg_$(step).nc", tracker_qg, current_time)
            write_particle_snapshot("snapshot_ybj_$(step).nc", tracker_ybj, current_time)
            write_particle_snapshot("snapshot_2d_$(step).nc", tracker_2d, current_time)
            
            # Compute some diagnostics
            qg_spread = compute_particle_spread(tracker_qg)
            ybj_spread = compute_particle_spread(tracker_ybj)
            twod_spread = compute_particle_spread(tracker_2d)
            
            println("    Particle spread - QG: $(qg_spread), YBJ: $(ybj_spread), 2D: $(twod_spread)")
        end
    end
    
    # 6. Save complete trajectories
    println("Saving complete particle trajectories...")
    
    write_particle_trajectories("trajectories_qg.nc", tracker_qg, 
                                metadata=Dict("vertical_velocity" => "QG omega equation"))
    write_particle_trajectories("trajectories_ybj.nc", tracker_ybj,
                                metadata=Dict("vertical_velocity" => "YBJ formulation"))
    write_particle_trajectories("trajectories_2d.nc", tracker_2d,
                                metadata=Dict("vertical_velocity" => "none (2D advection)"))
    
    # 7. Analysis and comparison
    println("Particle trajectory analysis:")
    
    # Compare final positions
    final_qg = (x=tracker_qg.particles.x, y=tracker_qg.particles.y, z=tracker_qg.particles.z)
    final_ybj = (x=tracker_ybj.particles.x, y=tracker_ybj.particles.y, z=tracker_ybj.particles.z)
    final_2d = (x=tracker_2d.particles.x, y=tracker_2d.particles.y, z=tracker_2d.particles.z)
    
    # Compute displacement statistics
    qg_disp = sqrt.(final_qg.x.^2 + final_qg.y.^2 + final_qg.z.^2)
    ybj_disp = sqrt.(final_ybj.x.^2 + final_ybj.y.^2 + final_ybj.z.^2)
    twod_disp = sqrt.(final_2d.x.^2 + final_2d.y.^2 + final_2d.z.^2)
    
    println("  Mean displacement - QG: $(mean(qg_disp)), YBJ: $(mean(ybj_disp)), 2D: $(mean(twod_disp))")
    println("  Max displacement - QG: $(maximum(qg_disp)), YBJ: $(maximum(ybj_disp)), 2D: $(maximum(twod_disp))")
    
    # Vertical displacement comparison
    qg_z_disp = abs.(final_qg.z .- z_level)
    ybj_z_disp = abs.(final_ybj.z .- z_level)
    
    println("  Vertical displacement - QG: $(mean(qg_z_disp)) ± $(std(qg_z_disp))")
    println("  Vertical displacement - YBJ: $(mean(ybj_z_disp)) ± $(std(ybj_z_disp))")
    
    println("\\n✅ Particle advection example completed!")
    println("\\nOutput files:")
    println("  - trajectories_qg.nc: Complete QG particle trajectories")
    println("  - trajectories_ybj.nc: Complete YBJ particle trajectories") 
    println("  - trajectories_2d.nc: Complete 2D particle trajectories")
    println("  - snapshot_*_*.nc: Particle snapshots at regular intervals")
    
    return (tracker_qg, tracker_ybj, tracker_2d)
end

"""
    compute_particle_spread(tracker)

Compute the RMS spread of particles from their initial positions.
"""
function compute_particle_spread(tracker::ParticleTracker)
    if length(tracker.particles.x_history) < 2
        return 0.0
    end
    
    # Initial positions
    x0 = tracker.particles.x_history[1]
    y0 = tracker.particles.y_history[1]
    z0 = tracker.particles.z_history[1]
    
    # Current positions
    x = tracker.particles.x
    y = tracker.particles.y
    z = tracker.particles.z
    
    # Displacement
    dx = x - x0
    dy = y - y0  
    dz = z - z0
    
    # RMS spread
    spread = sqrt(mean(dx.^2 + dy.^2 + dz.^2))
    
    return spread
end

function simple_particle_test()
    println("Simple Particle Advection Test")
    println("==============================")
    
    # Minimal setup for quick testing
    domain = create_domain_config(nx=32, ny=32, nz=8)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical, wave_type=:analytical
    )
    output = create_output_config(output_dir="./simple_test")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1, dt=1e-3
    )
    
    sim = setup_simulation(config)
    
    # Simple particle setup
    particle_config = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2, 
        z_level=π/4,
        nx_particles=4, ny_particles=4,
        use_ybj_w=true,
        use_3d_advection=true
    )
    
    tracker = ParticleTracker(particle_config, sim.grid)
    initialize_particles!(tracker, particle_config)
    
    println("Initialized $(tracker.particles.np) particles")
    
    # Run a few steps
    for step in 1:10
        if step == 1
            first_projection_step!(sim.state, sim.grid, sim.params, sim.plans)
        else
            leapfrog_step!(sim.state, sim.grid, sim.params, sim.plans)
        end
        
        compute_velocities!(sim.state, sim.grid; 
                           plans=sim.plans, 
                           params=sim.params, 
                           compute_w=true, 
                           use_ybj_w=true)
        
        advect_particles!(tracker, sim.state, sim.grid, sim.config.dt)
        
        if step % 5 == 0
            spread = compute_particle_spread(tracker)
            println("  Step $step: particle spread = $spread")
        end
    end
    
    write_particle_trajectories("simple_test_particles.nc", tracker)
    
    println("✅ Simple particle test completed!")
    
    return tracker
end

# Run the example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Uncomment the example you want to run:
    
    # Full comprehensive example (recommended)
    particle_advection_example()
    
    # Or simple quick test
    # simple_particle_test()
end