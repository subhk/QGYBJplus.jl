"""
Test script to verify that particles are advected by total velocity (QG + wave).

This script demonstrates the difference between QG-only and total velocity advection.
"""

using QGYBJ

function test_total_velocity()
    println("Testing Total Velocity Computation for Particles")
    println("===============================================")
    
    # Create a simple simulation with wave activity
    domain = create_domain_config(nx=32, ny=32, nz=8, Lx=2π, Ly=2π, Lz=π)
    stratification = create_stratification_config(:constant_N, N0=1.0)
    
    # Include both QG and wave initial conditions
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,    # Moderate QG flow
        wave_amplitude=0.05,  # Moderate wave amplitude
        random_seed=42
    )
    
    output = create_output_config(output_dir="./test_total_velocity")
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=0.1,  # Short test
        dt=1e-3
    )
    
    sim = setup_simulation(config)
    
    # Test QG-only velocity computation
    println("\\n1. Testing QG-only velocity computation...")
    compute_velocities!(sim.state, sim.grid; plans=sim.plans, compute_w=true)
    
    u_qg = copy(sim.state.u)
    v_qg = copy(sim.state.v)
    w_qg = copy(sim.state.w)
    
    println("   QG velocities computed: u_rms = $(sqrt(sum(abs2, u_qg)/length(u_qg)))")
    println("   QG velocities computed: v_rms = $(sqrt(sum(abs2, v_qg)/length(v_qg)))")
    println("   QG velocities computed: w_rms = $(sqrt(sum(abs2, w_qg)/length(w_qg)))")
    
    # Test total velocity computation  
    println("\\n2. Testing TOTAL velocity computation (QG + wave)...")
    compute_total_velocities!(sim.state, sim.grid; plans=sim.plans, compute_w=true)
    
    u_total = copy(sim.state.u)
    v_total = copy(sim.state.v)
    w_total = copy(sim.state.w)
    
    println("   Total velocities computed: u_rms = $(sqrt(sum(abs2, u_total)/length(u_total)))")
    println("   Total velocities computed: v_rms = $(sqrt(sum(abs2, v_total)/length(v_total)))")
    println("   Total velocities computed: w_rms = $(sqrt(sum(abs2, w_total)/length(w_total)))")
    
    # Compute wave velocity contributions
    u_wave = u_total .- u_qg
    v_wave = v_total .- v_qg
    
    println("\\n3. Wave velocity contributions:")
    println("   Wave u_rms = $(sqrt(sum(abs2, u_wave)/length(u_wave)))")
    println("   Wave v_rms = $(sqrt(sum(abs2, v_wave)/length(v_wave)))")
    
    # Check if wave velocities are non-zero (indicating they're being computed)
    wave_u_nonzero = sum(abs.(u_wave) .> 1e-12) > 0
    wave_v_nonzero = sum(abs.(v_wave) .> 1e-12) > 0
    
    if wave_u_nonzero || wave_v_nonzero
        println("   ✓ Wave velocities are non-zero - total velocity computation working!")
    else
        println("   ⚠ Wave velocities are zero - check wave amplitude A field")
    end
    
    # Test with particle advection
    println("\\n4. Testing particle advection with total velocity...")
    
    particle_config = create_particle_config(
        x_min=π/2, x_max=3π/2,
        y_min=π/2, y_max=3π/2,
        z_level=π/2,
        nx_particles=4, ny_particles=4,
        use_ybj_w=false,
        use_3d_advection=true
    )
    
    tracker = ParticleTracker(particle_config, sim.grid, sim.parallel_config)
    initialize_particles!(tracker, particle_config)
    
    # Store initial positions
    x0 = copy(tracker.particles.x)
    y0 = copy(tracker.particles.y)
    
    # Update velocity fields (this uses compute_total_velocities!)
    update_velocity_fields!(tracker, sim.state, sim.grid)
    
    # Test one step of advection with simulation time
    test_time = 0.01
    advect_particles!(tracker, sim.state, sim.grid, test_time, test_time)
    
    println("   Particle tracker initialized with $(tracker.particles.np) particles")
    println("   Velocity fields updated with total velocities")
    
    # Test velocity interpolation at a particle position
    if tracker.particles.np > 0
        x_test = tracker.particles.x[1]
        y_test = tracker.particles.y[1]  
        z_test = tracker.particles.z[1]
        
        u_interp, v_interp, w_interp = interpolate_velocity_at_position(x_test, y_test, z_test, tracker)
        
        println("   Sample interpolated velocity at ($(round(x_test,digits=2)), $(round(y_test,digits=2)), $(round(z_test,digits=2))):")
        println("     u = $(round(u_interp, digits=4))")
        println("     v = $(round(v_interp, digits=4))")
        println("     w = $(round(w_interp, digits=4))")
    end
    
    println("\\n✅ Total velocity computation test completed!")
    println("\\nParticles will now be advected by:")
    println("  • QG velocity: u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x")
    println("  • Wave velocity: u_wave, v_wave from Stokes drift")  
    println("  • Total: u_total = u_QG + u_wave, v_total = v_QG + v_wave")
    println("  • Vertical: w from QG omega equation or YBJ formulation")
    
    return true
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_total_velocity()
end