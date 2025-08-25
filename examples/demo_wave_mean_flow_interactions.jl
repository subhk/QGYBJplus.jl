"""
Demonstration of wave-mean flow interaction controls in QG-YBJ model.

This example shows how to use the new flags:
1. no_wave_feedback: Control whether waves affect the mean flow
2. fixed_mean_flow: Control whether the mean flow evolves in time

These controls enable different experimental setups:
- Pure wave dynamics with fixed background flow
- Mean flow evolution without wave feedback
- Full coupled wave-mean flow interactions
- Wave-only simulations (passive wave dynamics)
"""

using QGYBJ

function demo_coupled_wave_mean_flow()
    println("=== Full Coupled Wave-Mean Flow Interaction ===")
    
    # Standard setup with full coupling
    domain = create_domain_config(nx=128, ny=128, nz=64, Lx=4π, Ly=4π, Lz=2π)
    
    stratification = create_stratification_config(
        :tanh_profile,
        N_upper=0.01,
        N_lower=0.03,
        z_pycno=0.5,
        width=0.1
    )
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,    # Strong mean flow
        wave_amplitude=0.05,  # Moderate wave amplitude
        random_seed=1001
    )
    
    output = create_output_config(
        output_dir="./demo_coupled",
        psi_interval=1.0,
        wave_interval=1.0,
        diagnostics_interval=0.5
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=5e-4,
        total_time=20.0,
        
        # Full coupling - both flags false
        no_wave_feedback=false,   # Waves DO affect mean flow
        fixed_mean_flow=false     # Mean flow DOES evolve
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim; progress_callback=interaction_progress_callback)
    
    println("Coupled simulation completed - waves and mean flow interact")
    return sim
end

function demo_fixed_mean_flow()
    println("\n=== Wave Dynamics with Fixed Mean Flow ===")
    
    # Wave evolution in a prescribed, non-evolving mean flow
    domain = create_domain_config(nx=128, ny=128, nz=64, Lx=4π, Ly=4π, Lz=2π)
    
    stratification = create_stratification_config(
        :constant_N,
        N0=1.5
    )
    
    # Strong prescribed mean flow, small wave perturbations
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,     # Prescribed mean flow pattern
        wave_type=:random,        # Random wave field
        psi_amplitude=0.2,        # Strong mean flow
        wave_amplitude=0.01,      # Small wave amplitude
        random_seed=1002
    )
    
    output = create_output_config(
        output_dir="./demo_fixed_flow",
        psi_interval=2.0,
        wave_interval=0.5,        # More frequent wave output
        diagnostics_interval=0.2
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=1e-3,
        total_time=15.0,
        
        # Fixed mean flow setup
        no_wave_feedback=true,    # Waves do NOT affect mean flow
        fixed_mean_flow=true      # Mean flow does NOT evolve
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim; progress_callback=interaction_progress_callback)
    
    println("Fixed flow simulation completed - mean flow remained constant")
    return sim
end

function demo_no_wave_feedback()
    println("\n=== Mean Flow Evolution without Wave Feedback ===")
    
    # Mean flow evolves due to its own nonlinear dynamics, waves evolve separately
    domain = create_domain_config(nx=96, ny=96, nz=48, Lx=6π, Ly=6π, Lz=2π)
    
    stratification = create_stratification_config(
        :skewed_gaussian  # Complex stratification for interesting dynamics
    )
    
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:analytical,    # Prescribed wave pattern
        psi_amplitude=0.15,
        wave_amplitude=0.03,
        random_seed=1003
    )
    
    output = create_output_config(
        output_dir="./demo_no_feedback",
        psi_interval=1.0,
        wave_interval=1.0,
        diagnostics_interval=0.5
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=8e-4,
        total_time=25.0,
        
        # No wave feedback but mean flow evolves
        no_wave_feedback=true,    # Waves do NOT affect mean flow
        fixed_mean_flow=false     # Mean flow DOES evolve (from its own dynamics)
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim; progress_callback=interaction_progress_callback)
    
    println("No feedback simulation completed - mean flow evolved independently")
    return sim
end

function demo_wave_only_dynamics()
    println("\n=== Pure Wave Dynamics (Passive Waves) ===")
    
    # Waves evolve in a completely static background
    domain = create_domain_config(nx=64, ny=64, nz=32, Lx=2π, Ly=2π, Lz=π)
    
    stratification = create_stratification_config(
        :constant_N,
        N0=2.0  # Strong stratification
    )
    
    # Zero mean flow, waves only
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,     # Could be zero or simple pattern
        wave_type=:random,
        psi_amplitude=0.0,        # No mean flow
        wave_amplitude=0.1,       # Wave field only
        random_seed=1004
    )
    
    output = create_output_config(
        output_dir="./demo_wave_only",
        psi_interval=5.0,         # Infrequent psi output (should be constant)
        wave_interval=0.5,        # Frequent wave output
        diagnostics_interval=0.1
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=1e-3,
        total_time=10.0,
        
        # Wave-only setup
        no_wave_feedback=true,    # Waves do NOT affect mean flow
        fixed_mean_flow=true,     # Mean flow does NOT evolve
        linear=false              # Keep nonlinear wave dynamics
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim; progress_callback=interaction_progress_callback)
    
    println("Wave-only simulation completed - pure wave dynamics")
    return sim
end

function demo_linear_wave_evolution()
    println("\n=== Linear Wave Evolution in Fixed Flow ===")
    
    # Linear wave equation in prescribed mean flow
    domain = create_domain_config(nx=64, ny=64, nz=32, Lx=4π, Ly=4π, Lz=2π)
    
    stratification = create_stratification_config(
        :tanh_profile,
        N_upper=0.005,
        N_lower=0.025,
        z_pycno=0.4,
        width=0.08
    )
    
    # Prescribed jet-like mean flow
    initial_conditions = create_initial_condition_config(
        psi_type=:analytical,
        wave_type=:analytical,    # Single wave mode or wave packet
        psi_amplitude=0.3,        # Strong jet
        wave_amplitude=0.005,     # Small amplitude waves
        random_seed=1005
    )
    
    output = create_output_config(
        output_dir="./demo_linear_waves",
        psi_interval=10.0,        # Mean flow doesn't change
        wave_interval=0.25,       # High frequency wave output
        diagnostics_interval=0.1,
        save_velocities=true      # Save velocity fields for analysis
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=5e-4,
        total_time=12.0,
        
        # Linear wave dynamics in fixed flow
        no_wave_feedback=true,    # Waves do NOT affect mean flow
        fixed_mean_flow=true,     # Mean flow does NOT evolve
        linear=true               # Linear wave dynamics only
    )
    
    sim = setup_simulation(config)
    run_simulation!(sim; progress_callback=interaction_progress_callback)
    
    println("Linear wave simulation completed - linear dynamics in fixed flow")
    return sim
end

function interaction_progress_callback(sim)
    """Custom progress callback showing wave-mean flow interaction diagnostics."""
    
    # Only print from rank 0 in parallel runs
    if sim.parallel_config.use_mpi
        import MPI
        rank = MPI.Comm_rank(sim.parallel_config.comm)
        if rank != 0
            return
        end
    end
    
    # Print progress every 500 steps
    if sim.time_step % 500 == 0
        println("Step $(sim.time_step), t=$(sim.current_time)")
        
        # Show configuration flags
        config_info = []
        if sim.params.fixed_flow
            push!(config_info, "fixed_flow")
        end
        if sim.params.no_wave_feedback
            push!(config_info, "no_wave_feedback")
        end
        if sim.params.linear
            push!(config_info, "linear")
        end
        
        if !isempty(config_info)
            println("  Flags: $(join(config_info, ", "))")
        else
            println("  Flags: full_coupling")
        end
        
        # Could add energy diagnostics here
        if sim.time_step > 0
            println("  Simulation progressing normally")
        end
    end
end

function demo_parameter_comparison()
    println("\n=== Parameter Comparison Study ===")
    
    # Run the same initial condition with different interaction settings
    base_config = Dict(
        :domain => create_domain_config(nx=64, ny=64, nz=32),
        :stratification => create_stratification_config(:constant_N, N0=1.0),
        :initial_conditions => create_initial_condition_config(
            psi_type=:random, wave_type=:random,
            psi_amplitude=0.1, wave_amplitude=0.02, random_seed=2000
        ),
        :Ro => 0.1, :Fr => 0.1, :dt => 1e-3, :total_time => 5.0
    )
    
    # Different configurations to compare
    configs = [
        ("coupled", false, false, "Full coupling"),
        ("no_feedback", true, false, "No wave feedback"),
        ("fixed_flow", true, true, "Fixed mean flow"),
        ("linear_fixed", true, true, "Linear + fixed flow")
    ]
    
    results = Dict()
    
    for (name, no_feedback, fixed_flow, description) in configs
        println("  Running: $description")
        
        output = create_output_config(
            output_dir="./comparison_$name",
            psi_interval=1.0,
            wave_interval=1.0
        )
        
        config = create_model_config(
            base_config[:domain],
            base_config[:stratification], 
            base_config[:initial_conditions],
            output;
            base_config[:Ro], base_config[:Fr], 
            base_config[:dt], base_config[:total_time],
            no_wave_feedback=no_feedback,
            fixed_mean_flow=fixed_flow,
            linear=(name == "linear_fixed")
        )
        
        sim = setup_simulation(config)
        
        # Run short simulation for comparison
        run_simulation!(sim)
        
        # Store results for comparison
        results[name] = (
            description=description,
            final_time=sim.current_time,
            flags=(no_feedback, fixed_flow)
        )
    end
    
    println("\nComparison study completed:")
    for (name, result) in results
        println("  $(result.description): t_final = $(result.final_time)")
    end
    
    return results
end

function demo_legacy_compatibility()
    println("\n=== Legacy Compatibility Test ===")
    
    # Test backward compatibility with old no_feedback flag
    domain = create_domain_config(nx=32, ny=32, nz=16)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    output = create_output_config(output_dir="./demo_legacy")
    
    # Using old no_feedback flag (should still work)
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        total_time=2.0,
        no_feedback=true  # Legacy flag - should map to no_wave_feedback
    )
    
    sim = setup_simulation(config)
    
    # Check that legacy flag is properly handled
    if sim.params.no_wave_feedback && sim.params.no_feedback
        println("  Legacy no_feedback flag properly mapped to no_wave_feedback")
    else
        println("  Warning: Legacy flag mapping may have issues")
    end
    
    run_simulation!(sim)
    println("Legacy compatibility test completed")
    
    return sim
end

# Main demonstration function
function main()
    println("QG-YBJ Wave-Mean Flow Interaction Controls Demonstration")
    println("======================================================")
    
    try
        # Full demonstration of different interaction modes
        sim1 = demo_coupled_wave_mean_flow()
        sim2 = demo_fixed_mean_flow()
        sim3 = demo_no_wave_feedback()
        sim4 = demo_wave_only_dynamics()
        sim5 = demo_linear_wave_evolution()
        
        # Comparison study
        comparison_results = demo_parameter_comparison()
        
        # Legacy compatibility
        legacy_sim = demo_legacy_compatibility()
        
        println("\n=== All Wave-Mean Flow Demonstrations Completed ===")
        println("\nKey Results:")
        println("1. Coupled: Full wave-mean flow interaction")
        println("2. Fixed flow: Waves evolve in static background")
        println("3. No feedback: Independent evolution of waves and mean flow")
        println("4. Wave-only: Pure wave dynamics")
        println("5. Linear waves: Linear wave equation in prescribed flow")
        println("\nOutput directories contain NetCDF files for analysis")
        
    catch e
        println("Error during demonstration: $e")
        rethrow(e)
    end
end

# Run the demonstration if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end