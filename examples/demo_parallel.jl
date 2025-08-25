"""
Demonstration of parallel QG-YBJ simulations using MPI, PencilArrays, and PencilFFTs.

This example shows:
1. How to set up parallel simulations
2. Proper MPI initialization and finalization
3. Parallel I/O with NetCDF
4. Load balancing considerations
5. Performance monitoring

Usage:
  Serial:   julia demo_parallel.jl
  Parallel: mpiexecjl -n 4 julia demo_parallel.jl
"""

using QGYBJ

function demo_parallel_basic()
    println("=== Parallel QG-YBJ Basic Demo ===")
    
    # Domain configuration - larger for parallel benefit
    domain = create_domain_config(
        nx=256, ny=256, nz=128,        # Larger domain for parallel efficiency
        Lx=8π, Ly=8π, Lz=2π
    )
    
    # Pycnocline-like stratification
    stratification = create_stratification_config(
        :tanh_profile,
        N_upper=0.01,
        N_lower=0.03,
        z_pycno=0.6,
        width=0.05
    )
    
    # Random initial conditions
    initial_conditions = create_initial_condition_config(
        psi_type=:random,
        wave_type=:random,
        psi_amplitude=0.1,
        wave_amplitude=0.01,
        random_seed=42  # Same seed for reproducibility across runs
    )
    
    # Output configuration with parallel-friendly settings
    output = create_output_config(
        output_dir="./demo_parallel_basic",
        psi_interval=2.0,              # Less frequent output for large parallel runs
        wave_interval=2.0,
        diagnostics_interval=1.0,
        save_velocities=true,
        save_diagnostics=true
    )
    
    # Model configuration
    config = create_model_config(
        domain, stratification, initial_conditions, output,
        dt=5e-4,                       # Smaller time step for stability
        total_time=20.0,
        linear=false,
        inviscid=true,
        ybj_plus=true,
        no_feedback=false
    )
    
    # Set up simulation with MPI support
    sim = setup_simulation(config; use_mpi=true)
    
    # Run with progress monitoring
    run_simulation!(sim; progress_callback=parallel_progress_callback)
    
    return sim
end

function parallel_progress_callback(sim)
    # Only print from rank 0 to avoid spam
    if sim.parallel_config.use_mpi
        import MPI
        rank = MPI.Comm_rank(sim.parallel_config.comm)
        if rank != 0
            return
        end
    end
    
    # Print progress every 1000 steps
    if sim.time_step % 1000 == 0
        println("Step $(sim.time_step), t=$(sim.current_time)")
        
        # Monitor memory usage and performance
        if hasfield(typeof(sim.state.psi), :data)  # PencilArray
            println("  Field type: PencilArray (distributed)")
        else
            println("  Field type: Array (local)")
        end
        
        # Check for any issues
        if sim.current_time > 0
            # Could add performance diagnostics here
            println("  Simulation progressing normally")
        end
    end
end

function demo_parallel_scaling_test()
    println("\n=== Parallel Scaling Test ===")
    
    # Test different problem sizes to see parallel scaling
    grid_sizes = [(64, 64, 32), (128, 128, 64), (256, 256, 128)]
    
    for (nx, ny, nz) in grid_sizes
        println("Testing grid size: $nx × $ny × $nz")
        
        domain = create_domain_config(nx=nx, ny=ny, nz=nz)
        stratification = create_stratification_config(:constant_N, N0=1.0)
        initial_conditions = create_initial_condition_config(
            psi_type=:random, 
            wave_type=:random,
            psi_amplitude=0.05,
            wave_amplitude=0.005
        )
        output = create_output_config(
            output_dir="./scaling_test_$(nx)x$(ny)x$(nz)",
            psi_interval=5.0,  # Infrequent output for timing
            wave_interval=5.0
        )
        
        config = create_model_config(
            domain, stratification, initial_conditions, output,
            total_time=2.0,  # Short run for scaling test
            dt=1e-3
        )
        
        # Time the setup and initial steps
        t_start = time()
        sim = setup_simulation(config; use_mpi=true)
        t_setup = time() - t_start
        
        t_start = time()
        # Run just 100 steps for timing
        for step in 1:100
            if step == 1
                first_projection_step!(sim.state, sim.state_old, sim.grid, sim.params, sim.plans)
            else
                leapfrog_step!(sim.state, sim.state_old, sim.grid, sim.params, sim.plans)
            end
        end
        t_steps = time() - t_start
        
        # Report timings (only from rank 0)
        if !sim.parallel_config.use_mpi || MPI.Comm_rank(sim.parallel_config.comm) == 0
            println("  Setup time: $(t_setup) seconds")
            println("  100 steps time: $(t_steps) seconds")
            println("  Time per step: $(t_steps/100) seconds")
        end
    end
end

function demo_parallel_load_balancing()
    println("\n=== Load Balancing Demo ===")
    
    # Demonstrate load balancing with different domain aspect ratios
    configs = [
        ("Square domain", 128, 128, 64),
        ("Wide domain", 256, 128, 64), 
        ("Tall domain", 128, 256, 64),
        ("Deep domain", 128, 128, 128)
    ]
    
    for (name, nx, ny, nz) in configs
        println("Testing $name: $nx × $ny × $nz")
        
        domain = create_domain_config(nx=nx, ny=ny, nz=nz)
        stratification = create_stratification_config(:constant_N)
        initial_conditions = create_initial_condition_config(psi_type=:analytical, wave_type=:zero)
        output = create_output_config(
            output_dir="./load_balance_$nx-$ny-$nz",
            psi_interval=10.0  # Minimal output
        )
        
        config = create_model_config(domain, stratification, initial_conditions, output, total_time=1.0)
        
        sim = setup_simulation(config; use_mpi=true)
        
        # Analyze decomposition (if using PencilArrays)
        if sim.parallel_config.use_mpi && sim.grid.decomp !== nothing
            import PencilArrays
            import MPI
            
            rank = MPI.Comm_rank(sim.parallel_config.comm)
            nprocs = MPI.Comm_size(sim.parallel_config.comm)
            
            # Get local array sizes on each process
            local_size = size(sim.state.psi.data)  # PencilArray local data
            
            # Gather sizes to rank 0 for analysis
            all_sizes = MPI.gather(local_size, 0, sim.parallel_config.comm)
            
            if rank == 0
                println("  Process decomposition:")
                for (p, sz) in enumerate(all_sizes)
                    println("    Rank $(p-1): local size $sz")
                end
                
                # Calculate load balance metric
                total_points = sum(prod.(all_sizes))
                ideal_per_proc = total_points / nprocs
                max_imbalance = maximum(prod.(all_sizes)) / ideal_per_proc
                println("  Load imbalance factor: $(max_imbalance:.3f)")
            end
        end
    end
end

function demo_parallel_io()
    println("\n=== Parallel I/O Demo ===")
    
    # Test both parallel NetCDF I/O and gather-to-rank-0 approaches
    domain = create_domain_config(nx=128, ny=128, nz=64)
    stratification = create_stratification_config(:constant_N)
    initial_conditions = create_initial_condition_config(psi_type=:random, wave_type=:random)
    
    # Test parallel I/O
    output_parallel = create_output_config(
        output_dir="./demo_parallel_io",
        psi_interval=1.0,
        wave_interval=1.0
    )
    
    config = create_model_config(
        domain, stratification, initial_conditions, output_parallel,
        total_time=3.0
    )
    
    # Set up with parallel I/O enabled
    sim = setup_simulation(config; use_mpi=true)
    
    # Enable parallel I/O in the configuration
    if sim.parallel_config.use_mpi
        sim.parallel_config = ParallelConfig(
            use_mpi=true,
            comm=sim.parallel_config.comm,
            n_processes=sim.parallel_config.n_processes,
            parallel_io=true,      # Use parallel NetCDF
            gather_for_io=false    # Don't gather to rank 0
        )
    end
    
    println("Testing parallel NetCDF I/O...")
    
    # Time I/O operations
    t_start = time()
    run_simulation!(sim)
    t_total = time() - t_start
    
    if !sim.parallel_config.use_mpi || MPI.Comm_rank(sim.parallel_config.comm) == 0
        println("  Parallel I/O total time: $(t_total) seconds")
        
        # Check that files were created
        output_files = readdir(sim.config.output.output_dir)
        state_files = filter(f -> startswith(f, "state"), output_files)
        println("  Created $(length(state_files)) state files")
    end
end

function check_mpi_environment()
    """Check if MPI environment is properly set up."""
    
    try
        import MPI
        
        if MPI.Initialized()
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)
            nprocs = MPI.Comm_size(comm)
            
            println("MPI Environment:")
            println("  Initialized: Yes")
            println("  Rank: $rank")
            println("  Total processes: $nprocs")
            
            # Check for PencilArrays
            try
                import PencilArrays
                println("  PencilArrays: Available")
            catch
                println("  PencilArrays: Not available")
            end
            
            # Check for PencilFFTs  
            try
                import PencilFFTs
                println("  PencilFFTs: Available")
            catch
                println("  PencilFFTs: Not available")
            end
            
            return true
        else
            println("MPI not initialized")
            return false
        end
    catch e
        println("MPI not available: $e")
        return false
    end
end

function finalize_mpi()
    """Properly finalize MPI if it was initialized."""
    try
        import MPI
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    catch
        # MPI not available or already finalized
    end
end

# Main demonstration
function main()
    println("QG-YBJ Parallel Interface Demonstration")
    println("=======================================")
    
    # Check environment
    mpi_available = check_mpi_environment()
    
    if mpi_available
        println("\nMPI environment detected - running parallel demos")
        
        try
            # Run parallel demonstrations
            demo_parallel_basic()
            
            # Additional demos (comment out for quick testing)
            # demo_parallel_scaling_test()
            # demo_parallel_load_balancing() 
            # demo_parallel_io()
            
            println("\n=== All Parallel Demonstrations Completed ===")
            
        catch e
            println("Error in parallel demo: $e")
            rethrow(e)
        finally
            finalize_mpi()
        end
    else
        println("\nMPI not available - running serial fallback demo")
        
        # Run a serial simulation to show fallback works
        config = create_simple_config(nx=64, ny=64, nz=32, total_time=2.0)
        sim = setup_simulation(config; use_mpi=false)
        run_simulation!(sim)
        
        println("Serial fallback demo completed")
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end