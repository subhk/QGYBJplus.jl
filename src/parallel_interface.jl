"""
Enhanced parallel interface for QG-YBJ model using PencilArrays and PencilFFTs.

This module fixes several issues with the current parallelization setup:
1. Proper initialization of MPI and PencilArrays decomposition
2. Correct handling of distributed arrays in I/O operations
3. Integration with the new user interface system
4. Parallel-aware initialization and state management
"""

using ..QGYBJ: Grid, State, QGParams

"""
    ParallelConfig

Configuration for parallel execution.
"""
Base.@kwdef struct ParallelConfig
    use_mpi::Bool = false
    comm = nothing  # Will be set to MPI.COMM_WORLD if MPI is available
    n_processes::Int = 1
    
    # Pencil decomposition configuration
    pencil_dims::Tuple{Int,Int} = (0, 0)  # Auto-determine if (0,0)
    
    # I/O configuration for parallel runs
    parallel_io::Bool = true  # Use parallel NetCDF I/O
    gather_for_io::Bool = false  # Gather all data to rank 0 for I/O
end

"""
    setup_parallel_environment()

Initialize MPI and return parallel configuration.
"""
function setup_parallel_environment()
    config = ParallelConfig()
    
    try
        import MPI
        
        # Check if MPI is already initialized
        if !MPI.Initialized()
            MPI.Init()
        end
        
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        if rank == 0
            @info "MPI initialized with $nprocs processes"
        end
        
        config = ParallelConfig(
            use_mpi=true,
            comm=comm,
            n_processes=nprocs,
            parallel_io=true
        )
        
    catch e
        @info "MPI not available or failed to initialize: $e"
        config = ParallelConfig(use_mpi=false)
    end
    
    return config
end

"""
    init_parallel_grid(params::QGParams, pconfig::ParallelConfig)

Initialize grid with proper parallel decomposition.
"""
function init_parallel_grid(params::QGParams, pconfig::ParallelConfig)
    T = Float64
    nx, ny, nz = params.nx, params.ny, params.nz
    dx = params.Lx / nx
    dy = params.Ly / ny
    z = T.(collect(range(0, 2π; length=nz)))
    dz = diff(z)

    # Wavenumbers
    kx = T.([i <= nx÷2 ? (2π/params.Lx)*(i-1) : (2π/params.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/params.Ly)*(j-1) : (2π/params.Ly)*(j-1-ny) for j in 1:ny])

    # Initialize decomposition
    decomp = nothing
    kh2 = Array{T}(undef, nx, ny)
    
    if pconfig.use_mpi
        try
            import PencilArrays
            
            # Create pencil decomposition
            # For 3D (x,y,z) data, typically decompose in y and z dimensions
            decomp = PencilArrays.Pencil((nx, ny, nz), pconfig.comm)
            
            # Create distributed kh2 array
            kh2_pencil = PencilArrays.allocate_array(decomp, T)
            
            # Compute kh2 on each process's local portion
            local_indices = PencilArrays.range_local(decomp)
            for j in local_indices[2], i in local_indices[1]
                kh2_pencil[i,j,:] .= kx[i]^2 + ky[j]^2
            end
            
            kh2 = kh2_pencil
            
        catch e
            @warn "Failed to initialize PencilArrays decomposition: $e"
            # Fall back to serial
            @inbounds for j in 1:ny, i in 1:nx
                kh2[i,j] = kx[i]^2 + ky[j]^2
            end
        end
    else
        # Serial computation
        @inbounds for j in 1:ny, i in 1:nx
            kh2[i,j] = kx[i]^2 + ky[j]^2
        end
    end

    return Grid{T, typeof(kh2)}(nx, ny, nz, params.Lx, params.Ly, dx, dy, z, dz, kx, ky, kh2, decomp)
end

"""
    init_parallel_state(grid::Grid, pconfig::ParallelConfig; T=Float64)

Initialize state with distributed arrays when using MPI.
"""
function init_parallel_state(grid::Grid, pconfig::ParallelConfig; T=Float64)
    if grid.decomp !== nothing
        # Use PencilArrays for distributed storage
        import PencilArrays
        
        # Spectral fields (complex)
        q   = PencilArrays.allocate_array(grid.decomp, Complex{T}); fill!(q, 0)
        psi = PencilArrays.allocate_array(grid.decomp, Complex{T}); fill!(psi, 0)
        A   = PencilArrays.allocate_array(grid.decomp, Complex{T}); fill!(A, 0)
        B   = PencilArrays.allocate_array(grid.decomp, Complex{T}); fill!(B, 0)
        C   = PencilArrays.allocate_array(grid.decomp, Complex{T}); fill!(C, 0)
        
        # Real space fields
        u = PencilArrays.allocate_array(grid.decomp, T); fill!(u, 0)
        v = PencilArrays.allocate_array(grid.decomp, T); fill!(v, 0)
        w = PencilArrays.allocate_array(grid.decomp, T); fill!(w, 0)
        
    else
        # Serial arrays
        sz = (grid.nx, grid.ny, grid.nz)
        q   = Array{Complex{T}}(undef, sz); fill!(q, 0)
        psi = Array{Complex{T}}(undef, sz); fill!(psi, 0)
        A   = Array{Complex{T}}(undef, sz); fill!(A, 0)
        B   = Array{Complex{T}}(undef, sz); fill!(B, 0)
        C   = Array{Complex{T}}(undef, sz); fill!(C, 0)
        u   = Array{T}(undef, sz); fill!(u, 0)
        v   = Array{T}(undef, sz); fill!(v, 0)
        w   = Array{T}(undef, sz); fill!(w, 0)
    end
    
    return State{T, typeof(u), typeof(q)}(q, psi, A, B, C, u, v, w)
end


"""
    gather_array_for_io(arr, grid::Grid, pconfig::ParallelConfig)

Gather distributed array to rank 0 for I/O operations.
"""
function gather_array_for_io(arr, grid::Grid, pconfig::ParallelConfig)
    if grid.decomp === nothing || !pconfig.use_mpi
        return arr  # Already local
    end
    
    try
        import PencilArrays
        import MPI
        
        # Gather to a global array on rank 0
        gathered = PencilArrays.gather(arr)
        
        return gathered
    catch e
        @warn "Failed to gather array: $e"
        return arr
    end
end

# Parallel I/O functions moved to netcdf_io.jl for unified interface

"""
    parallel_initialization_from_config(config, pconfig)

Initialize model with parallel support from configuration.
"""
function parallel_initialization_from_config(config, pconfig)
    @info "Setting up parallel QG-YBJ simulation"
    
    # Create parameters (same as serial)
    T = Float64
    params = QGParams{T}(;
        nx = config.domain.nx,
        ny = config.domain.ny,
        nz = config.domain.nz,
        Lx = config.domain.Lx,
        Ly = config.domain.Ly,
        dt = config.dt,
        nt = ceil(Int, config.total_time / config.dt),
        f0 = config.f0,
        nu_h = config.nu_h,
        nu_v = config.nu_v,
        linear_vert_structure = 0,
        stratification = config.stratification.type,
        W2F = T(1e-6),
        gamma = T(1e-3),
        nuh1 = T(0.01), nuh2 = T(10.0), ilap1 = 2, ilap2 = 6,
        nuh1w = T(0.0), nuh2w = T(10.0), ilap1w = 2, ilap2w = 6,
        nuz = T(0.0),
        inviscid = config.inviscid,
        linear = config.linear,
        no_dispersion = config.no_dispersion,
        passive_scalar = config.passive_scalar,
        ybj_plus = config.ybj_plus,
        no_feedback = config.no_wave_feedback || config.no_feedback,
        fixed_flow = config.fixed_mean_flow,
        no_wave_feedback = config.no_wave_feedback,
        # Skewed Gaussian parameters (from config)
        N02_sg = config.stratification.N02_sg,
        N12_sg = config.stratification.N12_sg,
        sigma_sg = config.stratification.sigma_sg,
        z0_sg = config.stratification.z0_sg,
        alpha_sg = config.stratification.alpha_sg
    )
    
    # Initialize parallel grid and state
    grid = init_parallel_grid(params, pconfig)
    state = init_parallel_state(grid, pconfig)
    state_old = init_parallel_state(grid, pconfig)
    
    # Set up parallel transforms
    plans = plan_transforms!(grid, pconfig)
    
    # Initialize fields (need parallel-aware initialization)
    parallel_initialize_fields!(state, grid, plans, config, pconfig)
    
    # Set up output manager with parallel awareness  
    output_manager = OutputManager(config.output, params, pconfig)
    
    return params, grid, state, state_old, plans, output_manager
end

"""
    parallel_initialize_fields!(state, grid, plans, config, pconfig)

Initialize fields with parallel support.
"""
function parallel_initialize_fields!(state, grid, plans, config, pconfig)
    # This needs to handle distributed arrays properly
    # For now, initialize on each process locally and ensure consistency
    
    if config.initial_conditions.psi_type == :random
        # Each process needs the same random seed for consistency
        using Random
        Random.seed!(config.initial_conditions.random_seed)
        
        # Initialize only the local portion of the field
        init_parallel_random_psi!(state.psi, grid, config.initial_conditions.psi_amplitude, pconfig)
    end
    
    # Similar for wave fields
    if config.initial_conditions.wave_type == :random
        using Random
        Random.seed!(config.initial_conditions.random_seed + 1)  # Different seed
        
        init_parallel_random_waves!(state.B, grid, config.initial_conditions.wave_amplitude, pconfig)
    end
end

"""
    init_parallel_random_psi!(psik, grid, amplitude, pconfig)

Initialize random stream function with parallel support.
"""
function init_parallel_random_psi!(psik, grid, amplitude, pconfig)
    # Initialize random field ensuring consistency across processes
    # This is complex and requires careful handling of random number generation
    
    if grid.decomp === nothing
        # Serial fallback
        return init_random_psi!(psik, grid, amplitude)
    end
    
    # Parallel random initialization
    # Each process initializes its local portion with deterministic randomness
    import PencilArrays
    import Random
    
    local_ranges = PencilArrays.range_local(grid.decomp)
    
    # Generate deterministic random field based on global indices
    for k in local_ranges[3], j in local_ranges[2], i in local_ranges[1]
        # Use global indices (i,j,k) to seed local random generation
        local_seed = hash((i, j, k, config.initial_conditions.random_seed))
        Random.seed!(local_seed)
        
        # Simple random initialization (this would be more sophisticated in practice)
        psik[i, j, k] = amplitude * randn() * cis(2π * rand())
    end
end

"""
    init_parallel_random_waves!(Bk, grid, amplitude, pconfig)

Initialize random wave field with parallel support.
"""
function init_parallel_random_waves!(Bk, grid, amplitude, pconfig)
    # Similar to psi initialization but for wave field
    if grid.decomp === nothing
        return init_random_waves!(Bk, grid, amplitude)
    end
    
    import PencilArrays
    import Random
    
    local_ranges = PencilArrays.range_local(grid.decomp)
    
    for k in local_ranges[3], j in local_ranges[2], i in local_ranges[1]
        local_seed = hash((i, j, k, config.initial_conditions.random_seed, :waves))
        Random.seed!(local_seed)
        
        # Random complex field
        Bk[i, j, k] = amplitude * (randn() + im * randn()) * cis(2π * rand())
    end
end

# ParallelOutputManager removed - now using unified OutputManager in netcdf_io.jl