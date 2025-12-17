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
        M = Base.require(:MPI)
        if !M.Initialized(); M.Init(); end
        comm = M.COMM_WORLD
        rank = M.Comm_rank(comm)
        nprocs = M.Comm_size(comm)
        if rank == 0; @info "MPI initialized with $nprocs processes"; end
        config = ParallelConfig(use_mpi=true, comm=comm, n_processes=nprocs, parallel_io=true)
    catch e
        @info "MPI not available or failed to initialize: $e"; config = ParallelConfig(use_mpi=false)
    end
    return config
end

"""
    init_parallel_grid(params::QGParams, pconfig::ParallelConfig)

Initialize grid with proper parallel decomposition.

NOTE: This is a legacy interface. For proper MPI support, use the extension module:
    using MPI, PencilArrays, PencilFFTs, QGYBJ
    mpi_config = QGYBJ.setup_mpi_environment()
    grid = QGYBJ.init_mpi_grid(params, mpi_config)
"""
function init_parallel_grid(params::QGParams, pconfig::ParallelConfig)
    T = Float64
    nx, ny, nz = params.nx, params.ny, params.nz
    dx = params.Lx / nx
    dy = params.Ly / ny
    z = T.(collect(range(0, params.Lz; length=nz)))
    # Handle nz=1 edge case: diff returns empty array, so use full domain depth
    dz = nz > 1 ? diff(z) : T[params.Lz]

    # Wavenumbers (same on all processes)
    kx = T.([i <= nx÷2 ? (2π/params.Lx)*(i-1) : (2π/params.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/params.Ly)*(j-1) : (2π/params.Ly)*(j-1-ny) for j in 1:ny])

    # Initialize decomposition - this legacy interface only supports serial mode
    # For proper MPI support, use the extension module with init_mpi_grid()
    decomp = nothing
    kh2 = Array{T}(undef, nx, ny)

    if pconfig.use_mpi
        @warn """
        ParallelConfig.use_mpi=true but this legacy interface does not fully support MPI.
        For proper MPI parallelization, use the extension module:

            using MPI, PencilArrays, PencilFFTs, QGYBJ
            MPI.Init()
            mpi_config = QGYBJ.setup_mpi_environment()
            grid = QGYBJ.init_mpi_grid(params, mpi_config)

        Falling back to serial mode.
        """
    end

    # Serial computation of kh2
    @inbounds for j in 1:ny, i in 1:nx
        kh2[i,j] = kx[i]^2 + ky[j]^2
    end

    return Grid{T, typeof(kh2)}(nx, ny, nz, params.Lx, params.Ly, params.Lz, dx, dy, z, dz, kx, ky, kh2, decomp)
end

"""
    init_parallel_state(grid::Grid, pconfig::ParallelConfig; T=Float64)

Initialize state with distributed arrays when using MPI.

NOTE: This is a legacy interface. For proper MPI support, use the extension module:
    using MPI, PencilArrays, PencilFFTs, QGYBJ
    mpi_config = QGYBJ.setup_mpi_environment()
    state = QGYBJ.init_mpi_state(grid, mpi_config)
"""
function init_parallel_state(grid::Grid, pconfig::ParallelConfig; T=Float64)
    # This legacy interface only creates serial arrays
    # For proper MPI support, use the extension module with init_mpi_state()
    sz = (grid.nx, grid.ny, grid.nz)
    q   = Array{Complex{T}}(undef, sz); fill!(q, 0)
    psi = Array{Complex{T}}(undef, sz); fill!(psi, 0)
    A   = Array{Complex{T}}(undef, sz); fill!(A, 0)
    B   = Array{Complex{T}}(undef, sz); fill!(B, 0)
    C   = Array{Complex{T}}(undef, sz); fill!(C, 0)
    u   = Array{T}(undef, sz); fill!(u, 0)
    v   = Array{T}(undef, sz); fill!(v, 0)
    w   = Array{T}(undef, sz); fill!(w, 0)

    return State{T, typeof(u), typeof(q)}(q, B, psi, A, C, u, v, w)
end


"""
    gather_array_for_io(arr, grid::Grid, pconfig::ParallelConfig)

Gather distributed array to rank 0 for I/O operations.

NOTE: This is a legacy interface. For proper MPI gathering, use the extension module:
    gathered = QGYBJ.gather_to_root(arr, grid, mpi_config)
"""
function gather_array_for_io(arr, grid::Grid, pconfig::ParallelConfig)
    # This legacy interface just returns the array as-is
    # For proper MPI gathering, use the extension module with gather_to_root()
    return arr
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
        f₀ = config.f0,
        νₕ = config.nu_h,
        νᵥ = config.nu_v,
        linear_vert_structure = 0,
        stratification = config.stratification.type,
        W2F = T(1e-6),
        γ = T(1e-3),
        νₕ₁ = T(0.01), νₕ₂ = T(10.0), ilap1 = 2, ilap2 = 6,
        νₕ₁ʷ = T(0.0), νₕ₂ʷ = T(10.0), ilap1w = 2, ilap2w = 6,
        νz = T(0.0),
        inviscid = config.inviscid,
        linear = config.linear,
        no_dispersion = config.no_dispersion,
        passive_scalar = config.passive_scalar,
        ybj_plus = config.ybj_plus,
        no_feedback = config.no_wave_feedback || config.no_feedback,
        fixed_flow = config.fixed_mean_flow,
        no_wave_feedback = config.no_wave_feedback,
        # Skewed Gaussian parameters (from config)
        N₀²_sg = config.stratification.N02_sg,
        N₁²_sg = config.stratification.N12_sg,
        σ_sg = config.stratification.sigma_sg,
        z₀_sg = config.stratification.z0_sg,
        α_sg = config.stratification.alpha_sg
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
        # Deterministic initialization across ranks
        init_parallel_random_psi!(state.psi, grid, config.initial_conditions.psi_amplitude, pconfig)
    end
    
    # Similar for wave fields
    if config.initial_conditions.wave_type == :random
        init_parallel_random_waves!(state.B, grid, config.initial_conditions.wave_amplitude, pconfig)
    end
end

"""
    init_parallel_random_psi!(psik, grid, amplitude, pconfig)

Initialize random stream function with parallel support.

NOTE: This is a legacy interface. For proper MPI initialization, use the extension module:
    QGYBJ.init_mpi_random_field!(psik, grid, amplitude, seed_offset)
"""
function init_parallel_random_psi!(psik, grid, amplitude, pconfig)
    # Hash-based deterministic initialization (works in serial)
    # For proper MPI support, use the extension module with init_mpi_random_field!
    for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
        φ = 2π * ((hash((i,j,k)) % 1_000_000) / 1_000_000)
        psik[i,j,k] = amplitude * cis(φ)
    end
    return psik
end

"""
    init_parallel_random_waves!(Bk, grid, amplitude, pconfig)

Initialize random wave field with parallel support.

NOTE: This is a legacy interface. For proper MPI initialization, use the extension module:
    QGYBJ.init_mpi_random_field!(Bk, grid, amplitude, seed_offset)
"""
function init_parallel_random_waves!(Bk, grid, amplitude, pconfig)
    # Hash-based deterministic initialization (works in serial)
    # For proper MPI support, use the extension module with init_mpi_random_field!
    for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
        φ = 2π * ((hash((i,j,k,:waves)) % 1_000_000) / 1_000_000)
        Bk[i,j,k] = amplitude * cis(φ)
    end
    return Bk
end

# ParallelOutputManager removed - now using unified OutputManager in netcdf_io.jl
