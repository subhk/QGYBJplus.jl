"""
Unified particle advection module for QG-YBJ simulations.

This module provides Lagrangian particle tracking that automatically handles
both serial and parallel execution. Particles are advected using the TOTAL
velocity field (QG + wave velocities) with options for vertical velocity from either
QG omega equation or YBJ formulation.

Total velocity field includes:
- QG velocities: u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x
- Wave velocities: u_wave = Re(LA), v_wave = Im(LA) from YBJ+ eq (1.2)
  where L = ∂_z(f²/N²)∂_z and LA = B + (k_h²/4)A in spectral space
- Horizontal Stokes drift from Wagner & Young (2016) equation (3.16a):
  Full Jacobian form: J₀ = (LA)* ∂_{s*}(LA) - (f²/N²)(∂_{s*} A_z*) ∂_z(LA)
  where ∂_{s*} = (1/2)(∂_x + i∂_y) is the complex horizontal derivative.
  From eq (3.18): if₀ U^S = J₀, so u_S = Im(J₀)/f₀, v_S = -Re(J₀)/f₀
- Vertical Stokes drift from Wagner & Young (2016) eq (3.19)-(3.20):
  if₀w^S = K₀* - K₀, where K₀ = ∂(M*, M_s)/∂(z̃, s*) and M = (f₀²/N²)A_z
  Expanding: K₀ = M*_z · M_{ss*} - M*_{s*} · M_{sz}, giving w_S = -2·Im(K₀)/f₀
- Vertical velocity: w from QG omega equation or YBJ w₀ formulation

The system automatically detects MPI availability and handles:
- Domain decomposition and particle migration
- Periodic boundary conditions
- Distributed I/O
- Load balancing

Advanced features:
- Delayed particle release: Use particle_advec_time to control when particles start moving
- Time synchronization: Particles sync with simulation time when current_time is provided
- Multiple integration schemes: Euler, RK2, RK4 with adjustable interpolation methods
- Flexible I/O: Configurable save intervals and trajectory tracking
- Boundary handling: Periodic and reflective boundary conditions

Time synchronization:
- Pass current_time to advect_particles! for proper synchronization with fluid simulation
- particle_advec_time is compared against simulation time, not particle internal time
- Ensures particles respond to flow conditions at the correct simulation time
"""

module UnifiedParticleAdvection

# Bind names from parent module (QGYBJplus) without using/import
const _PARENT = Base.parentmodule(@__MODULE__)
const Grid = _PARENT.Grid
const State = _PARENT.State
const plan_transforms! = _PARENT.plan_transforms!
const compute_total_velocities! = _PARENT.compute_total_velocities!
const ParallelConfig = _PARENT.ParallelConfig

export ParticleConfig, ParticleState, ParticleTracker,
       create_particle_config, initialize_particles!,
       advect_particles!, interpolate_velocity_at_position,
       # Advanced interpolation methods
       InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
       interpolate_velocity_advanced,
       # 3D particle distributions
       ParticleConfig3D, ParticleDistribution, create_particle_config_3d,
       initialize_particles_3d!, UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM,
       create_uniform_3d_grid, create_layered_distribution, create_random_3d_distribution, create_custom_distribution,
       # Simplified particle initialization
       particles_in_box, particles_in_circle, particles_in_grid_3d, particles_in_layers,
       particles_random_3d, particles_custom,
       # Parallel utilities
       validate_particle_cfl

# Include advanced interpolation schemes FIRST (halo_exchange.jl depends on it)
include("interpolation_schemes.jl")
using .InterpolationSchemes

# Include halo exchange for cross-domain interpolation
include("halo_exchange.jl")
using .HaloExchange

# NOTE: particle_config.jl is included AFTER ParticleConfig struct definition below
# because it needs to access ParticleConfig from this module

"""
Configuration for particle initialization and advection.

Key parameters:
- Spatial domain: x_min/max, y_min/max, z_level for initial particle placement
- Particle count: nx_particles × ny_particles 
- Physics options: use_ybj_w (vertical velocity), use_3d_advection
- Timing control: particle_advec_time - when to start advecting particles
- Integration: method (:euler, :rk2, :rk4) and interpolation scheme
- Boundaries: periodic_x/y, reflect_z for boundary conditions
- I/O: save_interval and max_save_points for trajectory output

Advanced timing control:
- particle_advec_time=0.0: Start advecting immediately (default)
- particle_advec_time>0.0: Keep particles stationary until this time
- Useful for letting flow field develop before particle release
- Enables study of transient vs established flow patterns
"""
Base.@kwdef struct ParticleConfig{T<:AbstractFloat}
    # Spatial domain for particle initialization (x_max, y_max, z_level are REQUIRED)
    x_min::T = 0.0
    x_max::T           # REQUIRED - use G.Lx
    y_min::T = 0.0
    y_max::T           # REQUIRED - use G.Ly
    z_level::T         # REQUIRED - depth for particle initialization
    
    # Number of particles
    nx_particles::Int = 10
    ny_particles::Int = 10
    
    # Physics options
    use_ybj_w::Bool = false           # Use YBJ vs QG vertical velocity
    use_3d_advection::Bool = true     # Include vertical advection
    
    # Advection timing control
    particle_advec_time::T = 0.0      # Start advecting particles at this time (0.0 = from beginning)
    
    # Integration method
    integration_method::Symbol = :euler  # :euler, :rk2, :rk4
    
    # Interpolation method
    interpolation_method::InterpolationMethod = TRILINEAR  # TRILINEAR, TRICUBIC, ADAPTIVE
    
    # Boundary conditions
    periodic_x::Bool = true
    periodic_y::Bool = true
    reflect_z::Bool = true            # Reflect at vertical boundaries
    
    # I/O configuration - Controls particle trajectory saving rate
    save_interval::T = 0.1           # Time interval for saving trajectories (e.g., 0.1 = save every 0.1 time units)
    max_save_points::Int = 1000      # Maximum trajectory points to save per file
    auto_split_files::Bool = false   # Automatically create new files when max_save_points is reached
end

"""
    particles_in_box([T=Float32], z_level; x_max, y_max, x_min=0, y_min=0, nx=10, ny=10, kwargs...)

Create 2D particle distribution at a fixed z-level. Default precision is Float32 for memory efficiency.

# Arguments
- `T`: Optional type parameter (Float32 or Float64). Default: Float32
- `z_level`: The z-level (depth) for all particles
- `x_max, y_max`: Maximum domain bounds (REQUIRED)
- `x_min, y_min`: Minimum bounds (default: 0.0)
- `nx, ny`: Number of particles in each direction (default: 10 each)

# Examples
```julia
# Default Float32 precision (recommended for memory efficiency)
config = particles_in_box(500.0; x_max=G.Lx, y_max=G.Ly, nx=20, ny=20)

# Explicit Float64 if higher precision needed
config = particles_in_box(Float64, 500.0; x_max=G.Lx, y_max=G.Ly, nx=20, ny=20)
```
"""
function particles_in_box(::Type{T}, z_level::Real;
                          x_max::Real, y_max::Real,  # REQUIRED
                          x_min::Real=0.0, y_min::Real=0.0,
                          nx::Int=10, ny::Int=10,
                          kwargs...) where T<:AbstractFloat
    return ParticleConfig{T}(;
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        z_level=T(z_level), nx_particles=nx, ny_particles=ny,
        kwargs...)
end

# Default method uses Float32 for memory efficiency (50% less than Float64)
function particles_in_box(z_level::Real;
                          x_max::Real, y_max::Real,  # REQUIRED
                          x_min::Real=0.0, y_min::Real=0.0,
                          nx::Int=10, ny::Int=10,
                          kwargs...)
    return particles_in_box(Float32, z_level;
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        nx=nx, ny=ny, kwargs...)
end

# Legacy alias for backwards compatibility (default is now Float32 for memory efficiency)
create_particle_config(::Type{T}=Float32; kwargs...) where T = ParticleConfig{T}(; kwargs...)

# Include 3D particle configuration AFTER ParticleConfig is defined
# (particle_config.jl needs to access ParticleConfig from this module)
include("particle_config.jl")
using .EnhancedParticleConfig

create_particle_config_3d(::Type{T}=Float32; kwargs...) where {T<:AbstractFloat} =
    ParticleConfig3D{T}(; kwargs...)

create_uniform_3d_grid(; kwargs...) = particles_in_grid_3d(; kwargs...)
create_uniform_3d_grid(x_min::Real, x_max::Real, y_min::Real, y_max::Real,
                       z_max::Real, nx::Int, ny::Int, nz::Int; kwargs...) =
    particles_in_grid_3d(; x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                         z_max=z_max, nx=nx, ny=ny, nz=nz, kwargs...)
create_uniform_3d_grid(x_min::Real, x_max::Real, y_min::Real, y_max::Real,
                       z_min::Real, z_max::Real, nx::Int, ny::Int, nz::Int; kwargs...) =
    particles_in_grid_3d(; x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                         z_min=z_min, z_max=z_max, nx=nx, ny=ny, nz=nz, kwargs...)

create_layered_distribution(z_levels::Vector{<:Real}; kwargs...) =
    particles_in_layers(z_levels; kwargs...)
create_layered_distribution(x_min::Real, x_max::Real, y_min::Real, y_max::Real,
                            z_levels::Vector{<:Real}, nx::Int, ny::Int; kwargs...) =
    particles_in_layers(z_levels; x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                        nx=nx, ny=ny, kwargs...)

create_random_3d_distribution(n::Int; kwargs...) = particles_random_3d(n; kwargs...)
create_custom_distribution(positions; kwargs...) = particles_custom(positions; kwargs...)

"""
Particle state including positions, global IDs, velocities, and trajectory history.
"""
mutable struct ParticleState{T<:AbstractFloat}
    # Current state
    x::Vector{T}
    y::Vector{T} 
    z::Vector{T}
    id::Vector{Int}
    u::Vector{T}
    v::Vector{T}
    w::Vector{T}
    time::T
    np::Int
    
    # Trajectory history
    x_history::Vector{Vector{T}}
    y_history::Vector{Vector{T}}
    z_history::Vector{Vector{T}}
    id_history::Vector{Vector{Int}}
    time_history::Vector{T}
    
    function ParticleState{T}(np::Int) where T
        ids = collect(1:np)
        new{T}(
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            ids,
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            zero(T), np,
            Vector{T}[], Vector{T}[], Vector{T}[], Vector{Int}[], T[]
        )
    end
end

"""
Main particle tracker that handles both serial and parallel execution.
"""
mutable struct ParticleTracker{T<:AbstractFloat}
    config::ParticleConfig{T}
    particles::ParticleState{T}
    
    # Grid information
    nx::Int; ny::Int; nz::Int
    Lx::T; Ly::T; Lz::T
    x0::T; y0::T
    dx::T; dy::T; dz::T
    
    # Velocity field workspace (real space)
    u_field::Array{T,3}
    v_field::Array{T,3}
    w_field::Array{T,3}
    
    # Transform plans (for velocity computation)
    plans
    
    # Parallel information (automatically detected)
    comm::Any           # MPI communicator (nothing for serial)
    rank::Int          # MPI rank (0 for serial)
    nprocs::Int        # Number of MPI processes (1 for serial)
    is_parallel::Bool  # True if running in parallel
    
    # Domain decomposition info (for parallel)
    local_domain::Union{Nothing,NamedTuple}  # Local domain bounds
    
    # Particle migration buffers (for parallel)
    send_buffers::Vector{Vector{T}}
    recv_buffers::Vector{Vector{T}}
    send_buffers_id::Vector{Vector{Int}}
    recv_buffers_id::Vector{Vector{Int}}
    
    # Halo exchange system for cross-domain interpolation
    halo_info::Union{Nothing,HaloInfo{T}}
    
    # I/O settings
    save_counter::Int
    last_save_time::T
    is_io_rank::Bool
    gather_for_io::Bool
    
    # File splitting for large simulations
    output_file_sequence::Int          # Current file sequence number (0, 1, 2, ...)
    base_output_filename::String       # Base filename for automatic file splitting
    auto_file_splitting::Bool          # Enable automatic file splitting when max_save_points reached
    
    function ParticleTracker{T}(config::ParticleConfig{T}, grid::Grid, parallel_config=nothing) where T
        np = config.nx_particles * config.ny_particles
        particles = ParticleState{T}(np)
        
        # Use provided parallel config or detect environment
        if parallel_config !== nothing && parallel_config.use_mpi
            try
                M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
                comm = parallel_config.comm
                rank = M.Comm_rank(comm)
                nprocs = M.Comm_size(comm)
                is_parallel = true
            catch
                comm, rank, nprocs, is_parallel = detect_parallel_environment()
            end
        else
            comm, rank, nprocs, is_parallel = detect_parallel_environment()
        end
        
        # Set up domain decomposition if parallel (use MPI topology when available)
        topology = parallel_config !== nothing && hasfield(typeof(parallel_config), :topology) ?
                   parallel_config.topology : nothing
        local_domain = is_parallel ? compute_local_domain(grid, rank, nprocs; topology=topology) : nothing

        # Initialize buffers
        send_buffers = [T[] for _ in 1:nprocs]
        recv_buffers = [T[] for _ in 1:nprocs]
        send_buffers_id = [Int[] for _ in 1:nprocs]
        recv_buffers_id = [Int[] for _ in 1:nprocs]

        # Velocity field workspace - use LOCAL size in parallel mode
        if is_parallel && local_domain !== nothing
            nx_local = local_domain.nx_local
            ny_local = local_domain.ny_local
            u_field = zeros(T, grid.nz, nx_local, ny_local)
            v_field = zeros(T, grid.nz, nx_local, ny_local)
            w_field = zeros(T, grid.nz, nx_local, ny_local)
        else
            u_field = zeros(T, grid.nz, grid.nx, grid.ny)
            v_field = zeros(T, grid.nz, grid.nx, grid.ny)
            w_field = zeros(T, grid.nz, grid.nx, grid.ny)
        end

        # Set up transform plans (using unified interface)
        plans = plan_transforms!(grid, parallel_config)

        # Defer halo exchange setup until first velocity update
        # This allows us to get actual local dimensions from State arrays
        # (which may differ from 1D decomposition assumption in 2D pencil decomposition)
        halo_info = nothing  # Will be set up lazily in update_velocity_fields!

        new{T}(
            config, particles,
            grid.nx, grid.ny, grid.nz,
            grid.Lx, grid.Ly, grid.Lz,
            grid.x0, grid.y0,
            grid.Lx/grid.nx, grid.Ly/grid.ny, grid.dz[1],
            u_field, v_field, w_field, plans,
            comm, rank, nprocs, is_parallel,
            local_domain,
            send_buffers, recv_buffers,
            send_buffers_id, recv_buffers_id,
            halo_info,
            0, zero(T), rank == 0, true,
            0, "", false  # output_file_sequence, base_output_filename, auto_file_splitting
        )
    end
end

ParticleTracker(config::ParticleConfig{T}, grid::Grid, parallel_config=nothing) where T = ParticleTracker{T}(config, grid, parallel_config)

"""
    setup_halo_exchange_for_grid(grid, rank, nprocs, comm, T; local_dims=nothing, process_grid=nothing,
                                 periodic_x=true, periodic_y=true, interpolation_method=TRILINEAR)

Helper function to set up halo exchange system.

# Arguments
- `local_dims`: Tuple (nz_local, nx_local, ny_local) for 2D pencil decomposition.
                If nothing, assumes 1D decomposition in x.
- `process_grid`: Optional (px, py) process grid to match simulation topology.
- `periodic_x, periodic_y`: Boundary condition flags for halo exchange.
- `interpolation_method`: The interpolation scheme to use. Determines halo width:
                          - TRILINEAR: 1 halo cell
                          - TRICUBIC:  2 halo cells
                          - QUINTIC:   3 halo cells
                          - ADAPTIVE:  3 halo cells (supports all schemes)
"""
function setup_halo_exchange_for_grid(grid::Grid, rank::Int, nprocs::Int, comm, ::Type{T};
                                     local_dims::Union{Nothing,Tuple{Int,Int,Int}}=nothing,
                                     process_grid::Union{Nothing,Tuple{Int,Int}}=nothing,
                                     periodic_x::Bool=true,
                                     periodic_y::Bool=true,
                                     interpolation_method::InterpolationMethod=TRILINEAR) where T
    try
        return HaloInfo{T}(grid, rank, nprocs, comm;
                          local_dims=local_dims,
                          process_grid=process_grid,
                          periodic_x=periodic_x,
                          periodic_y=periodic_y,
                          interpolation_method=interpolation_method)
    catch e
        @warn "Failed to set up halo exchange: $e"
        return nothing
    end
end

"""
    detect_parallel_environment()

Automatically detect if running in parallel and get MPI information.
"""
function detect_parallel_environment()
    comm = nothing
    rank = 0
    nprocs = 1
    is_parallel = false

    try
        M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        if M.Initialized()
            comm = M.COMM_WORLD
            rank = M.Comm_rank(comm)
            nprocs = M.Comm_size(comm)
            is_parallel = nprocs > 1
        end
    catch
        # MPI not available
    end

    return comm, rank, nprocs, is_parallel
end

"""
    compute_local_domain(grid, rank, nprocs; topology=nothing)

Compute local domain bounds for MPI rank (1D or 2D decomposition).
"""
function compute_local_domain(grid::Grid, rank::Int, nprocs::Int; topology=nothing)
    # Determine process grid (px × py)
    px, py = if topology !== nothing
        topology
    elseif grid.decomp !== nothing && hasfield(typeof(grid.decomp), :topology)
        grid.decomp.topology
    else
        (nprocs, 1)
    end

    @assert px * py == nprocs "Process grid ($px × $py) must equal nprocs ($nprocs)"

    # Compute rank coordinates in process grid
    rank_x = rank % px
    rank_y = rank ÷ px

    # Local sizes for each dimension
    nx_local = compute_local_size(grid.nx, px, rank_x)
    ny_local = compute_local_size(grid.ny, py, rank_y)

    # Start indices (0-based) for each dimension
    x_start = compute_start_index(grid.nx, px, rank_x)
    y_start = compute_start_index(grid.ny, py, rank_y)

    # Convert to physical coordinates (respect domain origin)
    dx = grid.Lx / grid.nx
    dy = grid.Ly / grid.ny
    x_start_phys = grid.x0 + x_start * dx
    x_end_phys = grid.x0 + (x_start + nx_local) * dx
    y_start_phys = grid.y0 + y_start * dy
    y_end_phys = grid.y0 + (y_start + ny_local) * dy

    return (x_start=x_start_phys, x_end=x_end_phys,
            y_start=y_start_phys, y_end=y_end_phys,
            z_start=-grid.Lz, z_end=zero(grid.Lz),
            nx_local=nx_local, ny_local=ny_local,
            px=px, py=py, rank_x=rank_x, rank_y=rank_y)
end

"""
    initialize_particles!(tracker, config)

Initialize particles uniformly in specified region, handling both serial and parallel.
"""
function initialize_particles!(tracker::ParticleTracker{T}, 
                              config::ParticleConfig{T}) where T
    
    if tracker.is_parallel
        # Initialize only particles in local domain
        initialize_particles_parallel!(tracker, config)
    else
        # Initialize all particles (serial case)
        initialize_particles_serial!(tracker, config)
    end
    
    # Save initial state
    save_particle_state!(tracker)
    
    return tracker
end

"""
    initialize_particles!(tracker, config3d)

Initialize particles using enhanced 3D configuration.
"""
function initialize_particles!(tracker::ParticleTracker{T}, 
                              config::ParticleConfig3D{T}) where T
    
    # Use 3D initialization
    initialize_particles_3d!(tracker, config)

    tracker.config = EnhancedParticleConfig.convert_to_basic_config(config)
    
    # Save initial state
    save_particle_state!(tracker)
    
    return tracker
end

"""
Initialize particles for serial execution.
"""
function initialize_particles_serial!(tracker::ParticleTracker{T}, 
                                     config::ParticleConfig{T}) where T
    particles = tracker.particles
    
    # Create uniform grid of particles
    x_range = range(config.x_min, config.x_max, length=config.nx_particles+1)[1:end-1]
    y_range = range(config.y_min, config.y_max, length=config.ny_particles+1)[1:end-1]
    
    idx = 1
    for y in y_range, x in x_range
        particles.x[idx] = x
        particles.y[idx] = y
        particles.z[idx] = config.z_level
        particles.id[idx] = idx
        particles.u[idx] = 0.0
        particles.v[idx] = 0.0
        particles.w[idx] = 0.0
        idx += 1
    end
    
    particles.time = 0.0
    particles.np = length(particles.x)
    
    return tracker
end

"""
Initialize particles for parallel execution (only in local domain).
"""
function initialize_particles_parallel!(tracker::ParticleTracker{T},
                                       config::ParticleConfig{T}) where T
    local_domain = tracker.local_domain
    
    # Find intersection of particle region with local domain
    x_min = max(config.x_min, local_domain.x_start)
    x_max = min(config.x_max, local_domain.x_end)
    y_min = max(config.y_min, local_domain.y_start)
    y_max = min(config.y_max, local_domain.y_end)
    
    dxp = (config.x_max - config.x_min) / config.nx_particles
    dyp = (config.y_max - config.y_min) / config.ny_particles

    if x_min >= x_max || y_min >= y_max || dxp <= zero(T) || dyp <= zero(T)
        # No particles in this domain
        resize!(tracker.particles.x, 0)
        resize!(tracker.particles.y, 0)
        resize!(tracker.particles.z, 0)
        resize!(tracker.particles.id, 0)
        resize!(tracker.particles.u, 0)
        resize!(tracker.particles.v, 0)
        resize!(tracker.particles.w, 0)
        tracker.particles.np = 0
        return tracker
    end
    
    # Compute exact index ranges from the global particle grid
    tol = sqrt(eps(T)) * max(one(T), abs(config.x_max - config.x_min), abs(config.y_max - config.y_min))
    x_rel_min = (x_min - config.x_min) / dxp
    x_rel_max = (x_max - config.x_min) / dxp
    y_rel_min = (y_min - config.y_min) / dyp
    y_rel_max = (y_max - config.y_min) / dyp

    i_start = max(1, floor(Int, x_rel_min + tol) + 1)
    i_end = min(config.nx_particles, ceil(Int, x_rel_max - tol))
    j_start = max(1, floor(Int, y_rel_min + tol) + 1)
    j_end = min(config.ny_particles, ceil(Int, y_rel_max - tol))

    if i_start > i_end || j_start > j_end
        resize!(tracker.particles.x, 0)
        resize!(tracker.particles.y, 0)
        resize!(tracker.particles.z, 0)
        resize!(tracker.particles.id, 0)
        resize!(tracker.particles.u, 0)
        resize!(tracker.particles.v, 0)
        resize!(tracker.particles.w, 0)
        tracker.particles.np = 0
        return tracker
    end

    n_local = (i_end - i_start + 1) * (j_end - j_start + 1)
    
    # Resize arrays to actual local particle count
    resize!(tracker.particles.x, n_local)
    resize!(tracker.particles.y, n_local)
    resize!(tracker.particles.z, n_local)
    resize!(tracker.particles.id, n_local)
    resize!(tracker.particles.u, n_local)
    resize!(tracker.particles.v, n_local)
    resize!(tracker.particles.w, n_local)
    
    # Initialize local particles
    idx = 1
    for j in j_start:j_end, i in i_start:i_end
        tracker.particles.x[idx] = config.x_min + (i - 1) * dxp
        tracker.particles.y[idx] = config.y_min + (j - 1) * dyp
        tracker.particles.z[idx] = config.z_level
        tracker.particles.id[idx] = (j - 1) * config.nx_particles + i
        tracker.particles.u[idx] = 0.0
        tracker.particles.v[idx] = 0.0
        tracker.particles.w[idx] = 0.0
        idx += 1
    end
    
    tracker.particles.time = 0.0
    tracker.particles.np = n_local
    
    return tracker
end

"""
    advect_particles!(tracker, state, grid, dt, current_time=nothing; params=nothing, N2_profile=nothing)

Advect particles using unified serial/parallel interface.
Respects the particle_advec_time setting - particles remain stationary until this time.

Parameters:
- tracker: ParticleTracker instance
- state: Current fluid state
- grid: Grid information
- dt: Time step
- current_time: Current simulation time (if not provided, uses tracker's internal time)
- params: Model parameters (QGParams). Required for YBJ vertical velocity to get correct f₀, N².
- N2_profile: Optional N²(z) profile for nonuniform stratification. If not provided and
  `use_ybj_w=true`, will use constant N² from params, which may be inconsistent with the
  simulation's actual stratification.

# Important
When using YBJ vertical velocity (`use_ybj_w=true`) with variable stratification, you MUST
pass the same `N2_profile` used in the simulation. Otherwise, `compute_ybj_vertical_velocity!`
will re-invert B→A with constant N², giving inconsistent particle velocities.
"""
function advect_particles!(tracker::ParticleTracker{T},
                          state::State, grid::Grid, dt::T, current_time=nothing;
                          params=nothing, N2_profile=nothing) where T
    
    # Use simulation time if provided, otherwise use tracker's internal time
    if current_time !== nothing
        sim_time = T(current_time)
        # Update tracker time to match simulation
        tracker.particles.time = sim_time
    else
        sim_time = tracker.particles.time
    end
    
    # Check if we should start advecting particles yet
    advec_start_time = tracker.config.particle_advec_time
    
    if sim_time < advec_start_time
        # Particles remain stationary - only update time
        if current_time === nothing
            tracker.particles.time += dt
        else
            tracker.particles.time = T(current_time) + dt
        end
        
        # Still save state if needed (for tracking stationary phase)
        if should_save_particles(tracker)
            save_particle_state!(tracker)
        end
        
        return tracker
    end
    
    # Normal advection process starts here
    # Update velocity fields (pass params and N2_profile for consistent YBJ vertical velocity)
    update_velocity_fields!(tracker, state, grid; params=params, N2_profile=N2_profile)
    
    # Advect particles using chosen integration method
    if tracker.config.integration_method == :euler
        advect_euler!(tracker, dt)
    elseif tracker.config.integration_method == :rk2
        advect_rk2!(tracker, dt)
    elseif tracker.config.integration_method == :rk4
        advect_rk4!(tracker, dt)
    else
        error("Unknown integration method: $(tracker.config.integration_method)")
    end
    
    # Handle particle migration in parallel
    if tracker.is_parallel
        migrate_particles!(tracker)
    end
    
    # Apply boundary conditions
    apply_boundary_conditions!(tracker)
    
    # Update time (use simulation time if provided)
    if current_time !== nothing
        tracker.particles.time = T(current_time) + dt
    else
        tracker.particles.time += dt
    end
    
    # Save state if needed
    if should_save_particles(tracker)
        save_particle_state!(tracker)
    end
    
    return tracker
end

"""
    update_velocity_fields!(tracker, state, grid; params=nothing, N2_profile=nothing)

Update TOTAL velocity fields from fluid state (QG + wave velocities) and exchange halos if parallel.
Computes the complete velocity field needed for proper QG-YBJ particle advection.

Handles 2D pencil decomposition by getting actual local dimensions from State arrays.

# Arguments
- `params`: Model parameters (QGParams). Required for YBJ vertical velocity to get correct f₀, N².
- `N2_profile`: Optional N²(z) profile for nonuniform stratification. If not provided and
  `use_ybj_w=true`, will use constant N² from params, which may be inconsistent with the
  simulation's actual stratification.

# Important
When using YBJ vertical velocity (`use_ybj_w=true`) with variable stratification, you MUST
pass the same `N2_profile` used in the simulation. Otherwise, `compute_ybj_vertical_velocity!`
will re-invert B→A with constant N², giving inconsistent particle velocities.
"""
function update_velocity_fields!(tracker::ParticleTracker{T},
                                state::State, grid::Grid;
                                params=nothing, N2_profile=nothing) where T
    # Compute TOTAL velocities (QG + wave) with chosen vertical velocity formulation
    # Pass params and N2_profile to ensure consistent stratification handling
    compute_total_velocities!(state, grid;
                              plans=tracker.plans,
                              params=params,
                              compute_w=true,
                              use_ybj_w=tracker.config.use_ybj_w,
                              N2_profile=N2_profile)

    # Get actual local dimensions from State arrays
    # This handles both serial (full grid) and parallel (2D pencil decomposition)
    u_data = parent(state.u)
    v_data = parent(state.v)
    w_data = parent(state.w)
    local_dims = size(u_data)
    nz_local, nx_local, ny_local = local_dims

    # Resize tracker velocity fields if needed (first call or dimension change)
    if size(tracker.u_field) != local_dims
        tracker.u_field = zeros(T, nz_local, nx_local, ny_local)
        tracker.v_field = zeros(T, nz_local, nx_local, ny_local)
        tracker.w_field = zeros(T, nz_local, nx_local, ny_local)
    end

    # Copy velocity data
    tracker.u_field .= u_data
    tracker.v_field .= v_data
    tracker.w_field .= w_data

    # Lazily initialize halo exchange system with actual local dimensions
    # The halo width is determined by the interpolation method
    if tracker.is_parallel && tracker.halo_info === nothing
        process_grid = tracker.local_domain !== nothing && hasproperty(tracker.local_domain, :px) ?
                       (tracker.local_domain.px, tracker.local_domain.py) : nothing
        tracker.halo_info = setup_halo_exchange_for_grid(
            grid, tracker.rank, tracker.nprocs, tracker.comm, T;
            local_dims=local_dims,
            process_grid=process_grid,
            periodic_x=tracker.config.periodic_x,
            periodic_y=tracker.config.periodic_y,
            interpolation_method=tracker.config.interpolation_method
        )
    end

    # Exchange halo data for cross-domain interpolation
    if tracker.is_parallel && tracker.halo_info !== nothing
        exchange_velocity_halos!(tracker.halo_info,
                               tracker.u_field,
                               tracker.v_field,
                               tracker.w_field)
    end

    return tracker
end

"""
    validate_particle_cfl(tracker, max_velocity, dt)

Check if timestep satisfies CFL condition for particle advection in parallel mode.

For RK4 integration, intermediate positions can move up to dt*max_velocity from their
starting position. If this exceeds the halo region, interpolation will be inaccurate.

Returns true if timestep is safe, false if timestep may cause issues.

# Warning
If this returns false, consider:
- Reducing dt
- Increasing halo_width (use higher-order interpolation which has wider halos)
- Using Euler instead of RK4 (which only evaluates at current position)
"""
function validate_particle_cfl(tracker::ParticleTracker{T}, max_velocity::T, dt::T) where T
    if !tracker.is_parallel || tracker.halo_info === nothing
        return true  # No halo constraints for serial mode
    end

    hw = tracker.halo_info.halo_width
    dx = tracker.dx

    # For RK4, intermediate positions can be up to dt * max_velocity away
    # For safety, we require this to be less than halo_width * dx
    max_displacement = dt * max_velocity
    safe_displacement = hw * dx

    return max_displacement < safe_displacement
end

"""
    interpolate_velocity_at_position(x, y, z, tracker)

Interpolate velocity at particle position with advanced schemes and cross-domain capability.
"""
function interpolate_velocity_at_position(x::T, y::T, z::T,
                                        tracker::ParticleTracker{T}) where T
    
    # Use halo-aware interpolation if available (parallel case)
    if tracker.is_parallel && tracker.halo_info !== nothing
        return interpolate_velocity_with_halos_advanced(x, y, z, tracker, tracker.halo_info)
    end
    
    # Use advanced interpolation for serial case
    return interpolate_velocity_advanced_local(x, y, z, tracker)
end

"""
    interpolate_velocity_advanced_local(x, y, z, tracker)

Advanced interpolation for serial case using high-order schemes.
"""
function interpolate_velocity_advanced_local(x::T, y::T, z::T, 
                                           tracker::ParticleTracker{T}) where T
    
    # Set up grid info and boundary conditions
    z_min = -tracker.Lz
    grid_info = (dx=tracker.dx, dy=tracker.dy, dz=tracker.dz,
                Lx=tracker.Lx, Ly=tracker.Ly, Lz=tracker.Lz,
                z_min=z_min, z_max=zero(T), z0=z_min + tracker.dz / 2)
    
    boundary_conditions = (periodic_x=tracker.config.periodic_x,
                          periodic_y=tracker.config.periodic_y,
                          periodic_z=false)
    
    # Use advanced interpolation (shift to domain-relative coordinates)
    x_rel = x - tracker.x0
    y_rel = y - tracker.y0
    u_interp, v_interp, w_interp = interpolate_velocity_advanced(
        x_rel, y_rel, z,
        tracker.u_field, tracker.v_field, tracker.w_field,
        grid_info, boundary_conditions,
        tracker.config.interpolation_method
    )
    
    # For 2D advection, set w to zero
    if !tracker.config.use_3d_advection
        w_interp = 0.0
    end
    
    return u_interp, v_interp, w_interp
end

"""
    interpolate_velocity_with_halos_advanced(x, y, z, tracker, halo_info)

Advanced interpolation using halo data for parallel case.

The extended arrays have layout: [left_halo | local_data | right_halo] in BOTH x and y.
So index 1 is the left/bottom halo, and local data starts at index (halo_width + 1).
We need to offset both x_local and y_local by halo_width * dx/dy to account for this.

# 2D Decomposition Support
For 2D MPI decomposition (py > 1), both x and y coordinates must be converted to
local extended array coordinates. The halo regions contain neighbor data in both
directions, so periodic boundary conditions are handled by the halos, not by wrapping.
"""
function interpolate_velocity_with_halos_advanced(x::T, y::T, z::T,
                                                 tracker::ParticleTracker{T},
                                                 halo_info::HaloInfo{T}) where T

    # For high-order interpolation, we need extended halos
    method = tracker.config.interpolation_method
    if method == TRICUBIC || method == ADAPTIVE || method == QUINTIC
        # Check if we have enough halo width for the interpolation method
        min_halo = method == QUINTIC ? 3 : 2
        if halo_info.halo_width < min_halo
            @warn "Insufficient halo width ($( halo_info.halo_width)) for $(method) interpolation (needs $min_halo), falling back to trilinear"
            return interpolate_velocity_with_halos(x, y, z, tracker, halo_info)
        end

        hw = halo_info.halo_width
        hy = halo_info.is_2d_decomposition ? halo_info.halo_width : 0
        nz_ext, nx_ext, ny_ext = size(halo_info.u_extended)

        # Extended domain lengths for the padded arrays
        Lx_ext = nx_ext * tracker.dx
        Ly_ext = ny_ext * tracker.dy

        # Grid info uses extended domain lengths since we're interpolating in extended arrays
        z_min = -tracker.Lz
        grid_info = (dx=tracker.dx, dy=tracker.dy, dz=tracker.dz,
                    Lx=Lx_ext, Ly=Ly_ext, Lz=tracker.Lz,
                    z_min=z_min, z_max=zero(T), z0=z_min + tracker.dz / 2)

        # For extended arrays, disable periodic BCs where halos handle cross-boundary data
        boundary_conditions = (periodic_x=false,
                              periodic_y=halo_info.is_2d_decomposition ? false : halo_info.periodic_y,
                              periodic_z=false)

        # Convert global positions to extended array coordinates
        # Step 1: Apply periodic wrapping to global coordinates
        x_periodic = halo_info.periodic_x ? tracker.x0 + mod(x - tracker.x0, tracker.Lx) : x
        y_periodic = halo_info.periodic_y ? tracker.y0 + mod(y - tracker.y0, tracker.Ly) : y

        # Step 2: Compute local domain start positions (using HaloExchange functions)
        x_start = tracker.x0 + compute_start_index(halo_info.nx_global, halo_info.px, halo_info.rank_x) * tracker.dx
        y_start = tracker.y0 + compute_start_index(halo_info.ny_global, halo_info.py, halo_info.rank_y) * tracker.dy

        # Step 3: Convert to local coordinates relative to domain start
        # Then add hw/hy offsets to shift into extended array coordinate system
        # Extended array: [left_halo(hw) | local_data | right_halo(hw)]
        # Index 1 corresponds to position -hw*dx relative to local domain start
        x_local = x_periodic - x_start + hw * tracker.dx
        y_local = y_periodic - y_start + hy * tracker.dy

        u_interp, v_interp, w_interp = interpolate_velocity_advanced(
            x_local, y_local, z,
            halo_info.u_extended, halo_info.v_extended, halo_info.w_extended,
            grid_info, boundary_conditions,
            method
        )

        # For 2D advection, set w to zero
        if !tracker.config.use_3d_advection
            w_interp = zero(T)
        end

        return u_interp, v_interp, w_interp
    else
        # Use original halo interpolation for trilinear
        return interpolate_velocity_with_halos(x, y, z, tracker, halo_info)
    end
end

"""
Local velocity interpolation (fallback for compatibility).
"""
function interpolate_velocity_local(x::T, y::T, z::T, 
                                  tracker::ParticleTracker{T}) where T
    
    # Handle periodic boundaries (shift to domain-relative coordinates)
    x_rel = tracker.config.periodic_x ? mod(x - tracker.x0, tracker.Lx) : x - tracker.x0
    y_rel = tracker.config.periodic_y ? mod(y - tracker.y0, tracker.Ly) : y - tracker.y0
    z_min = -tracker.Lz
    z0 = z_min + tracker.dz / 2
    z_max = zero(T)
    z_clamped = clamp(z, z0, z_max)
    
    # Convert to grid indices (0-based for interpolation)
    fx = x_rel / tracker.dx
    fy = y_rel / tracker.dy  
    fz = (z_clamped - z0) / tracker.dz
    
    # Get integer and fractional parts
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    
    rx = fx - ix
    ry = fy - iy
    rz = fz - iz
    
    # Handle boundary indices with proper periodic wrapping
    if tracker.config.periodic_x
        ix1 = mod(ix, tracker.nx) + 1
        ix2 = mod(ix + 1, tracker.nx) + 1
    else
        ix1 = max(1, min(tracker.nx, ix + 1))
        ix2 = max(1, min(tracker.nx, ix + 2))
    end
    
    if tracker.config.periodic_y
        iy1 = mod(iy, tracker.ny) + 1
        iy2 = mod(iy + 1, tracker.ny) + 1
    else
        iy1 = max(1, min(tracker.ny, iy + 1))
        iy2 = max(1, min(tracker.ny, iy + 2))
    end
    
    # Z is never periodic
    iz1 = max(1, min(tracker.nz, iz + 1))
    iz2 = max(1, min(tracker.nz, iz + 2))
    
    # Trilinear interpolation
    # Bottom face (z1)
    u_z1_y1 = (1-rx) * tracker.u_field[iz1, ix1, iy1] + rx * tracker.u_field[iz1, ix2, iy1]
    u_z1_y2 = (1-rx) * tracker.u_field[iz1, ix1, iy2] + rx * tracker.u_field[iz1, ix2, iy2]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * tracker.v_field[iz1, ix1, iy1] + rx * tracker.v_field[iz1, ix2, iy1]
    v_z1_y2 = (1-rx) * tracker.v_field[iz1, ix1, iy2] + rx * tracker.v_field[iz1, ix2, iy2]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * tracker.w_field[iz1, ix1, iy1] + rx * tracker.w_field[iz1, ix2, iy1]
    w_z1_y2 = (1-rx) * tracker.w_field[iz1, ix1, iy2] + rx * tracker.w_field[iz1, ix2, iy2]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)
    u_z2_y1 = (1-rx) * tracker.u_field[iz2, ix1, iy1] + rx * tracker.u_field[iz2, ix2, iy1]
    u_z2_y2 = (1-rx) * tracker.u_field[iz2, ix1, iy2] + rx * tracker.u_field[iz2, ix2, iy2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * tracker.v_field[iz2, ix1, iy1] + rx * tracker.v_field[iz2, ix2, iy1]
    v_z2_y2 = (1-rx) * tracker.v_field[iz2, ix1, iy2] + rx * tracker.v_field[iz2, ix2, iy2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * tracker.w_field[iz2, ix1, iy1] + rx * tracker.w_field[iz2, ix2, iy1]
    w_z2_y2 = (1-rx) * tracker.w_field[iz2, ix1, iy2] + rx * tracker.w_field[iz2, ix2, iy2]
    w_z2 = (1-ry) * w_z2_y1 + ry * w_z2_y2
    
    # Final interpolation in z
    u_interp = (1-rz) * u_z1 + rz * u_z2
    v_interp = (1-rz) * v_z1 + rz * v_z2
    w_interp = (1-rz) * w_z1 + rz * w_z2
    
    # For 2D advection, set w to zero
    if !tracker.config.use_3d_advection
        w_interp = 0.0
    end
    
    return u_interp, v_interp, w_interp
end

# Integration methods

"""
    advect_euler!(tracker, dt)

Advect particles using simple Euler integration method: x = x + dt*u

This implements the basic Euler timestep:
- x_new = x_old + dt * u
- y_new = y_old + dt * v  
- z_new = z_old + dt * w

where (u,v,w) is the interpolated velocity at the current particle position.
"""
function advect_euler!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        x, y, z = particles.x[i], particles.y[i], particles.z[i]
        
        u, v, w = interpolate_velocity_at_position(x, y, z, tracker)
        
        # Euler timestep: x = x + dt*u
        particles.x[i] = x + dt * u
        particles.y[i] = y + dt * v
        particles.z[i] = z + dt * w
        
        particles.u[i] = u
        particles.v[i] = v
        particles.w[i] = w
    end
end

function advect_rk2!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        x0, y0, z0 = particles.x[i], particles.y[i], particles.z[i]
        
        # First stage
        u1, v1, w1 = interpolate_velocity_at_position(x0, y0, z0, tracker)
        
        # Midpoint
        x_mid = x0 + 0.5 * dt * u1
        y_mid = y0 + 0.5 * dt * v1
        z_mid = z0 + 0.5 * dt * w1
        
        # Second stage
        u2, v2, w2 = interpolate_velocity_at_position(x_mid, y_mid, z_mid, tracker)
        
        # Final update
        particles.x[i] = x0 + dt * u2
        particles.y[i] = y0 + dt * v2
        particles.z[i] = z0 + dt * w2
        
        particles.u[i] = u2
        particles.v[i] = v2
        particles.w[i] = w2
    end
end

function advect_rk4!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        x0, y0, z0 = particles.x[i], particles.y[i], particles.z[i]
        
        # Stage 1
        u1, v1, w1 = interpolate_velocity_at_position(x0, y0, z0, tracker)
        
        # Stage 2
        x_temp = x0 + 0.5 * dt * u1
        y_temp = y0 + 0.5 * dt * v1
        z_temp = z0 + 0.5 * dt * w1
        u2, v2, w2 = interpolate_velocity_at_position(x_temp, y_temp, z_temp, tracker)
        
        # Stage 3
        x_temp = x0 + 0.5 * dt * u2
        y_temp = y0 + 0.5 * dt * v2
        z_temp = z0 + 0.5 * dt * w2
        u3, v3, w3 = interpolate_velocity_at_position(x_temp, y_temp, z_temp, tracker)
        
        # Stage 4
        x_temp = x0 + dt * u3
        y_temp = y0 + dt * v3
        z_temp = z0 + dt * w3
        u4, v4, w4 = interpolate_velocity_at_position(x_temp, y_temp, z_temp, tracker)
        
        # Final update
        particles.x[i] = x0 + dt * (u1 + 2*u2 + 2*u3 + u4) / 6
        particles.y[i] = y0 + dt * (v1 + 2*v2 + 2*v3 + v4) / 6
        particles.z[i] = z0 + dt * (w1 + 2*w2 + 2*w3 + w4) / 6
        
        particles.u[i] = (u1 + 2*u2 + 2*u3 + u4) / 6
        particles.v[i] = (v1 + 2*v2 + 2*v3 + v4) / 6
        particles.w[i] = (w1 + 2*w2 + 2*w3 + w4) / 6
    end
end

"""
    migrate_particles!(tracker)

Handle particles that have moved outside local domain (parallel only).
"""
function migrate_particles!(tracker::ParticleTracker{T}) where T
    
    if !tracker.is_parallel || tracker.comm === nothing
        return tracker
    end
    
    if Base.find_package("MPI") === nothing
        @warn "MPI not available; cannot migrate particles"
        return tracker
    end
    try
        M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        particles = tracker.particles
        local_domain = tracker.local_domain
        
        # Clear send buffers
        for i in 1:tracker.nprocs
            empty!(tracker.send_buffers[i])
            empty!(tracker.send_buffers_id[i])
        end
        
        # Find particles that need migration
        keep_indices = Int[]
        
        use_2d = tracker.local_domain !== nothing &&
                 hasproperty(tracker.local_domain, :py) &&
                 tracker.local_domain.py > 1

        for i in 1:particles.np
            x = particles.x[i]
            y = particles.y[i]
            target_rank = use_2d ? find_target_rank(x, y, tracker) : find_target_rank(x, tracker)
            
            if target_rank == tracker.rank
                push!(keep_indices, i)
            else
                # Package particle for sending
                particle_data = [particles.x[i], particles.y[i], particles.z[i],
                               particles.u[i], particles.v[i], particles.w[i]]
                append!(tracker.send_buffers[target_rank + 1], particle_data)
                push!(tracker.send_buffers_id[target_rank + 1], particles.id[i])
            end
        end
        
        # Keep only local particles
        particles.x = particles.x[keep_indices]
        particles.y = particles.y[keep_indices]
        particles.z = particles.z[keep_indices]
        particles.id = particles.id[keep_indices]
        particles.u = particles.u[keep_indices]
        particles.v = particles.v[keep_indices]
        particles.w = particles.w[keep_indices]
        particles.np = length(keep_indices)
        
        # Exchange particles between ranks
        exchange_particles!(tracker)
        
    catch e
        @warn "Particle migration failed: $e"
    end
    
    return tracker
end

"""
Find which rank should own a particle at position (x, y).

Uses the same domain decomposition logic as compute_local_domain to ensure consistency.
Handles uneven division where first `remainder` ranks get one extra grid point.
"""
function find_target_rank(x::T, y::T, tracker::ParticleTracker{T}) where T
    x_rel = tracker.config.periodic_x ? mod(x - tracker.x0, tracker.Lx) : x - tracker.x0
    y_rel = tracker.config.periodic_y ? mod(y - tracker.y0, tracker.Ly) : y - tracker.y0

    # Grid parameters
    nx = tracker.nx
    ny = tracker.ny
    dx = tracker.dx
    dy = tracker.dy

    # Convert physical position to grid index (0-based)
    ix = floor(Int, x_rel / dx)
    iy = floor(Int, y_rel / dy)
    ix = clamp(ix, 0, nx - 1)
    iy = clamp(iy, 0, ny - 1)

    # Determine process grid
    if tracker.local_domain !== nothing && hasproperty(tracker.local_domain, :px)
        px = tracker.local_domain.px
        py = tracker.local_domain.py
    else
        px, py = compute_process_grid(tracker.nprocs)
    end

    # Map index to rank coordinate in each dimension
    rank_x = _rank_for_index(ix, nx, px)
    rank_y = _rank_for_index(iy, ny, py)

    return rank_y * px + rank_x
end

"""
Find which rank should own a particle at position x (1D decomposition fallback).
"""
function find_target_rank(x::T, tracker::ParticleTracker{T}) where T
    # Fallback for 1D decomposition in x
    x_rel = tracker.config.periodic_x ? mod(x - tracker.x0, tracker.Lx) : x - tracker.x0
    nx = tracker.nx
    dx = tracker.dx
    ix = floor(Int, x_rel / dx)
    ix = clamp(ix, 0, nx - 1)

    nprocs = tracker.nprocs
    return _rank_for_index(ix, nx, nprocs)
end

function _rank_for_index(idx::Int, n_global::Int, nprocs::Int)
    base = n_global ÷ nprocs
    remainder = n_global % nprocs
    if remainder == 0
        return min(nprocs - 1, idx ÷ base)
    end

    large_domain_end = remainder * (base + 1)
    if idx < large_domain_end
        return idx ÷ (base + 1)
    else
        return remainder + (idx - large_domain_end) ÷ base
    end
end

"""
Exchange particles between ranks using MPI with non-blocking communication.

Uses Isend/Irecv to avoid deadlock when all ranks try to communicate simultaneously.
"""
function exchange_particles!(tracker::ParticleTracker{T}) where T
    if Base.find_package("MPI") === nothing
        @warn "MPI not available; cannot exchange particles"
        return
    end
    try
        M = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        comm = tracker.comm
        nprocs = tracker.nprocs
        rank = tracker.rank

        # Send/receive particle counts using Alltoall
        # Each rank sends 1 element (particle count) to every other rank
        send_counts = [length(tracker.send_buffers[i]) ÷ 6 for i in 1:nprocs]
        send_counts_id = [length(tracker.send_buffers_id[i]) for i in 1:nprocs]
        @assert send_counts == send_counts_id "Particle data and id buffers are inconsistent"
        recv_counts = M.Alltoall(send_counts, 1, comm)

        # Post all non-blocking receives first
        recv_reqs = M.Request[]
        for other_rank in 0:nprocs-1
            if other_rank == rank
                continue
            end
            if recv_counts[other_rank + 1] > 0
                recv_data = Vector{T}(undef, recv_counts[other_rank + 1] * 6)
                tracker.recv_buffers[other_rank + 1] = recv_data
                recv_ids = Vector{Int}(undef, recv_counts[other_rank + 1])
                tracker.recv_buffers_id[other_rank + 1] = recv_ids

                req = M.Irecv!(recv_data, other_rank, other_rank, comm)  # Tag = sender rank
                req_id = M.Irecv!(recv_ids, other_rank, other_rank + nprocs, comm)
                push!(recv_reqs, req)
                push!(recv_reqs, req_id)
            end
        end

        # Post all non-blocking sends
        send_reqs = M.Request[]
        for other_rank in 0:nprocs-1
            if other_rank == rank
                continue
            end
            if !isempty(tracker.send_buffers[other_rank + 1])
                req = M.Isend(tracker.send_buffers[other_rank + 1], other_rank, rank, comm)  # Tag = my rank
                push!(send_reqs, req)
            end
            if !isempty(tracker.send_buffers_id[other_rank + 1])
                req_id = M.Isend(tracker.send_buffers_id[other_rank + 1], other_rank, rank + nprocs, comm)
                push!(send_reqs, req_id)
            end
        end

        # Wait for all receives to complete
        if !isempty(recv_reqs)
            M.Waitall(recv_reqs)
        end

        # Add received particles
        add_received_particles!(tracker)

        # Wait for all sends to complete before clearing buffers
        if !isempty(send_reqs)
            M.Waitall(send_reqs)
        end

        # Synchronize to ensure all particle exchanges are complete
        M.Barrier(comm)

    catch e
        @warn "Particle exchange failed: $e"
    end
end

"""
Add received particles to local collection.
"""
function add_received_particles!(tracker::ParticleTracker{T}) where T
    particles = tracker.particles

    for r in 1:tracker.nprocs
        rank_data = tracker.recv_buffers[r]
        rank_ids = tracker.recv_buffers_id[r]
        if !isempty(rank_data)
            n_new = length(rank_data) ÷ 6
            @assert length(rank_ids) == n_new "Received particle ids do not match data length"

            for p in 1:n_new
                idx = (p - 1) * 6
                push!(particles.x, rank_data[idx + 1])
                push!(particles.y, rank_data[idx + 2])
                push!(particles.z, rank_data[idx + 3])
                push!(particles.u, rank_data[idx + 4])
                push!(particles.v, rank_data[idx + 5])
                push!(particles.w, rank_data[idx + 6])
                push!(particles.id, rank_ids[p])
            end

            particles.np += n_new
        end
    end

    # Clear receive buffers
    for r in 1:tracker.nprocs
        empty!(tracker.recv_buffers[r])
        empty!(tracker.recv_buffers_id[r])
    end
end

"""
Apply boundary conditions to particles.
"""
function apply_boundary_conditions!(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    config = tracker.config
    
    @inbounds for i in 1:particles.np
        # Horizontal boundaries
        if config.periodic_x
            particles.x[i] = tracker.x0 + mod(particles.x[i] - tracker.x0, tracker.Lx)
        else
            particles.x[i] = clamp(particles.x[i], tracker.x0, tracker.x0 + tracker.Lx)
        end
        
        if config.periodic_y
            particles.y[i] = tracker.y0 + mod(particles.y[i] - tracker.y0, tracker.Ly)
        else
            particles.y[i] = clamp(particles.y[i], tracker.y0, tracker.y0 + tracker.Ly)
        end
        
        # Vertical boundaries (z ∈ [-Lz, 0])
        if config.reflect_z
            if particles.z[i] > 0
                particles.z[i] = -particles.z[i]
                particles.w[i] = -particles.w[i]
            elseif particles.z[i] < -tracker.Lz
                particles.z[i] = -2*tracker.Lz - particles.z[i]
                particles.w[i] = -particles.w[i]
            end
        else
            particles.z[i] = clamp(particles.z[i], -tracker.Lz, 0)
        end
    end
end

"""
Save current particle state to trajectory history.

If auto_split_files is enabled and max_save_points is reached, automatically
creates a new file and resets the trajectory history to continue saving.
"""
function save_particle_state!(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    
    # Check if we've reached max_save_points
    if length(particles.time_history) >= tracker.config.max_save_points
        if tracker.config.auto_split_files && !isempty(tracker.base_output_filename)
            # Save current trajectory segment to file
            split_and_save_trajectory_segment!(tracker)
            
            # Reset trajectory history for next segment
            empty!(particles.x_history)
            empty!(particles.y_history) 
            empty!(particles.z_history)
            empty!(particles.id_history)
            empty!(particles.time_history)
            
            tracker.output_file_sequence += 1
        else
            # Traditional behavior: stop saving when max reached
            return tracker
        end
    end
    
    # Save current state to history
    push!(particles.x_history, copy(particles.x))
    push!(particles.y_history, copy(particles.y))
    push!(particles.z_history, copy(particles.z))
    push!(particles.id_history, copy(particles.id))
    push!(particles.time_history, particles.time)
    
    tracker.save_counter += 1
    tracker.last_save_time = particles.time
    
    return tracker
end

"""
    split_and_save_trajectory_segment!(tracker)

Save current trajectory history to a sequentially numbered file and prepare for next segment.
Used internally by save_particle_state! when auto_split_files is enabled.
"""
function split_and_save_trajectory_segment!(tracker::ParticleTracker{T}) where T
    if isempty(tracker.particles.time_history)
        return tracker
    end
    
    # Create filename with sequence number
    base_name = tracker.base_output_filename
    if tracker.output_file_sequence == 0
        filename = "$(base_name).nc"
    else
        filename = "$(base_name)_part$(tracker.output_file_sequence).nc"
    end
    
    # Calculate time range for this segment
    start_time = tracker.particles.time_history[1]
    end_time = tracker.particles.time_history[end]
    n_points = length(tracker.particles.time_history)
    
    println("Auto-splitting: Saving trajectory segment to $filename")
    println("  Time range: $(round(start_time, digits=4)) - $(round(end_time, digits=4))")
    println("  Points: $n_points / $(tracker.config.max_save_points) (max)")
    
    # Create metadata for this segment
    metadata = Dict(
        "segment_number" => tracker.output_file_sequence,
        "start_time" => start_time,
        "end_time" => end_time,
        "points_in_segment" => n_points,
        "max_points_per_file" => tracker.config.max_save_points,
        "auto_split_enabled" => true
    )
    
    # Save this segment using existing I/O function
    try
        # Import the I/O module function
        write_particle_trajectories(filename, tracker; metadata=metadata)
        println("  ✅ Successfully saved segment $(tracker.output_file_sequence)")
    catch e
        @warn "Failed to save trajectory segment: $e"
    end
    
    return tracker
end

"""
    should_save_particles(tracker)

Check if it's time to save particle state based on save_interval.

Particles are advected every simulation timestep (dt), but positions are only 
saved to trajectory history at save_interval intervals. This provides:
- Independent control of simulation accuracy (via dt) and output frequency (via save_interval)
- Memory management for long simulations with many particles
- Reduced I/O overhead while maintaining high simulation fidelity

Returns true when: (current_time - last_save_time) >= save_interval
"""
function should_save_particles(tracker::ParticleTracker{T}) where T
    return (tracker.particles.time - tracker.last_save_time) >= tracker.config.save_interval
end

"""
    enable_auto_file_splitting!(tracker, base_filename; max_points_per_file=1000)

Enable automatic file splitting for long particle trajectories.

When enabled, the tracker will automatically create new files when max_save_points
is reached, allowing unlimited trajectory length with manageable file sizes.

Parameters:
- tracker: ParticleTracker instance
- base_filename: Base name for output files (without .nc extension)
- max_points_per_file: Maximum trajectory points per file (default: 1000)

Files created:
- "base_filename.nc" (first segment)
- "base_filename_part1.nc" (second segment)
- "base_filename_part2.nc" (third segment), etc.

Example:
```julia
tracker = ParticleTracker(config, grid, parallel_config)
enable_auto_file_splitting!(tracker, "long_simulation", max_points_per_file=500)

# Run long simulation - files will be created automatically
for step in 1:10000
    advect_particles!(tracker, state, grid, dt, current_time)
end

# Final segment saved with:
finalize_trajectory_files!(tracker)
```
"""
function enable_auto_file_splitting!(tracker::ParticleTracker{T}, base_filename::String;
                                     max_points_per_file::Int=1000) where T
    # Update tracker configuration
    tracker.base_output_filename = base_filename
    tracker.auto_file_splitting = true
    tracker.output_file_sequence = 0

    # Create new config with updated max_save_points and auto_split_files
    # Use keyword arguments as ParticleConfig uses @kwdef
    old_config = tracker.config
    new_config = ParticleConfig{T}(
        x_min = old_config.x_min,
        x_max = old_config.x_max,
        y_min = old_config.y_min,
        y_max = old_config.y_max,
        z_level = old_config.z_level,
        nx_particles = old_config.nx_particles,
        ny_particles = old_config.ny_particles,
        use_ybj_w = old_config.use_ybj_w,
        use_3d_advection = old_config.use_3d_advection,
        particle_advec_time = old_config.particle_advec_time,
        integration_method = old_config.integration_method,
        interpolation_method = old_config.interpolation_method,
        periodic_x = old_config.periodic_x,
        periodic_y = old_config.periodic_y,
        reflect_z = old_config.reflect_z,
        save_interval = old_config.save_interval,
        max_save_points = max_points_per_file,
        auto_split_files = true
    )

    tracker.config = new_config

    println("✅ Auto file splitting enabled:")
    println("  Base filename: $base_filename")
    println("  Max points per file: $max_points_per_file")
    println("  Files will be created: $(base_filename).nc, $(base_filename)_part1.nc, ...")

    return tracker
end

"""
    finalize_trajectory_files!(tracker; final_metadata=Dict())

Save final trajectory segment for auto-splitting trackers.

Call this at the end of long simulations to ensure the final trajectory 
segment is saved to file.
"""
function finalize_trajectory_files!(tracker::ParticleTracker; final_metadata::Dict=Dict())
    if !tracker.config.auto_split_files || isempty(tracker.particles.time_history)
        return tracker
    end
    
    println("Finalizing trajectory files...")
    
    # Save final segment
    split_and_save_trajectory_segment!(tracker)
    
    total_files = tracker.output_file_sequence + 1
    total_points = tracker.save_counter
    
    println("✅ Trajectory finalization complete:")
    println("  Total files created: $total_files")
    println("  Total trajectory points: $total_points")
    println("  Average points per file: $(round(total_points/total_files, digits=1))")
    
    return tracker
end

end # module UnifiedParticleAdvection

using .UnifiedParticleAdvection
