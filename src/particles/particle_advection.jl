"""
Unified particle advection module for QG-YBJ simulations.

This module provides Lagrangian particle tracking that automatically handles
both serial and parallel execution. Particles are advected using the TOTAL 
velocity field (QG + wave velocities) with options for vertical velocity from either 
QG omega equation or YBJ formulation.

Total velocity field includes:
- QG velocities: u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x
- Wave velocities: u_wave, v_wave from Stokes drift and wave corrections
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

# Bind names from parent module (QGYBJ) without using/import
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
       # 3D particle distributions
       ParticleConfig3D, ParticleDistribution, create_particle_config_3d,
       initialize_particles_3d!, UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM,
       create_uniform_3d_grid, create_layered_distribution, create_random_3d_distribution, create_custom_distribution

# Include halo exchange for cross-domain interpolation
include("halo_exchange.jl")
using .HaloExchange

# Include advanced interpolation schemes
include("interpolation_schemes.jl")
using .InterpolationSchemes

# Include 3D particle configuration
include("particle_config.jl")
using .EnhancedParticleConfig

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
    # Spatial domain for particle initialization
    x_min::T = 0.0
    x_max::T = 2π
    y_min::T = 0.0  
    y_max::T = 2π
    z_level::T = π/2  # Constant z-level for initialization
    
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
    particles_in_box(z_level; x_min, x_max, y_min, y_max, nx, ny, kwargs...)

Create particles uniformly distributed in a 2D rectangular box at a fixed z-level.

# Arguments
- `z_level`: Vertical level where particles are placed
- `x_min, x_max`: Horizontal x-domain bounds (default: 0 to 2π)
- `y_min, y_max`: Horizontal y-domain bounds (default: 0 to 2π)
- `nx, ny`: Number of particles in x and y directions (default: 10 each)

# Example
```julia
# 100 particles in a box [0,2π] × [0,2π] at z = π/2
config = particles_in_box(π/2; nx=10, ny=10)

# 64 particles in a smaller region
config = particles_in_box(1.0; x_min=π/2, x_max=3π/2, y_min=π/2, y_max=3π/2, nx=8, ny=8)

# With delayed release
config = particles_in_box(π/2; nx=10, ny=10, particle_advec_time=0.5)
```
"""
function particles_in_box(z_level::T;
                          x_min::T=T(0), x_max::T=T(2π),
                          y_min::T=T(0), y_max::T=T(2π),
                          nx::Int=10, ny::Int=10,
                          kwargs...) where T<:AbstractFloat
    return ParticleConfig{T}(;
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        z_level=z_level, nx_particles=nx, ny_particles=ny,
        kwargs...)
end

# Convenience method for non-typed call
function particles_in_box(z_level::Real;
                          x_min::Real=0.0, x_max::Real=2π,
                          y_min::Real=0.0, y_max::Real=2π,
                          nx::Int=10, ny::Int=10,
                          kwargs...)
    T = Float64
    return particles_in_box(T(z_level);
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        nx=nx, ny=ny, kwargs...)
end

# Legacy alias for backwards compatibility
create_particle_config(::Type{T}=Float64; kwargs...) where T = ParticleConfig{T}(; kwargs...)

"""
Particle state including positions, velocities, and trajectory history.
"""
mutable struct ParticleState{T<:AbstractFloat}
    # Current state
    x::Vector{T}
    y::Vector{T} 
    z::Vector{T}
    u::Vector{T}
    v::Vector{T}
    w::Vector{T}
    time::T
    np::Int
    
    # Trajectory history
    x_history::Vector{Vector{T}}
    y_history::Vector{Vector{T}}
    z_history::Vector{Vector{T}}
    time_history::Vector{T}
    
    function ParticleState{T}(np::Int) where T
        new{T}(
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            zero(T), np,
            Vector{T}[], Vector{T}[], Vector{T}[], T[]
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
                M = Base.require(:MPI)
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
        
        # Set up domain decomposition if parallel
        local_domain = is_parallel ? compute_local_domain(grid, rank, nprocs) : nothing

        # Initialize buffers
        send_buffers = [T[] for _ in 1:nprocs]
        recv_buffers = [T[] for _ in 1:nprocs]

        # Velocity field workspace - use LOCAL size in parallel mode
        if is_parallel && local_domain !== nothing
            nx_local = local_domain.nx_local
            u_field = zeros(T, nx_local, grid.ny, grid.nz)
            v_field = zeros(T, nx_local, grid.ny, grid.nz)
            w_field = zeros(T, nx_local, grid.ny, grid.nz)
        else
            u_field = zeros(T, grid.nx, grid.ny, grid.nz)
            v_field = zeros(T, grid.nx, grid.ny, grid.nz)
            w_field = zeros(T, grid.nx, grid.ny, grid.nz)
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
            grid.Lx/grid.nx, grid.Ly/grid.ny, grid.Lz/grid.nz,
            u_field, v_field, w_field, plans,
            comm, rank, nprocs, is_parallel,
            local_domain,
            send_buffers, recv_buffers,
            halo_info,
            0, zero(T), rank == 0, true,
            0, "", false  # output_file_sequence, base_output_filename, auto_file_splitting
        )
    end
end

ParticleTracker(config::ParticleConfig{T}, grid::Grid, parallel_config=nothing) where T = ParticleTracker{T}(config, grid, parallel_config)

"""
    setup_halo_exchange_for_grid(grid, rank, nprocs, comm, T; local_dims=nothing)

Helper function to set up halo exchange system.

# Arguments
- `local_dims`: Tuple (nx_local, ny_local, nz_local) for 2D pencil decomposition.
                If nothing, assumes 1D decomposition in x.
"""
function setup_halo_exchange_for_grid(grid::Grid, rank::Int, nprocs::Int, comm, ::Type{T};
                                     local_dims::Union{Nothing,Tuple{Int,Int,Int}}=nothing) where T
    try
        return HaloInfo{T}(grid, rank, nprocs, comm; local_dims=local_dims)
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
        M = Base.require(:MPI)
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
    compute_local_domain(grid, rank, nprocs)

Compute local domain bounds for MPI rank (1D decomposition in x).
"""
function compute_local_domain(grid::Grid, rank::Int, nprocs::Int)
    # Simple 1D decomposition in x-direction
    nx_local = grid.nx ÷ nprocs
    remainder = grid.nx % nprocs
    
    if rank < remainder
        nx_local += 1
        x_start = rank * nx_local
    else
        x_start = remainder * (nx_local + 1) + (rank - remainder) * nx_local
    end
    
    x_end = x_start + nx_local - 1
    
    # Convert to physical coordinates  
    dx = grid.Lx / grid.nx
    x_start_phys = x_start * dx
    x_end_phys = (x_end + 1) * dx
    
    return (x_start=x_start_phys, x_end=x_end_phys,
            y_start=0.0, y_end=grid.Ly,
            z_start=0.0, z_end=grid.Lz,
            nx_local=nx_local)
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
    
    if x_min >= x_max
        # No particles in this domain
        resize!(tracker.particles.x, 0)
        resize!(tracker.particles.y, 0)
        resize!(tracker.particles.z, 0)
        resize!(tracker.particles.u, 0)
        resize!(tracker.particles.v, 0)
        resize!(tracker.particles.w, 0)
        tracker.particles.np = 0
        return tracker
    end
    
    # Calculate local particle distribution
    x_frac = (x_max - x_min) / (config.x_max - config.x_min)
    nx_local = max(1, round(Int, config.nx_particles * x_frac))
    
    # Create local particle grid
    x_range = range(x_min, x_max, length=nx_local+1)[1:end-1]
    y_range = range(config.y_min, config.y_max, length=config.ny_particles+1)[1:end-1]
    
    n_local = nx_local * config.ny_particles
    
    # Resize arrays to actual local particle count
    resize!(tracker.particles.x, n_local)
    resize!(tracker.particles.y, n_local)
    resize!(tracker.particles.z, n_local)
    resize!(tracker.particles.u, n_local)
    resize!(tracker.particles.v, n_local)
    resize!(tracker.particles.w, n_local)
    
    # Initialize local particles
    idx = 1
    for y in y_range, x in x_range
        tracker.particles.x[idx] = x
        tracker.particles.y[idx] = y
        tracker.particles.z[idx] = config.z_level
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
    advect_particles!(tracker, state, grid, dt, current_time=nothing)

Advect particles using unified serial/parallel interface.
Respects the particle_advec_time setting - particles remain stationary until this time.

Parameters:
- tracker: ParticleTracker instance
- state: Current fluid state
- grid: Grid information  
- dt: Time step
- current_time: Current simulation time (if not provided, uses tracker's internal time)
"""
function advect_particles!(tracker::ParticleTracker{T}, 
                          state::State, grid::Grid, dt::T, current_time=nothing) where T
    
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
    # Update velocity fields
    update_velocity_fields!(tracker, state, grid)
    
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
    update_velocity_fields!(tracker, state, grid)

Update TOTAL velocity fields from fluid state (QG + wave velocities) and exchange halos if parallel.
Computes the complete velocity field needed for proper QG-YBJ particle advection.

Handles 2D pencil decomposition by getting actual local dimensions from State arrays.
"""
function update_velocity_fields!(tracker::ParticleTracker{T},
                                state::State, grid::Grid) where T
    # Compute TOTAL velocities (QG + wave) with chosen vertical velocity formulation
    compute_total_velocities!(state, grid;
                              plans=tracker.plans,
                              compute_w=true,
                              use_ybj_w=tracker.config.use_ybj_w)

    # Get actual local dimensions from State arrays
    # This handles both serial (full grid) and parallel (2D pencil decomposition)
    u_data = parent(state.u)
    v_data = parent(state.v)
    w_data = parent(state.w)
    local_dims = size(u_data)
    nx_local, ny_local, nz_local = local_dims

    # Resize tracker velocity fields if needed (first call or dimension change)
    if size(tracker.u_field) != local_dims
        tracker.u_field = zeros(T, nx_local, ny_local, nz_local)
        tracker.v_field = zeros(T, nx_local, ny_local, nz_local)
        tracker.w_field = zeros(T, nx_local, ny_local, nz_local)
    end

    # Copy velocity data
    tracker.u_field .= u_data
    tracker.v_field .= v_data
    tracker.w_field .= w_data

    # Lazily initialize halo exchange system with actual local dimensions
    if tracker.is_parallel && tracker.halo_info === nothing
        tracker.halo_info = setup_halo_exchange_for_grid(
            grid, tracker.rank, tracker.nprocs, tracker.comm, T;
            local_dims=local_dims
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
    grid_info = (dx=tracker.dx, dy=tracker.dy, dz=tracker.dz,
                Lx=tracker.Lx, Ly=tracker.Ly, Lz=tracker.Lz)
    
    boundary_conditions = (periodic_x=tracker.config.periodic_x,
                          periodic_y=tracker.config.periodic_y,
                          periodic_z=false)
    
    # Use advanced interpolation
    u_interp, v_interp, w_interp = interpolate_velocity_advanced(
        x, y, z,
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
"""
function interpolate_velocity_with_halos_advanced(x::T, y::T, z::T, 
                                                 tracker::ParticleTracker{T},
                                                 halo_info::HaloInfo{T}) where T
    
    # For high-order interpolation, we need extended halos
    if tracker.config.interpolation_method == TRICUBIC || tracker.config.interpolation_method == ADAPTIVE
        # Check if we have enough halo width for tricubic (needs at least 2)
        if halo_info.halo_width < 2
            @warn "Insufficient halo width for tricubic interpolation, falling back to trilinear"
            return interpolate_velocity_with_halos(x, y, z, tracker, halo_info)
        end
        
        # Use extended arrays for high-order interpolation
        grid_info = (dx=tracker.dx, dy=tracker.dy, dz=tracker.dz,
                    Lx=tracker.Lx, Ly=tracker.Ly, Lz=tracker.Lz)
        
        boundary_conditions = (periodic_x=tracker.config.periodic_x,
                              periodic_y=tracker.config.periodic_y,
                              periodic_z=false)
        
        # Adjust position for extended grid coordinates
        local_domain = tracker.local_domain
        x_local = x - local_domain.x_start
        
        u_interp, v_interp, w_interp = interpolate_velocity_advanced(
            x_local, y, z,
            halo_info.u_extended, halo_info.v_extended, halo_info.w_extended,
            grid_info, boundary_conditions,
            tracker.config.interpolation_method
        )
        
        # For 2D advection, set w to zero
        if !tracker.config.use_3d_advection
            w_interp = 0.0
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
    
    # Handle periodic boundaries
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x
    y_periodic = tracker.config.periodic_y ? mod(y, tracker.Ly) : y
    z_clamped = clamp(z, 0, tracker.Lz)
    
    # Convert to grid indices (0-based for interpolation)
    fx = x_periodic / tracker.dx
    fy = y_periodic / tracker.dy  
    fz = z_clamped / tracker.dz
    
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
    u_z1_y1 = (1-rx) * tracker.u_field[ix1,iy1,iz1] + rx * tracker.u_field[ix2,iy1,iz1]
    u_z1_y2 = (1-rx) * tracker.u_field[ix1,iy2,iz1] + rx * tracker.u_field[ix2,iy2,iz1]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * tracker.v_field[ix1,iy1,iz1] + rx * tracker.v_field[ix2,iy1,iz1]
    v_z1_y2 = (1-rx) * tracker.v_field[ix1,iy2,iz1] + rx * tracker.v_field[ix2,iy2,iz1]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * tracker.w_field[ix1,iy1,iz1] + rx * tracker.w_field[ix2,iy1,iz1]
    w_z1_y2 = (1-rx) * tracker.w_field[ix1,iy2,iz1] + rx * tracker.w_field[ix2,iy2,iz1]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)
    u_z2_y1 = (1-rx) * tracker.u_field[ix1,iy1,iz2] + rx * tracker.u_field[ix2,iy1,iz2]
    u_z2_y2 = (1-rx) * tracker.u_field[ix1,iy2,iz2] + rx * tracker.u_field[ix2,iy2,iz2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * tracker.v_field[ix1,iy1,iz2] + rx * tracker.v_field[ix2,iy1,iz2]
    v_z2_y2 = (1-rx) * tracker.v_field[ix1,iy2,iz2] + rx * tracker.v_field[ix2,iy2,iz2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * tracker.w_field[ix1,iy1,iz2] + rx * tracker.w_field[ix2,iy1,iz2]
    w_z2_y2 = (1-rx) * tracker.w_field[ix1,iy2,iz2] + rx * tracker.w_field[ix2,iy2,iz2]
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
        M = Base.require(:MPI)
        particles = tracker.particles
        local_domain = tracker.local_domain
        
        # Clear send buffers
        for i in 1:tracker.nprocs
            empty!(tracker.send_buffers[i])
        end
        
        # Find particles that need migration
        keep_indices = Int[]
        
        for i in 1:particles.np
            x = particles.x[i]
            target_rank = find_target_rank(x, tracker)
            
            if target_rank == tracker.rank
                push!(keep_indices, i)
            else
                # Package particle for sending
                particle_data = [particles.x[i], particles.y[i], particles.z[i],
                               particles.u[i], particles.v[i], particles.w[i]]
                append!(tracker.send_buffers[target_rank + 1], particle_data)
            end
        end
        
        # Keep only local particles
        particles.x = particles.x[keep_indices]
        particles.y = particles.y[keep_indices]
        particles.z = particles.z[keep_indices]
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
Find which rank should own a particle at position x.
"""
function find_target_rank(x::T, tracker::ParticleTracker{T}) where T
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x
    dx_rank = tracker.Lx / tracker.nprocs
    rank = min(tracker.nprocs - 1, floor(Int, x_periodic / dx_rank))
    return rank
end

"""
Exchange particles between ranks using MPI.
"""
function exchange_particles!(tracker::ParticleTracker{T}) where T
    if Base.find_package("MPI") === nothing
        @warn "MPI not available; cannot exchange particles"
        return
    end
    try
        M = Base.require(:MPI)
        comm = tracker.comm
        nprocs = tracker.nprocs
        
        # Send/receive particle counts
        send_counts = [length(tracker.send_buffers[i]) ÷ 6 for i in 1:nprocs]
        recv_counts = MPI.Alltoall(send_counts, comm)
        
        # Exchange particle data
        for other_rank in 0:nprocs-1
            if other_rank == tracker.rank
                continue
            end
            
            # Send to other_rank
            if !isempty(tracker.send_buffers[other_rank + 1])
                MPI.Send(tracker.send_buffers[other_rank + 1], other_rank, 0, comm)
            end
            
            # Receive from other_rank
            if recv_counts[other_rank + 1] > 0
                recv_data = Vector{T}(undef, recv_counts[other_rank + 1] * 6)
                MPI.Recv!(recv_data, other_rank, 0, comm)
                tracker.recv_buffers[other_rank + 1] = recv_data
            end
        end
        
        # Add received particles
        add_received_particles!(tracker)
        
    catch e
        @warn "Particle exchange failed: $e"
    end
end

"""
Add received particles to local collection.
"""
function add_received_particles!(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    
    for rank_data in tracker.recv_buffers
        if !isempty(rank_data)
            n_new = length(rank_data) ÷ 6
            
            for i in 1:n_new
                idx = (i-1) * 6
                push!(particles.x, rank_data[idx + 1])
                push!(particles.y, rank_data[idx + 2])
                push!(particles.z, rank_data[idx + 3])
                push!(particles.u, rank_data[idx + 4])
                push!(particles.v, rank_data[idx + 5])
                push!(particles.w, rank_data[idx + 6])
            end
            
            particles.np += n_new
        end
    end
    
    # Clear receive buffers
    for i in 1:tracker.nprocs
        empty!(tracker.recv_buffers[i])
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
            particles.x[i] = mod(particles.x[i], tracker.Lx)
        else
            particles.x[i] = clamp(particles.x[i], 0, tracker.Lx)
        end
        
        if config.periodic_y
            particles.y[i] = mod(particles.y[i], tracker.Ly)
        else
            particles.y[i] = clamp(particles.y[i], 0, tracker.Ly)
        end
        
        # Vertical boundaries
        if config.reflect_z
            if particles.z[i] < 0
                particles.z[i] = -particles.z[i]
                particles.w[i] = -particles.w[i]
            elseif particles.z[i] > tracker.Lz
                particles.z[i] = 2*tracker.Lz - particles.z[i]
                particles.w[i] = -particles.w[i]
            end
        else
            particles.z[i] = clamp(particles.z[i], 0, tracker.Lz)
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
function enable_auto_file_splitting!(tracker::ParticleTracker, base_filename::String; 
                                     max_points_per_file::Int=1000)
    # Update tracker configuration
    tracker.base_output_filename = base_filename
    tracker.auto_file_splitting = true
    tracker.output_file_sequence = 0
    
    # Update particle config for new max_save_points if provided
    if max_points_per_file != tracker.config.max_save_points
        # Create new config with updated max_save_points and auto_split_files
        new_config = ParticleConfig(
            tracker.config.x_min, tracker.config.x_max,
            tracker.config.y_min, tracker.config.y_max,
            tracker.config.z_min, tracker.config.z_max,
            tracker.config.nx_particles, tracker.config.ny_particles, tracker.config.nz_particles,
            tracker.config.z_level,
            
            # Copy other settings
            tracker.config.distribution_type, tracker.config.z_levels, tracker.config.particles_per_level,
            tracker.config.custom_x, tracker.config.custom_y, tracker.config.custom_z,
            tracker.config.particle_advec_time,
            tracker.config.use_ybj_w, tracker.config.use_3d_advection,
            tracker.config.integration_method, tracker.config.interpolation_method,
            tracker.config.periodic_x, tracker.config.periodic_y, tracker.config.reflect_z,
            
            # Updated I/O settings
            tracker.config.save_interval,
            max_points_per_file,      # New max_save_points
            true                      # Enable auto_split_files
        )
        
        tracker.config = new_config
    else
        # Just enable auto splitting with existing config
        tracker.config = ParticleConfig(
            tracker.config.x_min, tracker.config.x_max,
            tracker.config.y_min, tracker.config.y_max,
            tracker.config.z_min, tracker.config.z_max,
            tracker.config.nx_particles, tracker.config.ny_particles, tracker.config.nz_particles,
            tracker.config.z_level,
            
            # Copy other settings
            tracker.config.distribution_type, tracker.config.z_levels, tracker.config.particles_per_level,
            tracker.config.custom_x, tracker.config.custom_y, tracker.config.custom_z,
            tracker.config.particle_advec_time,
            tracker.config.use_ybj_w, tracker.config.use_3d_advection,
            tracker.config.integration_method, tracker.config.interpolation_method,
            tracker.config.periodic_x, tracker.config.periodic_y, tracker.config.reflect_z,
            
            # Updated I/O settings
            tracker.config.save_interval,
            tracker.config.max_save_points,
            true                      # Enable auto_split_files
        )
    end
    
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
