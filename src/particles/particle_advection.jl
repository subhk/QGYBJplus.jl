"""
Unified particle advection module for QG-YBJ+ simulations using GLM reconstruction.

This module implements Lagrangian particle tracking following the Generalized
Lagrangian Mean (GLM) framework. The physical particle position x(t) is decomposed as:

    x(t) = X(t) + ξ(X(t), t)                                          [eq. (1)]

where:
- X(t) is the Lagrangian-mean (slow) trajectory advected by QG flow
- ξ is the oscillatory wave displacement reconstructed from the YBJ+ wave field

# Operator Definitions (from PDF)

    L  (YBJ operator):   L  = ∂/∂z(f²/N² ∂/∂z)                        [eq. (4)]
    L⁺ (YBJ+ operator):  L⁺ = L - k_h²/4

Key relation: L = L⁺ + k_h²/4

# GLM Decomposition (following the PDF formulation)

1. **Mean trajectory**: The QG velocity IS the Lagrangian-mean flow:
   dX/dt = u^L_QG(X, t)                                               [eq. (2)]

   where u_QG = -∂ψ/∂y, v_QG = ∂ψ/∂x

2. **Wave velocity**: From equation (3), the instantaneous wave velocity is:
   u + iv = (LA) × e^{-ift}                                           [eq. (3)]

   where L is the YBJ operator (NOT L⁺). Since B = L⁺A:
   LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A

3. **Wave displacement**: Obtained by "dividing by frequency" [eq. (5)-(6)]:
   ξx + iξy = Re{(LA / (-if)) × e^{-ift}}                            [eq. (6)]

4. **Physical position**: The instantaneous "wiggly" trajectory:
   x(t) = X(t) + ξ(X(t), t)                                          [eq. (1)]

# Key Difference from Eulerian Approach

Since the QG velocity is the Lagrangian mean, we do NOT add:
- Wave velocity (Re(LA), Im(LA)) - this is captured by the displacement ξ
- Stokes drift - already included in the Lagrangian mean framework

The wave displacement ξ provides the inertial loops of radius ~|U|/f
riding on the mean drift X(t).

# Time stepping recipe (Section 1.4 of PDF)

1. Advect the slow mean position (Euler):                            [eq. (7)]
   X^{n+1} = X^n + Δt × u^L_QG(X^n, t^n)

2. Reconstruct wave displacement at (X^{n+1}, t^{n+1}):              [eq. (8)]
   ξ^{n+1} = Re{(LA(X^{n+1}, t^{n+1}) / (-if)) × e^{-if t^{n+1}}}

3. Output physical particle position:                                 [eq. (9)]
   x^{n+1} = X^{n+1} + ξ^{n+1}

# Features
- Automatic serial/parallel execution with MPI
- Domain decomposition and particle migration
- Euler time integration for particle advection
- Tricubic/quintic interpolation for smooth trajectories
- Delayed particle release via particle_advec_time
- Flexible I/O with trajectory saving
"""

module UnifiedParticleAdvection

# Bind names from parent module (QGYBJplus) without using/import
const _PARENT = Base.parentmodule(@__MODULE__)
const Grid = _PARENT.Grid
const State = _PARENT.State
const plan_transforms! = _PARENT.plan_transforms!
const compute_velocities! = _PARENT.compute_velocities!  # QG velocities only (Lagrangian mean)
const compute_wave_displacement! = _PARENT.compute_wave_displacement!  # Horizontal wave displacement ξ
const compute_vertical_wave_displacement! = _PARENT.compute_vertical_wave_displacement!  # Vertical wave displacement ξz
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
- Interpolation: interpolation_method for velocity interpolation
- Boundaries: periodic_x/y, reflect_z for boundary conditions
- I/O: save_interval and max_save_points for trajectory output

Advanced timing control:
- particle_advec_time=0.0: Start advecting immediately (default)
- particle_advec_time>0.0: Keep particles stationary until this time
- Useful for letting flow field develop before particle release
- Enables study of transient vs established flow patterns

Note: Time integration uses Euler method for simplicity and stability.
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

# GLM Position Decomposition
Following the GLM framework, we track both:
- Mean positions (x, y, z): The Lagrangian-mean trajectory X(t) advected by QG flow
- Wave displacements (xi_x, xi_y): The oscillatory displacement ξ from the wave field

The physical (wiggly) position is: x_phys = x + xi_x, y_phys = y + xi_y

# Wave Displacement (from PDF equations)

The wave displacement is computed from the wave velocity amplitude LA:
    ξx + iξy = Re{(LA / (-if)) × e^{-ift}}                           [eq. (6)]

where LA uses the YBJ operator L (NOT L⁺):
    L  = ∂/∂z(f²/N² ∂/∂z)                                            [eq. (4)]
    L⁺ = L - k_h²/4                                                   (YBJ+)

Since B = L⁺A and L = L⁺ + k_h²/4:
    LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A
"""
mutable struct ParticleState{T<:AbstractFloat}
    # Current mean positions (Lagrangian-mean trajectory X)
    x::Vector{T}      # Mean x position
    y::Vector{T}      # Mean y position
    z::Vector{T}      # Mean z position (no wave displacement in z for horizontal NIW)
    id::Vector{Int}

    # QG (Lagrangian-mean) velocities for advection
    u::Vector{T}      # u_QG = -∂ψ/∂y
    v::Vector{T}      # v_QG = ∂ψ/∂x
    w::Vector{T}      # w from omega equation

    # Wave displacement ξ (for computing physical position)
    xi_x::Vector{T}   # Horizontal wave displacement x-component
    xi_y::Vector{T}   # Horizontal wave displacement y-component
    xi_z::Vector{T}   # Vertical wave displacement (from equation 2.10)

    # Wave velocity amplitude LA (complex, stored as real/imag)
    # Used for computing ξ = Re{(LA/(-if)) × e^(-ift)}
    LA_real::Vector{T}  # Re(LA) at particle position
    LA_imag::Vector{T}  # Im(LA) at particle position

    # Vertical wave displacement coefficients at particle position
    # ξz = ξz_cos × cos(ft) + ξz_sin × sin(ft)
    xi_z_cos::Vector{T}  # Coefficient of cos(ft) at particle position
    xi_z_sin::Vector{T}  # Coefficient of sin(ft) at particle position

    time::T
    np::Int

    # Trajectory history (stores MEAN positions)
    x_history::Vector{Vector{T}}
    y_history::Vector{Vector{T}}
    z_history::Vector{Vector{T}}
    id_history::Vector{Vector{Int}}
    time_history::Vector{T}

    # Wave displacement history (for reconstructing wiggly trajectories)
    xi_x_history::Vector{Vector{T}}
    xi_y_history::Vector{Vector{T}}
    xi_z_history::Vector{Vector{T}}

    function ParticleState{T}(np::Int) where T
        ids = collect(1:np)
        new{T}(
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            ids,
            Vector{T}(undef, np), Vector{T}(undef, np), Vector{T}(undef, np),
            zeros(T, np), zeros(T, np), zeros(T, np),  # xi_x, xi_y, xi_z
            zeros(T, np), zeros(T, np),  # LA_real, LA_imag
            zeros(T, np), zeros(T, np),  # xi_z_cos, xi_z_sin
            zero(T), np,
            Vector{T}[], Vector{T}[], Vector{T}[], Vector{Int}[], T[],
            Vector{T}[], Vector{T}[], Vector{T}[]  # xi_x_history, xi_y_history, xi_z_history
        )
    end
end

"""
Main particle tracker that handles both serial and parallel execution.

# GLM Framework
The tracker advects mean positions X(t) using QG velocities and separately
tracks wave displacement ξ for reconstructing physical positions x = X + ξ.
"""
mutable struct ParticleTracker{T<:AbstractFloat}
    config::ParticleConfig{T}
    particles::ParticleState{T}

    # Grid information
    nx::Int; ny::Int; nz::Int
    Lx::T; Ly::T; Lz::T
    x0::T; y0::T
    dx::T; dy::T; dz::T

    # QG velocity field workspace (real space) - for mean position advection
    u_field::Array{T,3}
    v_field::Array{T,3}
    w_field::Array{T,3}

    # Wave velocity amplitude LA (complex) - for horizontal wave displacement ξ
    # LA = B + (k_h²/4)A in spectral space, transformed to physical space
    LA_real_field::Array{T,3}
    LA_imag_field::Array{T,3}

    # Vertical wave displacement coefficients - from equation (2.10)
    # ξz = ξz_cos × cos(ft) + ξz_sin × sin(ft)
    ξz_cos_field::Array{T,3}
    ξz_sin_field::Array{T,3}

    # Transform plans (for velocity computation)
    plans

    # Coriolis parameter for wave displacement computation
    f0::T

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
            LA_real_field = zeros(T, grid.nz, nx_local, ny_local)
            LA_imag_field = zeros(T, grid.nz, nx_local, ny_local)
            ξz_cos_field = zeros(T, grid.nz, nx_local, ny_local)
            ξz_sin_field = zeros(T, grid.nz, nx_local, ny_local)
        else
            u_field = zeros(T, grid.nz, grid.nx, grid.ny)
            v_field = zeros(T, grid.nz, grid.nx, grid.ny)
            w_field = zeros(T, grid.nz, grid.nx, grid.ny)
            LA_real_field = zeros(T, grid.nz, grid.nx, grid.ny)
            LA_imag_field = zeros(T, grid.nz, grid.nx, grid.ny)
            ξz_cos_field = zeros(T, grid.nz, grid.nx, grid.ny)
            ξz_sin_field = zeros(T, grid.nz, grid.nx, grid.ny)
        end

        # Set up transform plans (using unified interface)
        plans = plan_transforms!(grid, parallel_config)

        # Default Coriolis parameter (will be updated from params in advect_particles!)
        f0 = T(1e-4)

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
            u_field, v_field, w_field,
            LA_real_field, LA_imag_field,
            ξz_cos_field, ξz_sin_field,
            plans, f0,
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
        # Initialize wave displacement fields
        particles.xi_x[idx] = 0.0
        particles.xi_y[idx] = 0.0
        particles.xi_z[idx] = 0.0
        particles.LA_real[idx] = 0.0
        particles.LA_imag[idx] = 0.0
        particles.xi_z_cos[idx] = 0.0
        particles.xi_z_sin[idx] = 0.0
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
        resize!(tracker.particles.xi_x, 0)
        resize!(tracker.particles.xi_y, 0)
        resize!(tracker.particles.xi_z, 0)
        resize!(tracker.particles.LA_real, 0)
        resize!(tracker.particles.LA_imag, 0)
        resize!(tracker.particles.xi_z_cos, 0)
        resize!(tracker.particles.xi_z_sin, 0)
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
        resize!(tracker.particles.xi_x, 0)
        resize!(tracker.particles.xi_y, 0)
        resize!(tracker.particles.xi_z, 0)
        resize!(tracker.particles.LA_real, 0)
        resize!(tracker.particles.LA_imag, 0)
        resize!(tracker.particles.xi_z_cos, 0)
        resize!(tracker.particles.xi_z_sin, 0)
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
    resize!(tracker.particles.xi_x, n_local)
    resize!(tracker.particles.xi_y, n_local)
    resize!(tracker.particles.xi_z, n_local)
    resize!(tracker.particles.LA_real, n_local)
    resize!(tracker.particles.LA_imag, n_local)
    resize!(tracker.particles.xi_z_cos, n_local)
    resize!(tracker.particles.xi_z_sin, n_local)

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
        tracker.particles.xi_x[idx] = 0.0
        tracker.particles.xi_y[idx] = 0.0
        tracker.particles.xi_z[idx] = 0.0
        tracker.particles.LA_real[idx] = 0.0
        tracker.particles.LA_imag[idx] = 0.0
        tracker.particles.xi_z_cos[idx] = 0.0
        tracker.particles.xi_z_sin[idx] = 0.0
        idx += 1
    end
    
    tracker.particles.time = 0.0
    tracker.particles.np = n_local
    
    return tracker
end

"""
    advect_particles!(tracker, state, grid, dt, current_time=nothing; params=nothing, N2_profile=nothing)

Advect particles using GLM (Generalized Lagrangian Mean) framework.

# GLM Algorithm (following PDF Section 1.4)
1. Advect mean position X using QG velocity (Lagrangian-mean flow):
   X^{n+1} = X^n + Δt × u^L_QG(X^n, t^n)

2. Reconstruct wave displacement at (X^{n+1}, t^{n+1}):
   ξ^{n+1} = Re{(LA(X^{n+1}, t^{n+1}) / (-if)) × e^(-if t^{n+1})}

3. Physical position (available via particles.x + particles.xi_x, etc.):
   x^{n+1} = X^{n+1} + ξ^{n+1}

# Parameters
- tracker: ParticleTracker instance
- state: Current fluid state (contains B, A fields for wave displacement)
- grid: Grid information
- dt: Time step
- current_time: Current simulation time (required for wave phase computation)
- params: Model parameters (QGParams). Required for f₀.
- N2_profile: Optional N²(z) profile for nonuniform stratification.

# Note
The stored positions (particles.x, particles.y, particles.z) are the MEAN positions X.
The wave displacements (particles.xi_x, particles.xi_y) provide the oscillatory correction.
The physical (wiggly) trajectory is x = X + ξ.
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

    # ============================================================
    # Step 1: Update QG velocity fields (Lagrangian-mean flow only)
    # ============================================================
    update_velocity_fields!(tracker, state, grid; params=params, N2_profile=N2_profile)

    # ============================================================
    # Step 2: Advect MEAN positions using QG velocities (Euler)
    # ============================================================
    advect_euler!(tracker, dt)

    # Handle particle migration in parallel
    if tracker.is_parallel
        migrate_particles!(tracker)
    end

    # Apply boundary conditions to mean positions
    apply_boundary_conditions!(tracker)

    # Update time (use simulation time if provided)
    new_time = if current_time !== nothing
        T(current_time) + dt
    else
        tracker.particles.time + dt
    end
    tracker.particles.time = new_time

    # ============================================================
    # Step 3: Compute wave displacement at new positions
    # ============================================================
    # Update wave amplitude fields LA = B + (k_h²/4)A and vertical displacement coefficients
    update_wave_fields!(tracker, state, grid; params=params, N2_profile=N2_profile)

    # Compute wave displacement ξ = Re{(LA/(-if)) × e^(-ift)} at new time
    compute_particle_wave_displacement!(tracker, new_time)

    # Save state if needed (saves mean positions AND wave displacement)
    if should_save_particles(tracker)
        save_particle_state!(tracker)
    end

    return tracker
end

"""
    update_velocity_fields!(tracker, state, grid; params=nothing, N2_profile=nothing)

Update QG (Lagrangian-mean) velocity fields from fluid state and exchange halos if parallel.

# GLM Framework
In the Generalized Lagrangian Mean framework, the QG velocity IS the Lagrangian-mean flow.
Particles are advected by QG velocities ONLY:
    dX/dt = u^L_QG(X, t)

The wave contribution appears as a displacement ξ, not as an additional velocity.
This is computed separately by `update_wave_displacement!`.

# Arguments
- `params`: Model parameters (QGParams). Required for vertical velocity computation.
- `N2_profile`: Optional N²(z) profile for nonuniform stratification.
"""
function update_velocity_fields!(tracker::ParticleTracker{T},
                                state::State, grid::Grid;
                                params=nothing, N2_profile=nothing) where T
    # Compute ONLY QG velocities (Lagrangian-mean flow)
    # In GLM framework, we do NOT add wave velocity or Stokes drift to advection
    # Skip w computation entirely when use_3d_advection=false for better performance
    compute_velocities!(state, grid;
                        plans=tracker.plans,
                        params=params,
                        compute_w=tracker.config.use_3d_advection,
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
    update_wave_fields!(tracker, state, grid; params=nothing)

Update wave velocity amplitude fields LA for wave displacement computation.

# Operator Definitions (from PDF)
    L  (YBJ):  L  = ∂/∂z(f²/N² ∂/∂z)                [eq. (4)]
    L⁺ (YBJ+): L⁺ = L - k_h²/4

# Wave Velocity Amplitude
The instantaneous wave velocity is (equation 3):
    u + iv = (LA) × e^{-ift}

where L is the YBJ operator. Since B = L⁺A and L = L⁺ + k_h²/4:
    LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A

This function calls compute_wave_displacement! to compute LA in physical space
and stores it in tracker.LA_real_field and tracker.LA_imag_field for
interpolation to particle positions.

# Note
This should be called after update_velocity_fields! and before computing wave displacement.
"""
function update_wave_fields!(tracker::ParticleTracker{T},
                            state::State, grid::Grid;
                            params=nothing, N2_profile=nothing) where T
    # Compute horizontal wave displacement field LA in physical space
    # This function is defined in operators.jl
    compute_wave_displacement!(state, grid; plans=tracker.plans, params=params)

    # Compute vertical wave displacement coefficients
    # ξz = ξz_cos × cos(ft) + ξz_sin × sin(ft)
    # skip_inversion=true because compute_wave_displacement! already inverted A
    compute_vertical_wave_displacement!(state, grid, tracker.plans, params;
                                        N2_profile=N2_profile, skip_inversion=true)

    # Get actual local dimensions from State arrays
    local_dims = size(parent(state.u))
    nz_local, nx_local, ny_local = local_dims

    # Resize tracker fields if needed
    if size(tracker.LA_real_field) != local_dims
        tracker.LA_real_field = zeros(T, nz_local, nx_local, ny_local)
        tracker.LA_imag_field = zeros(T, nz_local, nx_local, ny_local)
        tracker.ξz_cos_field = zeros(T, nz_local, nx_local, ny_local)
        tracker.ξz_sin_field = zeros(T, nz_local, nx_local, ny_local)
    end

    # Copy horizontal wave displacement amplitude
    tracker.LA_real_field .= parent(state.LA_real)
    tracker.LA_imag_field .= parent(state.LA_imag)

    # Copy vertical wave displacement coefficients
    tracker.ξz_cos_field .= parent(state.ξz_cos)
    tracker.ξz_sin_field .= parent(state.ξz_sin)

    # Update f0 from params if available
    if params !== nothing && hasfield(typeof(params), :f₀)
        tracker.f0 = T(params.f₀)
    end

    return tracker
end

"""
    compute_particle_wave_displacement!(tracker, current_time)

Compute wave displacement ξ (horizontal and vertical) for all particles at current simulation time.

# Horizontal Wave Displacement Formula (from PDF)

The horizontal wave displacement is obtained from equation (6):
    ξx + iξy = Re{(LA / (-if)) × e^{-ift}}

where LA uses the YBJ operator L (NOT L⁺):
    L  = ∂/∂z(f²/N² ∂/∂z)                     [eq. (4)]
    L⁺ = L - k_h²/4                            (YBJ+)

Since B = L⁺A, we have:
    LA = (L⁺ + k_h²/4)A = B + (k_h²/4)A

# Vertical Wave Displacement Formula (equation 2.10)

The vertical wave displacement is:
    ξz = ξz_cos × cos(ft) + ξz_sin × sin(ft)

where ξz_cos and ξz_sin are precomputed coefficients stored in the tracker's
ξz_cos_field and ξz_sin_field (from compute_vertical_wave_displacement!).

# Physical Interpretation
The horizontal displacement ξ represents the inertial loops of radius ~|LA|/f
that particles trace while riding on the mean drift X(t). The vertical
displacement ξz captures wave-induced vertical oscillations from equation (2.10).

# Arguments
- `tracker`: ParticleTracker with updated LA and ξz fields (from update_wave_fields!)
- `current_time`: Current simulation time (for phase factor e^{-ift})
"""
function compute_particle_wave_displacement!(tracker::ParticleTracker{T},
                                            current_time::Real) where T
    particles = tracker.particles
    f0 = tracker.f0

    # Phase factors for wave displacement
    ft = f0 * T(current_time)
    cos_ft = cos(ft)
    sin_ft = sin(ft)

    # For horizontal displacement: e^(-ift)
    phase = -ft
    cos_phase = cos(phase)
    sin_phase = sin(phase)

    @inbounds for i in 1:particles.np
        x, y, z = particles.x[i], particles.y[i], particles.z[i]

        # ======== Horizontal wave displacement ========
        # Interpolate LA at particle position
        LA_r, LA_i = interpolate_LA_at_position(x, y, z, tracker)
        particles.LA_real[i] = LA_r
        particles.LA_imag[i] = LA_i

        # Compute horizontal wave displacement: ξ = Re{(LA / (-if)) × e^(-ift)}
        # LA / (-if) = LA × (i/f) = (LA_r + i·LA_i) × (i/f) = (-LA_i + i·LA_r) / f
        # So: LA / (-if) = (-LA_i/f) + i·(LA_r/f)
        disp_amp_real = -LA_i / f0
        disp_amp_imag = LA_r / f0

        # Multiply by e^(-ift) = cos(phase) + i·sin(phase)
        # (a + ib)(cos + i·sin) = (a·cos - b·sin) + i·(a·sin + b·cos)
        xi_complex_real = disp_amp_real * cos_phase - disp_amp_imag * sin_phase
        xi_complex_imag = disp_amp_real * sin_phase + disp_amp_imag * cos_phase

        # Horizontal displacement components
        particles.xi_x[i] = xi_complex_real
        particles.xi_y[i] = xi_complex_imag

        # ======== Vertical wave displacement ========
        # Interpolate ξz coefficients at particle position
        ξz_cos, ξz_sin = interpolate_xi_z_coeffs_at_position(x, y, z, tracker)
        particles.xi_z_cos[i] = ξz_cos
        particles.xi_z_sin[i] = ξz_sin

        # Compute vertical displacement: ξz = ξz_cos × cos(ft) + ξz_sin × sin(ft)
        particles.xi_z[i] = ξz_cos * cos_ft + ξz_sin * sin_ft
    end

    return tracker
end

"""
    interpolate_xi_z_coeffs_at_position(x, y, z, tracker)

Interpolate vertical wave displacement coefficients at particle position.
Returns (ξz_cos, ξz_sin) tuple.
"""
function interpolate_xi_z_coeffs_at_position(x::T, y::T, z::T,
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

    # Trilinear interpolation for ξz_cos
    c_z1_y1 = (1-rx) * tracker.ξz_cos_field[iz1, ix1, iy1] + rx * tracker.ξz_cos_field[iz1, ix2, iy1]
    c_z1_y2 = (1-rx) * tracker.ξz_cos_field[iz1, ix1, iy2] + rx * tracker.ξz_cos_field[iz1, ix2, iy2]
    c_z1 = (1-ry) * c_z1_y1 + ry * c_z1_y2

    c_z2_y1 = (1-rx) * tracker.ξz_cos_field[iz2, ix1, iy1] + rx * tracker.ξz_cos_field[iz2, ix2, iy1]
    c_z2_y2 = (1-rx) * tracker.ξz_cos_field[iz2, ix1, iy2] + rx * tracker.ξz_cos_field[iz2, ix2, iy2]
    c_z2 = (1-ry) * c_z2_y1 + ry * c_z2_y2

    ξz_cos = (1-rz) * c_z1 + rz * c_z2

    # Trilinear interpolation for ξz_sin
    s_z1_y1 = (1-rx) * tracker.ξz_sin_field[iz1, ix1, iy1] + rx * tracker.ξz_sin_field[iz1, ix2, iy1]
    s_z1_y2 = (1-rx) * tracker.ξz_sin_field[iz1, ix1, iy2] + rx * tracker.ξz_sin_field[iz1, ix2, iy2]
    s_z1 = (1-ry) * s_z1_y1 + ry * s_z1_y2

    s_z2_y1 = (1-rx) * tracker.ξz_sin_field[iz2, ix1, iy1] + rx * tracker.ξz_sin_field[iz2, ix2, iy1]
    s_z2_y2 = (1-rx) * tracker.ξz_sin_field[iz2, ix1, iy2] + rx * tracker.ξz_sin_field[iz2, ix2, iy2]
    s_z2 = (1-ry) * s_z2_y1 + ry * s_z2_y2

    ξz_sin = (1-rz) * s_z1 + rz * s_z2

    return ξz_cos, ξz_sin
end

"""
    interpolate_LA_at_position(x, y, z, tracker)

Interpolate wave velocity amplitude LA at particle position.
Returns (LA_real, LA_imag) tuple.
"""
function interpolate_LA_at_position(x::T, y::T, z::T,
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

    # Trilinear interpolation for LA_real
    LA_r_z1_y1 = (1-rx) * tracker.LA_real_field[iz1, ix1, iy1] + rx * tracker.LA_real_field[iz1, ix2, iy1]
    LA_r_z1_y2 = (1-rx) * tracker.LA_real_field[iz1, ix1, iy2] + rx * tracker.LA_real_field[iz1, ix2, iy2]
    LA_r_z1 = (1-ry) * LA_r_z1_y1 + ry * LA_r_z1_y2

    LA_r_z2_y1 = (1-rx) * tracker.LA_real_field[iz2, ix1, iy1] + rx * tracker.LA_real_field[iz2, ix2, iy1]
    LA_r_z2_y2 = (1-rx) * tracker.LA_real_field[iz2, ix1, iy2] + rx * tracker.LA_real_field[iz2, ix2, iy2]
    LA_r_z2 = (1-ry) * LA_r_z2_y1 + ry * LA_r_z2_y2

    LA_real = (1-rz) * LA_r_z1 + rz * LA_r_z2

    # Trilinear interpolation for LA_imag
    LA_i_z1_y1 = (1-rx) * tracker.LA_imag_field[iz1, ix1, iy1] + rx * tracker.LA_imag_field[iz1, ix2, iy1]
    LA_i_z1_y2 = (1-rx) * tracker.LA_imag_field[iz1, ix1, iy2] + rx * tracker.LA_imag_field[iz1, ix2, iy2]
    LA_i_z1 = (1-ry) * LA_i_z1_y1 + ry * LA_i_z1_y2

    LA_i_z2_y1 = (1-rx) * tracker.LA_imag_field[iz2, ix1, iy1] + rx * tracker.LA_imag_field[iz2, ix2, iy1]
    LA_i_z2_y2 = (1-rx) * tracker.LA_imag_field[iz2, ix1, iy2] + rx * tracker.LA_imag_field[iz2, ix2, iy2]
    LA_i_z2 = (1-ry) * LA_i_z2_y1 + ry * LA_i_z2_y2

    LA_imag = (1-rz) * LA_i_z1 + rz * LA_i_z2

    return LA_real, LA_imag
end

"""
    validate_particle_cfl(tracker, max_velocity, dt)

Check if timestep satisfies CFL condition for particle advection in parallel mode.

For Euler integration, particles move up to dt*max_velocity from their
starting position. If this exceeds the halo region, interpolation will be inaccurate.

Returns true if timestep is safe, false if timestep may cause issues.

# Warning
If this returns false, consider:
- Reducing dt
- Increasing halo_width (use higher-order interpolation which has wider halos)
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

# Integration method (Euler only)

"""
    advect_euler!(tracker, dt)

Advect particles using Euler integration method: x = x + dt*u

This implements the Euler timestep:
- x_new = x_old + dt * u
- y_new = y_old + dt * v
- z_new = z_old + dt * w

where (u,v,w) is the interpolated velocity at the current particle position.
"""
function advect_euler!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    use_3d = tracker.config.use_3d_advection

    @inbounds for i in 1:particles.np
        x, y, z = particles.x[i], particles.y[i], particles.z[i]

        u, v, w = interpolate_velocity_at_position(x, y, z, tracker)

        # Euler timestep: x = x + dt*u
        particles.x[i] = x + dt * u
        particles.y[i] = y + dt * v
        if use_3d
            particles.z[i] = z + dt * w
        end

        particles.u[i] = u
        particles.v[i] = v
        particles.w[i] = w
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

Saves both MEAN positions (x, y, z) and WAVE displacements (xi_x, xi_y).
The physical (wiggly) trajectory can be reconstructed as: x_phys = x + xi_x, y_phys = y + xi_y.

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
            empty!(particles.xi_x_history)
            empty!(particles.xi_y_history)
            empty!(particles.xi_z_history)

            tracker.output_file_sequence += 1
        else
            # Traditional behavior: stop saving when max reached
            return tracker
        end
    end

    # Save current mean positions to history
    push!(particles.x_history, copy(particles.x))
    push!(particles.y_history, copy(particles.y))
    push!(particles.z_history, copy(particles.z))
    push!(particles.id_history, copy(particles.id))
    push!(particles.time_history, particles.time)

    # Save wave displacement history (for reconstructing wiggly trajectories)
    push!(particles.xi_x_history, copy(particles.xi_x))
    push!(particles.xi_y_history, copy(particles.xi_y))
    push!(particles.xi_z_history, copy(particles.xi_z))

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
