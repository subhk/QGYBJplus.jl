"""
Enhanced particle configuration with support for multiple z-levels and 3D distributions.

This module extends the basic ParticleConfig to support:
- Multiple z-levels (layered distributions)
- 3D rectangular grids
- Mixed positioning patterns
- Custom particle density distributions
"""

module EnhancedParticleConfig

using ..InterpolationSchemes: InterpolationMethod, TRILINEAR

export ParticleConfig3D, ParticleDistribution, create_particle_config_3d,
       initialize_particles_3d!, UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM

"""
Available particle distribution patterns.
"""
@enum ParticleDistribution begin
    UNIFORM_GRID = 1  # Regular 3D grid (nx × ny × nz)
    LAYERED = 2       # Multiple 2D layers at different z-levels
    RANDOM_3D = 3     # Random distribution in 3D volume
    CUSTOM = 4        # User-provided positions
end

"""
Enhanced particle configuration supporting 3D distributions.
"""
Base.@kwdef struct ParticleConfig3D{T<:AbstractFloat}
    # Spatial domain for particle initialization
    x_min::T = 0.0
    x_max::T = 2π
    y_min::T = 0.0  
    y_max::T = 2π
    z_min::T = 0.0      # Minimum z-level
    z_max::T = π        # Maximum z-level
    
    # Particle distribution specification
    distribution_type::ParticleDistribution = UNIFORM_GRID
    
    # Number of particles (interpretation depends on distribution_type)
    nx_particles::Int = 10
    ny_particles::Int = 10
    nz_particles::Int = 5   # Number of z-levels or vertical particles
    
    # Multiple z-levels support
    z_levels::Vector{T} = T[]        # Specific z-levels (for LAYERED type)
    particles_per_level::Vector{Int} = Int[]  # Particles per level (optional)
    
    # Custom positions (for CUSTOM type)
    custom_x::Vector{T} = T[]
    custom_y::Vector{T} = T[]
    custom_z::Vector{T} = T[]
    
    # Physics options
    use_ybj_w::Bool = false
    use_3d_advection::Bool = true
    
    # Integration and interpolation methods
    integration_method::Symbol = :rk4
    interpolation_method::InterpolationMethod = TRILINEAR
    
    # Boundary conditions
    periodic_x::Bool = true
    periodic_y::Bool = true
    reflect_z::Bool = true
    
    # I/O configuration
    save_interval::T = 0.1
    max_save_points::Int = 1000
    
    # Validation
    function ParticleConfig3D{T}(x_min, x_max, y_min, y_max, z_min, z_max,
                                distribution_type, nx_particles, ny_particles, nz_particles,
                                z_levels, particles_per_level, custom_x, custom_y, custom_z,
                                use_ybj_w, use_3d_advection, integration_method, interpolation_method,
                                periodic_x, periodic_y, reflect_z, save_interval, max_save_points) where T
        
        # Basic validation
        @assert x_max > x_min "x_max must be greater than x_min"
        @assert y_max > y_min "y_max must be greater than y_min"
        @assert z_max > z_min "z_max must be greater than z_min"
        @assert nx_particles > 0 "nx_particles must be positive"
        @assert ny_particles > 0 "ny_particles must be positive"
        @assert nz_particles > 0 "nz_particles must be positive"
        
        # Distribution-specific validation
        if distribution_type == LAYERED
            if !isempty(z_levels)
                @assert all(z_min ≤ z ≤ z_max for z in z_levels) "All z_levels must be within [z_min, z_max]"
                @assert length(z_levels) == length(unique(z_levels)) "z_levels must be unique"
            end
        elseif distribution_type == CUSTOM
            @assert length(custom_x) == length(custom_y) == length(custom_z) "Custom position arrays must have same length"
            if !isempty(custom_x)
                @assert all(x_min ≤ x ≤ x_max for x in custom_x) "All custom_x must be within [x_min, x_max]"
                @assert all(y_min ≤ y ≤ y_max for y in custom_y) "All custom_y must be within [y_min, y_max]"
                @assert all(z_min ≤ z ≤ z_max for z in custom_z) "All custom_z must be within [z_min, z_max]"
            end
        end
        
        new{T}(x_min, x_max, y_min, y_max, z_min, z_max,
               distribution_type, nx_particles, ny_particles, nz_particles,
               z_levels, particles_per_level, custom_x, custom_y, custom_z,
               use_ybj_w, use_3d_advection, integration_method, interpolation_method,
               periodic_x, periodic_y, reflect_z, save_interval, max_save_points)
    end
end

"""
    create_particle_config_3d(; kwargs...)

Convenience constructor for 3D particle configurations.
"""
function create_particle_config_3d(::Type{T}=Float64; kwargs...) where T
    return ParticleConfig3D{T}(; kwargs...)
end

"""
    create_uniform_3d_grid(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)

Create uniform 3D grid configuration.
"""
function create_uniform_3d_grid(x_min::T, x_max::T, y_min::T, y_max::T, z_min::T, z_max::T,
                               nx::Int, ny::Int, nz::Int; kwargs...) where T
    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        distribution_type=UNIFORM_GRID,
        nx_particles=nx, ny_particles=ny, nz_particles=nz;
        kwargs...
    )
end

"""
    create_layered_distribution(x_min, x_max, y_min, y_max, z_levels, particles_per_level)

Create layered particle distribution at specific z-levels.
"""
function create_layered_distribution(x_min::T, x_max::T, y_min::T, y_max::T,
                                   z_levels::Vector{T}, nx::Int, ny::Int;
                                   particles_per_level::Vector{Int}=Int[], kwargs...) where T
    
    z_min = minimum(z_levels)
    z_max = maximum(z_levels)
    
    # Default: same number of particles per level
    if isempty(particles_per_level)
        particles_per_level = fill(nx * ny, length(z_levels))
    end
    
    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        distribution_type=LAYERED,
        nx_particles=nx, ny_particles=ny, nz_particles=length(z_levels),
        z_levels=z_levels, particles_per_level=particles_per_level;
        kwargs...
    )
end

"""
    create_random_3d_distribution(x_min, x_max, y_min, y_max, z_min, z_max, n_particles)

Create random 3D particle distribution.
"""
function create_random_3d_distribution(x_min::T, x_max::T, y_min::T, y_max::T, z_min::T, z_max::T,
                                      n_particles::Int; random_seed::Int=1234, kwargs...) where T
    
    # Generate random positions
    Random.seed!(random_seed)
    custom_x = x_min .+ (x_max - x_min) .* rand(T, n_particles)
    custom_y = y_min .+ (y_max - y_min) .* rand(T, n_particles)
    custom_z = z_min .+ (z_max - z_min) .* rand(T, n_particles)
    
    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        distribution_type=CUSTOM,
        nx_particles=n_particles, ny_particles=1, nz_particles=1,
        custom_x=custom_x, custom_y=custom_y, custom_z=custom_z;
        kwargs...
    )
end

"""
    create_custom_distribution(positions)

Create custom particle distribution from user-provided positions.
"""
function create_custom_distribution(positions::Vector{Tuple{T,T,T}}; kwargs...) where T
    
    custom_x = [pos[1] for pos in positions]
    custom_y = [pos[2] for pos in positions]  
    custom_z = [pos[3] for pos in positions]
    
    x_min, x_max = extrema(custom_x)
    y_min, y_max = extrema(custom_y)
    z_min, z_max = extrema(custom_z)
    
    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        distribution_type=CUSTOM,
        nx_particles=length(positions), ny_particles=1, nz_particles=1,
        custom_x=custom_x, custom_y=custom_y, custom_z=custom_z;
        kwargs...
    )
end

"""
    initialize_particles_3d!(tracker, config3d)

Initialize particles using 3D configuration.
"""
function initialize_particles_3d!(tracker, config::ParticleConfig3D{T}) where T
    
    if config.distribution_type == UNIFORM_GRID
        initialize_uniform_3d_grid!(tracker, config)
    elseif config.distribution_type == LAYERED
        initialize_layered_distribution!(tracker, config)
    elseif config.distribution_type == RANDOM_3D || config.distribution_type == CUSTOM
        initialize_custom_positions!(tracker, config)
    else
        error("Unknown distribution type: $(config.distribution_type)")
    end
    
    # Set initial velocities to zero
    fill!(tracker.particles.u, 0.0)
    fill!(tracker.particles.v, 0.0)
    fill!(tracker.particles.w, 0.0)
    
    # Set initial time
    tracker.particles.time = 0.0
    
    return tracker
end

"""
Initialize uniform 3D grid of particles.
"""
function initialize_uniform_3d_grid!(tracker, config::ParticleConfig3D{T}) where T
    
    # Create 3D grid
    x_range = range(config.x_min, config.x_max, length=config.nx_particles+1)[1:end-1]
    y_range = range(config.y_min, config.y_max, length=config.ny_particles+1)[1:end-1]
    z_range = range(config.z_min, config.z_max, length=config.nz_particles+1)[1:end-1]
    
    total_particles = config.nx_particles * config.ny_particles * config.nz_particles
    
    # Resize particle arrays
    resize!(tracker.particles.x, total_particles)
    resize!(tracker.particles.y, total_particles)
    resize!(tracker.particles.z, total_particles)
    resize!(tracker.particles.u, total_particles)
    resize!(tracker.particles.v, total_particles)
    resize!(tracker.particles.w, total_particles)
    
    # Fill particle positions
    idx = 1
    for z in z_range, y in y_range, x in x_range
        tracker.particles.x[idx] = x
        tracker.particles.y[idx] = y
        tracker.particles.z[idx] = z
        idx += 1
    end
    
    tracker.particles.np = total_particles
    
    return tracker
end

"""
Initialize layered particle distribution.
"""
function initialize_layered_distribution!(tracker, config::ParticleConfig3D{T}) where T
    
    z_levels = isempty(config.z_levels) ? 
               collect(range(config.z_min, config.z_max, length=config.nz_particles)) :
               config.z_levels
    
    # Calculate total particles
    if isempty(config.particles_per_level)
        particles_per_level = fill(config.nx_particles * config.ny_particles, length(z_levels))
    else
        particles_per_level = config.particles_per_level
    end
    
    total_particles = sum(particles_per_level)
    
    # Resize particle arrays
    resize!(tracker.particles.x, total_particles)
    resize!(tracker.particles.y, total_particles)
    resize!(tracker.particles.z, total_particles)
    resize!(tracker.particles.u, total_particles)
    resize!(tracker.particles.v, total_particles)
    resize!(tracker.particles.w, total_particles)
    
    # Fill particles layer by layer
    idx = 1
    for (level, z_level) in enumerate(z_levels)
        n_level = particles_per_level[level]
        
        # Estimate grid size for this level
        ny_level = config.ny_particles
        nx_level = div(n_level, ny_level)
        
        # Create 2D grid for this level
        x_range = range(config.x_min, config.x_max, length=nx_level+1)[1:end-1]
        y_range = range(config.y_min, config.y_max, length=ny_level+1)[1:end-1]
        
        # Place particles
        for y in y_range, x in x_range
            if idx <= total_particles
                tracker.particles.x[idx] = x
                tracker.particles.y[idx] = y
                tracker.particles.z[idx] = z_level
                idx += 1
            end
        end
    end
    
    tracker.particles.np = total_particles
    
    return tracker
end

"""
Initialize particles at custom positions.
"""
function initialize_custom_positions!(tracker, config::ParticleConfig3D{T}) where T
    
    n_particles = length(config.custom_x)
    
    # Resize particle arrays
    resize!(tracker.particles.x, n_particles)
    resize!(tracker.particles.y, n_particles)
    resize!(tracker.particles.z, n_particles)
    resize!(tracker.particles.u, n_particles)
    resize!(tracker.particles.v, n_particles)
    resize!(tracker.particles.w, n_particles)
    
    # Copy custom positions
    tracker.particles.x .= config.custom_x
    tracker.particles.y .= config.custom_y
    tracker.particles.z .= config.custom_z
    
    tracker.particles.np = n_particles
    
    return tracker
end

"""
    convert_to_basic_config(config3d)

Convert enhanced 3D config to basic ParticleConfig for compatibility.
"""
function convert_to_basic_config(config::ParticleConfig3D{T}) where T
    
    # For single z-level distributions, convert to basic config
    if config.distribution_type == LAYERED && length(config.z_levels) == 1
        z_level = config.z_levels[1]
    elseif config.distribution_type == UNIFORM_GRID && config.nz_particles == 1
        z_level = (config.z_min + config.z_max) / 2
    else
        # Multi-level distribution - use middle z-level as representative
        z_level = (config.z_min + config.z_max) / 2
    end
    
    # Import the basic ParticleConfig (assuming it's available)
    return ParticleConfig{T}(
        x_min=config.x_min, x_max=config.x_max,
        y_min=config.y_min, y_max=config.y_max,
        z_level=z_level,
        nx_particles=config.nx_particles, ny_particles=config.ny_particles,
        use_ybj_w=config.use_ybj_w, use_3d_advection=config.use_3d_advection,
        integration_method=config.integration_method,
        interpolation_method=config.interpolation_method,
        periodic_x=config.periodic_x, periodic_y=config.periodic_y, reflect_z=config.reflect_z,
        save_interval=config.save_interval, max_save_points=config.max_save_points
    )
end

end # module EnhancedParticleConfig

using .EnhancedParticleConfig