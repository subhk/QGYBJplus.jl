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
    create_uniform_horizontal_distribution(z_level; x_min, x_max, y_min, y_max, n_particles, kwargs...)

Create a uniform distribution of particles in a horizontal area at a single z-level.

This is the simplest way to initialize particles: specify a z-level, horizontal bounds,
and total number of particles. The particles are distributed uniformly in a grid pattern.

# Arguments
- `z_level::Real`: The vertical level where particles are placed
- `x_min::Real=0.0`: Minimum x-coordinate of the horizontal area
- `x_max::Real=2π`: Maximum x-coordinate of the horizontal area
- `y_min::Real=0.0`: Minimum y-coordinate of the horizontal area
- `y_max::Real=2π`: Maximum y-coordinate of the horizontal area
- `n_particles::Int=100`: Total number of particles (distributed as √n × √n grid)

# Alternative grid specification
Instead of `n_particles`, you can specify `nx` and `ny` directly:
- `nx::Int`: Number of particles in x-direction
- `ny::Int`: Number of particles in y-direction

# Returns
`ParticleConfig3D` configured for uniform horizontal distribution at the specified z-level.

# Example
```julia
# Simple: 100 particles uniformly in [0,2π]×[0,2π] at z=π/2
config = create_uniform_horizontal_distribution(π/2; n_particles=100)

# Custom area: 64 particles in a subregion at z=1.0
config = create_uniform_horizontal_distribution(1.0;
    x_min=π/4, x_max=3π/4,
    y_min=π/4, y_max=3π/4,
    n_particles=64
)

# Explicit grid: 10×8 = 80 particles at z=π
config = create_uniform_horizontal_distribution(π;
    x_min=0.0, x_max=2π,
    y_min=0.0, y_max=2π,
    nx=10, ny=8
)

# Initialize tracker with this config
initialize_particles_3d!(tracker, config)
```
"""
function create_uniform_horizontal_distribution(z_level::T;
                                               x_min::T=T(0.0), x_max::T=T(2π),
                                               y_min::T=T(0.0), y_max::T=T(2π),
                                               n_particles::Union{Int,Nothing}=nothing,
                                               nx::Union{Int,Nothing}=nothing,
                                               ny::Union{Int,Nothing}=nothing,
                                               kwargs...) where T<:AbstractFloat

    # Determine nx and ny from inputs
    if nx !== nothing && ny !== nothing
        # User specified nx and ny directly
        nx_particles = nx
        ny_particles = ny
    elseif n_particles !== nothing
        # Compute approximately square grid from total particle count
        nx_particles = round(Int, sqrt(n_particles))
        ny_particles = round(Int, n_particles / nx_particles)
        # Ensure we get at least n_particles (may be slightly more for non-square numbers)
        while nx_particles * ny_particles < n_particles
            ny_particles += 1
        end
    else
        # Default: 10×10 = 100 particles
        nx_particles = 10
        ny_particles = 10
    end

    @assert x_max > x_min "x_max must be greater than x_min"
    @assert y_max > y_min "y_max must be greater than y_min"
    @assert nx_particles > 0 "nx_particles must be positive"
    @assert ny_particles > 0 "ny_particles must be positive"

    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_level, z_max=z_level,  # Single z-level
        distribution_type=LAYERED,
        nx_particles=nx_particles,
        ny_particles=ny_particles,
        nz_particles=1,
        z_levels=T[z_level],  # Single z-level
        particles_per_level=Int[nx_particles * ny_particles];
        kwargs...
    )
end

# Convenience method with default Float64 type
function create_uniform_horizontal_distribution(z_level::Real;
                                               x_min::Real=0.0, x_max::Real=2π,
                                               y_min::Real=0.0, y_max::Real=2π,
                                               kwargs...)
    T = Float64
    return create_uniform_horizontal_distribution(T(z_level);
        x_min=T(x_min), x_max=T(x_max),
        y_min=T(y_min), y_max=T(y_max),
        kwargs...
    )
end

"""
    create_circular_distribution(z_level; x_center, y_center, radius, n_particles, pattern, kwargs...)

Create particles distributed in a circular disk at a single z-level.

Particles are distributed uniformly within a circular region using either a
sunflower (Fibonacci) pattern or concentric rings. The sunflower pattern
provides excellent uniform coverage and is commonly used in Lagrangian studies.

# Arguments
- `z_level::Real`: The vertical level where particles are placed
- `x_center::Real=π`: x-coordinate of circle center
- `y_center::Real=π`: y-coordinate of circle center
- `radius::Real=1.0`: Radius of the circular region
- `n_particles::Int=100`: Number of particles to place
- `pattern::Symbol=:sunflower`: Distribution pattern
  - `:sunflower` - Fibonacci/sunflower spiral (recommended, very uniform)
  - `:rings` - Concentric rings with uniform radial spacing
  - `:random` - Uniform random distribution within disk

# Returns
`ParticleConfig3D` configured with custom positions forming the circular distribution.

# Mathematical Background
The sunflower pattern uses the golden angle θ = π(3 - √5) ≈ 137.5° to achieve
uniform area coverage:
- θₙ = n × golden_angle
- rₙ = R × √(n/N)  (ensures uniform area density)

# Example
```julia
# 100 particles in a circle of radius 1.0 centered at (π, π) at z=π/2
config = create_circular_distribution(π/2;
    x_center=π, y_center=π,
    radius=1.0,
    n_particles=100
)

# Larger circle with more particles using concentric rings
config = create_circular_distribution(1.0;
    x_center=π, y_center=π,
    radius=2.0,
    n_particles=200,
    pattern=:rings
)

# Random distribution within disk
config = create_circular_distribution(π/4;
    x_center=3.0, y_center=3.0,
    radius=0.5,
    n_particles=50,
    pattern=:random
)

# Initialize tracker
initialize_particles_3d!(tracker, config)
```
"""
function create_circular_distribution(z_level::T;
                                     x_center::T=T(π),
                                     y_center::T=T(π),
                                     radius::T=T(1.0),
                                     n_particles::Int=100,
                                     pattern::Symbol=:sunflower,
                                     kwargs...) where T<:AbstractFloat

    @assert radius > 0 "radius must be positive"
    @assert n_particles > 0 "n_particles must be positive"

    # Generate particle positions based on pattern
    custom_x = Vector{T}(undef, n_particles)
    custom_y = Vector{T}(undef, n_particles)
    custom_z = fill(z_level, n_particles)

    if pattern == :sunflower
        # Sunflower/Fibonacci pattern - very uniform distribution
        # Golden angle in radians
        golden_angle = T(π) * (T(3) - sqrt(T(5)))

        for i in 1:n_particles
            # Radius: sqrt distribution for uniform area density
            r = radius * sqrt(T(i) / T(n_particles))
            # Angle: golden angle increment
            θ = T(i) * golden_angle

            custom_x[i] = x_center + r * cos(θ)
            custom_y[i] = y_center + r * sin(θ)
        end

    elseif pattern == :rings
        # Concentric rings with uniform radial spacing
        # Determine number of rings based on particle count
        n_rings = max(1, round(Int, sqrt(n_particles / π)))
        particles_placed = 0

        for ring in 1:n_rings
            # Radius of this ring
            r = radius * T(ring) / T(n_rings)

            # Number of particles on this ring (proportional to circumference)
            if ring == 1
                n_on_ring = max(1, round(Int, n_particles / (n_rings * n_rings)))
            else
                n_on_ring = max(1, round(Int, 2π * ring * n_particles / (π * n_rings * n_rings)))
            end

            # Don't exceed total particles
            n_on_ring = min(n_on_ring, n_particles - particles_placed)

            for j in 1:n_on_ring
                θ = T(2π) * T(j - 1) / T(n_on_ring)
                idx = particles_placed + j
                if idx <= n_particles
                    custom_x[idx] = x_center + r * cos(θ)
                    custom_y[idx] = y_center + r * sin(θ)
                end
            end
            particles_placed += n_on_ring

            if particles_placed >= n_particles
                break
            end
        end

        # Fill any remaining particles at center
        for i in (particles_placed + 1):n_particles
            custom_x[i] = x_center
            custom_y[i] = y_center
        end

    elseif pattern == :random
        # Uniform random distribution within disk
        # Use rejection sampling or sqrt(r) transformation
        for i in 1:n_particles
            # sqrt transformation for uniform area density
            r = radius * sqrt(rand(T))
            θ = T(2π) * rand(T)

            custom_x[i] = x_center + r * cos(θ)
            custom_y[i] = y_center + r * sin(θ)
        end

    else
        error("Unknown pattern: $pattern. Use :sunflower, :rings, or :random")
    end

    # Compute bounding box for the circular region
    x_min = x_center - radius
    x_max = x_center + radius
    y_min = y_center - radius
    y_max = y_center + radius

    return ParticleConfig3D{T}(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_level, z_max=z_level,
        distribution_type=CUSTOM,
        nx_particles=n_particles, ny_particles=1, nz_particles=1,
        custom_x=custom_x, custom_y=custom_y, custom_z=custom_z;
        kwargs...
    )
end

# Convenience method with default Float64 type
function create_circular_distribution(z_level::Real;
                                     x_center::Real=π,
                                     y_center::Real=π,
                                     radius::Real=1.0,
                                     kwargs...)
    T = Float64
    return create_circular_distribution(T(z_level);
        x_center=T(x_center),
        y_center=T(y_center),
        radius=T(radius),
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