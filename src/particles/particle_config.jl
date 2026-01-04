"""
Enhanced particle configuration with support for multiple z-levels and 3D distributions.

This module extends the basic ParticleConfig to support:
- Multiple z-levels (layered distributions)
- 3D rectangular grids
- Mixed positioning patterns
- Custom particle density distributions
"""

module EnhancedParticleConfig

using Random
using ..InterpolationSchemes: InterpolationMethod, TRILINEAR

# Access ParticleConfig from parent module (UnifiedParticleAdvection)
const _PARENT = parentmodule(@__MODULE__)
const ParticleConfig = _PARENT.ParticleConfig

export ParticleConfig3D, ParticleDistribution,
       initialize_particles_3d!, UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM,
       # Simplified particle initialization constructors
       particles_in_circle, particles_in_grid_3d, particles_in_layers,
       particles_random_3d, particles_custom

function _parallel_local_domain(tracker)
    if hasproperty(tracker, :is_parallel) && getproperty(tracker, :is_parallel) &&
       hasproperty(tracker, :local_domain)
        return getproperty(tracker, :local_domain)
    end
    return nothing
end

function _within_local_domain(x, y, local_domain, x_max_global, y_max_global)
    in_x = x >= local_domain.x_start &&
           (x < local_domain.x_end || (x == local_domain.x_end && local_domain.x_end == x_max_global))
    in_y = y >= local_domain.y_start &&
           (y < local_domain.y_end || (y == local_domain.y_end && local_domain.y_end == y_max_global))
    return in_x && in_y
end

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

Domain bounds (x_max, y_max, z_max) are REQUIRED - no defaults.
Use the Grid to get domain size: `x_max = G.Lx, y_max = G.Ly, z_max = G.Lz` (depth).
If you pass `z_min`, then `z_max` is treated as a coordinate (typically ≤ 0).
"""
Base.@kwdef struct ParticleConfig3D{T<:AbstractFloat}
    # Spatial domain for particle initialization (x_max, y_max, z_max are REQUIRED)
    x_min::T = 0.0
    x_max::T           # REQUIRED - use G.Lx
    y_min::T = 0.0
    y_max::T           # REQUIRED - use G.Ly
    z_min::T = 0.0
    z_max::T           # REQUIRED - pass depth if z_min omitted, else coordinate
    
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
    integration_method::Symbol = :euler
    interpolation_method::InterpolationMethod = TRILINEAR
    
    # Boundary conditions
    periodic_x::Bool = true
    periodic_y::Bool = true
    reflect_z::Bool = true
    
    # I/O configuration
    save_interval::T = 0.1
    max_save_points::Int = 1000

    # Random distribution seed
    seed::Int = 1234
    
    # Validation
    function ParticleConfig3D{T}(x_min, x_max, y_min, y_max, z_min, z_max,
                                distribution_type, nx_particles, ny_particles, nz_particles,
                                z_levels, particles_per_level, custom_x, custom_y, custom_z,
                                use_ybj_w, use_3d_advection, integration_method, interpolation_method,
                                periodic_x, periodic_y, reflect_z, save_interval, max_save_points,
                                seed) where T
        
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
            if !isempty(particles_per_level)
                expected_levels = isempty(z_levels) ? nz_particles : length(z_levels)
                @assert length(particles_per_level) == expected_levels "particles_per_level must match number of levels"
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
               periodic_x, periodic_y, reflect_z, save_interval, max_save_points, seed)
    end
end

"""
    particles_in_grid_3d(; x_max, y_max, z_max, nx, ny, nz, x_min=0, y_min=0, z_min=nothing, kwargs...)

Create particles uniformly distributed in a 3D rectangular grid.

# Arguments
- `x_max, y_max, z_max`: Domain bounds (REQUIRED - use G.Lx, G.Ly, G.Lz)
- `x_min, y_min`: Minimum bounds (default: 0.0)
- `z_min`: Minimum z (default: nothing → uses full depth with z ∈ [-z_max, 0])
- `nx, ny, nz`: Number of particles in each direction

# Example
```julia
# 1000 particles in a 10×10×10 3D grid
config = particles_in_grid_3d(x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, nx=10, ny=10, nz=10)

# Custom subdomain
config = particles_in_grid_3d(x_max=250e3, y_max=250e3, z_max=-500.0, z_min=-2000.0, nx=8, ny=8, nz=5)
```
"""
function particles_in_grid_3d(; x_max::Real, y_max::Real, z_max::Real,  # REQUIRED
                               x_min::Real=0.0, y_min::Real=0.0, z_min::Union{Real,Nothing}=nothing,
                               nx::Int=10, ny::Int=10, nz::Int=5,
                               precision::Type{T}=Float32,  # Float32 for memory efficiency (50% less memory than Float64)
                               kwargs...) where T<:AbstractFloat
    z_min_val = z_min === nothing ? -z_max : z_min
    z_max_val = z_min === nothing ? 0.0 : z_max
    return ParticleConfig3D{T}(
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        z_min=T(z_min_val), z_max=T(z_max_val),
        distribution_type=UNIFORM_GRID,
        nx_particles=nx, ny_particles=ny, nz_particles=nz;
        kwargs...
    )
end

"""
    particles_in_layers(z_levels; x_max, y_max, nx, ny, x_min=0, y_min=0, kwargs...)

Create particles distributed in 2D grids at multiple z-levels.

# Arguments
- `z_levels`: Vector of z-levels where particles are placed
- `x_max, y_max`: Domain bounds (REQUIRED - use G.Lx, G.Ly)
- `x_min, y_min`: Minimum bounds (default: 0.0)
- `nx, ny`: Number of particles per level in x and y (default: 10 each)

# Example
```julia
# 3 layers at depths 1000m, 2000m, 3000m with 10×10 particles each
config = particles_in_layers([-1000.0, -2000.0, -3000.0]; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Custom subdomain with 5 particles per side at each layer
config = particles_in_layers([-500.0, -1000.0, -1500.0]; x_max=250e3, y_max=250e3, nx=5, ny=5)
```
"""
function particles_in_layers(z_levels::Vector{<:Real};
                            x_max::Real, y_max::Real,  # REQUIRED
                            x_min::Real=0.0, y_min::Real=0.0,
                            nx::Int=10, ny::Int=10,
                            particles_per_level::Vector{Int}=Int[],
                            precision::Type{T}=Float32,  # Float32 for memory efficiency (50% less memory than Float64)
                            kwargs...) where T<:AbstractFloat
    z_levels_T = T.(z_levels)
    z_min = minimum(z_levels_T)
    z_max = maximum(z_levels_T)

    # Default: same number of particles per level
    if isempty(particles_per_level)
        particles_per_level = fill(nx * ny, length(z_levels_T))
    else
        @assert length(particles_per_level) == length(z_levels_T) "particles_per_level must match length(z_levels)"
    end

    return ParticleConfig3D{T}(
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        z_min=z_min, z_max=z_max,
        distribution_type=LAYERED,
        nx_particles=nx, ny_particles=ny, nz_particles=length(z_levels_T),
        z_levels=z_levels_T, particles_per_level=particles_per_level;
        kwargs...
    )
end

"""
    particles_random_3d(n; x_max, y_max, z_max, x_min=0, y_min=0, z_min=nothing, seed=1234, kwargs...)

Create randomly distributed particles in a 3D volume.

# Arguments
- `n`: Number of particles
- `x_max, y_max, z_max`: Domain bounds (REQUIRED - use G.Lx, G.Ly, G.Lz)
- `x_min, y_min`: Minimum bounds (default: 0.0)
- `z_min`: Minimum z (default: nothing → uses full depth with z ∈ [-z_max, 0])
- `seed`: Random seed for reproducibility (default: 1234)

# Example
```julia
# 500 random particles in the full domain
config = particles_random_3d(500; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz)

# 1000 random particles in a subdomain
config = particles_random_3d(1000; x_max=250e3, y_max=250e3, z_max=-500.0, z_min=-2000.0)
```
"""
function particles_random_3d(n::Int;
                            x_max::Real, y_max::Real, z_max::Real,  # REQUIRED
                            x_min::Real=0.0, y_min::Real=0.0, z_min::Union{Real,Nothing}=nothing,
                            seed::Int=1234,
                            precision::Type{T}=Float32,  # Float32 for memory efficiency (50% less memory than Float64)
                            kwargs...) where T<:AbstractFloat
    z_min_val = z_min === nothing ? -z_max : z_min
    z_max_val = z_min === nothing ? 0.0 : z_max
    return ParticleConfig3D{T}(
        x_min=T(x_min), x_max=T(x_max), y_min=T(y_min), y_max=T(y_max),
        z_min=T(z_min_val), z_max=T(z_max_val),
        distribution_type=RANDOM_3D,
        nx_particles=n, ny_particles=1, nz_particles=1,
        seed=seed;
        kwargs...
    )
end

"""
    particles_custom(positions; kwargs...)

Create particles at user-specified positions.

# Arguments
- `positions`: Vector of (x, y, z) tuples

# Example
```julia
# 4 particles at specific locations
config = particles_custom([(1.0, 1.0, 0.5), (2.0, 2.0, 1.0), (3.0, 1.5, 0.75), (1.5, 3.0, 1.25)])
```
"""
function particles_custom(positions::Vector{<:Tuple{Real,Real,Real}};
                         precision::Type{T}=Float32,  # Float32 for memory efficiency (50% less memory than Float64)
                         kwargs...) where T<:AbstractFloat

    custom_x = T[pos[1] for pos in positions]
    custom_y = T[pos[2] for pos in positions]
    custom_z = T[pos[3] for pos in positions]

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
    particles_in_circle(z; center, radius, n, pattern, kwargs...)

Create particles distributed in a circular disk at a fixed z-level.

# Arguments
- `z`: Vertical level where particles are placed
- `center`: (x, y) center of circle (default: (π, π))
- `radius`: Radius of the circular region (default: 1.0)
- `n`: Number of particles (default: 100)
- `pattern`: Distribution pattern (default: :sunflower)
  - `:sunflower` - Fibonacci spiral (very uniform, recommended)
  - `:rings` - Concentric rings
  - `:random` - Uniform random within disk

# Example
```julia
# 100 particles in a circle at z = -π/2
config = particles_in_circle(-π/2; radius=1.0, n=100)

# Custom center and larger circle
config = particles_in_circle(-1.0; center=(2.0, 2.0), radius=2.0, n=200)

# Random distribution
config = particles_in_circle(-π/4; radius=0.5, n=50, pattern=:random)
```
"""
function particles_in_circle(z_level::T;
                            center::Tuple{T,T}=(T(π), T(π)),
                            radius::T=T(1.0),
                            n::Int=100,
                            pattern::Symbol=:sunflower,
                            kwargs...) where T<:AbstractFloat
    x_center, y_center = center
    n_particles = n

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

# Convenience method with default Float32 type for memory efficiency
function particles_in_circle(z_level::Real;
                            center::Tuple{Real,Real}=(π, π),
                            radius::Real=1.0,
                            n::Int=100,
                            pattern::Symbol=:sunflower,
                            kwargs...)
    T = Float32  # Float32 for memory efficiency (50% less memory than Float64)
    return particles_in_circle(T(z_level);
        center=(T(center[1]), T(center[2])),
        radius=T(radius),
        n=n,
        pattern=pattern,
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
    elseif config.distribution_type == RANDOM_3D
        initialize_random_positions!(tracker, config)
    elseif config.distribution_type == CUSTOM
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

    local_domain = _parallel_local_domain(tracker)
    dxp = (config.x_max - config.x_min) / config.nx_particles
    dyp = (config.y_max - config.y_min) / config.ny_particles
    dzp = (config.z_max - config.z_min) / config.nz_particles

    if local_domain !== nothing
        tol = sqrt(eps(T)) * max(one(T), abs(config.x_max - config.x_min), abs(config.y_max - config.y_min))
        x_rel_min = (local_domain.x_start - config.x_min) / dxp
        x_rel_max = (local_domain.x_end - config.x_min) / dxp
        y_rel_min = (local_domain.y_start - config.y_min) / dyp
        y_rel_max = (local_domain.y_end - config.y_min) / dyp

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

        total_particles = (i_end - i_start + 1) * (j_end - j_start + 1) * config.nz_particles

        resize!(tracker.particles.x, total_particles)
        resize!(tracker.particles.y, total_particles)
        resize!(tracker.particles.z, total_particles)
        resize!(tracker.particles.id, total_particles)
        resize!(tracker.particles.u, total_particles)
        resize!(tracker.particles.v, total_particles)
        resize!(tracker.particles.w, total_particles)

        idx = 1
        for k in 1:config.nz_particles, j in j_start:j_end, i in i_start:i_end
            tracker.particles.x[idx] = config.x_min + (i - 1) * dxp
            tracker.particles.y[idx] = config.y_min + (j - 1) * dyp
            tracker.particles.z[idx] = config.z_min + (k - 1) * dzp
            tracker.particles.id[idx] = (k - 1) * config.nx_particles * config.ny_particles +
                                        (j - 1) * config.nx_particles + i
            idx += 1
        end

        tracker.particles.np = total_particles
        return tracker
    end

    # Create 3D grid (serial or no local domain)
    x_range = range(config.x_min, config.x_max, length=config.nx_particles+1)[1:end-1]
    y_range = range(config.y_min, config.y_max, length=config.ny_particles+1)[1:end-1]
    z_range = range(config.z_min, config.z_max, length=config.nz_particles+1)[1:end-1]

    total_particles = config.nx_particles * config.ny_particles * config.nz_particles

    # Resize particle arrays
    resize!(tracker.particles.x, total_particles)
    resize!(tracker.particles.y, total_particles)
    resize!(tracker.particles.z, total_particles)
    resize!(tracker.particles.id, total_particles)
    resize!(tracker.particles.u, total_particles)
    resize!(tracker.particles.v, total_particles)
    resize!(tracker.particles.w, total_particles)

    # Fill particle positions
    idx = 1
    for z in z_range, y in y_range, x in x_range
        tracker.particles.x[idx] = x
        tracker.particles.y[idx] = y
        tracker.particles.z[idx] = z
        tracker.particles.id[idx] = idx
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
    @assert length(particles_per_level) == length(z_levels) "particles_per_level must match number of layers"
    
    local_domain = _parallel_local_domain(tracker)
    total_particles = 0
    n_levels = length(z_levels)

    nx_levels = Vector{Int}(undef, n_levels)
    ny_levels = Vector{Int}(undef, n_levels)
    i_starts = Vector{Int}(undef, n_levels)
    i_ends = Vector{Int}(undef, n_levels)
    j_starts = Vector{Int}(undef, n_levels)
    j_ends = Vector{Int}(undef, n_levels)
    level_counts = Vector{Int}(undef, n_levels)
    level_offsets = Vector{Int}(undef, n_levels)

    tol = sqrt(eps(T)) * max(one(T), abs(config.x_max - config.x_min), abs(config.y_max - config.y_min))
    
    offset = 0
    for level in 1:n_levels
        level_offsets[level] = offset
        n_level = particles_per_level[level]
        offset += n_level
        if n_level <= 0
            nx_levels[level] = 0
            ny_levels[level] = 0
            i_starts[level] = 1
            i_ends[level] = 0
            j_starts[level] = 1
            j_ends[level] = 0
            level_counts[level] = 0
            continue
        end

        aspect = config.nx_particles / config.ny_particles
        ny_level = max(1, round(Int, sqrt(n_level / aspect)))
        nx_level = max(1, ceil(Int, n_level / ny_level))

        if local_domain !== nothing
            dxp = (config.x_max - config.x_min) / nx_level
            dyp = (config.y_max - config.y_min) / ny_level
            x_rel_min = (local_domain.x_start - config.x_min) / dxp
            x_rel_max = (local_domain.x_end - config.x_min) / dxp
            y_rel_min = (local_domain.y_start - config.y_min) / dyp
            y_rel_max = (local_domain.y_end - config.y_min) / dyp

            i_start = max(1, floor(Int, x_rel_min + tol) + 1)
            i_end = min(nx_level, ceil(Int, x_rel_max - tol))
            j_start = max(1, floor(Int, y_rel_min + tol) + 1)
            j_end = min(ny_level, ceil(Int, y_rel_max - tol))
        else
            i_start = 1
            i_end = nx_level
            j_start = 1
            j_end = ny_level
        end

        count = 0
        if i_start <= i_end && j_start <= j_end
            for j in j_start:j_end
                max_i = min(i_end, n_level - (j - 1) * nx_level)
                if max_i >= i_start
                    count += max_i - i_start + 1
                end
            end
        end

        nx_levels[level] = nx_level
        ny_levels[level] = ny_level
        i_starts[level] = i_start
        i_ends[level] = i_end
        j_starts[level] = j_start
        j_ends[level] = j_end
        level_counts[level] = count
        total_particles += count
    end

    # Resize particle arrays
    resize!(tracker.particles.x, total_particles)
    resize!(tracker.particles.y, total_particles)
    resize!(tracker.particles.z, total_particles)
    resize!(tracker.particles.id, total_particles)
    resize!(tracker.particles.u, total_particles)
    resize!(tracker.particles.v, total_particles)
    resize!(tracker.particles.w, total_particles)
    
    # Fill particles layer by layer
    idx = 1
    for (level, z_level) in enumerate(z_levels)
        n_level = particles_per_level[level]
        if level_counts[level] == 0
            continue
        end

        nx_level = nx_levels[level]
        ny_level = ny_levels[level]
        i_start = i_starts[level]
        i_end = i_ends[level]
        j_start = j_starts[level]
        j_end = j_ends[level]

        dxp = (config.x_max - config.x_min) / nx_level
        dyp = (config.y_max - config.y_min) / ny_level

        for j in j_start:j_end
            max_i = min(i_end, n_level - (j - 1) * nx_level)
            if max_i < i_start
                continue
            end
            for i in i_start:max_i
                tracker.particles.x[idx] = config.x_min + (i - 1) * dxp
                tracker.particles.y[idx] = config.y_min + (j - 1) * dyp
                tracker.particles.z[idx] = z_level
                tracker.particles.id[idx] = level_offsets[level] + (j - 1) * nx_level + i
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

    local_domain = _parallel_local_domain(tracker)
    if local_domain !== nothing
        keep = Int[]
        for i in 1:n_particles
            x = config.periodic_x ? tracker.x0 + mod(config.custom_x[i] - tracker.x0, tracker.Lx) : config.custom_x[i]
            y = config.periodic_y ? tracker.y0 + mod(config.custom_y[i] - tracker.y0, tracker.Ly) : config.custom_y[i]
            if _within_local_domain(x, y, local_domain, tracker.x0 + tracker.Lx, tracker.y0 + tracker.Ly)
                push!(keep, i)
            end
        end

        n_local = length(keep)
        resize!(tracker.particles.x, n_local)
        resize!(tracker.particles.y, n_local)
        resize!(tracker.particles.z, n_local)
        resize!(tracker.particles.id, n_local)
        resize!(tracker.particles.u, n_local)
        resize!(tracker.particles.v, n_local)
        resize!(tracker.particles.w, n_local)

        for (j, idx) in enumerate(keep)
            tracker.particles.x[j] = config.custom_x[idx]
            tracker.particles.y[j] = config.custom_y[idx]
            tracker.particles.z[j] = config.custom_z[idx]
            tracker.particles.id[j] = idx
        end

        tracker.particles.np = n_local
        return tracker
    end

    # Resize particle arrays
    resize!(tracker.particles.x, n_particles)
    resize!(tracker.particles.y, n_particles)
    resize!(tracker.particles.z, n_particles)
    resize!(tracker.particles.id, n_particles)
    resize!(tracker.particles.u, n_particles)
    resize!(tracker.particles.v, n_particles)
    resize!(tracker.particles.w, n_particles)

    # Copy custom positions
    tracker.particles.x .= config.custom_x
    tracker.particles.y .= config.custom_y
    tracker.particles.z .= config.custom_z
    tracker.particles.id .= collect(1:n_particles)

    tracker.particles.np = n_particles
    
    return tracker
end

"""
Initialize particles at random positions within the 3D domain.
"""
function initialize_random_positions!(tracker, config::ParticleConfig3D{T}) where T
    n_particles = config.nx_particles * config.ny_particles * config.nz_particles
    if n_particles <= 0
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

    rng = MersenneTwister(config.seed)
    xs = T(config.x_min) .+ (T(config.x_max) - T(config.x_min)) .* rand(rng, T, n_particles)
    ys = T(config.y_min) .+ (T(config.y_max) - T(config.y_min)) .* rand(rng, T, n_particles)
    zs = T(config.z_min) .+ (T(config.z_max) - T(config.z_min)) .* rand(rng, T, n_particles)

    local_domain = _parallel_local_domain(tracker)
    if local_domain !== nothing
        keep = Int[]
        for i in 1:n_particles
            x = config.periodic_x ? tracker.x0 + mod(xs[i] - tracker.x0, tracker.Lx) : xs[i]
            y = config.periodic_y ? tracker.y0 + mod(ys[i] - tracker.y0, tracker.Ly) : ys[i]
            if _within_local_domain(x, y, local_domain, tracker.x0 + tracker.Lx, tracker.y0 + tracker.Ly)
                push!(keep, i)
            end
        end

        n_local = length(keep)
        resize!(tracker.particles.x, n_local)
        resize!(tracker.particles.y, n_local)
        resize!(tracker.particles.z, n_local)
        resize!(tracker.particles.id, n_local)
        resize!(tracker.particles.u, n_local)
        resize!(tracker.particles.v, n_local)
        resize!(tracker.particles.w, n_local)

        for (j, idx) in enumerate(keep)
            tracker.particles.x[j] = xs[idx]
            tracker.particles.y[j] = ys[idx]
            tracker.particles.z[j] = zs[idx]
            tracker.particles.id[j] = idx
        end

        tracker.particles.np = n_local
        return tracker
    end

    resize!(tracker.particles.x, n_particles)
    resize!(tracker.particles.y, n_particles)
    resize!(tracker.particles.z, n_particles)
    resize!(tracker.particles.id, n_particles)
    resize!(tracker.particles.u, n_particles)
    resize!(tracker.particles.v, n_particles)
    resize!(tracker.particles.w, n_particles)

    tracker.particles.x .= xs
    tracker.particles.y .= ys
    tracker.particles.z .= zs
    tracker.particles.id .= collect(1:n_particles)

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
