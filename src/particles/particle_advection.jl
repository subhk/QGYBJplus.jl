"""
Particle advection module for QG-YBJ model.

This module implements Lagrangian particle tracking using the total velocity field
(QG + YBJ) including horizontal velocities (u,v) and vertical velocity (w) from
either QG omega equation or YBJ formulation.
"""

module ParticleAdvection

using ..QGYBJ: Grid, State, compute_velocities!

export ParticleConfig, ParticleState, ParticleTracker,
       create_particle_config, initialize_particles!,
       advect_particles!, interpolate_velocity

"""
Configuration for particle initialization and advection.
"""
Base.@kwdef struct ParticleConfig{T}
    # Horizontal region definition
    x_min::T = 0.0              # Minimum x coordinate
    x_max::T = 2π               # Maximum x coordinate  
    y_min::T = 0.0              # Minimum y coordinate
    y_max::T = 2π               # Maximum y coordinate
    
    # Vertical level
    z_level::T = π/2            # Constant z level for particle initialization
    
    # Number of particles
    nx_particles::Int = 10      # Number of particles in x direction
    ny_particles::Int = 10      # Number of particles in y direction
    
    # Advection options
    use_ybj_w::Bool = false     # Use YBJ vertical velocity (vs QG omega equation)
    use_3d_advection::Bool = true  # Include vertical advection (vs 2D only)
    
    # Time stepping
    dt_particles::T = 1e-3      # Particle time step (can be different from fluid)
    integration_method::Symbol = :rk4  # :euler, :rk2, :rk4
    
    # Boundary conditions for particles
    periodic_x::Bool = true     # Periodic in x direction
    periodic_y::Bool = true     # Periodic in y direction
    reflect_z::Bool = true      # Reflect at top/bottom boundaries
    
    # Output options
    save_interval::T = 1.0      # Time interval for saving particle positions
    max_save_points::Int = 1000 # Maximum number of saved time points
end

"""
State of a single particle or collection of particles.
"""
mutable struct ParticleState{T,A}
    # Positions
    x::A    # x coordinates [np]
    y::A    # y coordinates [np] 
    z::A    # z coordinates [np]
    
    # Velocities (for diagnostics)
    u::A    # x velocity [np]
    v::A    # y velocity [np]
    w::A    # z velocity [np]
    
    # Time tracking
    time::T              # Current time
    dt::T                # Time step
    
    # Trajectory storage
    x_history::Vector{A} # History of x positions
    y_history::Vector{A} # History of y positions  
    z_history::Vector{A} # History of z positions
    time_history::Vector{T} # History of times
    
    # Number of particles
    np::Int
end

"""
Main particle tracker that manages particle advection.
"""
mutable struct ParticleTracker{T,A}
    config::ParticleConfig{T}
    particles::ParticleState{T,A}
    
    # Grid and domain info (copied from fluid grid)
    Lx::T
    Ly::T
    Lz::T
    nx::Int
    ny::Int
    nz::Int
    
    # For interpolation
    dx::T
    dy::T
    dz::T
    
    # Velocity fields (workspace)
    u_field::A  # [nx, ny, nz]
    v_field::A  # [nx, ny, nz]
    w_field::A  # [nx, ny, nz]
    
    # Output control
    last_save_time::T
    save_counter::Int
end

"""
    create_particle_config(; kwargs...)

Create particle configuration with specified parameters.

# Examples
```julia
# 2D particle tracking at mid-depth
config = create_particle_config(
    x_min=0.0, x_max=2π, y_min=0.0, y_max=2π,
    z_level=π/2,
    nx_particles=20, ny_particles=20,
    use_3d_advection=false
)

# 3D tracking with YBJ vertical velocity
config = create_particle_config(
    nx_particles=50, ny_particles=50,
    use_ybj_w=true,
    use_3d_advection=true,
    integration_method=:rk4
)
```
"""
function create_particle_config(; kwargs...)
    return ParticleConfig(; kwargs...)
end

"""
    initialize_particles!(tracker, config, grid)

Initialize particles uniformly in the specified horizontal region.
"""
function initialize_particles!(config::ParticleConfig{T}, grid::Grid) where T
    np = config.nx_particles * config.ny_particles
    
    # Create uniform grid in horizontal region
    x_range = range(config.x_min, config.x_max, length=config.nx_particles)
    y_range = range(config.y_min, config.y_max, length=config.ny_particles)
    
    # Initialize position arrays
    x_particles = zeros(T, np)
    y_particles = zeros(T, np)
    z_particles = fill(T(config.z_level), np)
    
    # Fill positions uniformly
    idx = 1
    for i in 1:config.nx_particles
        for j in 1:config.ny_particles
            x_particles[idx] = x_range[i]
            y_particles[idx] = y_range[j]
            idx += 1
        end
    end
    
    # Initialize velocity arrays (will be filled during advection)
    u_particles = zeros(T, np)
    v_particles = zeros(T, np)
    w_particles = zeros(T, np)
    
    # Create particle state
    particles = ParticleState{T,Vector{T}}(
        x_particles, y_particles, z_particles,
        u_particles, v_particles, w_particles,
        T(0), T(config.dt_particles),
        Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{T}(),
        np
    )
    
    # Create tracker
    tracker = ParticleTracker(
        config, particles,
        grid.Lx, grid.Ly, grid.Lz, grid.nx, grid.ny, grid.nz,
        grid.Lx/grid.nx, grid.Ly/grid.ny, grid.Lz/grid.nz,
        zeros(T, grid.nx, grid.ny, grid.nz),
        zeros(T, grid.nx, grid.ny, grid.nz),
        zeros(T, grid.nx, grid.ny, grid.nz),
        T(0), 0
    )
    
    # Save initial positions
    save_particle_state!(tracker)
    
    return tracker
end

"""
    interpolate_velocity(x, y, z, u_field, v_field, w_field, grid, tracker)

Interpolate velocity at particle position using trilinear interpolation.
"""
function interpolate_velocity(x::T, y::T, z::T, 
                            u_field, v_field, w_field,
                            tracker::ParticleTracker{T}) where T
    
    # Handle periodic boundaries
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x
    y_periodic = tracker.config.periodic_y ? mod(y, tracker.Ly) : y
    
    # Clamp z to domain bounds
    z_clamped = clamp(z, 0, tracker.Lz)
    
    # Convert to grid indices (0-based for interpolation)
    fx = x_periodic / tracker.dx
    fy = y_periodic / tracker.dy
    fz = z_clamped / tracker.dz
    
    # Get integer parts and fractional parts
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    iz = floor(Int, fz)
    
    rx = fx - ix
    ry = fy - iy
    rz = fz - iz
    
    # Handle boundary indices (1-based indexing in Julia)
    ix1 = max(1, min(tracker.nx, ix + 1))
    iy1 = max(1, min(tracker.ny, iy + 1))
    iz1 = max(1, min(tracker.nz, iz + 1))
    
    ix2 = max(1, min(tracker.nx, ix + 2))
    iy2 = max(1, min(tracker.ny, iy + 2))
    iz2 = max(1, min(tracker.nz, iz + 2))
    
    # Trilinear interpolation
    # Bottom face (z1)
    u_z1_y1 = (1-rx) * u_field[ix1,iy1,iz1] + rx * u_field[ix2,iy1,iz1]
    u_z1_y2 = (1-rx) * u_field[ix1,iy2,iz1] + rx * u_field[ix2,iy2,iz1]
    u_z1 = (1-ry) * u_z1_y1 + ry * u_z1_y2
    
    v_z1_y1 = (1-rx) * v_field[ix1,iy1,iz1] + rx * v_field[ix2,iy1,iz1]
    v_z1_y2 = (1-rx) * v_field[ix1,iy2,iz1] + rx * v_field[ix2,iy2,iz1]
    v_z1 = (1-ry) * v_z1_y1 + ry * v_z1_y2
    
    w_z1_y1 = (1-rx) * w_field[ix1,iy1,iz1] + rx * w_field[ix2,iy1,iz1]
    w_z1_y2 = (1-rx) * w_field[ix1,iy2,iz1] + rx * w_field[ix2,iy2,iz1]
    w_z1 = (1-ry) * w_z1_y1 + ry * w_z1_y2
    
    # Top face (z2)  
    u_z2_y1 = (1-rx) * u_field[ix1,iy1,iz2] + rx * u_field[ix2,iy1,iz2]
    u_z2_y2 = (1-rx) * u_field[ix1,iy2,iz2] + rx * u_field[ix2,iy2,iz2]
    u_z2 = (1-ry) * u_z2_y1 + ry * u_z2_y2
    
    v_z2_y1 = (1-rx) * v_field[ix1,iy1,iz2] + rx * v_field[ix2,iy1,iz2]
    v_z2_y2 = (1-rx) * v_field[ix1,iy2,iz2] + rx * v_field[ix2,iy2,iz2]
    v_z2 = (1-ry) * v_z2_y1 + ry * v_z2_y2
    
    w_z2_y1 = (1-rx) * w_field[ix1,iy1,iz2] + rx * w_field[ix2,iy1,iz2]
    w_z2_y2 = (1-rx) * w_field[ix1,iy2,iz2] + rx * w_field[ix2,iy2,iz2]
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

"""
    update_velocity_fields!(tracker, state, grid, plans, params)

Update velocity fields from fluid state for particle advection.
"""
function update_velocity_fields!(tracker::ParticleTracker, state::State, 
                                grid::Grid, plans, params)
    # Compute velocities with chosen vertical velocity formulation
    compute_velocities!(state, grid; 
                       plans=plans, 
                       params=params,
                       compute_w=true,
                       use_ybj_w=tracker.config.use_ybj_w)
    
    # Copy to tracker workspace (already in real space)
    tracker.u_field .= state.u
    tracker.v_field .= state.v
    tracker.w_field .= state.w
    
    return tracker
end

"""
    advect_particles!(tracker, dt)

Advect particles using specified integration method.
"""
function advect_particles!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    if tracker.config.integration_method == :euler
        advect_euler!(tracker, dt)
    elseif tracker.config.integration_method == :rk2
        advect_rk2!(tracker, dt)
    elseif tracker.config.integration_method == :rk4
        advect_rk4!(tracker, dt)
    else
        error("Unknown integration method: $(tracker.config.integration_method)")
    end
    
    # Apply boundary conditions
    apply_boundary_conditions!(tracker)
    
    # Update time
    particles.time += dt
    
    return tracker
end

"""
Forward Euler advection: x^{n+1} = x^n + dt * u(x^n)
"""
function advect_euler!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        # Get velocity at current position
        u, v, w = interpolate_velocity(particles.x[i], particles.y[i], particles.z[i],
                                     tracker.u_field, tracker.v_field, tracker.w_field,
                                     tracker)
        
        # Update positions
        particles.x[i] += dt * u
        particles.y[i] += dt * v
        particles.z[i] += dt * w
        
        # Store velocities for diagnostics
        particles.u[i] = u
        particles.v[i] = v
        particles.w[i] = w
    end
end

"""
Second-order Runge-Kutta (midpoint method).
"""
function advect_rk2!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        x0, y0, z0 = particles.x[i], particles.y[i], particles.z[i]
        
        # First stage
        u1, v1, w1 = interpolate_velocity(x0, y0, z0,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Midpoint positions
        x_mid = x0 + 0.5 * dt * u1
        y_mid = y0 + 0.5 * dt * v1
        z_mid = z0 + 0.5 * dt * w1
        
        # Second stage
        u2, v2, w2 = interpolate_velocity(x_mid, y_mid, z_mid,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Final update
        particles.x[i] = x0 + dt * u2
        particles.y[i] = y0 + dt * v2
        particles.z[i] = z0 + dt * w2
        
        # Store final velocities
        particles.u[i] = u2
        particles.v[i] = v2
        particles.w[i] = w2
    end
end

"""
Fourth-order Runge-Kutta advection.
"""
function advect_rk4!(tracker::ParticleTracker{T}, dt::T) where T
    particles = tracker.particles
    
    @inbounds for i in 1:particles.np
        x0, y0, z0 = particles.x[i], particles.y[i], particles.z[i]
        
        # Stage 1
        u1, v1, w1 = interpolate_velocity(x0, y0, z0,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Stage 2
        x_temp = x0 + 0.5 * dt * u1
        y_temp = y0 + 0.5 * dt * v1
        z_temp = z0 + 0.5 * dt * w1
        u2, v2, w2 = interpolate_velocity(x_temp, y_temp, z_temp,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Stage 3
        x_temp = x0 + 0.5 * dt * u2
        y_temp = y0 + 0.5 * dt * v2
        z_temp = z0 + 0.5 * dt * w2
        u3, v3, w3 = interpolate_velocity(x_temp, y_temp, z_temp,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Stage 4
        x_temp = x0 + dt * u3
        y_temp = y0 + dt * v3
        z_temp = z0 + dt * w3
        u4, v4, w4 = interpolate_velocity(x_temp, y_temp, z_temp,
                                         tracker.u_field, tracker.v_field, tracker.w_field,
                                         tracker)
        
        # Final update
        particles.x[i] = x0 + (dt/6) * (u1 + 2*u2 + 2*u3 + u4)
        particles.y[i] = y0 + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
        particles.z[i] = z0 + (dt/6) * (w1 + 2*w2 + 2*w3 + w4)
        
        # Store final velocities (use stage 1 for simplicity)
        particles.u[i] = u1
        particles.v[i] = v1
        particles.w[i] = w1
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
                particles.w[i] = -particles.w[i]  # Reflect velocity
            elseif particles.z[i] > tracker.Lz
                particles.z[i] = 2*tracker.Lz - particles.z[i]
                particles.w[i] = -particles.w[i]  # Reflect velocity
            end
        else
            particles.z[i] = clamp(particles.z[i], 0, tracker.Lz)
        end
    end
end

"""
Save current particle state to trajectory history.
"""
function save_particle_state!(tracker::ParticleTracker)
    particles = tracker.particles
    
    # Check if we should save
    if length(particles.time_history) >= tracker.config.max_save_points
        return  # Don't save more points
    end
    
    # Save current state
    push!(particles.x_history, copy(particles.x))
    push!(particles.y_history, copy(particles.y))
    push!(particles.z_history, copy(particles.z))
    push!(particles.time_history, particles.time)
    
    tracker.save_counter += 1
    tracker.last_save_time = particles.time
    
    return tracker
end

"""
Check if it's time to save particle state.
"""
function should_save_particles(tracker::ParticleTracker)
    return (tracker.particles.time - tracker.last_save_time) >= tracker.config.save_interval
end

end # module ParticleAdvection

using .ParticleAdvection