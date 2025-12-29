# [Particle API](@id api-particles)

```@meta
CurrentModule = QGYBJplus
```

This page documents the particle advection API for Lagrangian tracking.

## Core Types

```@docs
ParticleConfig
ParticleState
ParticleTracker
```

## Particle Initialization Constructors

Simple, intuitive functions for creating particle distributions:

```@docs
particles_in_box
particles_in_circle
particles_in_grid_3d
particles_in_layers
particles_random_3d
particles_custom
```

## Initialization and Advection

```@docs
initialize_particles!
advect_particles!
interpolate_velocity_at_position
```

## I/O Functions

```@docs
write_particle_trajectories
read_particle_trajectories
write_particle_snapshot
write_particle_trajectories_by_zlevel
```

## Interpolation Methods

```@docs
InterpolationMethod
```

## Parallel Utilities

```@docs
validate_particle_cfl
```

## 3D Particle Types

```@docs
ParticleConfig3D
ParticleDistribution
initialize_particles_3d!
```

## Quick Reference

| Constructor | Description | Example |
|:------------|:------------|:--------|
| `particles_in_box(z; x_max, y_max, ...)` | 2D box at fixed z | `particles_in_box(500.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)` |
| `particles_in_circle(z; ...)` | Circular disk | `particles_in_circle(1.0; radius=0.5, n=100)` |
| `particles_in_grid_3d(; x_max, y_max, z_max, ...)` | 3D grid | `particles_in_grid_3d(; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, nx=10, ny=10, nz=5)` |
| `particles_in_layers(zs; x_max, y_max, ...)` | Multiple z-levels | `particles_in_layers([500.0, 1000.0, 1500.0]; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)` |
| `particles_random_3d(n; x_max, y_max, z_max, ...)` | Random 3D | `particles_random_3d(500; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz)` |
| `particles_custom(pos; ...)` | Custom positions | `particles_custom([(1.0,1.0,0.5), ...])` |

## Usage Example

```julia
using QGYBJplus

# Setup model
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(par)

# Create particle configuration (100 particles in a box at z = 2000m)
# NOTE: x_max, y_max are REQUIRED - use G.Lx, G.Ly from grid
pconfig = particles_in_box(2000.0;
    x_max=G.Lx, y_max=G.Ly,  # REQUIRED
    nx=10, ny=10,
    integration_method=:rk4,
    save_interval=0.1
)

# Or use a circular distribution (no x_max/y_max needed - computed from center/radius)
pconfig = particles_in_circle(2000.0; center=(G.Lx/2, G.Ly/2), radius=50e3, n=100)

# Or multiple z-levels (x_max, y_max REQUIRED)
pconfig = particles_in_layers([1000.0, 2000.0, 3000.0]; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Create tracker and initialize
tracker = ParticleTracker(pconfig, G)
initialize_particles!(tracker, pconfig)

# Advection loop
dt = par.dt
for step in 1:1000
    compute_velocities!(S, G, plans)
    advect_particles!(tracker, S, G, dt, step * dt)
end

# Write trajectories
write_particle_trajectories("trajectories.nc", tracker)
```
