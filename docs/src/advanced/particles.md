# [Particle Advection](@id particles)

```@meta
CurrentModule = QGYBJ
```

This page describes Lagrangian particle tracking in QGYBJ.jl, including the physics, numerical algorithms, and parallel implementation.

## Overview

Particle tracking allows you to:

- Follow **fluid parcels** as they move with the flow
- Compute **Lagrangian statistics** (dispersion, diffusivity)
- Track **tracer concentrations** along trajectories
- Study **mixing** and **transport** in QG-YBJ+ dynamics

## Physics of Particle Advection

### Total Velocity Field

In QG-YBJ+ dynamics, particles are advected by the **total velocity field** consisting of:

**1. Geostrophic Flow** (from streamfunction ψ):
```math
u_{QG} = -\frac{\partial \psi}{\partial y}, \quad v_{QG} = \frac{\partial \psi}{\partial x}
```

**2. Wave-Induced Stokes Drift** (from wave amplitude A):
```math
u_{wave} = 2\,\text{Re}\left[A^* \frac{\partial A}{\partial x}\right] = \frac{\partial |A|^2}{\partial x}
```
```math
v_{wave} = 2\,\text{Re}\left[A^* \frac{\partial A}{\partial y}\right] = \frac{\partial |A|^2}{\partial y}
```

**3. Vertical Velocity** (two options):

*QG Omega Equation:*
```math
\nabla^2 w + \frac{N^2}{f^2}\frac{\partial^2 w}{\partial z^2} = 2\,J(\psi_z, \nabla^2\psi)
```

*YBJ Formulation:*
```math
w = -\frac{f^2}{N^2}\left[\left(\frac{\partial A}{\partial x}\right)_z - i\left(\frac{\partial A}{\partial y}\right)_z\right] + \text{c.c.}
```

### Total Velocity

The complete velocity used for particle advection is:
```math
\mathbf{u}_{total} = (u_{QG} + u_{wave},\; v_{QG} + v_{wave},\; w)
```

## Quick Start

### Basic Setup

```julia
using QGYBJ

# Create simulation configuration
config = SimulationConfig(
    domain = DomainConfig(nx=128, ny=128, nz=64),
    # ... other settings
)

# Set up simulation
sim = setup_simulation(config)

# Create particle configuration
particle_config = create_particle_config(
    x_min = 0.0, x_max = 2π,
    y_min = 0.0, y_max = 2π,
    z_level = π/2,           # Vertical level
    nx_particles = 10,       # 10×10 = 100 particles
    ny_particles = 10,
    integration_method = :rk4,
    interpolation_method = TRILINEAR,
    save_interval = 0.1
)

# Create particle tracker
tracker = ParticleTracker(particle_config, sim.grid)
initialize_particles!(tracker, particle_config)

# Advect particles during simulation
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, dt, sim.current_time)
end

# Save trajectories
write_particle_trajectories("particles.nc", tracker)
```

## Time Integration Methods

Three integration schemes are available:

### Euler Method (1st order)
```math
\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t \cdot \mathbf{u}(\mathbf{x}_n, t_n)
```

```julia
config = create_particle_config(integration_method = :euler, ...)
```

### RK2 Midpoint Method (2nd order)
```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{u}(\mathbf{x}_n, t_n) \\
\mathbf{x}_{mid} &= \mathbf{x}_n + \frac{\Delta t}{2} \mathbf{k}_1 \\
\mathbf{k}_2 &= \mathbf{u}(\mathbf{x}_{mid}, t_n + \frac{\Delta t}{2}) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \Delta t \cdot \mathbf{k}_2
\end{aligned}
```

```julia
config = create_particle_config(integration_method = :rk2, ...)
```

### RK4 Classical Method (4th order)
```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{u}(\mathbf{x}_n, t_n) \\
\mathbf{k}_2 &= \mathbf{u}(\mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_1, t_n + \frac{\Delta t}{2}) \\
\mathbf{k}_3 &= \mathbf{u}(\mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_2, t_n + \frac{\Delta t}{2}) \\
\mathbf{k}_4 &= \mathbf{u}(\mathbf{x}_n + \Delta t\,\mathbf{k}_3, t_n + \Delta t) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
```

```julia
config = create_particle_config(integration_method = :rk4, ...)
```

| Method | Order | Velocity Evaluations/Step | Recommended Use |
|:-------|:------|:--------------------------|:----------------|
| `:euler` | 1 | 1 | Quick tests, large dt |
| `:rk2` | 2 | 2 | Balance of speed/accuracy |
| `:rk4` | 4 | 4 | High accuracy studies |

## Interpolation Methods

Velocity must be interpolated from the grid to particle positions.

### Trilinear (Default)
- **Stencil**: 2×2×2 = 8 points
- **Order**: O(h²)
- **Smoothness**: C⁰ continuous

```julia
config = create_particle_config(interpolation_method = TRILINEAR, ...)
```

### Tricubic
- **Stencil**: 4×4×4 = 64 points (Catmull-Rom splines)
- **Order**: O(h⁴)
- **Smoothness**: C¹ continuous

```julia
config = create_particle_config(interpolation_method = TRICUBIC, ...)
```

### Quintic
- **Stencil**: 6×6×6 = 216 points (B-splines)
- **Order**: O(h⁶)
- **Smoothness**: C⁴ continuous

```julia
config = create_particle_config(interpolation_method = QUINTIC, ...)
```

### Adaptive
Automatically selects trilinear or tricubic based on local field smoothness.

```julia
config = create_particle_config(interpolation_method = ADAPTIVE, ...)
```

| Method | Points | Error | Best For |
|:-------|:-------|:------|:---------|
| `TRILINEAR` | 8 | O(h²) | Speed, rough fields |
| `TRICUBIC` | 64 | O(h⁴) | Accuracy, smooth fields |
| `QUINTIC` | 216 | O(h⁶) | Highest accuracy |
| `ADAPTIVE` | 8-64 | Variable | Mixed conditions |

## Particle Initialization

### 2D Uniform Grid

```julia
config = create_particle_config(
    x_min = π/4, x_max = 7π/4,
    y_min = π/4, y_max = 7π/4,
    z_level = π,              # Single z-level
    nx_particles = 20,
    ny_particles = 20         # 400 particles total
)
```

### 3D Layered Distribution

```julia
# Particles at multiple vertical levels
config = create_layered_distribution(
    0.0, 2π,        # x range
    0.0, 2π,        # y range
    [π/4, π/2, 3π/4],  # z levels
    10, 10          # nx, ny per level
)
# Creates 300 particles (10×10×3)
```

### 3D Random Distribution

```julia
config = create_random_3d_distribution(
    0.0, 2π,        # x range
    0.0, 2π,        # y range
    0.0, 2π,        # z range
    500             # total particles
)
```

### Custom Positions

```julia
x_custom = [1.0, 2.0, 3.0]
y_custom = [1.0, 2.0, 3.0]
z_custom = [π/2, π/2, π/2]

config = create_custom_distribution(x_custom, y_custom, z_custom)
```

## Boundary Conditions

### Horizontal (Periodic)
Particles wrap around domain edges:
```julia
x_new = mod(x, Lx)
y_new = mod(y, Ly)
```

### Vertical (Reflective)
Particles bounce off top and bottom:
```julia
if z < 0
    z = -z
    w = -w  # Reverse vertical velocity
elseif z > Lz
    z = 2*Lz - z
    w = -w
end
```

Configure via:
```julia
config = create_particle_config(
    periodic_x = true,
    periodic_y = true,
    reflect_z = true,     # Reflective vertical BCs
    ...
)
```

## Delayed Particle Release

Start advecting particles after the flow has developed:

```julia
config = create_particle_config(
    particle_advec_time = 1.0,  # Start advecting at t=1.0
    ...
)
```

Particles remain stationary until `current_time >= particle_advec_time`.

## Trajectory Output

### Save Interval

Control how often positions are recorded:
```julia
config = create_particle_config(
    save_interval = 0.1,      # Save every 0.1 time units
    max_save_points = 1000,   # Max points per file
    ...
)
```

### Automatic File Splitting

For long simulations:
```julia
tracker = ParticleTracker(config, grid)
enable_auto_file_splitting!(tracker, "long_run", max_points_per_file=500)

# Files created: long_run.nc, long_run_part1.nc, long_run_part2.nc, ...
```

### Writing Trajectories

```julia
# Standard output
write_particle_trajectories("particles.nc", tracker)

# With metadata
write_particle_trajectories("particles.nc", tracker;
    metadata = Dict("experiment" => "test1", "description" => "...")
)

# By z-level (for layered distributions)
write_particle_trajectories_by_zlevel("particles", tracker)
# Creates: particles_z0.nc, particles_z1.nc, ...
```

## Parallel Algorithm

When running with MPI, particle advection uses domain decomposition.

### Domain Decomposition

The domain is split in the x-direction across MPI ranks:

```
┌─────────────────────────────────────────────────────┐
│           Domain: [0, Lx] × [0, Ly] × [0, Lz]       │
│                                                     │
│   ┌──────────┬──────────┬──────────┬──────────┐     │
│   │  Rank 0  │  Rank 1  │  Rank 2  │  Rank 3  │     │
│   │x∈[0,Lx/4)│x∈[Lx/4,  │x∈[Lx/2,  │x∈[3Lx/4, │     │
│   │          │   Lx/2)  │  3Lx/4)  │   Lx)    │     │
│   └──────────┴──────────┴──────────┴──────────┘     │
│                                                     │
│   Each rank owns particles within its x-range       │
└─────────────────────────────────────────────────────┘
```

### Halo Exchange

For interpolation near domain boundaries, velocity data is exchanged between neighbors:

```
┌─────────────────────────────────────────────────────────────┐
│                     HALO EXCHANGE                           │
│                                                             │
│   Rank 0                        Rank 1                      │
│   ┌─────────────────┐          ┌─────────────────┐          │
│   │ Local │  Right  │          │ Left  │ Local   │          │
│   │ Data  │  Halo   │  ←────→  │ Halo  │ Data    │          │
│   │       │ (ghost) │          │(ghost)│         │          │
│   └───────┴─────────┘          └───────┴─────────┘          │
│                                                             │
│   • Rank 0 sends RIGHT edge → Rank 1's LEFT halo            │
│   • Rank 1 sends LEFT edge  → Rank 0's RIGHT halo           │
│                                                             │
│   Halo width = 2 cells (enough for trilinear/tricubic)      │
└─────────────────────────────────────────────────────────────┘
```

### Particle Migration

When particles cross domain boundaries, they are transferred:

```
┌─────────────────────────────────────────────────────────────┐
│                   PARTICLE MIGRATION                        │
│                                                             │
│   1. After advection, check each particle's position        │
│   2. If x outside local domain → pack into send buffer      │
│   3. MPI.Alltoall to exchange particle counts               │
│   4. MPI.Send/Recv to transfer particle data                │
│   5. Unpack received particles into local collection        │
│                                                             │
│   Particle data transferred: [x, y, z, u, v, w]             │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Timestep Workflow

```
┌───────────────────────────────────────────────────────────────┐
│                  PARALLEL ADVECTION TIMESTEP                  │
│                                                               │
│  1. UPDATE VELOCITY FIELDS                                    │
│     • Compute QG velocities (distributed FFT)                 │
│     • Solve omega equation (tridiagonal in z)                 │
│     • Add wave Stokes drift                                   │
│     • Exchange velocity halos (MPI)                           │
│                              ↓                                │
│  2. ADVECT PARTICLES (each rank processes local particles)    │
│     • Interpolate velocity (use halo for boundary particles)  │
│     • Time integration (Euler/RK2/RK4)                        │
│                              ↓                                │
│  3. MIGRATE PARTICLES                                         │
│     • Identify particles that left local domain               │
│     • Exchange particle data between ranks (MPI)              │
│                              ↓                                │
│  4. APPLY BOUNDARY CONDITIONS                                 │
│     • Periodic wrap in x, y                                   │
│     • Reflective bounce in z                                  │
│                              ↓                                │
│  5. SAVE TRAJECTORIES (if save_interval reached)              │
│     • Each rank saves local particles, or                     │
│     • Gather to rank 0 for unified output                     │
└───────────────────────────────────────────────────────────────┘
```

### Using Parallel Particles

```julia
using MPI
using QGYBJ

MPI.Init()

# Set up parallel configuration
parallel_config = setup_parallel_environment()

# Create particle tracker with parallel support
tracker = ParticleTracker(particle_config, grid, parallel_config)
initialize_particles!(tracker, particle_config)

# Advection automatically handles:
# - Halo exchange for boundary interpolation
# - Particle migration between ranks
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, dt, sim.current_time)
end

MPI.Finalize()
```

## Key Data Structures

### ParticleConfig

```julia
struct ParticleConfig{T}
    # Spatial domain
    x_min, x_max, y_min, y_max::T
    z_level::T

    # Particle count
    nx_particles, ny_particles::Int

    # Physics
    use_ybj_w::Bool           # YBJ vs QG vertical velocity
    use_3d_advection::Bool    # Include vertical advection

    # Timing
    particle_advec_time::T    # Delayed start time

    # Numerics
    integration_method::Symbol        # :euler, :rk2, :rk4
    interpolation_method::InterpolationMethod  # TRILINEAR, etc.

    # Boundaries
    periodic_x, periodic_y::Bool
    reflect_z::Bool

    # I/O
    save_interval::T
    max_save_points::Int
end
```

### ParticleTracker

```julia
mutable struct ParticleTracker{T}
    config::ParticleConfig{T}
    particles::ParticleState{T}   # x, y, z, u, v, w arrays

    # Grid info
    nx, ny, nz::Int
    Lx, Ly, Lz, dx, dy, dz::T

    # Velocity workspace
    u_field, v_field, w_field::Array{T,3}

    # MPI info (for parallel)
    comm, rank, nprocs
    local_domain::NamedTuple
    halo_info::HaloInfo{T}
    send_buffers, recv_buffers::Vector{Vector{T}}

    # I/O
    save_counter::Int
    last_save_time::T
end
```

## Performance Considerations

| Aspect | Serial | Parallel |
|:-------|:-------|:---------|
| Velocity computation | O(N) | O(N/P) per rank |
| Interpolation | O(Np × stencil) | O(Np/P × stencil) |
| Halo exchange | N/A | O(ny × nz × halo_width) |
| Migration | N/A | O(Np_crossing) |

**Tips:**
- Use `TRILINEAR` for speed, `TRICUBIC` for accuracy
- RK4 costs 4× more than Euler but is much more accurate
- Halo exchange overhead is small for typical particle counts
- Migration cost depends on flow strength near boundaries

## Visualization

### Plot Particle Positions

```julia
using Plots

# 2D scatter plot
scatter(tracker.particles.x, tracker.particles.y,
    markersize=2, alpha=0.6,
    xlabel="x", ylabel="y",
    title="Particle Distribution"
)
```

### Plot Trajectories

```julia
# Load saved trajectories
using NCDatasets
ds = NCDataset("particles.nc")
x_hist = ds["x"][:]  # (np, ntime)
y_hist = ds["y"][:]
close(ds)

# Plot first 50 particle tracks
p = plot(legend=false)
for i in 1:50
    plot!(p, x_hist[i,:], y_hist[i,:], alpha=0.3)
end
display(p)
```

### Animation

```julia
anim = @animate for t in 1:10:size(x_hist, 2)
    scatter(x_hist[:,t], y_hist[:,t],
        markersize=2, xlim=(0,2π), ylim=(0,2π),
        title="t = $(t)")
end
gif(anim, "particles.gif", fps=20)
```

## API Reference

See the [Particle API Reference](../api/particles.md) for complete documentation of:

- [`ParticleConfig`](@ref) - Configuration options
- [`ParticleState`](@ref) - Particle positions and velocities
- [`ParticleTracker`](@ref) - Main tracking object
- [`create_particle_config`](@ref) - Configuration constructor
- [`initialize_particles!`](@ref) - Initialize particle positions
- [`advect_particles!`](@ref) - Advect particles one timestep
- [`interpolate_velocity_at_position`](@ref) - Velocity interpolation
- [`write_particle_trajectories`](@ref) - Save to NetCDF
