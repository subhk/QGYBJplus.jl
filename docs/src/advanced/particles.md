# [Particle Advection](@id particles)

```@meta
CurrentModule = QGYBJplus
```

This page describes Lagrangian particle tracking in QGYBJ+.jl, including the physics, numerical algorithms, and parallel implementation.

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

*Horizontal Stokes Drift* (Xie & Vanneste 2015):
```math
u_S = \text{Im}\left[A^* \frac{\partial A}{\partial x}\right] = |A|^2 \frac{\partial \phi}{\partial x}
```
```math
v_S = \text{Im}\left[A^* \frac{\partial A}{\partial y}\right] = |A|^2 \frac{\partial \phi}{\partial y}
```

*Vertical Stokes Drift:*
```math
w_S = \text{Im}\left[A^* \frac{\partial A}{\partial z}\right] = |A|^2 \frac{\partial \phi}{\partial z}
```

where ``\phi`` is the wave phase (``A = |A|e^{i\phi}``). The Stokes drift represents particle drift in the direction of wave propagation (phase gradient), weighted by wave intensity.

The vertical derivative ∂A/∂z is computed by `invert_B_to_A!` and stored in `S.C`.

**3. QG Vertical Velocity** (from omega equation):
```math
\nabla^2 w_{QG} + \frac{f^2}{N^2}\frac{\partial^2 w_{QG}}{\partial z^2} = \frac{2f}{N^2}\,J(\psi_z, \nabla^2\psi)
```

**4. YBJ Vertical Velocity** (alternative wave-induced formulation):
```math
w_{YBJ} = -\frac{f^2}{N^2}\left[\left(\frac{\partial A}{\partial x}\right)_z - i\left(\frac{\partial A}{\partial y}\right)_z\right] + \text{c.c.}
```
This is controlled by the `use_ybj_w` option. When `use_ybj_w=true`, this wave-induced vertical velocity is used instead of solving the QG omega equation.

### Total Velocity

The complete velocity used for particle advection is:
```math
\mathbf{u}_{total} = (u_{QG} + u_{wave},\; v_{QG} + v_{wave},\; w + w_{wave})
```
where $w$ is either $w_{QG}$ (from omega equation) or $w_{YBJ}$ (wave-induced) depending on the `use_ybj_w` setting.

This includes both horizontal and **vertical Stokes drift**, ensuring particles are correctly advected by the full wave-induced velocity field.

## Quick Start

### Co-Evolution with Fluid (Recommended)

Particles can be passed directly to the timestep functions to co-evolve with the wave and mean flow equations. This ensures particles use the same `dt` as the fluid simulation:

```julia
using QGYBJplus

# Setup model first
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(par)

# Create particle configuration (100 particles, default Euler integration)
# NOTE: x_max, y_max are REQUIRED - use G.Lx, G.Ly from grid
particle_config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Create and initialize particle tracker
tracker = ParticleTracker(particle_config, G)
initialize_particles!(tracker, particle_config)

# Particles co-evolve automatically with the fluid
# Option 1: Leapfrog time stepping
first_projection_step!(S, G, par, plans; a=a, particle_tracker=tracker, current_time=0.0)
for step in 1:nsteps
    current_time = step * par.dt
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a,
                   particle_tracker=tracker, current_time=current_time)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end

# Option 2: IMEX time stepping
first_imex_step!(S, G, par, plans, imex_ws; a=a, particle_tracker=tracker, current_time=0.0)
for step in 1:nsteps
    current_time = step * par.dt
    imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a=a,
                  particle_tracker=tracker, current_time=current_time)
    Sn, Snp1 = Snp1, Sn
end

# Save trajectories
write_particle_trajectories("particles.nc", tracker)
```

### Manual Advection (Alternative)

For more control, particles can be advected manually:

```julia
# Create particle configuration
# NOTE: x_max, y_max are REQUIRED - use G.Lx, G.Ly from grid
particle_config = particles_in_box(2000.0;
    x_max=G.Lx, y_max=G.Ly,  # REQUIRED
    nx=10, ny=10,
    save_interval=0.1
)

# Create particle tracker
tracker = ParticleTracker(particle_config, sim.grid)
initialize_particles!(tracker, particle_config)

# Advect particles manually after each timestep
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, par.dt, sim.current_time)
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
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, integration_method=:euler)
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
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, integration_method=:rk2)
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
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, integration_method=:rk4)
```

| Method | Order | Velocity Evaluations/Step | Recommended Use |
|:-------|:------|:--------------------------|:----------------|
| `:euler` (default) | 1 | 1 | Co-evolution with fluid, large dt |
| `:rk2` | 2 | 2 | Balance of speed/accuracy |
| `:rk4` | 4 | 4 | High accuracy studies |

## Interpolation Methods

Velocity must be interpolated from the grid to particle positions.

### Trilinear (Default)
- **Stencil**: 2×2×2 = 8 points
- **Order**: O(h²)
- **Smoothness**: C⁰ continuous

```julia
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=TRILINEAR)
```

### Tricubic
- **Stencil**: 4×4×4 = 64 points (Catmull-Rom splines)
- **Order**: O(h⁴)
- **Smoothness**: C¹ continuous

```julia
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=TRICUBIC)
```

### Quintic
- **Stencil**: 6×6×6 = 216 points (B-splines)
- **Order**: O(h⁶)
- **Smoothness**: C⁴ continuous

```julia
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=QUINTIC)
```

### Adaptive
Automatically selects trilinear or tricubic based on local field smoothness.

```julia
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=ADAPTIVE)
```

| Method | Points | Error | Best For |
|:-------|:-------|:------|:---------|
| `TRILINEAR` | 8 | O(h²) | Speed, rough fields |
| `TRICUBIC` | 64 | O(h⁴) | Accuracy, smooth fields |
| `QUINTIC` | 216 | O(h⁶) | Highest accuracy |
| `ADAPTIVE` | 8-64 | Variable | Mixed conditions |

## Particle Initialization

QGYBJ+.jl provides simple, intuitive constructors for initializing particles:

| Constructor | Description |
|:------------|:------------|
| `particles_in_box(z; ...)` | Uniform grid in a 2D rectangular box at fixed z |
| `particles_in_circle(z; ...)` | Circular disk at fixed z (sunflower/rings/random) |
| `particles_in_grid_3d(; ...)` | Uniform 3D rectangular grid |
| `particles_in_layers(z_levels; ...)` | Multiple 2D grids at different z-levels |
| `particles_random_3d(n; ...)` | Random distribution in 3D volume |
| `particles_custom(positions; ...)` | User-specified positions |

### Particles in a Box (2D at fixed z)

```julia
# 100 particles (10×10) in a box at z = 2000m
# NOTE: x_max, y_max are REQUIRED
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Custom subdomain
config = particles_in_box(2000.0;
    x_min=100e3, x_max=400e3,  # subset of domain
    y_min=100e3, y_max=400e3,
    nx=20, ny=20               # 400 particles
)
```

### Particles in a Circle (2D at fixed z)

```julia
# 100 particles in a circle of radius 1.0 at z = π/2
config = particles_in_circle(π/2; radius=1.0, n=100)

# Custom center and pattern
config = particles_in_circle(1.0;
    center=(2.0, 2.0),        # Circle center
    radius=1.5,
    n=200,
    pattern=:sunflower        # :sunflower, :rings, or :random
)
```

**Available patterns:**
- `:sunflower` - Fibonacci spiral (very uniform, recommended)
- `:rings` - Concentric rings
- `:random` - Uniform random within disk

### Particles in a 3D Grid

```julia
# 500 particles in a 10×10×5 grid
# NOTE: x_max, y_max, z_max are REQUIRED
config = particles_in_grid_3d(; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, nx=10, ny=10, nz=5)

# Custom subdomain
config = particles_in_grid_3d(;
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    z_min=500.0, z_max=2500.0,
    nx=8, ny=8, nz=4
)
```

### Particles in Layers (multiple z-levels)

```julia
# 300 particles at 3 z-levels (10×10 per level)
# NOTE: x_max, y_max are REQUIRED
config = particles_in_layers([1000.0, 2000.0, 3000.0]; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Custom horizontal subdomain
config = particles_in_layers([500.0, 1000.0, 1500.0, 2000.0];
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    nx=5, ny=5
)
```

### Random 3D Distribution

```julia
# 500 random particles in full domain
# NOTE: x_max, y_max, z_max are REQUIRED
config = particles_random_3d(500; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz)

# Custom subdomain with seed
config = particles_random_3d(1000;
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    z_min=500.0, z_max=2500.0,
    seed=42
)
```

### Custom Positions

```julia
# Particles at specific (x, y, z) locations
config = particles_custom([
    (1.0, 1.0, 0.5),
    (2.0, 2.0, 1.0),
    (3.0, 1.5, 0.75),
    (1.5, 3.0, 1.25)
])
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
config = particles_in_box(2000.0;
    x_max=G.Lx, y_max=G.Ly,
    periodic_x=true,
    periodic_y=true,
    reflect_z=true      # Reflective vertical BCs
)
```

## Delayed Particle Release

Start advecting particles after the flow has developed:

```julia
config = particles_in_box(2000.0; x_max=G.Lx, y_max=G.Ly, particle_advec_time=100.0)  # Start at t=100.0
```

Particles remain stationary until `current_time >= particle_advec_time`.

## Trajectory Output

### Save Interval

Control how often positions are recorded:
```julia
config = particles_in_box(2000.0;
    x_max=G.Lx, y_max=G.Ly,
    save_interval=10.0,       # Save every 10.0 time units
    max_save_points=1000      # Max points per file
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

The domain is split in both x and y according to the MPI process grid (px × py).
Each rank owns a tile in the horizontal plane and the full z-range.

Example for a 2×2 topology:

```
┌─────────────────────────┬─────────────────────────┐
│ Rank (0,1)               │ Rank (1,1)              │
│ x∈[0, Lx/2), y∈[Ly/2, Ly) │ x∈[Lx/2, Lx), y∈[Ly/2, Ly) │
├─────────────────────────┼─────────────────────────┤
│ Rank (0,0)               │ Rank (1,0)              │
│ x∈[0, Lx/2), y∈[0, Ly/2)  │ x∈[Lx/2, Lx), y∈[0, Ly/2)  │
└─────────────────────────┴─────────────────────────┘
```

Particles belong to the rank that owns their (x, y) position.

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
│   Halo width depends on interpolation: 1 (trilinear), 2     │
│   (tricubic), 3 (quintic/adaptive)                          │
└─────────────────────────────────────────────────────────────┘
```

For 2D topologies (py > 1), halos are exchanged in both x and y directions,
including corner halos needed by wider stencils.

### Particle Migration

When particles cross domain boundaries, they are transferred:

```
┌─────────────────────────────────────────────────────────────┐
│                   PARTICLE MIGRATION                        │
│                                                             │
│   1. After advection, check each particle's position        │
│   2. If (x,y) outside local domain → pack into send buffer  │
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
│     • Exchange velocity halos in x/y (and corners for 2D)     │
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
using QGYBJplus

MPI.Init()

# Set up parallel configuration
parallel_config = setup_mpi_environment()

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
mutable struct ParticleTracker{T}  # Simplified view (omits I/O bookkeeping)
    config::ParticleConfig{T}
    particles::ParticleState{T}   # x, y, z, u, v, w arrays

    # Grid info
    nx, ny, nz::Int
    Lx, Ly, Lz, dx, dy, dz::T

    # Velocity workspace
    u_field, v_field, w_field::Array{T,3}

    # MPI info (for parallel)
    comm, rank, nprocs, is_parallel
    local_domain::NamedTuple  # x/y bounds, local sizes, topology info
    halo_info::HaloInfo{T}
    send_buffers, recv_buffers::Vector{Vector{T}}
    is_io_rank::Bool
    gather_for_io::Bool
end
```

## Performance Considerations

| Aspect | Serial | Parallel |
|:-------|:-------|:---------|
| Velocity computation | O(N) | O(N/P) per rank |
| Interpolation | O(Np × stencil) | O(Np/P × stencil) |
| Halo exchange | N/A | O((nx_local + ny_local) × nz × halo_width) |
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

**Types:**
- [`ParticleConfig`](@ref) - Configuration options
- [`ParticleState`](@ref) - Particle positions and velocities
- [`ParticleTracker`](@ref) - Main tracking object

**Initialization Constructors:**
- [`particles_in_box`](@ref) - 2D box at fixed z-level
- [`particles_in_circle`](@ref) - Circular disk at fixed z-level
- [`particles_in_grid_3d`](@ref) - Uniform 3D grid
- [`particles_in_layers`](@ref) - Multiple z-levels
- [`particles_random_3d`](@ref) - Random 3D distribution
- [`particles_custom`](@ref) - User-specified positions

**Core Functions:**
- [`initialize_particles!`](@ref) - Initialize particle positions
- [`advect_particles!`](@ref) - Advect particles one timestep
- [`interpolate_velocity_at_position`](@ref) - Velocity interpolation
- [`write_particle_trajectories`](@ref) - Save to NetCDF
