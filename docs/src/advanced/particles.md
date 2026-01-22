# [Particle Advection](@id particles)

```@meta
CurrentModule = QGYBJplus
```

This page describes Lagrangian particle tracking in QGYBJ+.jl using the Generalized Lagrangian Mean (GLM) framework.

## Overview

Particle tracking allows you to:

- Follow **fluid parcels** as they move with the flow
- Compute **Lagrangian statistics** (dispersion, diffusivity)
- Track **tracer concentrations** along trajectories
- Study **mixing** and **transport** in QG-YBJ+ dynamics

## Generalized Lagrangian Mean (GLM) Framework

QGYBJ+.jl implements particle advection using the **Generalized Lagrangian Mean (GLM)** framework, which cleanly separates the slowly-evolving mean trajectory from fast wave oscillations.

### Position Decomposition

The physical particle position ``\mathbf{x}(t)`` is decomposed into:

```math
\mathbf{x}(t) = \mathbf{X}(t) + \boldsymbol{\xi}(\mathbf{X}(t), t)
```

where:
- ``\mathbf{X}(t)`` is the **mean position** (Lagrangian-mean trajectory)
- ``\boldsymbol{\xi}`` is the **wave displacement** (fast oscillatory component)

In component form:
```math
x = X + \xi_x, \quad y = Y + \xi_y, \quad z = Z + \xi_z
```

### Mean Position Advection

The mean position evolves according to the **QG velocity** (which is the Lagrangian-mean flow):

```math
\frac{d\mathbf{X}}{dt} = \mathbf{u}_{QG}(\mathbf{X}, t)
```

where the QG velocities are:
```math
u_{QG} = -\frac{\partial \psi}{\partial y}, \quad v_{QG} = \frac{\partial \psi}{\partial x}
```

The mean vertical position ``Z`` is advected by ``w_{QG}`` from the omega equation (or remains constant if `use_3d_advection=false`).

This is time-stepped using **Euler method**:
```math
\mathbf{X}^{n+1} = \mathbf{X}^n + \Delta t \cdot \mathbf{u}_{QG}(\mathbf{X}^n, t^n)
```

### Wave Displacement

The wave displacement is **not** time-stepped—it is reconstructed diagnostically from the wave field at each output time.

#### Horizontal Wave Displacement

From the wave velocity amplitude ``LA``:
```math
\xi_x + i\xi_y = \text{Re}\left\{\frac{LA}{-if_0} e^{-if_0 t}\right\}
```

where ``LA`` uses the YBJ operator:
- ``L = \partial_z(f_0^2/N^2)\partial_z``
- ``L^+ = L - k_h^2/4`` (YBJ+ operator)
- Since ``L^+A`` is the evolved variable: ``LA = L^+A + (k_h^2/4)A``

#### Vertical Wave Displacement (Equation 2.10)

From Asselin & Young (2019) equation (2.10):
```math
\xi_z = \xi_{z,\cos} \cos(f_0 t) + \xi_{z,\sin} \sin(f_0 t)
```

where:
```math
\xi_{z,\cos} = \frac{f_0}{N^2} w_{\sin}, \quad \xi_{z,\sin} = -\frac{f_0}{N^2} w_{\cos}
```

with:
```math
w_{\cos} = \text{Re}(\partial_x A_z) + \text{Im}(\partial_y A_z), \quad
w_{\sin} = \text{Im}(\partial_x A_z) - \text{Re}(\partial_y A_z)
```

### Why GLM?

| Aspect | Benefit |
|:-------|:--------|
| **Numerical stability** | Mean trajectory uses larger ``\Delta t`` (not constrained by wave period) |
| **Physical clarity** | Separates slow (QG) and fast (wave) dynamics |
| **Efficiency** | Wave displacement is diagnostic, not prognostic |
| **Accuracy** | No accumulation of phase errors in wave oscillations |

---

## Quick Start

### Co-Evolution with Fluid (Recommended)

Particles can be passed to the timestep functions to co-evolve with the fluid:

```julia
using QGYBJplus

# Setup model
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(par)

# Create particle configuration (100 particles at depth 2000 m)
particle_config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Create and initialize particle tracker
tracker = ParticleTracker(particle_config, G)
initialize_particles!(tracker, particle_config)

# Particles co-evolve with the fluid
first_projection_step!(S, G, par, plans; a=a, particle_tracker=tracker, current_time=0.0)
for step in 1:nsteps
    current_time = step * par.dt
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a,
                   particle_tracker=tracker, current_time=current_time)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end

# Save trajectories
write_particle_trajectories("particles.nc", tracker)
```

### Manual Advection

For more control:

```julia
# Create and initialize tracker
tracker = ParticleTracker(particle_config, G)
initialize_particles!(tracker, particle_config)

# Advect manually after each timestep
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, par.dt, sim.current_time;
                      params=par)
end

write_particle_trajectories("particles.nc", tracker)
```

---

## Configuration Options

### 2D vs 3D Advection

| Setting | Behavior |
|:--------|:---------|
| `use_3d_advection = true` (default) | Mean position Z is advected by ``w_{QG}`` |
| `use_3d_advection = false` | Z remains at initial depth (only ``\xi_z`` varies) |

When `use_3d_advection = false`, the physical vertical position is:
```math
z(t) = Z_0 + \xi_z(t)
```
where ``Z_0`` is the initial depth and ``\xi_z(t)`` is the oscillating wave displacement.

```julia
# Particles stay at fixed mean depth, only wave displacement varies
config = particles_in_box(-2000.0;
    x_max=G.Lx, y_max=G.Ly,
    use_3d_advection=false
)
```

### Vertical Velocity Source

| Setting | Vertical velocity for mean advection |
|:--------|:-------------------------------------|
| `use_ybj_w = false` (default) | QG omega equation |
| `use_ybj_w = true` | YBJ wave-induced ``w`` |

Note: The vertical wave displacement ``\xi_z`` is always computed from equation (2.10), regardless of this setting.

---

## Interpolation Methods

Velocity must be interpolated from the grid to particle positions.

| Method | Stencil | Order | Best For |
|:-------|:--------|:------|:---------|
| `TRILINEAR` (default) | 8 points | O(h²) | Speed, rough fields |
| `TRICUBIC` | 64 points | O(h⁴) | Accuracy, smooth fields |
| `QUINTIC` | 216 points | O(h⁶) | Highest accuracy |
| `ADAPTIVE` | 8-64 | Variable | Mixed conditions |

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly,
                          interpolation_method=TRICUBIC)
```

---

## Particle Initialization

### Simple Constructors

| Constructor | Description |
|:------------|:------------|
| `particles_in_box(z; ...)` | Uniform grid at fixed z |
| `particles_in_circle(z; ...)` | Circular disk at fixed z |
| `particles_in_grid_3d(; ...)` | Uniform 3D grid |
| `particles_in_layers(z_levels; ...)` | Multiple z-levels |
| `particles_random_3d(n; ...)` | Random 3D distribution |
| `particles_custom(positions; ...)` | User-specified positions |

Note: Vertical coordinate is `z ∈ [-Lz, 0]` with `z = 0` at the surface.

### Examples

```julia
# 100 particles at depth 2000 m
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Circular patch
config = particles_in_circle(-1000.0; center=(250e3, 250e3), radius=50e3, n=200)

# 3D grid
config = particles_in_grid_3d(; x_max=G.Lx, y_max=G.Ly, z_max=0.0,
                               z_min=-G.Lz, nx=10, ny=10, nz=5)

# Multiple layers
config = particles_in_layers([-500.0, -1000.0, -2000.0];
                              x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)
```

---

## Boundary Conditions

### Horizontal (Periodic)
```julia
x_new = mod(x - x0, Lx) + x0
y_new = mod(y - y0, Ly) + y0
```

### Vertical (Reflective)
Particles bounce off top (z=0) and bottom (z=-Lz):
```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly,
    periodic_x=true,
    periodic_y=true,
    reflect_z=true
)
```

---

## Trajectory Output

### Save Interval

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly,
    save_interval=10.0,       # Save every 10 time units
    max_save_points=1000      # Max points per file
)
```

### Automatic File Splitting

For long simulations:
```julia
tracker = ParticleTracker(config, grid)
enable_auto_file_splitting!(tracker, "long_run", max_points_per_file=500)

# Files: long_run.nc, long_run_part1.nc, long_run_part2.nc, ...
```

### Delayed Release

Start advecting after flow develops:
```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly,
    particle_advec_time=100.0  # Start at t=100
)
```

---

## Output Variables

The trajectory file contains:

| Variable | Description |
|:---------|:------------|
| `x, y, z` | Mean positions ``(X, Y, Z)`` |
| `xi_x, xi_y, xi_z` | Wave displacements ``(\xi_x, \xi_y, \xi_z)`` |
| `time` | Time stamps |
| `id` | Particle IDs |

To reconstruct physical positions:
```julia
x_physical = x + xi_x
y_physical = y + xi_y
z_physical = z + xi_z
```

---

## Parallel Execution

When running with MPI, particle advection uses domain decomposition automatically.

### Key Features
- **Halo exchange**: Velocity data exchanged for boundary interpolation
- **Particle migration**: Particles transferred when they cross domain boundaries
- **Consistent topology**: Uses the same MPI decomposition as the fluid solver

```julia
using MPI
using QGYBJplus

MPI.Init()
parallel_config = setup_mpi_environment()

tracker = ParticleTracker(particle_config, grid, parallel_config)
initialize_particles!(tracker, particle_config)

# Advection handles halo exchange and migration automatically
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, dt, current_time)
end

MPI.Finalize()
```

---

## Visualization

### Plot Positions

```julia
using Plots

scatter(tracker.particles.x, tracker.particles.y,
    markersize=2, xlabel="x", ylabel="y", title="Particle Distribution")
```

### Plot Trajectories

```julia
using NCDatasets

ds = NCDataset("particles.nc")
x_hist = ds["x"][:]  # (np, ntime)
y_hist = ds["y"][:]

# Plot first 50 tracks
plot(legend=false)
for i in 1:50
    plot!(x_hist[i,:], y_hist[i,:], alpha=0.3)
end
```

### Animation

```julia
anim = @animate for t in 1:10:size(x_hist, 2)
    scatter(x_hist[:,t], y_hist[:,t], markersize=2,
        xlim=(0, Lx), ylim=(0, Ly), title="t = $(t)")
end
gif(anim, "particles.gif", fps=20)
```

---

## Key Data Structures

### ParticleConfig

```julia
struct ParticleConfig{T}
    x_min, x_max, y_min, y_max::T  # Horizontal domain
    z_level::T                      # Initial depth
    nx_particles, ny_particles::Int # Particle count

    use_ybj_w::Bool                # Vertical velocity source
    use_3d_advection::Bool         # Include vertical mean advection
    particle_advec_time::T         # Delayed start time
    interpolation_method           # TRILINEAR, TRICUBIC, etc.

    periodic_x, periodic_y::Bool   # Horizontal BCs
    reflect_z::Bool                # Vertical BCs

    save_interval::T               # Output frequency
    max_save_points::Int           # Max points per file
end
```

### ParticleState

Contains particle data:
- Mean positions: `x, y, z`
- Particle IDs: `id`
- QG velocities: `u, v, w`
- Horizontal wave displacement: `xi_x, xi_y`
- Vertical wave displacement: `xi_z`
- Wave amplitude: `LA_real, LA_imag`
- Vertical displacement coefficients: `xi_z_cos, xi_z_sin`

---

## References

- **Asselin, O. & Young, W. R.** (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Fluid Mech.*, 876, 428-448.
- **Wagner, G. L. & Young, W. R.** (2016). A three-component model for the coupled evolution of near-inertial waves, quasi-geostrophic flow and the near-inertial second harmonic. *J. Fluid Mech.*, 802, 806-837.

## API Reference

See the [Particle API Reference](../api/particles.md) for complete documentation.
