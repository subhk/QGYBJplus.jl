# [Parallel Particle Algorithm](@id parallel-particles)

```@meta
CurrentModule = QGYBJplus
```

This page provides detailed technical documentation of the parallel particle advection algorithm in QGYBJ+.jl.

## Overview

The parallel particle algorithm enables efficient Lagrangian particle tracking on distributed-memory systems using MPI. Key features:

- **2D domain decomposition** in x and y (px × py topology)
- **Halo exchange** in x/y (plus corners) for cross-boundary interpolation
- **Automatic particle migration** between MPI ranks
- **Non-blocking MPI communication** for efficiency
- **Load balancing** as particles redistribute

## Domain Decomposition

### 2D Tile Decomposition

The physical domain `[0, Lx] × [0, Ly] × [-Lz, 0]` is partitioned into tiles across
x and y according to the MPI process grid (px × py). Each rank owns a contiguous
tile in x/y and the full z-range.

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

- Each rank owns contiguous x and y ranges
- Full z dimension on each rank
- Particles belong to the rank containing their (x, y) position

### Local Domain Calculation

```julia
function compute_local_domain(grid::Grid, rank::Int, nprocs::Int; topology=nothing)
    # Determine process grid (px × py)
    px, py = if topology !== nothing
        topology
    elseif grid.decomp !== nothing && hasfield(typeof(grid.decomp), :topology)
        grid.decomp.topology
    else
        (nprocs, 1)
    end

    rank_x = rank % px
    rank_y = rank ÷ px

    nx_local = compute_local_size(grid.nx, px, rank_x)
    ny_local = compute_local_size(grid.ny, py, rank_y)

    x_start = compute_start_index(grid.nx, px, rank_x)
    y_start = compute_start_index(grid.ny, py, rank_y)

    dx = grid.Lx / grid.nx
    dy = grid.Ly / grid.ny

    x_start_phys = x_start * dx
    x_end_phys = (x_start + nx_local) * dx
    y_start_phys = y_start * dy
    y_end_phys = (y_start + ny_local) * dy

    return (
        x_start = x_start_phys,
        x_end = x_end_phys,
        y_start = y_start_phys,
        y_end = y_end_phys,
        z_start = -grid.Lz,
        z_end = 0.0,
        nx_local = nx_local,
        ny_local = ny_local,
        px = px,
        py = py,
        rank_x = rank_x,
        rank_y = rank_y
    )
end
```

**Example**: 256×256 grid points, topology (2,2) → local 128×128 tiles per rank.

## Halo Exchange

### Purpose

When a particle is near a domain boundary, velocity interpolation requires data from the neighboring rank. **Halo regions** (ghost cells) store copies of neighbor data.

### Halo Width Requirements

The halo width is automatically determined by the interpolation method:

| Interpolation | Stencil | Halo Width | Accuracy |
|:--------------|:--------|:-----------|:---------|
| TRILINEAR | 2×2×2 | 1 cell | O(h²) |
| TRICUBIC | 4×4×4 | 2 cells | O(h⁴) |
| QUINTIC | 6×6×6 | 3 cells | O(h⁶) |
| ADAPTIVE | varies | 3 cells | varies |

The `required_halo_width()` function computes this automatically:

```julia
using QGYBJplus: required_halo_width, TRILINEAR, TRICUBIC, QUINTIC

required_halo_width(TRILINEAR)  # → 1
required_halo_width(TRICUBIC)   # → 2
required_halo_width(QUINTIC)    # → 3
```

### Extended Array Structure

```
┌─────────────────────────────────────────────────────────────┐
│            EXTENDED ARRAY LAYOUT (local + halos)            │
│                                                             │
│   y (north)                                                  │
│      ▲                                                      │
│      │   ┌───────────────────────────────┐                  │
│      │   │           top halo           │                  │
│      │   ├───────────┬───────────┬───────┤                  │
│      │   │ left halo │   local   │ right │                  │
│      │   │           │   data    │ halo  │                  │
│      │   ├───────────┴───────────┴───────┤                  │
│      │   │         bottom halo           │                  │
│      │   └───────────────────────────────┘                  │
│                                                             │
│   hw = halo_width (depends on interpolation method)         │
│   Extended size: (nz_local, nx_local + 2*hw, ny_local + 2*hw)│
└─────────────────────────────────────────────────────────────┘
```

### Communication Pattern (Periodic Boundaries)

For periodic domains, neighbor relationships wrap in x and y. In 2D, each rank
has up to 8 neighbors:

```
    NW(7) --- N(4) --- NE(8)
      |        |        |
    W(1) --- local --- E(2)
      |        |        |
    SW(5) --- S(3) --- SE(6)
```

With `periodic_x`/`periodic_y` enabled, edge neighbors wrap around (e.g., west of
rank_x=0 is rank_x=px-1). For py==1, only W/E neighbors are used.

### Implementation

```julia
function exchange_velocity_halos!(halo_info, u_field, v_field, w_field)
    # 1. Copy local data to center of extended arrays
    copy_local_to_extended!(halo_info, u_field, v_field, w_field)

    # 2. Pack boundary data into send buffers (W/E/S/N and corners)
    #    For py==1, only W/E buffers are used.

    # 3. Post non-blocking receives from all neighbors (up to 8)

    # 4. Send to neighbors (non-blocking)

    # 5. Wait for receives and unpack into halo regions
    MPI.Waitall(recv_reqs)

    # 6. Wait for sends to complete
    MPI.Waitall(send_reqs)
end
```

### Buffer Layout

Each buffer contains packed velocity components:

```
Buffer layout: [u₁, v₁, w₁, u₂, v₂, w₂, ..., uₙ, vₙ, wₙ]

where n depends on the buffer type:
- X-direction buffers: halo_width × ny_local × nz_local
- Y-direction buffers: nx_local × halo_width × nz_local
- Corner buffers: halo_width × halo_width × nz_local
```

## Particle Migration

### When Migration Occurs

After advection, particles may have moved outside their owning rank's domain:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     PARTICLE CROSSING BOUNDARY                         │
│                                                                        │
│   Before advection:                                                    │
│   ┌───────────────────────┬───────────────────────┐                    │
│   │       RANK 0          │       RANK 1          │                    │
│   │                    •  │                       │                    │
│   │                   ↗   │                       │                    │
│   │     Particle moving   │                       │                    │
│   │     toward boundary   │                       │                    │
│   └───────────────────────┴───────────────────────┘                    │
│                                                                        │
│   After advection:                                                     │
│   ┌───────────────────────┬───────────────────────┐                    │
│   │       RANK 0          │       RANK 1          │                    │
│   │                       │  •                    │                    │
│   │                       │  ↑ Particle now in    │                    │
│   │                       │    Rank 1's domain    │                    │
│   └───────────────────────┴───────────────────────┘                    │
│                                                                        │
│   → Particle must be migrated from Rank 0 to Rank 1                    │
└────────────────────────────────────────────────────────────────────────┘
```

The same logic applies across y-boundaries when py > 1.

### Migration Algorithm

```julia
function migrate_particles!(tracker)
    particles = tracker.particles
    local_domain = tracker.local_domain

    # 1. Clear send buffers
    for buf in tracker.send_buffers
        empty!(buf)
    end

    # 2. Identify particles to keep vs migrate
    keep_indices = Int[]

    for i in 1:particles.np
        x = particles.x[i]
        y = particles.y[i]
        target_rank = find_target_rank(x, y, tracker)

        if target_rank == tracker.rank
            # Particle stays local
            push!(keep_indices, i)
        else
            # Package particle for migration: [x, y, z, u, v, w]
            particle_data = [
                particles.x[i], particles.y[i], particles.z[i],
                particles.u[i], particles.v[i], particles.w[i]
            ]
            append!(tracker.send_buffers[target_rank + 1], particle_data)
        end
    end

    # 3. Remove migrated particles from local arrays
    particles.x = particles.x[keep_indices]
    particles.y = particles.y[keep_indices]
    # ... same for z, u, v, w
    particles.np = length(keep_indices)

    # 4. Exchange particle data via MPI
    exchange_particles!(tracker)

    # 5. Add received particles to local collection
    add_received_particles!(tracker)
end

function find_target_rank(x, y, tracker)
    # Handle periodic boundaries
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x
    y_periodic = tracker.config.periodic_y ? mod(y, tracker.Ly) : y

    # Convert to grid indices (0-based)
    ix = floor(Int, x_periodic / tracker.dx)
    iy = floor(Int, y_periodic / tracker.dy)
    ix = min(ix, tracker.nx - 1)
    iy = min(iy, tracker.ny - 1)

    # Determine process grid (px × py)
    px, py = tracker.local_domain === nothing ?
             compute_process_grid(tracker.nprocs) :
             (tracker.local_domain.px, tracker.local_domain.py)

    # Map indices to rank coordinates (handles uneven division)
    rank_x = _rank_for_index(ix, tracker.nx, px)
    rank_y = _rank_for_index(iy, tracker.ny, py)

    return rank_y * px + rank_x
end
```

### All-to-All Communication

```julia
function exchange_particles!(tracker)
    comm = tracker.comm
    nprocs = tracker.nprocs

    # 1. Exchange particle counts (how many to send to each rank)
    send_counts = [length(tracker.send_buffers[i]) ÷ 6 for i in 1:nprocs]
    recv_counts = MPI.Alltoall(send_counts, comm)

    # 2. Point-to-point sends/receives for particle data
    for other_rank in 0:nprocs-1
        if other_rank == tracker.rank
            continue
        end

        # Send to other_rank
        if !isempty(tracker.send_buffers[other_rank + 1])
            MPI.Send(tracker.send_buffers[other_rank + 1], other_rank, tag=0, comm)
        end

        # Receive from other_rank
        if recv_counts[other_rank + 1] > 0
            n_values = recv_counts[other_rank + 1] * 6
            recv_data = Vector{T}(undef, n_values)
            MPI.Recv!(recv_data, other_rank, tag=0, comm)
            tracker.recv_buffers[other_rank + 1] = recv_data
        end
    end
end
```

## Complete Parallel Timestep

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              PARALLEL PARTICLE ADVECTION TIMESTEP                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: UPDATE VELOCITY FIELDS                                         │  │
│  │                                                                        │  │
│  │   • Compute geostrophic velocities (distributed FFT):                  │  │
│  │       û = -i·kᵧ·ψ̂,  v̂ = i·kₓ·ψ̂                                         │  │
│  │                                                                        │  │
│  │   • Solve omega equation for w (tridiagonal in z):                     │  │
│  │       ∇²w + (f²/N²)∂²w/∂z² = (2f/N²)·J(ψ_z, ∇²ψ)                       │  │
│  │                                                                        │  │
│  │   • Add wave Stokes drift (horizontal + vertical):                     │  │
│  │       u += Im[A*·∂A/∂x],  v += Im[A*·∂A/∂y],  w += Im[A*·∂A/∂z]        │  │
│  │                                                                        │  │
│  │   • Exchange velocity halos (MPI non-blocking)                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │ 
│                                ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │ STEP 2: ADVECT PARTICLES (each rank independently)              │         │
│  │                                                                 │         │
│  │   For each local particle:                                      │         │
│  │     1. Interpolate velocity at (x, y, z)                        │         │
│  │        • Use extended arrays (halos) if near boundary           │         │
│  │        • Trilinear/Tricubic/Quintic interpolation               │         │
│  │                                                                 │         │
│  │     2. Time integration:                                        │         │
│  │        • Euler:  x_{n+1} = x_n + dt·u                           │         │
│  │        • RK2:    Midpoint method                                │         │
│  │        • RK4:    Classical 4th order                            │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │ STEP 3: MIGRATE PARTICLES                                       │         │
│  │                                                                 │         │
│  │   1. Identify particles outside local domain                    │         │
│  │   2. Pack outgoing particles into send buffers                  │         │
│  │   3. MPI.Alltoall - exchange particle counts                    │         │
│  │   4. MPI.Send/Recv - transfer particle data                     │         │
│  │   5. Unpack incoming particles into local arrays                │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │ STEP 4: APPLY BOUNDARY CONDITIONS                               │         │
│  │                                                                 │         │
│  │   Horizontal (periodic):  x = mod(x, Lx),  y = mod(y, Ly)       │         │
│  │                                                                 │         │
│  │   Vertical (reflective):                                        │         │
│  │     if z > 0:    z = -z,         w = -w                          │         │
│  │     if z < -Lz:  z = -2·Lz - z, w = -w                           │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │ STEP 5: SAVE TRAJECTORIES (if save_interval reached)            │         │
│  │                                                                 │         │
│  │   Option A: Each rank saves local particles independently       │         │
│  │   Option B: Gather all particles to rank 0, unified output      │         │
│  └─────────────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Velocity Data Flow: QG + Wave

This section explains how particles obtain the combined QG flow and wave Stokes drift velocities in parallel mode.

### Overview

Particles are advected by the **total velocity field**:
```
u_total = u_QG + u_wave + u_Stokes
v_total = v_QG + v_wave + v_Stokes
w_total = w_QG + w_Stokes
```

where:
- **QG velocities**: Geostrophic flow from streamfunction ψ, vertical from omega equation
- **Wave velocity**: Backrotated wave orbital velocity Re(LA), Im(LA) from YBJ+ equation
- **Stokes drift**: Wave-induced drift from full Jacobian (Wagner & Young 2016)

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    VELOCITY DATA FLOW FOR PARTICLE ADVECTION                      │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: advect_particles!(tracker, state, grid, dt)                         │ │
│  │                            ↓                                                │ │
│  │         calls update_velocity_fields!(tracker, state, grid)                 │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                ↓                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: compute_total_velocities!(state, grid; plans=tracker.plans, ...)    │ │
│  │                                                                             │ │
│  │   ┌───────────────────────────────────────────────────────────────────────┐ │ │
│  │   │ compute_velocities!(S, G) → QG velocities                              │ │ │
│  │   │                                                                       │ │ │
│  │   │   HORIZONTAL (geostrophic):                                           │ │ │
│  │   │     û = -i·kᵧ·ψ̂   →  u_QG = -∂ψ/∂y                                    │ │ │
│  │   │     v̂ =  i·kₓ·ψ̂   →  v_QG = +∂ψ/∂x                                    │ │ │
│  │   │                                                                       │ │ │
│  │   │   VERTICAL (omega equation):                                          │ │ │
│  │   │     ∇²w + (f²/N²)∂²w/∂z² = (2f/N²)·J(ψ_z, ∇²ψ)                        │ │ │
│  │   │     → Tridiagonal solve per (kₓ,kᵧ) → w_QG                            │ │ │
│  │   │                                                                       │ │ │
│  │   │   State.u = u_QG, State.v = v_QG, State.w = w_QG                      │ │ │
│  │   └───────────────────────────────────────────────────────────────────────┘ │ │
│  │                               ↓                                             │ │
│  │   ┌───────────────────────────────────────────────────────────────────────┐ │ │
│  │   │ compute_wave_velocities!(S, G) → Wave velocity + Stokes drift         │ │ │
│  │   │                                                                       │ │ │
│  │   │   WAVE VELOCITY (Asselin & Young 2019, eq 1.2):                       │ │ │
│  │   │     LA = B + (k_h²/4)·A  in spectral space (YBJ+ relation)           │ │ │
│  │   │     u_wave = Re(LA), v_wave = Im(LA)                                  │ │ │
│  │   │                                                                       │ │ │
│  │   │   HORIZONTAL STOKES DRIFT (Wagner & Young 2016, eq 3.16a, 3.18):     │ │ │
│  │   │     J₀ = (LA)*·∂_{s*}(LA) - (f₀²/N²)·(∂_{s*}A_z*)·∂_z(LA)           │ │ │
│  │   │     u_S = Im(J₀)/f₀,  v_S = -Re(J₀)/f₀                               │ │ │
│  │   │                                                                       │ │ │
│  │   │   VERTICAL STOKES DRIFT (Wagner & Young 2016, eq 3.19-3.20):         │ │ │
│  │   │     w_S = -2·Im(K₀)/f₀  where K₀ = M*_z·M_{ss*} - M*_{s*}·M_{sz}    │ │ │
│  │   │                                                                       │ │ │
│  │   │   State.u += u_wave + u_S  →  u_QG + u_wave + u_S (TOTAL)            │ │ │
│  │   │   State.v += v_wave + v_S  →  v_QG + v_wave + v_S (TOTAL)            │ │ │
│  │   │   State.w += w_S           →  w_QG + w_S (TOTAL)                      │ │ │
│  │   └───────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                ↓                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: Copy to tracker (LOCAL data only in parallel)                       │ │
│  │                                                                             │ │
│  │   u_data = parent(state.u)   # LOCAL portion of distributed array          │ │
│  │   v_data = parent(state.v)                                                  │ │
│  │   w_data = parent(state.w)                                                  │ │
│  │                                                                             │ │
│  │   tracker.u_field .= u_data  # Copy TOTAL velocities to tracker            │ │
│  │   tracker.v_field .= v_data                                                 │ │
│  │   tracker.w_field .= w_data                                                 │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                ↓                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 4: Halo Exchange (PARALLEL MODE ONLY)                                  │ │
│  │                                                                             │ │
│  │   exchange_velocity_halos!(halo_info, u_field, v_field, w_field)            │ │
│  │                                                                             │ │
│  │   Rank 0           Rank 1           Rank 2                                  │ │
│  │   [u,v,w]    →     [u,v,w]    →     [u,v,w]                                │ │
│  │      ↓                ↓                ↓                                    │ │
│  │   [H|local|H]     [H|local|H]     [H|local|H]                              │ │
│  │      └──send──→──recv──┘──send──→──recv──┘                                 │ │
│  │                                                                             │ │
│  │   Extended arrays now contain:                                              │ │
│  │   • Left halo: neighbor's RIGHT edge velocities                             │ │
│  │   • Local: own velocity data (QG + wave)                                   │ │
│  │   • Right halo: neighbor's LEFT edge velocities                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                ↓                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 5: Particle Interpolation                                              │ │
│  │                                                                             │ │
│  │   For each particle at (x, y, z):                                           │ │
│  │     (u, v, w) = interpolate_velocity_at_position(x, y, z, tracker)          │ │
│  │                                                                             │ │
│  │   • Uses extended arrays (local + halos)                                    │ │
│  │   • TRILINEAR / TRICUBIC / QUINTIC interpolation                           │ │
│  │   • Returns TOTAL velocity (QG + wave) at particle position                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

#### 1. State Contains TOTAL Velocities

After `compute_total_velocities!`, the State arrays contain the **sum** of QG and wave contributions:

```julia
# In compute_total_velocities!:
compute_velocities!(S, G; ...)      # Sets S.u, S.v, S.w to QG values
compute_wave_velocities!(S, G; ...) # ADDS Stokes drift to S.u, S.v, S.w
```

#### 2. Parallel Data Distribution (PencilArrays)

In MPI parallel mode:
- `state.u`, `state.v`, `state.w` are **distributed arrays** (PencilArrays)
- Each rank owns a tile: `[x_start, x_end] × [y_start, y_end] × [-Lz, 0]`
- `parent(state.u)` extracts only the **local portion**

```julia
# In update_velocity_fields!:
u_data = parent(state.u)  # Shape: (nz, nx_local, ny_local) - LOCAL data only
tracker.u_field .= u_data # Copy to tracker's workspace
```

#### 3. Wave Data Source

The wave amplitude `A` and its derivatives come from the simulation state:

| Field | Location | Description |
|:------|:---------|:------------|
| `State.A` | Spectral | Wave amplitude Â(kₓ, kᵧ, z) |
| `State.C` | Spectral | A_z = ∂A/∂z (set by `invert_B_to_A!`) |

The Stokes drift computation:
1. Computes ∂A/∂x, ∂A/∂y in spectral space: `i·kₓ·Â`, `i·kᵧ·Â`
2. Transforms A, ∂A/∂x, ∂A/∂y, ∂A/∂z to physical space
3. Computes `Im[A* · ∂A/∂(x,y,z)]` pointwise in physical space
4. Adds result to existing QG velocities

#### 4. Halo Exchange Enables Cross-Boundary Interpolation

After halo exchange, each rank's extended arrays contain:

```
Extended Array: halos in x and y around the local tile
                (left/right + bottom/top + corners)
```

This allows particles near domain boundaries to interpolate velocities using neighbor data without additional MPI communication during advection.

### Code Path Summary

```julia
advect_particles!(tracker, state, grid, dt)
  └─→ update_velocity_fields!(tracker, state, grid)
        ├─→ compute_total_velocities!(state, grid)
        │     ├─→ compute_velocities!()      # QG: u,v from ψ; w from omega eqn
        │     └─→ compute_wave_velocities!() # Stokes: Im[A*·∇A] added to u,v,w
        ├─→ tracker.u_field .= parent(state.u)  # Copy LOCAL total velocity
        └─→ exchange_velocity_halos!()          # Fill halos from neighbors
  └─→ advect_euler!/advect_rk2!/advect_rk4!()
        └─→ interpolate_velocity_at_position()  # Uses extended arrays
```

## CFL Stability for Particle Advection

### Halo Constraint

For multi-stage integration methods (RK2, RK4), intermediate particle positions are evaluated during a single timestep. If a particle moves beyond the halo region, velocity interpolation becomes inaccurate (values are clamped to boundary).

**Stability requirement:**
```
max_velocity × dt < halo_width × dx
```

where:
- `max_velocity`: Maximum expected flow velocity
- `dt`: Timestep
- `halo_width`: Ghost cell count (1, 2, or 3 depending on interpolation)
- `dx`: Grid spacing

### Validation Function

Use `validate_particle_cfl` to check if your timestep is appropriate:

```julia
using QGYBJplus: validate_particle_cfl

# Estimate maximum velocity in your simulation
max_velocity = 0.1  # m/s

# Check if timestep is safe
if !validate_particle_cfl(tracker, max_velocity, dt)
    @warn "Timestep may exceed halo region; consider reducing dt"
end
```

### Recommendations

1. **Use higher-order interpolation**: TRICUBIC (hw=2) or QUINTIC (hw=3) provide larger halo regions
2. **Reduce timestep**: If dt is too large for the flow velocities
3. **Use Euler integration**: Only evaluates velocity at current position (no intermediate stages)

**Example CFL calculation:**
```julia
# Grid: 256 points, Lx = 2π, using TRICUBIC (hw=2)
dx = 2π / 256  ≈ 0.0245
halo_width = 2
safe_displacement = halo_width * dx ≈ 0.049

# If max velocity = 0.1 m/s, max safe dt:
dt_max = safe_displacement / max_velocity ≈ 0.49
```

## Scalability Analysis

### Communication Costs

| Operation | Data Volume | Frequency |
|:----------|:------------|:----------|
| Halo exchange | O((nx_local + ny_local) × nz × halo_width × 3) | Every timestep |
| Migration Alltoall | O(nprocs) integers | Every timestep |
| Migration Send/Recv | O(Np_crossing × 6) floats | Every timestep |

### Scaling Characteristics

**Strong Scaling** (fixed problem size, varying ranks):
- Velocity computation: ~O(N/P) per rank
- Particle advection: ~O(Np/P) per rank
- Communication: O(boundary_size) - increases relative importance

**Weak Scaling** (fixed load per rank):
- Ideal scaling if particles are uniformly distributed
- Load imbalance if particles cluster in few ranks

### Load Balancing Considerations

Particle load can become imbalanced if:
1. Flow concentrates particles (e.g., in eddies)
2. Initial distribution is non-uniform
3. Some ranks have many boundary crossings

**Mitigation strategies:**
- Use sufficiently many particles for statistical averaging
- Initialize particles uniformly
- Monitor particle count per rank

## Data Structures

### HaloInfo

```julia
mutable struct HaloInfo{T}
    halo_width::Int

    # Extended arrays with halos (x and y)
    u_extended::Array{T,3}       # Size: (nz_local, nx_local + 2*hw, ny_local + 2*hw)
    v_extended::Array{T,3}
    w_extended::Array{T,3}

    # Local data position in extended arrays
    local_start::NTuple{3,Int}   # (1, hw+1, hw+1)
    local_end::NTuple{3,Int}     # (nz_local, hw+nx_local, hw+ny_local)

    # Neighbor ranks (W, E, S, N, SW, SE, NW, NE)
    neighbors::Vector{Int}

    # Communication buffers (x/y edges + corners)
    send_west::Vector{T}
    send_east::Vector{T}
    recv_west::Vector{T}
    recv_east::Vector{T}
    send_south::Vector{T}
    send_north::Vector{T}
    recv_south::Vector{T}
    recv_north::Vector{T}
    send_sw::Vector{T}
    send_se::Vector{T}
    send_nw::Vector{T}
    send_ne::Vector{T}
    recv_sw::Vector{T}
    recv_se::Vector{T}
    recv_nw::Vector{T}
    recv_ne::Vector{T}

    # MPI + topology info
    comm::Any
    rank::Int
    nprocs::Int
    nx_global::Int
    ny_global::Int
    nz_global::Int
    nx_local::Int
    ny_local::Int
    nz_local::Int
    px::Int
    py::Int
    rank_x::Int
    rank_y::Int
    periodic_x::Bool
    periodic_y::Bool
    is_2d_decomposition::Bool
end
```

### ParticleTracker (Parallel Fields)

```julia
mutable struct ParticleTracker{T}
    # ... common fields ...

    # MPI configuration
    comm::Any                    # MPI.COMM_WORLD
    rank::Int                    # This process's rank
    nprocs::Int                  # Total processes
    is_parallel::Bool            # true if nprocs > 1

    # Domain decomposition
    local_domain::NamedTuple     # (x_start, x_end, y_start, y_end, nx_local, ny_local, px, py, rank_x, rank_y, ...)

    # Halo exchange
    halo_info::HaloInfo{T}       # Extended arrays + buffers

    # Migration buffers (one per rank)
    send_buffers::Vector{Vector{T}}  # Length: nprocs
    recv_buffers::Vector{Vector{T}}  # Length: nprocs

    # I/O configuration
    is_io_rank::Bool             # true if rank == 0
    gather_for_io::Bool          # Gather particles to rank 0 for output
end
```

## Usage Example

```julia
using MPI
using QGYBJplus

# Initialize MPI
MPI.Init()
# Set up parallel configuration
mpi_config = setup_mpi_environment()

# Create distributed grid and state
par = default_params(Lx=2π, Ly=2π, Lz=1.0, nx=64, ny=64, nz=32)
grid = init_mpi_grid(par, mpi_config)
plans = plan_mpi_transforms(grid, mpi_config)
state = init_mpi_state(grid, plans, mpi_config)

# Create particle configuration
particle_config = create_particle_config(
    x_min = 0.0, x_max = 2π,
    y_min = 0.0, y_max = 2π,
    z_level = π,
    nx_particles = 100,
    ny_particles = 100,
    integration_method = :rk4,
    interpolation_method = TRILINEAR
)

# Create particle tracker with parallel support
tracker = ParticleTracker(particle_config, grid, mpi_config)
initialize_particles!(tracker, particle_config)

# Main simulation loop (advance fluid state separately)
for step in 1:nsteps
    current_time = step * par.dt
    advect_particles!(tracker, state, grid, par.dt, current_time)
end

# Save trajectories (rank 0 gathers and writes, or each rank writes independently)
if mpi_config.is_root || !tracker.gather_for_io
    write_particle_trajectories("particles_rank$(mpi_config.rank).nc", tracker)
end

MPI.Finalize()
```

## Debugging Tips

### Check Particle Distribution

```julia
# Print particle count per rank
println("Rank $rank has $(tracker.particles.np) particles")
MPI.Barrier(comm)
```

### Verify Halo Exchange

```julia
# After halo exchange, check extended array values
if rank == 0
    println("Left halo (should be from neighbor): ", halo_info.u_extended[1:2, 1, 1])
    println("Local data start: ", halo_info.u_extended[3:4, 1, 1])
end
```

### Monitor Migration

```julia
# Track particles crossing boundaries
n_migrated = sum(length.(tracker.send_buffers)) ÷ 6
println("Rank $rank migrating $n_migrated particles")
```

## See Also

- [Particle Advection](particles.md) - General particle documentation
- [MPI Parallelization](parallel.md) - Overall parallel architecture
- [API Reference](../api/particles.md) - Function documentation
