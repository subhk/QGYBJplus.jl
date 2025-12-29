# [Parallel Particle Algorithm](@id parallel-particles)

```@meta
CurrentModule = QGYBJplus
```

This page provides detailed technical documentation of the parallel particle advection algorithm in QGYBJ+.jl.

## Overview

The parallel particle algorithm enables efficient Lagrangian particle tracking on distributed-memory systems using MPI. Key features:

- **1D domain decomposition** in x-direction
- **Halo exchange** for cross-boundary velocity interpolation
- **Automatic particle migration** between MPI ranks
- **Non-blocking MPI communication** for efficiency
- **Load balancing** as particles redistribute

## Domain Decomposition

### Slab Decomposition

The physical domain `[0, Lx] × [0, Ly] × [0, Lz]` is partitioned into slabs along the x-direction:

```
         Physical Domain
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   x=0                                                     x=Lx  │
│    │                                                        │   │
│    ▼                                                        ▼   │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│   │          │          │          │          │          │      │
│   │  Rank 0  │  Rank 1  │  Rank 2  │  Rank 3  │  Rank 4  │      │
│   │          │          │          │          │          │      │
│   │ x ∈ [0,  │ x ∈ [L/5,│ x ∈ [2L/5│ x ∈ [3L/5│ x ∈ [4L/5│      │
│   │    L/5)  │   2L/5)  │   3L/5)  │   4L/5)  │    L)    │      │
│   │          │          │          │          │          │      │
│   └──────────┴──────────┴──────────┴──────────┴──────────┘      │
│                                                                 │
│   • Each rank owns a contiguous x-range                         │
│   • Full y and z dimensions on each rank                        │
│   • Particles "belong" to the rank containing their x-position  │
└─────────────────────────────────────────────────────────────────┘
```

### Local Domain Calculation

```julia
function compute_local_domain(grid::Grid, rank::Int, nprocs::Int)
    # Base points per rank
    nx_local = grid.nx ÷ nprocs
    remainder = grid.nx % nprocs

    # Handle uneven division (first 'remainder' ranks get +1 point)
    if rank < remainder
        nx_local += 1
        x_start = rank * nx_local
    else
        x_start = remainder * (nx_local + 1) + (rank - remainder) * nx_local
    end

    x_end = x_start + nx_local - 1

    # Convert grid indices to physical coordinates
    dx = grid.Lx / grid.nx
    x_start_phys = x_start * dx
    x_end_phys = (x_end + 1) * dx

    return (
        x_start = x_start_phys,
        x_end = x_end_phys,
        y_start = 0.0,
        y_end = grid.Ly,
        z_start = 0.0,
        z_end = grid.Lz,
        nx_local = nx_local
    )
end
```

**Example**: 256 grid points, 4 ranks → 64 points per rank

| Rank | Grid Indices | Physical Range (Lx=2π) |
|:-----|:-------------|:-----------------------|
| 0 | [0, 63] | [0, π/2) |
| 1 | [64, 127] | [π/2, π) |
| 2 | [128, 191] | [π, 3π/2) |
| 3 | [192, 255] | [3π/2, 2π) |

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
┌────────────────────────────────────────────────────────────────────────┐
│                    EXTENDED ARRAY LAYOUT (Rank 1)                      │
│                                                                        │
│   Index:   1    2   ...  hw   hw+1  ...  hw+nx  hw+nx+1 ... hw+nx+hw   │
│           ┌────┬────┬───┬────┬─────┬───┬───────┬───────┬───┬────────┐  │
│           │    │    │   │    │     │   │       │       │   │        │  │
│           │ Left Halo   │ ←── Local Data ──→   │ Right Halo│        │  │
│           │ (from R0)   │      (owned)         │ (from R2) │        │  │
│           └────┴────┴───┴────┴─────┴───┴───────┴───────┴───┴────────┘  │
│                                                                        │
│   hw = halo_width (depends on interpolation method)                    │
│   nx = local grid points                                               │
│                                                                        │
│   Total extended size: nx + 2*hw                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### Communication Pattern (Periodic Boundaries)

For doubly-periodic domains, the halo exchange wraps around:

```
┌────────────────────────────────────────────────────────────────────────┐
│                  PERIODIC HALO EXCHANGE COMMUNICATION                  │
│                                                                        │
│   RANK 0                 RANK 1                 RANK 2 (last)          │
│   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐       │
│   │ L │ Local│ R │       │ L │ Local│ R │       │ L │ Local│ R │       │
│   │   │      │   │       │   │      │   │       │   │      │   │       │
│   └───┴──────┴───┘       └───┴──────┴───┘       └───┴──────┴───┘       │
│     ↑                                                      │           │
│     └──────────────── Periodic wrap ──────────────────────┘           │
│                                                                        │
│   Interior Communication (same as before):                             │
│   ═════════════════════════════════════                                │
│   Rank 0 → Rank 1:  send_right → recv_left                             │
│   Rank 1 → Rank 0:  send_left  → recv_right                            │
│   Rank 1 → Rank 2:  send_right → recv_left                             │
│   Rank 2 → Rank 1:  send_left  → recv_right                            │
│                                                                        │
│   Periodic Boundary Communication:                                     │
│   ════════════════════════════════                                     │
│   Rank 2 → Rank 0:  send_right (R2's right edge) → recv_left (R0)      │
│   Rank 0 → Rank 2:  send_left  (R0's left edge)  → recv_right (R2)     │
│                                                                        │
│   After exchange:                                                      │
│   • R0's left halo contains R2's right edge data (periodic wrap!)      │
│   • R2's right halo contains R0's left edge data (periodic wrap!)      │
│   • All particles can interpolate correctly near boundaries            │
└────────────────────────────────────────────────────────────────────────┘
```

**Key**: With `periodic_x=true` (default), rank 0's left neighbor is the last rank,
and the last rank's right neighbor is rank 0. This ensures correct velocity
interpolation for particles near the periodic boundaries.

### Implementation

```julia
function exchange_velocity_halos!(halo_info, u_field, v_field, w_field)
    # 1. Copy local data to center of extended arrays
    copy_local_to_extended!(halo_info, u_field, v_field, w_field)

    # 2. Pack boundary data into send buffers
    pack_halo_data!(halo_info)
    # send_left  ← our LEFT edge  (for left neighbor's RIGHT halo)
    # send_right ← our RIGHT edge (for right neighbor's LEFT halo)

    # 3. Post non-blocking receives
    if left_neighbor >= 0
        MPI.Irecv!(recv_left, left_neighbor, tag=0, comm)
    end
    if right_neighbor >= 0
        MPI.Irecv!(recv_right, right_neighbor, tag=1, comm)
    end

    # 4. Send to neighbors (non-blocking)
    if right_neighbor >= 0
        MPI.Isend(send_right, right_neighbor, tag=0, comm)
    end
    if left_neighbor >= 0
        MPI.Isend(send_left, left_neighbor, tag=1, comm)
    end

    # 5. Wait for receives and unpack
    MPI.Waitall(recv_reqs)
    unpack_halo_data!(halo_info)
    # recv_left  → our LEFT halo region
    # recv_right → our RIGHT halo region

    # 6. Wait for sends to complete
    MPI.Waitall(send_reqs)
end
```

### Buffer Layout

Each buffer contains packed velocity components:

```
Buffer layout: [u₁, v₁, w₁, u₂, v₂, w₂, ..., uₙ, vₙ, wₙ]

where n = halo_width × ny × nz

Total buffer size = 3 × halo_width × ny × nz × sizeof(T)
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
        target_rank = find_target_rank(x, tracker)

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

function find_target_rank(x, tracker)
    # Handle periodic boundary
    x_periodic = tracker.config.periodic_x ? mod(x, tracker.Lx) : x

    # Determine owning rank based on x-position
    dx_rank = tracker.Lx / tracker.nprocs
    rank = min(tracker.nprocs - 1, floor(Int, x_periodic / dx_rank))

    return rank
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
│  │     if z < 0:   z = -z,        w = -w                           │         │
│  │     if z > Lz:  z = 2·Lz - z,  w = -w                           │         │
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
u_total = u_QG + u_Stokes
v_total = v_QG + v_Stokes
w_total = w_QG + w_Stokes
```

where:
- **QG velocities**: Geostrophic flow from streamfunction ψ, vertical from omega equation
- **Stokes drift**: Wave-induced drift from near-inertial wave amplitude A

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
│  │   │ compute_wave_velocities!(S, G) → Wave Stokes drift (ADDS to QG)       │ │ │
│  │   │                                                                       │ │ │
│  │   │   From wave amplitude A = |A|·e^{iφ}:                                 │ │ │
│  │   │     ∂A/∂x, ∂A/∂y in spectral: i·kₓ·Â, i·kᵧ·Â                         │ │ │
│  │   │     ∂A/∂z from S.C (computed by invert_B_to_A!)                       │ │ │
│  │   │                                                                       │ │ │
│  │   │   Transform to physical space, then:                                  │ │ │
│  │   │     u_S = Im[A*·∂A/∂x] = |A|²·∂φ/∂x  (phase gradient × intensity)    │ │ │
│  │   │     v_S = Im[A*·∂A/∂y] = |A|²·∂φ/∂y                                   │ │ │
│  │   │     w_S = Im[A*·∂A/∂z] = |A|²·∂φ/∂z                                   │ │ │
│  │   │                                                                       │ │ │
│  │   │   State.u += u_S  →  State.u = u_QG + u_S (TOTAL)                     │ │ │
│  │   │   State.v += v_S  →  State.v = v_QG + v_S (TOTAL)                     │ │ │
│  │   │   State.w += w_S  →  State.w = w_QG + w_S (TOTAL)                     │ │ │
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
- Each rank owns a slab: `[x_start, x_end] × [0, Ly] × [0, Lz]`
- `parent(state.u)` extracts only the **local portion**

```julia
# In update_velocity_fields!:
u_data = parent(state.u)  # Shape: (nz, nx_local, ny) - LOCAL data only
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
Extended Array: [left_halo | local_data | right_halo]
                     ↑           ↑            ↑
              from rank-1   owned data   from rank+1
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
| Halo exchange | O(ny × nz × halo_width × 3) per neighbor | Every timestep |
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
    halo_width::Int              # Ghost cell width (default: 2)

    # Extended arrays with halos
    u_extended::Array{T,3}       # Size: (nx_local + 2*hw, ny, nz)
    v_extended::Array{T,3}
    w_extended::Array{T,3}

    # Local data position in extended arrays
    local_start::NTuple{3,Int}   # (hw+1, 1, 1)
    local_end::NTuple{3,Int}     # (hw+nx, ny, nz)

    # Neighbor ranks (-1 if at boundary)
    left_neighbor::Int
    right_neighbor::Int

    # Communication buffers
    send_left::Vector{T}         # Size: 3 × hw × ny × nz
    send_right::Vector{T}
    recv_left::Vector{T}
    recv_right::Vector{T}

    # MPI info
    comm::Any
    rank::Int
    nprocs::Int
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
    local_domain::NamedTuple     # (x_start, x_end, y_start, y_end, ...)

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
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Set up parallel configuration
parallel_config = ParallelConfig(
    use_mpi = true,
    comm = comm,
    n_processes = nprocs
)

# Create simulation (distributed grid and state)
config = SimulationConfig(...)
sim = setup_simulation(config; parallel_config=parallel_config)

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
tracker = ParticleTracker(particle_config, sim.grid, parallel_config)
initialize_particles!(tracker, particle_config)

# Main simulation loop
for step in 1:nsteps
    # Advance fluid state
    timestep!(sim)

    # Advect particles (handles halo exchange + migration automatically)
    advect_particles!(tracker, sim.state, sim.grid, dt, sim.current_time)
end

# Save trajectories (rank 0 gathers and writes, or each rank writes independently)
if rank == 0 || !tracker.gather_for_io
    write_particle_trajectories("particles_rank$(rank).nc", tracker)
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
