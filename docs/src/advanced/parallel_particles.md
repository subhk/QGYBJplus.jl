# [Parallel Particle Algorithm](@id parallel-particles)

```@meta
CurrentModule = QGYBJ
```

This page provides detailed technical documentation of the parallel particle advection algorithm in QGYBJ.jl.

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
│                                                                  │
│   x=0                                                     x=Lx  │
│    │                                                        │    │
│    ▼                                                        ▼    │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────┐     │
│   │          │          │          │          │          │      │
│   │  Rank 0  │  Rank 1  │  Rank 2  │  Rank 3  │  Rank 4  │      │
│   │          │          │          │          │          │      │
│   │ x ∈ [0,  │ x ∈ [L/5,│ x ∈ [2L/5│ x ∈ [3L/5│ x ∈ [4L/5│      │
│   │    L/5)  │   2L/5)  │   3L/5)  │   4L/5)  │    L)    │      │
│   │          │          │          │          │          │      │
│   └──────────┴──────────┴──────────┴──────────┴──────────┘     │
│                                                                  │
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

### Extended Array Structure

```
┌────────────────────────────────────────────────────────────────────────┐
│                    EXTENDED ARRAY LAYOUT (Rank 1)                      │
│                                                                         │
│   Index:   1    2   ...  hw   hw+1  ...  hw+nx  hw+nx+1 ... hw+nx+hw   │
│           ┌────┬────┬───┬────┬─────┬───┬───────┬───────┬───┬────────┐  │
│           │    │    │   │    │     │   │       │       │   │        │  │
│           │ Left Halo  │ ←── Local Data ──→    │ Right Halo │        │  │
│           │ (from R0)  │      (owned)          │ (from R2)  │        │  │
│           └────┴────┴───┴────┴─────┴───┴───────┴───────┴───┴────────┘  │
│                                                                         │
│   hw = halo_width (default: 2)                                         │
│   nx = local grid points                                                │
│                                                                         │
│   Total extended size: nx + 2*hw                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Communication Pattern

```
┌────────────────────────────────────────────────────────────────────────┐
│                      HALO EXCHANGE COMMUNICATION                        │
│                                                                         │
│   RANK 0                 RANK 1                 RANK 2                  │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐          │
│   │ L │ Local│ R │       │ L │ Local│ R │       │ L │ Local│ R │        │
│   │   │      │   │       │   │      │   │       │   │      │   │        │
│   └───┴──────┴───┘       └───┴──────┴───┘       └───┴──────┴───┘        │
│                                                                         │
│   Communication:                                                        │
│   ═══════════════                                                       │
│                                                                         │
│   Rank 0 → Rank 1:  send_right (R0's right edge) → recv_left (R1)      │
│   Rank 1 → Rank 0:  send_left  (R1's left edge)  → recv_right (R0)     │
│                                                                         │
│   Rank 1 → Rank 2:  send_right (R1's right edge) → recv_left (R2)      │
│   Rank 2 → Rank 1:  send_left  (R2's left edge)  → recv_right (R1)     │
│                                                                         │
│   After exchange:                                                       │
│   • R1's left halo contains R0's right edge data                       │
│   • R1's right halo contains R2's left edge data                       │
│   • Particles in R1 can interpolate using neighbor data                │
└────────────────────────────────────────────────────────────────────────┘
```

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
│                     PARTICLE CROSSING BOUNDARY                          │
│                                                                         │
│   Before advection:                                                     │
│   ┌───────────────────────┬───────────────────────┐                    │
│   │       RANK 0          │       RANK 1          │                    │
│   │                    •  │                       │                    │
│   │                   ↗   │                       │                    │
│   │     Particle moving   │                       │                    │
│   │     toward boundary   │                       │                    │
│   └───────────────────────┴───────────────────────┘                    │
│                                                                         │
│   After advection:                                                      │
│   ┌───────────────────────┬───────────────────────┐                    │
│   │       RANK 0          │       RANK 1          │                    │
│   │                       │  •                    │                    │
│   │                       │  ↑ Particle now in    │                    │
│   │                       │    Rank 1's domain    │                    │
│   └───────────────────────┴───────────────────────┘                    │
│                                                                         │
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
┌────────────────────────────────────────────────────────────────────────┐
│              PARALLEL PARTICLE ADVECTION TIMESTEP                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: UPDATE VELOCITY FIELDS                                   │   │
│  │                                                                   │   │
│  │   • Compute geostrophic velocities (distributed FFT):            │   │
│  │       û = -i·kᵧ·ψ̂,  v̂ = i·kₓ·ψ̂                                  │   │
│  │                                                                   │   │
│  │   • Solve omega equation for w (tridiagonal in z):               │   │
│  │       ∇²w + (N²/f²)∂²w/∂z² = 2·J(ψ_z, ∇²ψ)                       │   │
│  │                                                                   │   │
│  │   • Add wave Stokes drift:                                       │   │
│  │       u += 2·Re[A*·∂A/∂x],  v += 2·Re[A*·∂A/∂y]                  │   │
│  │                                                                   │   │
│  │   • Exchange velocity halos (MPI non-blocking)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: ADVECT PARTICLES (each rank independently)               │   │
│  │                                                                   │   │
│  │   For each local particle:                                       │   │
│  │     1. Interpolate velocity at (x, y, z)                         │   │
│  │        • Use extended arrays (halos) if near boundary            │   │
│  │        • Trilinear/Tricubic/Quintic interpolation                │   │
│  │                                                                   │   │
│  │     2. Time integration:                                         │   │
│  │        • Euler:  x_{n+1} = x_n + dt·u                            │   │
│  │        • RK2:    Midpoint method                                 │   │
│  │        • RK4:    Classical 4th order                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: MIGRATE PARTICLES                                        │   │
│  │                                                                   │   │
│  │   1. Identify particles outside local domain                     │   │
│  │   2. Pack outgoing particles into send buffers                   │   │
│  │   3. MPI.Alltoall - exchange particle counts                     │   │
│  │   4. MPI.Send/Recv - transfer particle data                      │   │
│  │   5. Unpack incoming particles into local arrays                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: APPLY BOUNDARY CONDITIONS                                │   │
│  │                                                                   │   │
│  │   Horizontal (periodic):  x = mod(x, Lx),  y = mod(y, Ly)        │   │
│  │                                                                   │   │
│  │   Vertical (reflective):                                         │   │
│  │     if z < 0:   z = -z,        w = -w                            │   │
│  │     if z > Lz:  z = 2·Lz - z,  w = -w                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: SAVE TRAJECTORIES (if save_interval reached)             │   │
│  │                                                                   │   │
│  │   Option A: Each rank saves local particles independently        │   │
│  │   Option B: Gather all particles to rank 0, unified output       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
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
using QGYBJ

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
