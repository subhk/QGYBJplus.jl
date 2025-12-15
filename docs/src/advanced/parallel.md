# [MPI Parallelization](@id parallel)

```@meta
CurrentModule = QGYBJ
```

This page describes how to run QGYBJ.jl on distributed memory systems using MPI with **2D pencil decomposition** via PencilArrays and PencilFFTs.

## Overview

QGYBJ.jl uses **2D pencil decomposition** for optimal parallel scalability:

```
           Serial (Full Domain)                2D Pencil Decomposition
        ┌─────────────────────┐         ┌───────┬───────┬───────┐
        │                     │         │  P0   │  P1   │  P2   │
        │                     │         ├───────┼───────┼───────┤
        │    nx × ny × nz     │   →     │  P3   │  P4   │  P5   │
        │                     │         ├───────┼───────┼───────┤
        │                     │         │  P6   │  P7   │  P8   │
        └─────────────────────┘         └───────┴───────┴───────┘
                                              9 processes in 3×3 grid
```

The domain is distributed across a 2D process grid (px × py), allowing scaling to O(N²) processes for an N³ grid.

## Key Concept: Pencil Configurations

Different operations require data arranged differently. QGYBJ uses **three pencil configurations**:

| Configuration | `decomp_dims` | Distribution | Local Dim | Use Case |
|:--------------|:--------------|:-------------|:----------|:---------|
| **xy-pencil** | (2, 3) | y,z distributed | x | Horizontal FFTs |
| **xz-pencil** | (1, 3) | x,z distributed | y | Intermediate transpose |
| **z-pencil** | (1, 2) | x,y distributed | **z** | Vertical operations |

**Why three configurations?**
- Horizontal FFTs need consecutive x-data → xy-pencil
- Vertical tridiagonal solves need all z-data → z-pencil
- **PencilArrays constraint**: Transpose requires decomp_dims to differ by at most ONE dimension
- Since (2,3)→(1,2) differs by TWO dimensions, we need intermediate xz-pencil (1,3)
- **Two-step transpose** automatically switches between configurations

## Requirements

Install MPI-related packages:

```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

Ensure you have a working MPI installation (OpenMPI, MPICH, Intel MPI, etc.).

## Quick Start

### Basic MPI Script

```julia
# parallel_run.jl
using MPI
using PencilArrays
using PencilFFTs
using QGYBJ

# Initialize MPI
MPI.Init()

# Setup MPI environment with automatic 2D topology
mpi_config = QGYBJ.setup_mpi_environment()

# Create parameters
params = default_params(nx=256, ny=256, nz=128)

# Initialize distributed grid and state
grid = QGYBJ.init_mpi_grid(params, mpi_config)
state = QGYBJ.init_mpi_state(grid, mpi_config)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

# Plan parallel FFTs
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# Get physics coefficients
a_vec = a_ell_ut(params, grid)
dealias = dealias_mask(grid)

# Time stepping loop
dt = 0.001
for step in 1:1000
    # All operations automatically handle 2D decomposition
    invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
    compute_velocities!(state, grid; plans=plans, params=params, workspace=workspace)
    leapfrog_step!(state, state, state, grid, params, plans;
                   a=a_vec, dealias_mask=dealias, workspace=workspace)

    if step % 100 == 0 && mpi_config.is_root
        println("Step $step completed")
    end
end

if mpi_config.is_root
    println("Simulation complete!")
end

MPI.Finalize()
```

### Running

```bash
# Run on 16 processes (auto-decomposed to 4×4 grid)
mpiexec -n 16 julia --project parallel_run.jl

# Run on 64 processes (auto-decomposed to 8×8 grid)
mpiexec -n 64 julia --project parallel_run.jl
```

## 2D Pencil Decomposition Details

### Process Topology

The `setup_mpi_environment()` function automatically computes an optimal 2D process grid:

```julia
# Automatic topology (recommended)
mpi_config = QGYBJ.setup_mpi_environment()

# For 16 processes: creates (4, 4) topology
# For 12 processes: creates (3, 4) topology

# Manual topology specification
mpi_config = QGYBJ.setup_mpi_environment(topology=(4, 4))
```

### PencilDecomp Structure

The decomposition is stored in `grid.decomp`:

```julia
# Access decomposition info
decomp = grid.decomp

# Three pencil configurations
decomp.pencil_xy  # decomp_dims=(2,3): y,z distributed, x local - for FFTs
decomp.pencil_xz  # decomp_dims=(1,3): x,z distributed, y local - INTERMEDIATE
decomp.pencil_z   # decomp_dims=(1,2): x,y distributed, z local - for vertical ops

# Local index ranges for each configuration
decomp.local_range_xy  # (1:nx, y_start:y_end, z_start:z_end)
decomp.local_range_xz  # (x_start:x_end, 1:ny, z_start:z_end)
decomp.local_range_z   # (x_start:x_end, y_start:y_end, 1:nz)

# Global dimensions and topology
decomp.global_dims  # (nx, ny, nz)
decomp.topology     # (px, py) process grid
```

### How Transposes Work

QGYBJ uses **two-step transpose** because PencilArrays requires that pencil decompositions differ by at most one dimension. Since xy-pencil (2,3) and z-pencil (1,2) differ in both dimensions, we use an intermediate xz-pencil (1,3):

```
                         QGYBJ Two-Step Transpose Architecture

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         xy-pencil (2,3)                                     │
    │                    y,z distributed; x local                                 │
    │                  Used for: FFTs, horizontal operations                      │
    └─────────────────────────────┬───────────────────────────────────────────────┘
                                  │
                    transpose!(buffer, src)  [changes 2→1]
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         xz-pencil (1,3)                                     │
    │                    x,z distributed; y local                                 │
    │                       INTERMEDIATE for transposes                           │
    └─────────────────────────────┬───────────────────────────────────────────────┘
                                  │
                    transpose!(dst, buffer)  [changes 3→2]
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          z-pencil (1,2)                                     │
    │                    x,y distributed; z local                                 │
    │               Used for: Tridiagonal solves, vertical ops                    │
    └─────────────────────────────────────────────────────────────────────────────┘

    Transpose path satisfies PencilArrays constraint (max 1 dim change per step):
    • xy→z: (2,3) → (1,3) → (1,2)  ✓
    • z→xy: (1,2) → (1,3) → (2,3)  ✓
```

The two-step transpose is handled automatically by `transpose_to_z_pencil!` and `transpose_to_xy_pencil!`.

**Example:** Inverting q to ψ requires a vertical tridiagonal solve:

```julia
function invert_q_to_psi!(S, G; a, workspace)
    # Detect if 2D decomposition is active
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

    if need_transpose
        # 1. Two-step transpose: xy(2,3) → xz(1,3) → z(1,2)
        transpose_to_z_pencil!(workspace.q_z, S.q, G)

        # 2. Solve tridiagonal system (z now fully local!)
        # ... Thomas algorithm for each (i,j) column ...

        # 3. Two-step transpose back: z(1,2) → xz(1,3) → xy(2,3)
        transpose_to_xy_pencil!(S.psi, workspace.psi_z, G)
    else
        # Serial: direct solve
        _invert_q_to_psi_direct!(S, G, a)
    end
end
```

## Workspace Arrays

For 2D decomposition, pre-allocated z-pencil workspace arrays avoid repeated allocation:

```julia
# Initialize workspace once
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

# Workspace contains z-pencil arrays:
# workspace.q_z     - for q transpose
# workspace.psi_z   - for ψ transpose
# workspace.B_z     - for B transpose
# workspace.A_z     - for A transpose
# workspace.C_z     - for C transpose
# workspace.work_z  - general workspace

# Pass workspace to functions
invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
invert_B_to_A!(state, grid, params, a_vec; workspace=workspace)
```

## Index Mapping

### Local to Global Indices

When iterating over distributed arrays, use local indices but map to global for wavenumbers:

```julia
# For xy-pencil arrays (default)
for k_local in axes(arr, 3)
    for j_local in axes(arr, 2)
        for i_local in axes(arr, 1)
            # Map to global indices for wavenumber lookup
            i_global = local_to_global(i_local, 1, grid)
            j_global = local_to_global(j_local, 2, grid)

            kx = grid.kx[i_global]
            ky = grid.ky[j_global]

            # Use local indices for array access
            arr[i_local, j_local, k_local] = ...
        end
    end
end

# For z-pencil arrays (after transpose)
i_global = local_to_global_z(i_local, 1, grid)
j_global = local_to_global_z(j_local, 2, grid)
```

### Helper Functions

```julia
# Get local index ranges
local_range = get_local_range(grid)       # xy-pencil
local_range_z = get_local_range_z(grid)   # z-pencil

# Get local dimensions of an array
nx_local, ny_local, nz_local = get_local_dims(arr)

# Check if array is distributed
is_distributed = is_parallel_array(arr)

# Access underlying data (works for both Array and PencilArray)
data = parent(arr)
```

## Parallel FFTs

PencilFFTs handles the distributed FFTs automatically:

```julia
# Plan creation
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# Forward FFT (physical → spectral)
fft_forward!(dst, src, plans)

# Backward FFT (spectral → physical)
fft_backward!(dst, src, plans)
```

PencilFFTs automatically handles:
- X-FFT in xy-pencil
- Transpose to appropriate configuration for Y-FFT
- Transpose back to output configuration

## Communication Patterns

### Global Reductions

```julia
# Sum a value across all processes
local_energy = flow_kinetic_energy(state.u, state.v)
global_energy = QGYBJ.mpi_reduce_sum(local_energy, mpi_config)

# Only root has the correct sum
if mpi_config.is_root
    println("Total KE: $global_energy")
end
```

### Gather/Scatter

```julia
# Gather full field to root process
global_psi = QGYBJ.gather_to_root(state.psi, grid, mpi_config)
# Returns full array on rank 0, nothing on others

# Scatter from root to all processes
distributed_psi = QGYBJ.scatter_from_root(initial_psi, grid, mpi_config)
```

### Synchronization

```julia
# Barrier - wait for all processes
QGYBJ.mpi_barrier(mpi_config)
```

## Complete Example

```julia
# full_mpi_simulation.jl
using MPI
using PencilArrays
using PencilFFTs
using QGYBJ

function main()
    MPI.Init()

    # Setup MPI with 2D decomposition
    mpi_config = QGYBJ.setup_mpi_environment()

    if mpi_config.is_root
        println("Running on $(mpi_config.nprocs) processes")
        println("Topology: $(mpi_config.topology)")
    end

    # Parameters
    params = QGParams(
        nx = 256, ny = 256, nz = 128,
        Lx = 2π, Ly = 2π,
        f0 = 1.0,
        stratification = :constant_N,
        ybj_plus = true
    )

    # Initialize distributed grid, state, workspace
    grid = QGYBJ.init_mpi_grid(params, mpi_config)
    state = QGYBJ.init_mpi_state(grid, mpi_config)
    workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

    # FFT plans
    plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

    # Physics setup
    a_vec = a_ell_ut(params, grid)
    dealias = dealias_mask(grid)

    # Initialize with random field (deterministic across processes)
    QGYBJ.init_mpi_random_field!(state.psi, grid, 1.0, 42)
    QGYBJ.init_mpi_random_field!(state.B, grid, 0.1, 123)

    # Compute initial q from psi
    invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)

    # Time stepping
    dt = 0.001
    nsteps = 10000
    output_interval = 100

    for step in 1:nsteps
        # Main physics
        invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
        compute_velocities!(state, grid; plans=plans, params=params, workspace=workspace)

        # Time step
        leapfrog_step!(state, state, state, grid, params, plans;
                       a=a_vec, dealias_mask=dealias, workspace=workspace)

        # Diagnostics
        if step % output_interval == 0
            local_ke = flow_kinetic_energy(state.u, state.v)
            global_ke = QGYBJ.mpi_reduce_sum(local_ke, mpi_config)

            if mpi_config.is_root
                println("Step $step / $nsteps: KE = $global_ke")
            end
        end
    end

    # Final output
    if mpi_config.is_root
        println("Simulation complete!")
    end

    MPI.Finalize()
end

main()
```

## Allocating Distributed Arrays

```julia
# Allocate array in xy-pencil configuration (for FFTs)
# decomp_dims=(2,3): y,z distributed; x local
arr_xy = QGYBJ.allocate_xy_pencil(grid, ComplexF64)

# Allocate array in xz-pencil configuration (intermediate for transposes)
# decomp_dims=(1,3): x,z distributed; y local
arr_xz = QGYBJ.allocate_xz_pencil(grid, ComplexF64)

# Allocate array in z-pencil configuration (for vertical ops)
# decomp_dims=(1,2): x,y distributed; z local
arr_z = QGYBJ.allocate_z_pencil(grid, ComplexF64)
```

In serial mode, all three functions return standard `Array{T,3}` of size `(nx, ny, nz)`.

## Performance Considerations

### Scaling

| Regime | Description |
|:-------|:------------|
| Strong scaling | Fixed problem size, increase processes |
| Weak scaling | Fixed work per process, increase total size |

QGYBJ.jl with 2D decomposition scales to O(N²) processes for N³ grid.

### Communication Costs

| Operation | Communication Pattern | Cost |
|:----------|:---------------------|:-----|
| Transpose xy↔z | **Two-step** all-to-all via xz-pencil | 2 × O(N³/P) data moved |
| FFT | Internal transposes | Handled by PencilFFTs |
| Reduction | Global sum | O(log P) |

**Note:** The two-step transpose (xy↔xz↔z) doubles the communication compared to a single transpose, but is required by PencilArrays' constraint that decomp_dims can only differ by one dimension per transpose.

### Optimization Tips

1. **Use power-of-2 process counts** for optimal topology
2. **Match decomposition to problem**: More processes in larger dimensions
3. **Minimize transpose frequency**: Batch vertical operations
4. **Pre-allocate workspace**: Avoid allocation in time loop

```julia
# Good: Power of 2
mpiexec -n 16 julia script.jl   # 4×4 topology
mpiexec -n 64 julia script.jl   # 8×8 topology

# May be slower: Awkward factorization
mpiexec -n 15 julia script.jl   # 3×5 topology
```

## Job Submission Scripts

### SLURM

```bash
#!/bin/bash
#SBATCH --job-name=qgybj
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load julia openmpi

export JULIA_NUM_THREADS=1  # Use MPI parallelism
mpiexec -n 64 julia --project run_simulation.jl
```

### PBS

```bash
#!/bin/bash
#PBS -N qgybj
#PBS -l nodes=4:ppn=16
#PBS -l walltime=24:00:00
#PBS -l mem=128gb

module load julia openmpi

cd $PBS_O_WORKDIR
mpiexec -n 64 julia --project run_simulation.jl
```

## Troubleshooting

### MPI Not Found

```julia
# Use system MPI instead of Julia's binary
using MPIPreferences
MPIPreferences.use_system_binary()
# Restart Julia
```

### Memory Errors

- Local arrays too large → increase process count
- Check with: `size(parent(state.psi))` on each rank

### Deadlock

- Ensure ALL ranks call collective operations (gather, reduce, barrier)
- Check for mismatched send/receive

### Debugging

```julia
function debug_print(msg, mpi_config)
    for r in 0:(mpi_config.nprocs-1)
        if mpi_config.rank == r
            println("[Rank $r] $msg")
            flush(stdout)
        end
        QGYBJ.mpi_barrier(mpi_config)
    end
end
```

## API Reference

The following MPI functions are provided:

### Setup Functions
- `setup_mpi_environment` - Initialize MPI environment and configuration
- `init_mpi_grid` - Create grid with 2D pencil decomposition
- `init_mpi_state` - Create distributed state arrays
- `init_mpi_workspace` - Allocate workspace for z-pencil operations
- `plan_mpi_transforms` - Create PencilFFT plans

### Communication Functions
- `gather_to_root` - Collect distributed array to rank 0
- `scatter_from_root` - Distribute array from rank 0
- `mpi_barrier` - Synchronize all processes
- `mpi_reduce_sum` - Sum values across all processes

### Transpose Functions
- `transpose_to_z_pencil!` - Two-step transpose: xy(2,3) → xz(1,3) → z(1,2)
- `transpose_to_xy_pencil!` - Two-step transpose: z(1,2) → xz(1,3) → xy(2,3)

### Index Mapping Functions
- `local_to_global` - Map local index to global (xy-pencil), with dimension argument
- `local_to_global_z` - Map local index to global (z-pencil)
- `range_local` - Get local index range (from PencilArrays)
- `range_remote` - Get remote (global) index range (from PencilArrays)

### Helper Functions
- `get_kh2` - Get horizontal wavenumber squared array
- `is_dealiased` - Check if a mode is within dealiasing radius

### Array Allocation
- `allocate_xy_pencil` - Allocate array in xy-pencil layout (for FFTs)
- `allocate_xz_pencil` - Allocate array in xz-pencil layout (intermediate)
- `allocate_z_pencil` - Allocate array in z-pencil layout (for vertical ops)
