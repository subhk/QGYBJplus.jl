# [MPI Parallelization](@id parallel)

```@meta
CurrentModule = QGYBJplus
```

Run QGYBJ+.jl on distributed memory systems using 2D pencil decomposition.

!!! note "When to Use"
    Recommended for grids ≥256³ or when memory is limited. For smaller problems, use threading: `julia -t auto`.

## Quick Start

```julia
# parallel_run.jl
using MPI, PencilArrays, PencilFFTs, QGYBJplus

MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()

params = default_params(Lx=1000e3, Ly=1000e3, Lz=5000.0, nx=256, ny=256, nz=128)
grid = QGYBJplus.init_mpi_grid(params, mpi_config)
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
state = QGYBJplus.init_mpi_state(grid, plans, mpi_config)
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)

a_vec = a_ell_ut(params, grid)

for step in 1:1000
    invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
    leapfrog_step!(state, state, state, grid, params, plans; a=a_vec, workspace=workspace)
end

MPI.Finalize()
```

Run with:
```bash
mpiexec -n 16 julia --project parallel_run.jl
```

## Requirements

```julia
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

System MPI required: `brew install open-mpi` (macOS) or `apt install libopenmpi-dev` (Ubuntu).

## Scaling

| Processes | Topology | Grid Size |
|:----------|:---------|:----------|
| 4 | 2×2 | 128³ |
| 16 | 4×4 | 256³ |
| 64 | 8×8 | 512³ |

Use powers of 2 for optimal performance.

## Key Concepts

**2D Pencil Decomposition**: Domain split across px × py process grid. z-dimension stays local for efficient vertical solves.

**Workspace**: Pre-allocate once to avoid repeated allocation:
```julia
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)
```

**State Copies**: Use `copy_state(S)` not `deepcopy(S)` to preserve pencil topology.

## Key Functions

| Function | Purpose |
|:---------|:--------|
| `setup_mpi_environment()` | Initialize MPI config |
| `init_mpi_grid()` | Create distributed grid |
| `plan_mpi_transforms()` | Create PencilFFT plans |
| `init_mpi_state()` | Create distributed state |
| `init_mpi_workspace()` | Allocate workspace |
| `copy_state()` | Copy state (preserves topology) |
| `mpi_reduce_sum()` | Sum across processes |

## Global Reductions

```julia
local_ke = flow_kinetic_energy(state.u, state.v)
global_ke = QGYBJplus.mpi_reduce_sum(local_ke, mpi_config)
if mpi_config.is_root
    println("Total KE: $global_ke")
end
```

## Job Scripts

### SLURM
```bash
#!/bin/bash
#SBATCH --nodes=4 --ntasks-per-node=16
mpiexec -n 64 julia --project script.jl
```

## Troubleshooting

| Problem | Solution |
|:--------|:---------|
| Pencil topology mismatch | Use `copy_state(S)` not `deepcopy(S)` |
| Deadlock | All ranks must call collective operations |
| Segfaults | Use `size(parent(arr))` for array dimensions |

See [Troubleshooting](@ref troubleshooting) for more details.
