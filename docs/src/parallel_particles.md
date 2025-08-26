## Parallel & Particles

### Parallel Execution (MPI + Pencil Arrays)

QGYBJ.jl includes a parallel interface that uses MPI + PencilArrays/PencilFFTs
if available. The serial path (FFTW) is the default.

Outline:
- Initialize MPI and call `setup_parallel_environment()`
- Build `ParallelConfig` and pass it into `setup_simulation(config; use_mpi=true)`
- Internally, the grid/state are distributed and transforms use PencilFFTs

### Particles

There is a unified particle advection system that can advect particles in
serial or parallel. The API provides:
- `ParticleConfig`, `ParticleState`, `ParticleTracker`
- `initialize_particles!`, `advect_particles!`
- Interpolation methods: `TRILINEAR`, `TRICUBIC`, `ADAPTIVE`, `QUINTIC`
- 3D distributions: uniform grid, layered, random, or custom

See examples in the `examples/` folder for end‑to‑end scripts.

