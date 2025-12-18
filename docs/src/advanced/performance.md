# [Performance Tips](@id performance)

```@meta
CurrentModule = QGYBJ
```

This page provides guidance for optimizing QGYBJ.jl performance.

## Quick Performance Wins

### 1. Use FFTW Wisdom

Pre-compute optimal FFT plans:

```julia
using FFTW

# Measure FFT plans (slower startup, faster runtime)
FFTW.set_num_threads(Threads.nthreads())
plans = plan_transforms!(grid; flags=FFTW.MEASURE)

# For production: save and load wisdom
FFTW.export_wisdom("fftw_wisdom.txt")
# Later:
FFTW.import_wisdom("fftw_wisdom.txt")
```

### 2. Enable Multi-threading

```bash
# Set threads before running Julia
export JULIA_NUM_THREADS=8
julia --threads=8 simulation.jl
```

In Julia:
```julia
# Check thread count
Threads.nthreads()

# FFTW uses its own threading
FFTW.set_num_threads(8)
```

### 3. Use Appropriate Precision

```julia
# Double precision (default, most accurate)
state = init_state(grid)

# Single precision (2x memory savings, faster)
# Note: Requires initializing parameters and grid with Float32
```

!!! warning
    Single precision may cause numerical instabilities for long simulations.

## Memory Optimization

### Pre-allocation

All arrays are pre-allocated to avoid GC:

```julia
# All arrays are pre-allocated in setup_model()
G, S, plans, a_ell = setup_model(params)

# Reused every time step
leapfrog_step!(S, G, params, plans, a_ell)
```

### Memory Usage Estimate

```julia
function estimate_memory(nx, ny, nz; T=Float64)
    # Main arrays
    complex_size = sizeof(Complex{T}) * (nx÷2+1) * ny * nz
    real_size = sizeof(T) * nx * ny * nz

    n_complex = 10  # psi, q, B, A, etc.
    n_real = 5      # u, v, work arrays

    total = n_complex * complex_size + n_real * real_size
    return total / 1e9  # GB
end

println("Memory: ", estimate_memory(256, 256, 128), " GB")
```

### Memory-Efficient Output

Write in chunks rather than storing everything:

```julia
# Don't do this (stores all data)
all_psi = zeros(nx, ny, nz, nsteps)

# Do this (stream to disk)
for step = 1:nsteps
    timestep!(state, ...)
    if step % output_interval == 0
        write_to_disk(state.psi, step)
    end
end
```

## Computational Bottlenecks

### Profiling

```julia
using Profile

@profile run_simulation(config)
Profile.print(mincount=100)

# Flamegraph visualization
using ProfileSVG
ProfileSVG.save("profile.svg")
```

### Typical Time Distribution

| Operation | Fraction | Optimization |
|:----------|:---------|:-------------|
| FFT/IFFT | 40-50% | FFTW wisdom, threading |
| Tridiagonal solves | 20-30% | Pre-factorization |
| Array operations | 15-25% | Loop fusion with `@.` |
| I/O | 5-10% | Buffering, compression |

### Timing Individual Components

```julia
using BenchmarkTools

# Time FFT
@btime fft_forward!($work.tmp_k, $work.tmp, $plans)

# Time elliptic solve
@btime invert_q_to_psi!($state, $grid, $params, $a_ell)

# Time full step
@btime leapfrog_step!($S, $G, $params, $plans, $a_ell)
```

## Numerical Efficiency

### Time Step Selection

Use the largest stable time step:

```julia
# CFL-based time step
u_max = maximum(abs.(state.u))
dt_cfl = 0.5 * grid.dx / u_max

# Use slightly smaller for safety
dt = 0.8 * dt_cfl
```

### Adaptive Time Stepping

```julia
function adaptive_dt(state, grid, params; cfl=0.5, dt_max=0.01)
    u_max = maximum(abs.(state.u))
    v_max = maximum(abs.(state.v))

    dx = grid.Lx / grid.nx
    dy = grid.Ly / grid.ny

    dt = cfl * min(dx/u_max, dy/v_max)
    return min(dt, dt_max)
end
```

### Dissipation Tuning

Too much dissipation wastes resolution. Too little causes instability.

```julia
# Minimal dissipation for given resolution
nu_h2 = 1e-4 * (2π/nx)^8  # Scales with grid spacing

# Or based on energy pile-up check
E_k = horizontal_energy_spectrum(state.psi, grid)
if E_k[end] > 0.01 * maximum(E_k)
    @warn "Energy piling up at small scales, increase dissipation"
end
```

## Loop Optimization

### Broadcasting

Use `@.` for fused operations:

```julia
# Slow (multiple allocations)
result = a .+ b .* c

# Fast (single pass, no allocation)
@. result = a + b * c
```

### In-place Operations

```julia
# Allocating
b = fft(a)

# In-place
mul!(b, plan, a)
```

### Loop Order

Julia is column-major (like Fortran):

```julia
# Fast (memory-contiguous)
for k in 1:nz
    for j in 1:ny
        for i in 1:nx
            a[i, j, k] = ...
        end
    end
end

# Slow (cache-unfriendly)
for i in 1:nx
    for j in 1:ny
        for k in 1:nz
            a[i, j, k] = ...
        end
    end
end
```

### SIMD and LoopVectorization

```julia
using LoopVectorization

# Auto-vectorized loop
@turbo for i in eachindex(a)
    a[i] = b[i] * c[i] + d[i]
end
```

## GPU Acceleration

### CUDA Support

```julia
using CUDA

# Note: GPU support is experimental/future feature
# Move arrays to GPU
S_gpu = cu(S)
G_gpu = cu(G)

# GPU FFT plans
plans_gpu = plan_gpu_transforms!(G_gpu)

# Run on GPU
leapfrog_step!(S_gpu, G_gpu, params, plans_gpu, a_ell_gpu)
```

### When to Use GPU

| Scenario | Recommendation |
|:---------|:---------------|
| nx, ny < 256 | CPU often faster |
| nx, ny ≥ 512 | GPU beneficial |
| Many particles | GPU for interpolation |
| MPI cluster | CPU per node |

## Parallelization Strategy

### Shared Memory (OpenMP/Threads)

Best for:
- Single node
- nx × ny < 512²
- Memory-bound operations

```julia
# Set threads
Threads.@threads for k in 1:nz
    process_layer!(state, k)
end
```

### Distributed Memory (MPI)

Best for:
- Multiple nodes
- nx × ny ≥ 512²
- Large domains

```bash
mpiexec -n 64 julia simulation.jl
```

### Hybrid (MPI + Threads)

```bash
# 4 MPI ranks × 8 threads each = 32 cores
export JULIA_NUM_THREADS=8
mpiexec -n 4 julia simulation.jl
```

## I/O Performance

### Buffered Output

```julia
# Write less frequently
output_interval = 1000  # Not every step

# Buffer multiple snapshots
buffer_size = 10
output_buffer = zeros(nx, ny, nz, buffer_size)
```

### Compression

```julia
using NCDatasets

# Compressed NetCDF
defVar(ds, "psi", Float64, ("x", "y", "z", "t");
    deflatelevel = 5,
    chunksizes = (nx, ny, nz, 1)
)
```

### Parallel I/O

```julia
# HDF5 with MPI I/O
using HDF5

h5open("output.h5", "w", mpi_comm) do fid
    write_mpi(fid, "psi", local_psi, global_dims, local_range)
end
```

## Benchmarking Guide

### Full Benchmark Suite

```julia
function benchmark_simulation(nx, ny, nz; nsteps=100, Lx=500e3, Ly=500e3, Lz=4000.0)
    params = default_params(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz)
    G, S, plans, a_ell = setup_model(params)

    # Initialize
    init_random_psi!(S, G; amplitude=0.1)
    compute_q_from_psi!(S, G, plans, a_ell)

    # Warm-up
    first_projection_step!(S, G, params, plans, a_ell)
    for _ in 1:9
        leapfrog_step!(S, G, params, plans, a_ell)
    end

    # Timed run
    t_start = time()
    for _ in 1:nsteps
        leapfrog_step!(S, G, params, plans, a_ell)
    end
    t_end = time()

    dt_avg = (t_end - t_start) / nsteps
    throughput = nx * ny * nz / dt_avg / 1e6  # Million cells/second

    return (dt_avg=dt_avg, throughput=throughput)
end

# Run benchmark
result = benchmark_simulation(128, 128, 64)
println("Time per step: $(result.dt_avg*1000) ms")
println("Throughput: $(result.throughput) Mcells/s")
```

### Scaling Tests

```julia
# Strong scaling
for nprocs in [1, 2, 4, 8, 16]
    # Run with mpiexec -n $nprocs
    run_benchmark(256, 256, 128)
end

# Weak scaling
for n in [64, 128, 256, 512]
    run_benchmark(n, n, n÷2)
end
```

## Summary Checklist

1. **Before running:**
   - [ ] Set `JULIA_NUM_THREADS` appropriately
   - [ ] Enable FFTW threading
   - [ ] Use FFTW.MEASURE for plans
   - [ ] Pre-allocate all arrays

2. **For production:**
   - [ ] Profile to find bottlenecks
   - [ ] Use appropriate time step (CFL)
   - [ ] Minimize I/O frequency
   - [ ] Enable compression for output

3. **For large runs:**
   - [ ] Use MPI for distributed memory
   - [ ] Consider GPU for large grids
   - [ ] Use parallel I/O
   - [ ] Monitor memory usage
