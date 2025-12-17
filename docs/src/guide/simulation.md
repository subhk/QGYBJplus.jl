# [Running Simulations](@id running)

```@meta
CurrentModule = QGYBJ
```

This page explains how to run and monitor QGYBJ.jl simulations.

## Quick Start

### Simple Interface

```julia
using QGYBJ

config = create_simple_config(
    nx=64, ny=64, nz=32,
    dt=0.001,
    total_time=10.0
)

result = run_simple_simulation(config)
```

### Manual Control

```julia
# Setup with domain size (REQUIRED)
grid = Grid(nx=64, ny=64, nz=32)
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km
state = create_state(grid)
initialize_random_flow!(state, grid)
initialize_random_waves!(state, grid)

work = create_work_arrays(grid)
plans = plan_transforms!(grid)
a_ell = setup_elliptic_matrices(grid, params)

# Time loop
dt = 0.001
nsteps = 10000

for step = 1:nsteps
    timestep!(state, grid, params, work, plans, a_ell, dt)
end
```

## Time Stepping

### The `timestep!` Function

```julia
timestep!(state, grid, params, work, plans, a_ell, dt)
```

This performs one AB3 time step, including:
1. Compute nonlinear terms (Jacobians, refraction)
2. Apply integrating factors for diffusion
3. Update prognostic variables (q, B)
4. Invert elliptic equations (q→ψ, B→A)
5. Compute velocities

### Adaptive Time Stepping

```julia
function adaptive_timestep!(state, grid, params, work, plans, a_ell;
                            cfl=0.5, dt_min=1e-6, dt_max=0.01)
    # Compute CFL-based dt
    u_max = maximum(abs.(state.u))
    v_max = maximum(abs.(state.v))
    dt = cfl * min(grid.dx/u_max, grid.dy/v_max)

    # Clamp to limits
    dt = clamp(dt, dt_min, dt_max)

    # Take step
    timestep!(state, grid, params, work, plans, a_ell, dt)

    return dt
end
```

## Progress Monitoring

### Basic Progress

```julia
for step = 1:nsteps
    timestep!(...)

    if step % 100 == 0
        println("Step $step / $nsteps ($(100*step/nsteps)%)")
    end
end
```

### With Diagnostics

```julia
for step = 1:nsteps
    timestep!(...)

    if step % 100 == 0
        KE = flow_kinetic_energy(state.u, state.v, grid)
        WE = wave_energy(state.B, state.A, grid)[2]
        println("Step $step: KE=$KE, WE=$WE")
    end
end
```

### Progress Bar

```julia
using ProgressMeter

@showprogress for step = 1:nsteps
    timestep!(...)
end
```

## Checkpointing

### Save Checkpoints

```julia
checkpoint_interval = 1000

for step = 1:nsteps
    timestep!(...)

    if step % checkpoint_interval == 0
        filename = "checkpoint_$(lpad(step, 8, '0')).jld2"
        @save filename state grid params step
    end
end
```

### Restart from Checkpoint

```julia
using JLD2

# Load checkpoint
@load "checkpoint_00005000.jld2" state grid params step

# Continue simulation
for step = step+1:nsteps
    timestep!(...)
end
```

## Stability Monitoring

### CFL Check

```julia
function check_cfl(state, grid, dt)
    u_max = maximum(abs.(state.u))
    v_max = maximum(abs.(state.v))
    cfl = dt * max(u_max/grid.dx, v_max/grid.dy)
    return cfl
end

for step = 1:nsteps
    timestep!(...)

    cfl = check_cfl(state, grid, dt)
    if cfl > 1.0
        @warn "CFL > 1 at step $step: $cfl"
    end
end
```

### Energy Conservation

```julia
E0 = flow_total_energy(state, grid, params)

for step = 1:nsteps
    timestep!(...)

    if step % 100 == 0
        E = flow_total_energy(state, grid, params)
        dE = (E - E0) / E0
        if abs(dE) > 0.1
            @warn "Energy drift: $dE at step $step"
        end
    end
end
```

## Output During Simulation

### Snapshots

```julia
output_interval = 100
output_steps = Int[]
output_times = Float64[]

for step = 1:nsteps
    timestep!(...)
    time = step * dt

    if step % output_interval == 0
        push!(output_steps, step)
        push!(output_times, time)

        # Save to NetCDF
        write_snapshot(state, grid, step, time, "output/")
    end
end
```

### Time Series

```julia
KE_history = Float64[]
WE_history = Float64[]
time_history = Float64[]

for step = 1:nsteps
    timestep!(...)

    push!(time_history, step * dt)
    push!(KE_history, flow_kinetic_energy(state.u, state.v, grid))
    push!(WE_history, wave_energy(state.B, state.A, grid)[2])
end
```

## Parallel Execution

### Multi-threaded

```julia
# Set before starting Julia
export JULIA_NUM_THREADS=8

# Or check in code
println("Using $(Threads.nthreads()) threads")
```

### MPI

```julia
using MPI
MPI.Init()

# Run with: mpiexec -n 16 julia simulation.jl
```

See [MPI Parallelization](@ref parallel) for details.

## Common Patterns

### Production Run Template

```julia
using QGYBJ
using JLD2

function run_production(;
    nx, ny, nz, dt, nsteps,
    Lx, Ly, Lz,  # Domain size is REQUIRED
    output_interval=100,
    checkpoint_interval=1000,
    output_dir="output"
)
    # Setup
    mkpath(output_dir)
    grid = Grid(nx=nx, ny=ny, nz=nz)
    params = default_params(Lx=Lx, Ly=Ly, Lz=Lz)  # Domain size is REQUIRED
    state = create_state(grid)
    initialize_random_flow!(state, grid)
    initialize_random_waves!(state, grid)

    work = create_work_arrays(grid)
    plans = plan_transforms!(grid)
    a_ell = setup_elliptic_matrices(grid, params)

    # Time loop
    for step = 1:nsteps
        timestep!(state, grid, params, work, plans, a_ell, dt)

        # Output
        if step % output_interval == 0
            write_snapshot(state, grid, step, step*dt, output_dir)
        end

        # Checkpoint
        if step % checkpoint_interval == 0
            @save "$output_dir/checkpoint.jld2" state grid params step
        end
    end

    return state
end
```

## Troubleshooting

### NaN Values

```julia
if any(isnan, state.psi)
    error("NaN detected at step $step")
end
```

### Instability

- Reduce time step
- Increase dissipation
- Check initial conditions for sharp gradients

### Memory Issues

- Reduce grid size
- Use checkpointing
- Enable MPI for distribution
