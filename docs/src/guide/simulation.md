# [Running Simulations](@id running)

```@meta
CurrentModule = QGYBJplus
```

This page explains how to run and monitor QGYBJ+.jl simulations.

## Quick Start

### Simple Interface

```julia
using QGYBJplus

config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # Domain size (REQUIRED)
    nx=64, ny=64, nz=32,
    dt=0.001,
    total_time=10.0
)

result = run_simple_simulation(config)
```

### Manual Control

```julia
using QGYBJplus

# Setup with domain size (REQUIRED)
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.001, nt=10000
)
G, S, plans, a_ell = setup_model(par)

# Initialize
init_random_psi!(S, G; amplitude=0.1)
compute_q_from_psi!(S, G, plans, a_ell)

# Time loop
first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

## Time Stepping

### Available Schemes

| Scheme | Description | Best For |
|:-------|:------------|:---------|
| `:leapfrog` | Explicit, 2nd order, Robert-Asselin filter | dt ≤ 2f/N² (~2s), academic tests |
| `:imex_cn` | IMEX Crank-Nicolson with operator splitting | **Large dt (~20s), production runs** |

### Using `run_simulation!` (Recommended)

```julia
# Leapfrog (default)
run_simulation!(S, G, par, plans;
    timestepper=:leapfrog, ...)

# IMEX-CN (10x larger timestep)
run_simulation!(S, G, par, plans;
    timestepper=:imex_cn, ...)
```

### Manual Time Loop (Leapfrog)

```julia
# First step uses forward Euler
first_projection_step!(S, G, par, plans, a_ell)

# Subsequent steps use leapfrog
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

### Manual Time Loop (IMEX-CN)

```julia
imex_ws = init_imex_workspace(S, G)
Snp1 = copy_state(S)

for step = 1:par.nt
    imex_cn_step!(Snp1, S, G, par, plans, imex_ws;
                  a=a_ell, dealias_mask=L,
                  workspace=workspace, N2_profile=N2)

    # Copy for next step (only 2 time levels needed)
    parent(S.L⁺A) .= parent(Snp1.L⁺A)
    parent(S.A) .= parent(Snp1.A)
    parent(S.q) .= parent(Snp1.q)
    parent(S.psi) .= parent(Snp1.psi)
end
```

### Choosing a Timestep

| Scheme | Stability Constraint | Typical dt |
|:-------|:--------------------|:-----------|
| Leapfrog | dt ≤ min(2f/N², dx/U) | ~2s (dispersion-limited) |
| IMEX-CN | dt ≤ dx/U | ~20s (advection CFL only) |

Each time step performs:
1. Compute nonlinear terms (Jacobians, refraction)
2. Apply dissipation via integrating factors
3. Update prognostic variables (q, B)
4. Invert elliptic equations (q→ψ, B→A)
5. Compute velocities

## Progress Monitoring

### Basic Progress

```julia
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)

    if step % 100 == 0
        println("Step $step / $(par.nt) ($(100*step/par.nt)%)")
    end
end
```

### With Diagnostics

```julia
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)

    if step % 100 == 0
        compute_velocities!(S, G, plans)
        KE = flow_kinetic_energy(S.u, S.v)
        WE_B, WE_A = wave_energy(S.L⁺A, S.A)
        println("Step $step: KE=$KE, WE_B=$WE_B")
    end
end
```

### Progress Bar

```julia
using ProgressMeter

@showprogress for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

## Checkpointing

### Save Checkpoints

```julia
using JLD2

checkpoint_interval = 1000

first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)

    if step % checkpoint_interval == 0
        filename = "checkpoint_$(lpad(step, 8, '0')).jld2"
        @save filename S G par step
    end
end
```

### Restart from Checkpoint

```julia
using JLD2

# Load checkpoint
@load "checkpoint_00005000.jld2" S G par step

# Continue simulation
for step = step+1:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

## Stability Monitoring

### CFL Check

```julia
function check_cfl(S, G, dt)
    u_max = maximum(abs.(S.u))
    v_max = maximum(abs.(S.v))
    cfl = dt * max(u_max/G.dx, v_max/G.dy)
    return cfl
end

for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
    compute_velocities!(S, G, plans)

    cfl = check_cfl(S, G, par.dt)
    if cfl > 1.0
        @warn "CFL > 1 at step $step: $cfl"
    end
end
```

### Energy Conservation

```julia
compute_velocities!(S, G, plans)
E0 = flow_kinetic_energy(S.u, S.v)

for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)

    if step % 100 == 0
        compute_velocities!(S, G, plans)
        E = flow_kinetic_energy(S.u, S.v)
        dE = (E - E0) / E0
        if abs(dE) > 0.1
            @warn "Energy drift: $dE at step $step"
        end
    end
end
```

## Output During Simulation

### Snapshots with NetCDF

```julia
using QGYBJplus

output_interval = 100

first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
    time = step * par.dt

    if step % output_interval == 0
        # Save streamfunction
        ncdump_psi(S, G, step, time, "output/")

        # Save wave envelope
        ncdump_la(S, G, step, time, "output/")
    end
end
```

### Time Series

```julia
KE_history = Float64[]
WE_history = Float64[]
time_history = Float64[]

for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)

    push!(time_history, step * par.dt)
    compute_velocities!(S, G, plans)
    push!(KE_history, flow_kinetic_energy(S.u, S.v))
    push!(WE_history, wave_energy(S.L⁺A, S.A)[1])
end
```

## Using the High-Level API

### QGYBJSimulation

```julia
using QGYBJplus

# Create configuration
domain = create_domain_config(
    nx=64, ny=64, nz=32,
    Lx=500e3, Ly=500e3, Lz=4000.0
)

strat = create_stratification_config(type=:constant_N)

model = create_model_config(
    ybj_plus=true,
    inviscid=false
)

output = create_output_config(
    output_dir="output",
    output_interval=100
)

# Setup simulation
sim = setup_simulation(domain, strat; model=model, output=output)

# Run
run_simulation!(sim, dt=0.001, nsteps=10000)
```

## Parallel Execution

### Multi-threaded (FFTs)

```julia
# Set before starting Julia
export JULIA_NUM_THREADS=8

# Or check in code
println("Using $(Threads.nthreads()) threads")
```

### MPI with 2D Pencil Decomposition

```julia
using MPI, PencilArrays, PencilFFTs, QGYBJplus

MPI.Init()
mpi_config = QGYBJplus.setup_mpi_environment()

# Setup distributed simulation
params = default_params(
    Lx=1000e3, Ly=1000e3, Lz=5000.0,
    nx=256, ny=256, nz=128
)
grid = QGYBJplus.init_mpi_grid(params, mpi_config)
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)
state = QGYBJplus.init_mpi_state(grid, plans, mpi_config)
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)

# Run with: mpiexec -n 16 julia simulation.jl

MPI.Finalize()
```

See [MPI Parallelization](@ref parallel) for details.

## Common Patterns

### Production Run Template

```julia
using QGYBJplus
using JLD2

function run_production(;
    Lx, Ly, Lz,              # Domain size (REQUIRED)
    nx, ny, nz,
    dt, nsteps,
    output_interval=100,
    checkpoint_interval=1000,
    output_dir="output"
)
    # Setup
    mkpath(output_dir)
    par = default_params(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nx=nx, ny=ny, nz=nz,
        dt=dt, nt=nsteps
    )
    G, S, plans, a_ell = setup_model(par)

    # Initialize
    init_random_psi!(S, G; amplitude=0.1)
    compute_q_from_psi!(S, G, plans, a_ell)

    # Time loop
    first_projection_step!(S, G, par, plans, a_ell)
    for step = 2:nsteps
        leapfrog_step!(S, G, par, plans, a_ell)

        # Output
        if step % output_interval == 0
            ncdump_psi(S, G, step, step*dt, output_dir)
        end

        # Checkpoint
        if step % checkpoint_interval == 0
            @save "$output_dir/checkpoint.jld2" S G par step
        end
    end

    return S
end

# Run
run_production(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.001, nsteps=10000
)
```

## Troubleshooting

### NaN Values

```julia
if any(isnan, S.psi)
    error("NaN detected at step $step")
end
```

### Instability

- **Switch to IMEX-CN**: Use `timestepper=:imex_cn` for unconditional dispersion stability
- Reduce time step (`dt`)
- Increase dissipation (`νₕ₂`, `ilap2`)
- Check initial conditions for sharp gradients

### Memory Issues

- Reduce grid size
- Use checkpointing
- Enable MPI for distribution across nodes
