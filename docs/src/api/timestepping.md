# [Time Stepping](@id api-timestepping)

```@meta
CurrentModule = QGYBJ
```

This page documents the time integration functions.

## Main Time Stepping Scheme

QGYBJ.jl uses a **Leapfrog scheme with Robert-Asselin filter** for time integration. This provides second-order accuracy while maintaining stability through computational mode damping.

### Overview

The time stepping consists of two functions:
1. `first_projection_step!` - Forward Euler initialization
2. `leapfrog_step!` - Main leapfrog integration with Robert-Asselin filter

## Forward Euler Projection Step

```@docs
first_projection_step!
```

**Purpose:** Initialize the leapfrog scheme by providing values at times n and n-1.

**Algorithm:**
1. Compute tendencies at time n (advection, refraction, diffusion)
2. Apply physics switches (linear, inviscid, etc.)
3. Forward Euler update with integrating factors
4. Wave feedback (optional)
5. Diagnostic inversions (q → ψ → u, v)

**Usage:**
```julia
# Serial mode
first_projection_step!(state, grid, params, plans; a=a_ell, dealias_mask=L)

# Parallel mode (2D decomposition)
first_projection_step!(state, grid, params, plans; a=a_ell, dealias_mask=L, workspace=workspace)
```

## Leapfrog Step with Robert-Asselin Filter

```@docs
leapfrog_step!
```

**The Leapfrog scheme:**
```math
\phi^{n+1} = \phi^{n-1} + 2\Delta t \times F^n
```

**Robert-Asselin filter (damps computational mode):**
```math
\tilde{\phi}^n = \phi^n + \gamma(\phi^{n-1} - 2\phi^n + \phi^{n+1})
```

**With integrating factor for hyperdiffusion:**
```math
\phi^{n+1} = \phi^{n-1} \times e^{-2\lambda\Delta t} + 2\Delta t \times F^n \times e^{-\lambda\Delta t}
```

**Usage:**
```julia
# Serial mode
leapfrog_step!(Snp1, Sn, Snm1, grid, params, plans; a=a_ell, dealias_mask=L)

# Parallel mode (2D decomposition)
leapfrog_step!(Snp1, Sn, Snm1, grid, params, plans; a=a_ell, dealias_mask=L, workspace=workspace)

# Time level rotation after each step
Snm1, Sn, Snp1 = Sn, Snp1, Snm1
```

## Forward Euler Update

Used for the first (projection) step:
```math
\phi^{n+1} = \left[\phi^n - \Delta t \times F\right] \times e^{-\lambda\Delta t}
```

The integrating factor `e^{-λΔt}` handles hyperdiffusion exactly.

## Tendency Computation

Each time step computes the following tendencies:

**QG Potential Vorticity:**
```math
F_q = -J(\psi, q) + \nu_z \frac{\partial^2 q}{\partial z^2}
```

**Wave Envelope (real/imaginary parts):**
```math
F_{BR} = -J(\psi, BR) - \frac{k_h^2}{2}A_I + \frac{1}{2}BI \times \zeta
```
```math
F_{BI} = -J(\psi, BI) + \frac{k_h^2}{2}A_R - \frac{1}{2}BR \times \zeta
```

### Nonlinear Terms

```@docs
convol_waqg!
refraction_waqg!
```

**Advection:** `convol_waqg!` computes J(ψ, q), J(ψ, BR), J(ψ, BI)

**Refraction:** `refraction_waqg!` computes B × ζ where ζ = ∇²ψ

### Vertical Diffusion

```@docs
dissipation_q_nv!
```

Computes νz ∂²q/∂z² using the tridiagonal solver. Automatically handles 2D decomposition transposes.

## Integrating Factors

### Purpose

For stiff hyperdiffusion terms, we use an integrating factor approach:
```math
\tilde{\phi} = \phi \times e^{\nu k^{2p} t}
```

This allows exact treatment of the linear diffusion while using explicit time stepping.

### Function

```@docs
int_factor
```

**Usage:**
```julia
# Compute factor for a spectral mode
If = int_factor(kx, ky, params; waves=false)   # For mean flow (q)
Ifw = int_factor(kx, ky, params; waves=true)   # For waves (B)

# Apply in time stepping
q_new = q_old * exp(-2*If) - 2*dt * tendency * exp(-If)   # Leapfrog
q_new = q_old * exp(-If) - dt * tendency                    # Euler
```

## Complete Simulation Loop

### Setup and Run

```julia
using QGYBJ

# Initialize
params = default_params(nx=64, ny=64, nz=32)
grid = init_grid(params)
plans = plan_transforms!(grid)
a_ell = a_ell_ut(params, grid)
L = dealias_mask(params, grid)

# Create three state arrays for leapfrog
Snm1 = init_state(grid)  # n-1
Sn = init_state(grid)    # n
Snp1 = init_state(grid)  # n+1

# Initialize with random fields
init_random_psi!(Sn, grid, params, plans; a=a_ell)

# Projection step (Forward Euler initialization)
first_projection_step!(Sn, grid, params, plans; a=a_ell, dealias_mask=L)

# Copy for n-1 state
copy_state!(Snm1, Sn)

# Main time loop
for iter in 1:nsteps
    leapfrog_step!(Snp1, Sn, Snm1, grid, params, plans; a=a_ell, dealias_mask=L)

    # Rotate time levels
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end
```

### Parallel Mode (2D Decomposition)

```julia
using MPI, PencilArrays, PencilFFTs, QGYBJ

MPI.Init()
mpi_config = QGYBJ.setup_mpi_environment()

# Initialize with MPI
params = default_params(nx=256, ny=256, nz=128)
grid = QGYBJ.init_mpi_grid(params, mpi_config)
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

a_ell = a_ell_ut(params, grid)
L = dealias_mask(params, grid)

# Create states
Snm1 = QGYBJ.init_mpi_state(grid, mpi_config)
Sn = QGYBJ.init_mpi_state(grid, mpi_config)
Snp1 = QGYBJ.init_mpi_state(grid, mpi_config)

# Initialize
init_random_psi!(Sn, grid, params, plans; a=a_ell)
first_projection_step!(Sn, grid, params, plans; a=a_ell, dealias_mask=L, workspace=workspace)
copy_state!(Snm1, Sn)

# Main loop with workspace for 2D decomposition
for iter in 1:nsteps
    leapfrog_step!(Snp1, Sn, Snm1, grid, params, plans;
                   a=a_ell, dealias_mask=L, workspace=workspace)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end

MPI.Finalize()
```

## CFL Condition

### Stability Constraint

```julia
function compute_cfl(state, grid, dt)
    u_max = maximum(abs.(state.u))
    v_max = maximum(abs.(state.v))
    return dt * max(u_max/grid.dx, v_max/grid.dy)
end
```

For stability, CFL < 1 is required. Recommended: CFL ≈ 0.5.

### Adaptive Time Stepping

```julia
function adaptive_dt(state, grid; cfl_target=0.5, dt_max=0.01)
    u_max = maximum(abs.(state.u)) + 1e-10  # Avoid division by zero
    v_max = maximum(abs.(state.v)) + 1e-10

    dt = cfl_target * min(grid.dx/u_max, grid.dy/v_max)
    return min(dt, dt_max)
end
```

## Robert-Asselin Filter Parameter

The filter coefficient γ (`γ` in `QGParams`) controls damping of the computational mode:

- **Too large** (γ > 0.01): Excessive damping, accuracy loss
- **Too small** (γ < 0.0001): Computational mode growth
- **Recommended**: γ ≈ 0.001 (default)

```julia
params = default_params(nx=64, ny=64, nz=32; γ=0.001)
```

Note: Type `\gamma<tab>` in the Julia REPL to enter `γ`.

## Time Level Management

The leapfrog scheme requires three time levels:

| Variable | Description | Usage |
|:---------|:------------|:------|
| `Snm1` | State at n-1 | Input, receives filtered n values |
| `Sn` | State at n | Input (unchanged) |
| `Snp1` | State at n+1 | Output |

After each step, rotate the pointers:
```julia
Snm1, Sn, Snp1 = Sn, Snp1, Snm1
```

This avoids data copying by just swapping references.

## Physics Switches

The time stepping respects these `QGParams` switches:

| Switch | Effect |
|:-------|:-------|
| `linear` | Zero nonlinear advection J(ψ, q), J(ψ, B) |
| `inviscid` | Zero vertical diffusion νz ∂²q/∂z² |
| `passive_scalar` | Waves as passive tracers (no dispersion/refraction) |
| `no_dispersion` | Zero wave dispersion (A = 0) |
| `fixed_flow` | Mean flow doesn't evolve (q unchanged) |
| `no_wave_feedback` | No qʷ feedback term |

## Performance

### Timing Breakdown

Typical distribution:
| Component | Fraction |
|:----------|:---------|
| FFTs | 40-50% |
| Elliptic solves | 20-30% |
| Array operations | 15-25% |
| Transpose operations (2D) | 5-10% |

### 2D Decomposition Notes

When using 2D decomposition:
- Pass `workspace` argument to avoid repeated allocation
- Vertical operations (inversions, diffusion) use automatic transposes
- The workspace contains pre-allocated z-pencil arrays

```julia
# Pre-allocate workspace (once)
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

# Reuse for all steps
for step in 1:nsteps
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                   a=a, dealias_mask=L, workspace=workspace)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end
```

## API Summary

All time stepping functions documented above:
- `first_projection_step!` - Forward Euler initialization step
- `leapfrog_step!` - Main leapfrog integration with Robert-Asselin filter
- `convol_waqg!` - Nonlinear advection computation
- `refraction_waqg!` - Wave refraction term
- `dissipation_q_nv!` - Vertical diffusion
- `int_factor` - Integrating factor for hyperdiffusion
- `compute_qw!` - Wave feedback term (see [Physics API](physics.md))
