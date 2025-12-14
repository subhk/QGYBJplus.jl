# [Time Stepping](@id api-timestepping)

```@meta
CurrentModule = QGYBJ
```

This page documents the time integration functions.

## Main Time Stepper

### timestep!

```@docs
timestep!
```

The main time stepping function. Performs one AB3 step.

**Algorithm:**
1. Compute nonlinear tendencies
2. Apply integrating factors for diffusion
3. AB3 update for prognostic variables
4. Invert elliptic equations
5. Update diagnostic variables

**Usage:**
```julia
timestep!(state, grid, params, work, plans, a_ell, dt)
```

## Time Integration Schemes

### Adams-Bashforth 3rd Order

```@docs
ab3_step!
```

The AB3 scheme:
```math
q^{n+1} = q^n + \Delta t\left(\frac{23}{12}F^n - \frac{16}{12}F^{n-1} + \frac{5}{12}F^{n-2}\right)
```

**Usage:**
```julia
ab3_step!(q_new, q, rq, rq_old, rq_old2, dt)
```

### Adams-Bashforth 2nd Order

```@docs
ab2_step!
```

Used for the second time step:
```math
q^{n+1} = q^n + \Delta t\left(\frac{3}{2}F^n - \frac{1}{2}F^{n-1}\right)
```

### Forward Euler

```@docs
euler_step!
```

Used for the first time step:
```math
q^{n+1} = q^n + \Delta t \cdot F^n
```

## Tendency Computation

### compute_rhs!

```@docs
compute_rhs_qg!
compute_rhs_wave!
```

Computes the right-hand side tendencies:

**QG:**
```math
F_q = -J(\psi, q) - J(\psi, q^w) + \text{dissipation}
```

**Wave:**
```math
F_B = -J(\psi, B) - B\frac{\partial\zeta}{\partial t} - i\frac{N^2}{2f_0}\nabla^2 A + \text{dissipation}
```

## Integrating Factors

### Purpose

For stiff diffusion terms, we transform:
```math
\tilde{q} = q \cdot e^{\nu k^{2p} t}
```

### Functions

```@docs
compute_integrating_factors
apply_integrating_factor!
remove_integrating_factor!
```

**Usage:**
```julia
# Setup (once)
IF_q, IF_B = compute_integrating_factors(grid, params, dt)

# Each step
apply_integrating_factor!(q, IF_q)
# ... time step ...
remove_integrating_factor!(q, IF_q)
```

## Startup Procedure

### First Steps

```@docs
startup_ab3!
```

AB3 requires history. For the first two steps:

1. **Step 1**: Forward Euler
2. **Step 2**: AB2

```julia
# Automatic handling
for step = 1:nsteps
    timestep!(state, grid, params, work, plans, a_ell, dt)
    # First 2 steps use Euler/AB2 automatically
end
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

## Sub-stepping

For very stiff problems:

```@docs
substep!
```

```julia
# Take N substeps within one outer step
n_sub = 4
dt_sub = dt / n_sub

for _ in 1:n_sub
    substep!(state, grid, params, work, plans, a_ell, dt_sub)
end
```

## State History

### Managing History Arrays

The state contains history for AB3:

```julia
# Current tendency
state.rq_new

# Previous tendencies
state.rq_old   # n-1
state.rq_old2  # n-2
```

### Rotating History

```@docs
rotate_history!
```

Called automatically at end of timestep:
```julia
state.rq_old2 .= state.rq_old
state.rq_old .= state.rq_new
```

## Performance

### Timing Breakdown

Typical distribution:
| Component | Fraction |
|:----------|:---------|
| FFTs | 40-50% |
| Elliptic solves | 20-30% |
| Array operations | 15-25% |
| History management | 5% |

### Optimization

```julia
# Pre-compute integrating factors
IF_q, IF_B = compute_integrating_factors(grid, params, dt)

# Reuse for all steps (if dt is constant)
for step = 1:nsteps
    timestep_with_IF!(state, ..., IF_q, IF_B)
end
```

## API Reference

```@docs
timestep!
ab3_step!
ab2_step!
euler_step!
compute_rhs_qg!
compute_rhs_wave!
compute_integrating_factors
apply_integrating_factor!
startup_ab3!
```
