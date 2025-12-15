# [Numerical Methods](@id numerical-methods)

```@meta
CurrentModule = QGYBJ
```

This page describes the numerical algorithms used in QGYBJ.jl, including the 2D pencil decomposition strategy for parallel execution.

## Spatial Discretization

### Horizontal: Pseudo-Spectral Method

The model uses a **pseudo-spectral** approach in the horizontal:

| Operation | Space | Method |
|:----------|:------|:-------|
| Linear derivatives | Spectral | Multiply by `ik_x`, `ik_y` |
| Nonlinear products | Physical | Pointwise multiplication |
| Transform | Both | FFT / IFFT |

#### Advantages
- **Spectral accuracy** for smooth fields
- **Efficient** O(N log N) via FFT
- **No numerical diffusion** from derivatives

#### Dealiasing

Nonlinear products create aliasing errors. We use the **2/3 rule**:

```math
k_{max} = \frac{2}{3} \cdot \frac{N}{2}
```

Modes with `|k| > k_max` are set to zero after each nonlinear term.

```julia
# Apply dealiasing mask
mask = dealias_mask(grid)
@. field_k *= mask
```

### Vertical: Finite Differences

The vertical direction uses **second-order centered differences**:

```math
\frac{\partial f}{\partial z}\bigg|_k \approx \frac{f_{k+1} - f_{k-1}}{2\Delta z}
```

```math
\frac{\partial^2 f}{\partial z^2}\bigg|_k \approx \frac{f_{k+1} - 2f_k + f_{k-1}}{\Delta z^2}
```

For variable coefficients (stratification):

```math
\frac{\partial}{\partial z}\left(a(z)\frac{\partial f}{\partial z}\right) \approx \frac{a_{k+1/2}(f_{k+1} - f_k) - a_{k-1/2}(f_k - f_{k-1})}{\Delta z^2}
```

## Time Integration

### Leapfrog with Robert-Asselin Filter and Integrating Factors

The primary time stepping scheme is **leapfrog** with Robert-Asselin filtering and **integrating factors** for hyperdiffusion.

#### Forward Euler (First Step)

The first step uses forward Euler to bootstrap the leapfrog scheme:

```math
q^{n+1} = \left[ q^n - \Delta t \cdot J(\psi, q)^n + \Delta t \cdot D_q^n \right] \cdot e^{-\lambda \Delta t}
```

For the wave envelope (in real/imaginary form):

```math
B_R^{n+1} = \left[ B_R^n - \Delta t \cdot J(\psi, B_R) - \Delta t \cdot \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_I + \Delta t \cdot \frac{1}{2} r_{BI} \right] \cdot e^{-\lambda_w \Delta t}
```

```math
B_I^{n+1} = \left[ B_I^n - \Delta t \cdot J(\psi, B_I) + \Delta t \cdot \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_R - \Delta t \cdot \frac{1}{2} r_{BR} \right] \cdot e^{-\lambda_w \Delta t}
```

#### Leapfrog (Subsequent Steps)

Subsequent steps use centered leapfrog with integrating factors:

```math
q^{n+1} = q^{n-1} \cdot e^{-2\lambda \Delta t} - 2\Delta t \cdot J(\psi, q)^n \cdot e^{-\lambda \Delta t} + 2\Delta t \cdot D_q^n \cdot e^{-2\lambda \Delta t}
```

```math
B_R^{n+1} = B_R^{n-1} \cdot e^{-2\lambda_w \Delta t} - 2\Delta t \cdot \left[ J(\psi, B_R) + \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_I - \frac{1}{2} r_{BI} \right]^n \cdot e^{-\lambda_w \Delta t}
```

```math
B_I^{n+1} = B_I^{n-1} \cdot e^{-2\lambda_w \Delta t} - 2\Delta t \cdot \left[ J(\psi, B_I) - \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_R + \frac{1}{2} r_{BR} \right]^n \cdot e^{-\lambda_w \Delta t}
```

#### Robert-Asselin Filter

The Robert-Asselin filter damps the computational mode that can grow with leapfrog:

```math
\tilde{q}^n = q^n + \gamma \left( q^{n-1} - 2q^n + q^{n+1} \right)
```

where ``\gamma \approx 0.001`` (typically very small to minimize physical mode damping).

```julia
# First step: Forward Euler
first_projection_step!(state, grid, params, plans; a=a_vec, dealias_mask=mask)

# Subsequent steps: Leapfrog with Robert-Asselin
leapfrog_step!(state_np1, state_n, state_nm1, grid, params, plans;
               a=a_vec, dealias_mask=mask)
```

### Integrating Factor Method

The integrating factor ``\lambda`` handles hyperdiffusion exactly:

```math
\lambda = \nu_{h1} \left( |k_x|^{2p_1} + |k_y|^{2p_1} \right) + \nu_{h2} \left( |k_x|^{2p_2} + |k_y|^{2p_2} \right)
```

where:
- ``\nu_{h1}, p_1``: First hyperdiffusion operator (often for large-scale drag)
- ``\nu_{h2}, p_2``: Second hyperdiffusion operator (often ``p_2 = 4`` for small-scale dissipation)

The wave field has its own integrating factor ``\lambda_w`` with potentially different coefficients.

**Advantages of integrating factors:**
- Hyperdiffusion is treated **exactly** (no stability restriction)
- Allows much larger time steps than explicit diffusion treatment
- Second-order accuracy preserved for advective terms

## Elliptic Inversions

### Tridiagonal Systems

Both QG (q -> psi) and YBJ+ (B -> A) inversions lead to tridiagonal systems at each horizontal wavenumber (kx, ky):

```math
a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k
```

### Thomas Algorithm

We solve these in O(N) operations using the Thomas algorithm:

**Forward sweep:**
```julia
for k = 2:nz
    w = a[k] / b[k-1]
    b[k] = b[k] - w * c[k-1]
    d[k] = d[k] - w * d[k-1]
end
```

**Back substitution:**
```julia
x[nz] = d[nz] / b[nz]
for k = nz-1:-1:1
    x[k] = (d[k] - c[k] * x[k+1]) / b[k]
end
```

### Key Inversions

| Function | Solves | Physical Meaning |
|:---------|:-------|:-----------------|
| `invert_q_to_psi!` | nabla²psi + (f²/N²)d²psi/dz² = q | PV to streamfunction |
| `invert_B_to_A!` | L⁺A = B | Wave envelope to amplitude |
| `invert_helmholtz!` | nabla²phi - lambda*phi = f | General Helmholtz |

## FFT Implementation

### Serial Mode: FFTW

We use FFTW with **measured** plans for optimal performance:

```julia
# Create optimized plans
plans = plan_transforms!(grid)

# Forward FFT (physical -> spectral)
fft_forward!(dst, src, plans)

# Backward FFT (spectral -> physical)
fft_backward!(dst, src, plans)
```

Plan creation is expensive (~seconds) but execution is fast.

### Parallel Mode: PencilFFTs

For MPI parallel execution, we use PencilFFTs which handles distributed FFTs:

```julia
# Create parallel FFT plans
plans = QGYBJ.plan_mpi_transforms(grid, mpi_config)

# Same interface as serial
fft_forward!(dst, src, plans)
fft_backward!(dst, src, plans)
```

PencilFFTs automatically handles the transposes needed for distributed FFTs.

## 2D Pencil Decomposition

### The Challenge

The model requires two types of operations:
1. **Horizontal FFTs**: Need consecutive x and y data
2. **Vertical solves**: Need all z data at each (x,y) point

With 2D decomposition, no single configuration has all data local.

### Solution: Dual Pencil Configurations

QGYBJ.jl uses two pencil configurations:

| Configuration | Local Dimension | Distributed Dimensions | Use |
|:--------------|:----------------|:-----------------------|:----|
| **xy-pencil** | x | y, z | Horizontal FFTs |
| **z-pencil** | z | x, y | Vertical operations |

```
    xy-pencil                           z-pencil
   (x local)                          (z local)
┌─────────────────┐               ┌─────────────────┐
│ x: FULL         │               │ x: distributed  │
│ y: distributed  │  <----->      │ y: distributed  │
│ z: distributed  │  transpose    │ z: FULL         │
└─────────────────┘               └─────────────────┘
```

### Transpose Operations

Functions requiring vertical operations follow this pattern:

```julia
function some_vertical_operation!(S, G; workspace=nothing)
    # Check if 2D decomposition is active
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

    if need_transpose
        # 1. Transpose from xy-pencil to z-pencil
        transpose_to_z_pencil!(workspace.field_z, S.field, G)

        # 2. Perform vertical operation (z now fully local)
        _vertical_operation_on_z_pencil!(workspace.result_z, workspace.field_z, ...)

        # 3. Transpose result back to xy-pencil
        transpose_to_xy_pencil!(S.result, workspace.result_z, G)
    else
        # Serial mode: direct vertical operation
        _vertical_operation_direct!(S, G, ...)
    end
end
```

### Functions Using This Pattern

| Function | What it does | Needs z local? |
|:---------|:-------------|:---------------|
| `invert_q_to_psi!` | PV inversion | Yes (tridiagonal) |
| `invert_B_to_A!` | Wave amplitude recovery | Yes (tridiagonal) |
| `invert_helmholtz!` | General Helmholtz | Yes (tridiagonal) |
| `compute_vertical_velocity!` | Omega equation | Yes (tridiagonal) |
| `compute_ybj_vertical_velocity!` | YBJ w formula | Yes (vertical derivative) |
| `dissipation_q_nv!` | Numerical dissipation | Yes (vertical terms) |
| `sumB!` | Sum B over depth | Yes (vertical sum) |
| `compute_sigma` | YBJ sigma term | Yes (vertical operations) |
| `compute_A!` | Compute A from B | Yes (vertical operations) |
| `omega_eqn_rhs!` | RHS of omega equation | Yes (vertical derivatives) |

### Workspace Arrays

To avoid repeated allocation, pre-allocate z-pencil workspace:

```julia
# Initialize once
workspace = QGYBJ.init_mpi_workspace(grid, mpi_config)

# Contents:
# workspace.q_z, workspace.psi_z, workspace.B_z,
# workspace.A_z, workspace.C_z, workspace.work_z

# Pass to functions
invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
```

## Jacobian Computation

### Pseudo-Spectral Method

The Jacobian `J(a, b) = da/dx * db/dy - da/dy * db/dx` is computed:

1. Compute derivatives in spectral space:
   ```julia
   ax_k = im * kx .* a_k
   ay_k = im * ky .* a_k
   bx_k = im * kx .* b_k
   by_k = im * ky .* b_k
   ```

2. Transform to physical space:
   ```julia
   fft_backward!(ax, ax_k, plans)
   # ... etc
   ```

3. Compute product in physical space:
   ```julia
   J = ax .* by - ay .* bx
   ```

4. Transform back and dealias:
   ```julia
   fft_forward!(J_k, J, plans)
   J_k .*= dealias_mask
   ```

### Conservation Properties

The pseudo-spectral Jacobian conserves:
- **Circulation**: int J(a,b) dA = 0
- **Energy** (to machine precision in inviscid limit)
- **Enstrophy** (to machine precision in inviscid limit)

## Stability Constraints

### CFL Condition

For advection terms:

```math
\Delta t < \frac{\Delta x}{\max|u|} \approx \frac{L/N}{U}
```

### Diffusion Stability

With integrating factors, there is **no diffusion stability limit**.

Without integrating factors, explicit diffusion requires:

```math
\Delta t < \frac{\Delta x^{2p}}{2\nu}
```

For hyperdiffusion (p=4), this is very restrictive.

### Recommended Time Steps

| Resolution | Typical dt |
|:-----------|:-----------|
| 64^3 | 0.001 - 0.01 |
| 128^3 | 0.0005 - 0.005 |
| 256^3 | 0.0002 - 0.002 |
| 512^3 | 0.0001 - 0.001 |

## Memory Layout

### Array Ordering

Julia uses **column-major** ordering (Fortran-style):

```julia
# Fast index first for cache efficiency
for k = 1:nz
    for j = 1:ny
        for i = 1:nx
            field[i, j, k] = ...
        end
    end
end
```

### Complex Arrays

Spectral fields are stored as `Array{ComplexF64, 3}`:

```julia
# Spectral field dimensions
psi_k = zeros(ComplexF64, nx, ny, nz)
```

### PencilArrays (Parallel)

In parallel mode, arrays are `PencilArray{T,3}`:

```julia
# Access underlying data
data = parent(arr)

# Local dimensions
nx_local, ny_local, nz_local = size(data)
```

## Accuracy Verification

### Order of Accuracy

| Component | Spatial Order | Temporal Order |
|:----------|:--------------|:---------------|
| Horizontal derivatives | Spectral | - |
| Vertical derivatives | 2nd | - |
| Elliptic solvers | 2nd (vertical) | - |
| Time stepping (Leapfrog) | - | 2nd |
| Integrating factors | - | Exact |

### Conservation Tests

Run with inviscid settings to verify:
- Energy conservation (< 10^-10 relative change)
- Enstrophy conservation (< 10^-10 relative change)

```julia
# Check energy conservation
KE_initial = flow_kinetic_energy(state.u, state.v)
# ... run simulation ...
KE_final = flow_kinetic_energy(state.u, state.v)
println("Relative change: ", abs(KE_final - KE_initial) / KE_initial)
```

## Performance Optimization

### Key Optimizations

1. **Pre-allocated work arrays**: No allocations in time loop
2. **FFTW planning**: Measured plans for optimal performance
3. **Loop fusion**: `@.` macro for element-wise operations
4. **In-place operations**: Minimize memory allocation
5. **Workspace reuse**: Pre-allocated z-pencil arrays for transposes

### Profiling

```julia
using Profile

# Profile time stepping
@profile for _ in 1:100
    leapfrog_step!(state_np1, state_n, state_nm1, grid, params, plans;
                   a=a_vec, dealias_mask=mask, workspace=workspace)
end

Profile.print()
```

Typical hotspots:
- FFT transforms (~40-50%)
- Tridiagonal solves (~20-30%)
- Transpose operations (~10-20% in parallel)
- Array operations (~10-20%)

### Parallel Scaling

| Processes | Expected Speedup | Limiting Factor |
|:----------|:-----------------|:----------------|
| 1-16 | Near linear | - |
| 16-64 | Good | Transpose overhead |
| 64-256 | Moderate | Communication |
| 256+ | Diminishing | Problem size dependent |

## References

- Canuto, C., et al. (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer.
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*. Springer.
- PencilArrays.jl documentation: https://jipolanco.github.io/PencilArrays.jl/
- PencilFFTs.jl documentation: https://jipolanco.github.io/PencilFFTs.jl/
