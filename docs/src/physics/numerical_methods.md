# [Numerical Methods](@id numerical-methods)

```@meta
CurrentModule = QGYBJplus
```

This page describes the numerical algorithms used in QGYBJ+.jl, including the 2D pencil decomposition strategy for parallel execution.

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

Nonlinear products create aliasing errors. We use the **radial 2/3 rule**:

```math
k_{max} = \frac{\min(N_x, N_y)}{3}
```

Modes with ``k_x^2 + k_y^2 > k_{max}^2`` are set to zero after each nonlinear term. The radial cutoff ensures isotropic treatment of modes.

```julia
# Apply dealiasing mask (radial cutoff)
mask = dealias_mask(grid)  # Returns 2D array of 0s and 1s
@. field_k *= mask
```

#### Hyperdiffusion Helper Functions

For dimensional simulations, use the helper functions to compute appropriate hyperdiffusion coefficients:

```julia
# Compute 4th order hyperdiffusion for 10-step e-folding at grid scale
hd = compute_hyperdiff_params(
    nx=128, ny=128, Lx=70e3, Ly=70e3, dt=10.0,
    order=4, efold_steps=10
)

# Returns: (ν=..., ilap=2, order=4)
# ν is in m⁴/s for 4th order (biharmonic ∇⁴)
```

The coefficient is computed such that the grid-scale mode decays by a factor of ``e`` in the specified number of time steps.

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
B_R^{n+1} = \left[ B_R^n - \Delta t \cdot J(\psi, B_R) - \Delta t \cdot \frac{N^2 k_h^2}{2 f_0} A_I + \Delta t \cdot \frac{1}{2} r_{BI} \right] \cdot e^{-\lambda_w \Delta t}
```

```math
B_I^{n+1} = \left[ B_I^n - \Delta t \cdot J(\psi, B_I) + \Delta t \cdot \frac{N^2 k_h^2}{2 f_0} A_R - \Delta t \cdot \frac{1}{2} r_{BR} \right] \cdot e^{-\lambda_w \Delta t}
```

where ``N^2`` is the buoyancy frequency squared and ``f_0`` is the Coriolis parameter.

#### Leapfrog (Subsequent Steps)

Subsequent steps use centered leapfrog with integrating factors:

```math
q^{n+1} = q^{n-1} \cdot e^{-2\lambda \Delta t} - 2\Delta t \cdot J(\psi, q)^n \cdot e^{-\lambda \Delta t} + 2\Delta t \cdot D_q^n \cdot e^{-2\lambda \Delta t}
```

```math
B_R^{n+1} = B_R^{n-1} \cdot e^{-2\lambda_w \Delta t} - 2\Delta t \cdot \left[ J(\psi, B_R) + \frac{N^2 k_h^2}{2 f_0} A_I - \frac{1}{2} r_{BI} \right]^n \cdot e^{-\lambda_w \Delta t}
```

```math
B_I^{n+1} = B_I^{n-1} \cdot e^{-2\lambda_w \Delta t} - 2\Delta t \cdot \left[ J(\psi, B_I) - \frac{N^2 k_h^2}{2 f_0} A_R + \frac{1}{2} r_{BR} \right]^n \cdot e^{-\lambda_w \Delta t}
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
\lambda = \nu_{h1} \left( |k_x|^{2 \cdot ilap1} + |k_y|^{2 \cdot ilap1} \right) + \nu_{h2} \left( |k_x|^{2 \cdot ilap2} + |k_y|^{2 \cdot ilap2} \right)
```

where (using Unicode parameters):
- ``\nu_{h1}`` (`νₕ₁`), `ilap1`: First hyperdiffusion operator (default: biharmonic with ilap1=2)
- ``\nu_{h2}`` (`νₕ₂`), `ilap2`: Second hyperdiffusion operator (default: hyper-6 with ilap2=6)

The wave field has its own integrating factor ``\lambda_w`` with potentially different coefficients.

**Advantages of integrating factors:**
- Hyperdiffusion is treated **exactly** (no stability restriction)
- Allows much larger time steps than explicit diffusion treatment
- Second-order accuracy preserved for advective terms

### Second-Order IMEX-CNAB with Strang Splitting

For applications where the dispersion CFL constraint (dt ≤ 2f/N² ≈ 2s) is limiting, we provide a **second-order IMEX-CNAB** scheme that treats dispersion implicitly and uses Strang splitting for refraction.

#### The Challenge: Refraction Instability

The YBJ+ equation includes a refraction term ``-(i/2)\zeta B`` which, when discretized with forward Euler, is unconditionally unstable:

```math
|1 - i \Delta t \zeta/2| = \sqrt{1 + (\Delta t \zeta/2)^2} > 1
```

This amplifies energy regardless of time step size.

#### Solution: Strang Splitting + IMEX-CNAB

We use **Strang splitting** for refraction (second-order) combined with **Adams-Bashforth 2** for advection (second-order):

**Stage 1 - First Half-Refraction (Strang):**
```math
B^* = B^n \times \exp(-i \frac{\Delta t}{2} \frac{\zeta}{2})
```

**Stage 2 - IMEX-CNAB for Advection + Dispersion:**
```math
B^{**} - \frac{\Delta t}{2} i \alpha_{\text{disp}} k_h^2 A^{n+1} = B^* + \frac{\Delta t}{2} i \alpha_{\text{disp}} k_h^2 A^* + \frac{3\Delta t}{2} N^n - \frac{\Delta t}{2} N^{n-1}
```

where ``N^n = -J(\psi^n, B^n)`` is the advection tendency at time ``n``, and Adams-Bashforth 2 extrapolation ``\frac{3}{2}N^n - \frac{1}{2}N^{n-1}`` provides second-order accuracy.

**Stage 3 - Second Half-Refraction (Strang):**
```math
B^{n+1} = B^{**} \times \exp(-i \frac{\Delta t}{2} \frac{\zeta}{2})
```

Since ``|\exp(-i \theta)| = 1`` for real ``\theta``, refraction is **exactly energy-preserving**.

**Mean-Flow Update (q):** The PV equation is advanced with Adams–Bashforth 2 on
``-J(\psi,q) + \text{diffusion}``, using the integrating factor for hyperdiffusion.
To keep the coupled system second-order, the second refraction half-step uses
``\psi^{n+1}`` predicted from the updated ``q`` (and ``q^w`` when wave feedback is enabled).

#### Critical: Consistent A*

After applying refraction to get ``B^*``, we must compute ``A^* = (L^+)^{-1} B^*`` (not use ``A^n``). Using ``A^n`` with ``B^*`` breaks the consistency required by IMEX-CN, causing instability.

#### Modified Elliptic Problem

Substituting ``B = L^+ A`` into the IMEX-CN equation:
```math
(L^+ - \beta) A^{n+1} = \text{RHS}
```
where ``\beta = (\Delta t/2) \cdot i \cdot \alpha_{\text{disp}} \cdot k_h^2``.

This is a tridiagonal system for each ``(k_x, k_y)`` mode, solved with the Thomas algorithm.

#### Temporal Accuracy

| Component | Method | Order |
|:----------|:-------|:------|
| Refraction | Strang splitting | 2nd |
| Dispersion | Crank-Nicolson | 2nd |
| Advection | Adams-Bashforth 2 | 2nd |
| **Overall** | **IMEX-CNAB** | **2nd** |

Note: The first time step uses forward Euler for advection (AB2 bootstrap), so the very first step is first-order.

#### Stability Summary

| Term | Treatment | Stability |
|:-----|:----------|:----------|
| Refraction | Exact integrating factor | Unconditionally stable |
| Dispersion | Implicit Crank-Nicolson | Unconditionally stable |
| Advection | Explicit Adams-Bashforth 2 | CFL: ``\Delta t < \Delta x / U_{\max}`` |

For typical oceanographic parameters (U ≈ 0.3 m/s, dx ≈ 300m), this allows **dt ≈ 20s** vs **dt ≈ 2s** for explicit leapfrog—a **10x speedup**.

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
plans = QGYBJplus.plan_mpi_transforms(grid, mpi_config)

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

QGYBJ+.jl uses two pencil configurations:

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
workspace = QGYBJplus.init_mpi_workspace(grid, mpi_config)

# Contents:
# workspace.q_z, workspace.psi_z, workspace.B_z,
# workspace.A_z, workspace.C_z, workspace.work_z

# Pass to functions
invert_q_to_psi!(state, grid; a=a_vec, workspace=workspace)
```

## Jacobian/Advection Computation

### Divergence Form (convol_waqg)

For advection terms like ``J(\psi, q)``, QGYBJ+.jl uses the **divergence form**:

```math
J(\psi, q) = \frac{\partial(uq)}{\partial x} + \frac{\partial(vq)}{\partial y} = ik_x \widehat{uq} + ik_y \widehat{vq}
```

where ``u = -\partial\psi/\partial y`` and ``v = \partial\psi/\partial x`` are the geostrophic velocities.

### Algorithm (convol_waqg)

1. Precompute velocities in real space: ``u_r``, ``v_r``

2. Transform field to real space:
   ```julia
   fft_backward!(qr, qk, plans)
   ```

3. Compute products in real space:
   ```julia
   uterm = u_r .* qr
   vterm = v_r .* qr
   ```

4. Transform back and compute divergence:
   ```julia
   fft_forward!(uterm_k, uterm, plans)
   fft_forward!(vterm_k, vterm, plans)
   J_k = im * kx .* uterm_k + im * ky .* vterm_k
   ```

5. Apply dealiasing:
   ```julia
   J_k[.!dealias_mask] .= 0
   ```

6. Normalize (for unnormalized FFT):
   ```julia
   J_k ./= (nx * ny)
   ```

### Conservation Properties

The pseudo-spectral advection conserves:
- **Circulation**: ``\int J(\psi, q) \, dA = 0``
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
            field[k, i, j] = ...
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
