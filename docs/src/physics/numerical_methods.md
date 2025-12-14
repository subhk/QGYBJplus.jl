# [Numerical Methods](@id numerical-methods)

```@meta
CurrentModule = QGYBJ
```

This page describes the numerical algorithms used in QGYBJ.jl.

## Spatial Discretization

### Horizontal: Pseudo-Spectral Method

The model uses a **pseudo-spectral** approach in the horizontal:

| Operation | Space | Method |
|:----------|:------|:-------|
| Linear derivatives | Spectral | Multiply by ``ik_x``, ``ik_y`` |
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

Modes with ``|k| > k_{max}`` are set to zero after each nonlinear term.

```julia
# Apply dealiasing mask
@. field_k *= dealias_mask
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

### Adams-Bashforth 3rd Order (AB3)

For the nonlinear terms, we use explicit AB3:

```math
N^{n+1} = N^n + \Delta t\left(\frac{23}{12}F^n - \frac{16}{12}F^{n-1} + \frac{5}{12}F^{n-2}\right)
```

where ``F`` represents nonlinear tendencies (Jacobians, refraction).

#### Startup Procedure
- Step 1: Forward Euler
- Step 2: AB2 (two-level)
- Step 3+: Full AB3

### Integrating Factor Method

For linear (diffusive) terms, we use **integrating factors** to avoid stiffness:

```math
\tilde{q} = q \cdot e^{\nu k^{2p} t}
```

The transformed equation has no linear term:

```math
\frac{\partial \tilde{q}}{\partial t} = e^{\nu k^{2p} t} \cdot N(q)
```

This allows much larger time steps than explicit treatment of diffusion.

#### Implementation

```julia
# Pre-compute integrating factors
IF = exp.(nu .* kh.^(2p) .* dt)
IFh = exp.(0.5 .* nu .* kh.^(2p) .* dt)

# Time stepping with integrating factor
q_new = IF .* q_old + dt * tendency
```

## Elliptic Inversions

### Tridiagonal Systems

Both QG (q → ψ) and YBJ+ (B → A) inversions lead to tridiagonal systems:

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

### Pre-factorization

For efficiency, we **pre-factor** the tridiagonal matrices:

```julia
# Setup phase (once)
a_ell = setup_elliptic_matrices(grid, params)

# Solve phase (each time step)
invert_q_to_psi!(state, grid, params, a_ell)
```

This reuses the factored matrices across all ``(k_x, k_y)`` wavenumbers.

## FFT Implementation

### FFTW Planning

We use FFTW with **measured** plans for optimal performance:

```julia
# Create optimized plans
plan_forward = plan_rfft(field, flags=FFTW.MEASURE)
plan_backward = plan_irfft(field_k, nx, flags=FFTW.MEASURE)
```

Plan creation is expensive (~seconds) but execution is fast.

### Real-to-Complex Transforms

For real fields, we use `rfft`/`irfft`:

| Transform | Input Size | Output Size |
|:----------|:-----------|:------------|
| rfft | (nx, ny) | (nx÷2+1, ny) |
| irfft | (nx÷2+1, ny) | (nx, ny) |

This reduces memory by ~2x compared to complex FFTs.

### In-Place Transforms

Where possible, we use in-place transforms:

```julia
# In-place forward transform
mul!(field_k, plan_forward, field)

# In-place backward transform
mul!(field, plan_backward, field_k)
```

## Jacobian Computation

### Arakawa Method

The Jacobian ``J(a, b)`` is computed pseudo-spectrally:

1. Compute derivatives in spectral space:
   ```julia
   ax_k = im * kx .* a_k
   ay_k = im * ky .* a_k
   bx_k = im * kx .* b_k
   by_k = im * ky .* b_k
   ```

2. Transform to physical space:
   ```julia
   ax = irfft(ax_k)
   ay = irfft(ay_k)
   # ... etc
   ```

3. Compute product in physical space:
   ```julia
   J = ax .* by - ay .* bx
   ```

4. Transform back and dealias:
   ```julia
   J_k = rfft(J) .* dealias_mask
   ```

### Conservation Properties

The pseudo-spectral Jacobian conserves:
- **Circulation**: ``\int J(a,b) \, dA = 0``
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

| Resolution | Typical ``\Delta t`` |
|:-----------|:---------------------|
| 64³ | 0.001 - 0.01 |
| 128³ | 0.0005 - 0.005 |
| 256³ | 0.0002 - 0.002 |
| 512³ | 0.0001 - 0.001 |

## Memory Layout

### Array Ordering

Julia uses **column-major** ordering (Fortran-style):

```julia
# Fast index first
for k = 1:nz
    for j = 1:ny
        for i = 1:nx
            field[i, j, k] = ...
        end
    end
end
```

Horizontal loops are innermost for cache efficiency.

### Complex Arrays

Spectral fields are stored as `Array{ComplexF64, 3}`:

```julia
# Spectral field dimensions
psi_k = zeros(ComplexF64, nx÷2+1, ny, nz)
```

## Parallelization

### Serial Execution

By default, QGYBJ.jl runs serially with multi-threaded BLAS/FFTW.

### MPI Parallel

With MPI enabled, the domain is decomposed using **pencil decomposition**:

```
        Pencil-X          Pencil-Y          Pencil-Z
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ proc 0 │ proc 1 │  │   proc 0      │  │   proc 0      │
    │───────┼───────│  │───────────────│  │───────────────│
    │ proc 2 │ proc 3 │  │   proc 1      │  │   proc 1      │
    │───────┼───────│  │───────────────│  │───────────────│
    │ proc 4 │ proc 5 │  │   proc 2      │  │   proc 2      │
    └───────────────┘  └───────────────┘  └───────────────┘
```

See [MPI Parallelization](@ref parallel) for details.

## Accuracy Verification

### Order of Accuracy

| Component | Spatial Order | Temporal Order |
|:----------|:--------------|:---------------|
| Horizontal derivatives | Spectral | - |
| Vertical derivatives | 2nd | - |
| Elliptic solvers | 2nd (vertical) | - |
| Time stepping (AB3) | - | 3rd |
| Integrating factors | - | Exact |

### Conservation Tests

Run with `inviscid=true` to verify:
- Energy conservation (should be < 10⁻¹⁰ relative change)
- Enstrophy conservation (should be < 10⁻¹⁰ relative change)

## Performance Optimization

### Key Optimizations

1. **Pre-allocated work arrays**: No allocations in time loop
2. **FFTW planning**: Measured plans for optimal performance
3. **Loop fusion**: `@.` macro for element-wise operations
4. **Cache blocking**: Vertical loops chunked for L1 cache

### Profiling

```julia
using Profile
@profile run_simulation(config)
Profile.print()
```

Typical hotspots:
- FFT transforms (~40-50%)
- Tridiagonal solves (~20-30%)
- Array operations (~20-30%)

## References

- Canuto, C., et al. (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer.
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*. Springer.
