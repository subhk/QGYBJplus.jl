# [Numerical methods](@id numerical-methods)

```@meta
CurrentModule = QGYBJplus
```

## Horizontal discretization

Horizontal derivatives are pseudo-spectral. Linear derivatives multiply
Fourier coefficients by `im*kx` or `im*ky`; nonlinear products are evaluated
in physical space and transformed back.

The two-thirds mask retains the integer mode indices inside an ellipse,

```math
\begin{aligned}
\frac{i_x^2}{K_x^2} + \frac{i_y^2}{K_y^2} &< 1, && 3 \mid n_x \text{ or } 3 \mid n_y, \\
\frac{i_x^2}{K_x^2} + \frac{i_y^2}{K_y^2} &\leq 1, && \text{otherwise},
\end{aligned}
\qquad
K_x = \left\lfloor\frac{n_x}{3}\right\rfloor,\quad
K_y = \left\lfloor\frac{n_y}{3}\right\rfloor,
```

which is a disk on a square grid. Every retained mode then has
``|i_x| < n_x/3`` and ``|i_y| < n_y/3``, so a quadratic product of retained
modes never aliases onto a retained mode. The strict inequality excludes the
exact ``|k|=n/3`` boundary when a dimension is divisible by three, where two
cutoff modes could otherwise alias onto the opposite retained boundary. A
dimension with fewer than three points keeps only its mean.

```julia
mask = dealias_mask(model.grid)
is_dealiased(2, 1, model.grid)
```

## Vertical discretization

Vertical nodes are uniform cell centers. Second-order differences discretize
vertical derivatives and the variable-coefficient operator

```math
\partial_z\!\left(a(z)\,\partial_z \phi\right),
\qquad a(z)=f^2/N^2(z).
```

At every horizontal Fourier mode, elliptic inversions reduce to a tridiagonal
vertical solve. The pointwise `N²` profile is sampled at cell centers, while
the conservative flux uses
`a_ell[k] = f² / N²(grid.z_faces[k + 1])` at each cell's upper face. Every
vertical column is solved by one kernel work item with the Thomas algorithm,
so all horizontal modes are solved in parallel on the CPU or GPU.

## ETD-RK2

For

```math
u_t = Lu + N(u),
```

the two stages are

```math
\begin{aligned}
a &= e^{hL}u_n + h\varphi_1(hL)N(u_n),\\
u_{n+1} &= e^{hL}u_n + h\varphi_1(hL)N(u_n)
          + h\varphi_2(hL)\left[N(a)-N(u_n)\right].
\end{aligned}
```

The diagonal linear term is horizontal hyperdiffusion,

```math
\lambda = \nu_1(k_x^2+k_y^2)^{p_1/2}
          + \nu_2(k_x^2+k_y^2)^{p_2/2}.
```

Here each ``p_i`` is the configured total (positive, even) derivative order.

The implementation evaluates `exp(-λh)`, `hφ₁(-λh)`, and `hφ₂(-λh)` with
cancellation-safe series near zero. Advection, refraction, dispersion, and
vertical diffusion are evaluated at both stages.

```julia
simulation = Simulation(model; Δt=10.0, stop_iteration=100)
run!(simulation)
```

For manual control:

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
step!(model, timestepper)
```

Manual `step!` calls do not update a simulation clock, particles, output, or
diagnostic schedules. Use `run!` for ordinary simulations.

## Stability

The horizontal dissipative term is integrated exactly and does not impose an
explicit diffusion limit. Choose `Δt` for advective CFL, explicit dispersion,
and vertical-diffusion constraints.

Dispersion is explicit, and a two-stage Runge–Kutta step amplifies an
oscillation of frequency ``\omega`` by ``\sqrt{1+(\omega\Delta t)^4/4}`` per
step. Under YBJ⁺ the fastest envelope frequency is ``2f``, carried by the
depth-mean part of `B` (``LA`` has no depth mean, so ``B=\nabla^2A/4``
there), so keep ``2f\Delta t`` small: at ``2f\Delta t = 0.1`` the
amplification is ``1+1.3\times10^{-5}`` per step, while at ``2f\Delta t \gtrsim
0.5`` round-off in that mode grows visibly over long runs. The original YBJ
dispersion is unbounded at high horizontal wavenumber and weak stratification,
which is the stiffness YBJ⁺ removes.

Dimensional helper functions can target an e-folding time at a selected
horizontal scale:

```julia
hyperdiffusion = compute_hyperdiff_params(
    nx=128,
    ny=128,
    Lx=70e3,
    Ly=70e3,
    dt=10.0,
    order=4,
    efold_steps=10,
)
```

## Transforms and kernels

`ModelRuntime` owns the transform plans, reusable work arrays, coefficient
profiles, and the dealiasing mask, all on the model's architecture. Horizontal
transforms are one batched 2D FFT over every vertical level (FFTW on the CPU,
cuFFT on CUDA GPUs), and every other array loop is a KernelAbstractions
kernel. Application operators accept the model:

```julia
invert_q_to_psi!(model)
invert_B_to_A!(model)
compute_velocities!(model)
```

See [CPU and GPU execution](@ref architectures).

## Array layout

All model arrays use `(z, x, y)` ordering. Spectral fields are complex and
physical velocities are real. NetCDF output converts to `(x, y, z)`.

## Verification

The test suite checks ETD integrating-factor damping, workspace reuse,
second-order convergence, conservation properties, and that the GPU code path
(run on the CPU with JLArrays) reproduces the CPU model.
