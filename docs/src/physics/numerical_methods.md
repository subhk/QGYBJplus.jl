# [Numerical methods](@id numerical-methods)

```@meta
CurrentModule = QGYBJplus
```

## Horizontal discretization

Horizontal derivatives are pseudo-spectral. Linear derivatives multiply
Fourier coefficients by `im*kx` or `im*ky`; nonlinear products are evaluated
in physical space and transformed back.

The radial two-thirds mask retains indices satisfying

```math
i_x^2 + i_y^2 \leq \left\lfloor\frac{\min(n_x,n_y)}{3}\right\rfloor^2.
```

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
vertical solve. Under MPI the runtime transposes between the spectral output
pencil and a pencil with the full vertical column local.

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
\lambda = \nu_1(k_x^2+k_y^2)^{p_1}
          + \nu_2(k_x^2+k_y^2)^{p_2}.
```

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
and vertical-diffusion constraints. Dimensional helper functions can target an
e-folding time at a selected horizontal scale:

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

## Distributed transforms

`ModelRuntime` owns transform plans, decomposition metadata, transpose
destinations, and reusable work arrays. Application operators accept the model
and select the correct local layout automatically:

```julia
invert_q_to_psi!(model)
invert_B_to_A!(model)
compute_velocities!(model)
```

Use `get_local_range_physical(model)` and
`get_local_range_spectral(model)` only when writing custom distributed loops.

## Array layout

All model arrays use `(z, x, y)` ordering. Spectral fields are complex and
physical velocities are real. NetCDF output converts to `(x, y, z)`.

## Verification

The test suite checks ETD integrating-factor damping, workspace reuse,
one-/multi-rank transform equivalence, MPI rank-independent results, and
periodic particle migration.
