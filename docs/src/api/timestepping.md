# Time stepping

QGYBJplus uses a second-order exponential time-differencing Runge–Kutta method
(ETD-RK2). Horizontal flow and wave hyperdiffusion are integrated exactly;
advection, refraction, dispersion, and vertical diffusion are evaluated at two
explicit Runge–Kutta stages.

For a semilinear system ``u_t = Lu + N(u)``, the update is

```math
\begin{aligned}
a &= e^{hL}u_n + h\varphi_1(hL)N(u_n),\\
u_{n+1} &= a + h\varphi_2(hL)\left[N(a)-N(u_n)\right],
\end{aligned}
```

where ``\varphi_1(z)=(e^z-1)/z`` and
``\varphi_2(z)=(e^z-1-z)/z^2``. The implementation uses series expansions near
zero to avoid cancellation.

## High-level use

[`run_simulation!`](@ref) and [`run!`](@ref) always use ETD-RK2:

```julia
run_simulation!(state, grid, params, plans;
                output_config=output_config,
                N2_profile=N2_profile)
```

## Low-level use

```julia
a = a_ell_ut(params, grid)
mask = dealias_mask(grid)
next_state = copy_state(state)
rk_workspace = ExpRK2Workspace(state, plans; G=grid)

exp_rk2_step!(next_state, state, grid, params, plans;
              a=a,
              dealias_mask=mask,
              timestep_workspace=rk_workspace)
```

For MPI runs, pass the reusable [`MPIWorkspace`](@ref) as `workspace` and the
[`ExpRK2Workspace`](@ref) as `timestep_workspace`.

## API

```@docs
exp_rk2_step!
ExpRK2Workspace
```
