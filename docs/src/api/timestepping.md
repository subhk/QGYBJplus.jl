# Time Stepping

```@meta
CurrentModule = QGYBJplus
```

QGYBJplus uses one production time stepper: a second-order exponential
Runge-Kutta method, exposed as `exp_rk2_step!` and used automatically by
`run!` and `run_simulation!`.

The equations remain dimensional. Horizontal hyperdiffusion is handled exactly
with integrating factors, while advection, refraction, dispersion, and vertical
PV diffusion are evaluated explicitly by the RK stages.

## High-Level Use

Most users should configure a model and run it:

```julia
grid = RectilinearGrid(size=(64, 64, 32),
                       x=(-250e3, 250e3),
                       y=(-250e3, 250e3),
                       z=(-4000.0, 0.0))

model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))

simulation = Simulation(model; Δt=300.0, stop_iteration=200)
run!(simulation)
```

There is no `timestepper` keyword. The simulation path always uses exponential
RK2.

## Low-Level Step

For development or tests, call the stepper directly:

```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0,
                     nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(par)
L = dealias_mask(G)

Snp1 = copy_state(S)
exp_rk2_step!(Snp1, S, G, par, plans; a=a, dealias_mask=L)
```

## API

```@docs
exp_rk2_step!
```
