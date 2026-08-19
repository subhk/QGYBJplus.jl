# [Running Simulations](@id running)

```@meta
CurrentModule = QGYBJplus
```

QGYBJplus uses second-order exponential Runge–Kutta (ETD-RK2) time stepping.
There is no runtime timestepper selector.

## Quick start

```julia
using QGYBJplus

grid = RectilinearGrid(size=(64, 64, 32),
                       extent=(500e3, 500e3, 4000.0), centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5),
                   flow=:evolving,
                   feedback=:wave_mean)

set!(model;
     ψ=(x, y, z) -> 1e3 * sin(2π*x/500e3) * cos(2π*y/500e3),
     waves=SurfaceWave(amplitude=0.1, scale=30.0))

simulation = Simulation(model;
                        Δt=20.0,
                        stop_time=86400.0,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(3600.0),
                                            fields=(:ψ, :waves)),
                        diagnostics=IterationInterval(100))
run!(simulation)
```

`QGYBJModel` owns the model state and numerical machinery. `Simulation` adds
the run clock, output, and diagnostics schedule. MPI decomposition is automatic;
launch the same script with `mpiexecjl -n N julia --project=. script.jl` for a
distributed run.

The lower-level driver remains available for custom integration loops:

```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0,
                     nx=64, ny=64, nz=32, dt=20.0, nt=1000)
G, S, plans, a = setup_model(par)

run_simulation!(S, G, par, plans; print_progress=true)
```

## Manual ETD-RK2 loop

Use the low-level stepper when adding custom diagnostics or particle handling:

```julia
mask = dealias_mask(G)
Sn = copy_state(S)
Snp1 = copy_state(S)
rk_workspace = ExpRK2Workspace(Sn, plans; G=G)

for step in 1:par.nt
    exp_rk2_step!(Snp1, Sn, G, par, plans;
                  a=a,
                  dealias_mask=mask,
                  timestep_workspace=rk_workspace,
                  current_time=(step - 1) * par.dt)
    Sn, Snp1 = Snp1, Sn
end
```

The two states are rotated after every step. `Sn` is therefore always the
current solution and `Snp1` is the destination buffer.

## Configured simulation

The configuration API owns setup, output, diagnostics, and MPI resources:

```julia
config = create_simple_config(
    nx=128, ny=128, nz=64,
    Lx=500e3, Ly=500e3, Lz=4000.0,
    dt=100.0, total_time=86400.0,
)

simulation = setup_simulation(config)
run_simulation!(simulation)
```

## MPI

For a distributed low-level loop, initialize an [`MPIWorkspace`](@ref) once and
pass it separately from the ETD workspace:

```julia
workspace = init_mpi_workspace(G, mpi_config)
rk_workspace = ExpRK2Workspace(Sn, plans; G=G)

exp_rk2_step!(Snp1, Sn, G, par, plans;
              a=a, dealias_mask=mask,
              workspace=workspace,
              timestep_workspace=rk_workspace)
```

See [MPI Parallelization](@ref parallel) for launch and decomposition details.

## Stability

ETD-RK2 integrates horizontal hyperdiffusion exactly, so that term does not
impose an explicit stability limit. Advection, refraction, dispersion, and
vertical diffusion remain explicit. Reduce `dt` if the advective CFL number or
the explicit wave/vertical-diffusion limit is too large.
