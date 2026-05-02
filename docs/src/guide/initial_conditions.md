# [Initial Conditions](@id initial-conditions)

```@meta
CurrentModule = QGYBJplus
```

Use `set!` to initialize fields in high-level scripts.

## Mean Flow from a Streamfunction

```julia
ψᵢ = (x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3)

set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic)
```

`pv_method = :barotropic` computes the barotropic PV from the streamfunction.
This is the most common setup for fixed-flow YBJ or YBJ+ experiments.

## Surface-Confined Waves

```julia
set!(model;
     waves = SurfaceWave(amplitude = 0.05,
                         scale = 500.0,
                         profile = :gaussian))
```

The amplitude is dimensional velocity scale in m/s. The scale is the vertical
decay scale in meters.

## Flow and Waves Together

```julia
set!(model;
     ψ = ψᵢ,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05, scale = 500.0))
```

## Low-Level Initialization

For numerical development, the lower-level state API is still available:

```julia
params = default_params(Lx = 500e3, Ly = 500e3, Lz = 4000.0,
                        nx = 64, ny = 64, nz = 32)

grid, state, plans, a_ell = setup_model(params)

init_random_psi!(state, grid; amplitude = 0.1, seed = 1234)
compute_q_from_psi!(state, grid, plans, a_ell)
init_surface_waves!(state, grid, params; amplitude = 0.05, scale = 500.0)
```

Most user scripts should prefer `set!` because it keeps grid, transforms,
state, and physics bookkeeping inside the model object.

## From NetCDF

For advanced workflows, read initial fields and assign them to the low-level
state:

```julia
ψ = read_initial_psi("initial_psi.nc", grid, plans)
B = read_initial_waves("initial_waves.nc", grid, plans)

state.psi .= ψ
state.L⁺A .= B
```

When running with MPI, pass the MPI configuration so rank 0 reads and scatters
the data:

```julia
ψ = read_initial_psi("initial_psi.nc", grid, plans; parallel_config = mpi_config)
```

## Quick Checks

After initializing the flow, compute velocities before inspecting physical
velocity diagnostics:

```julia
compute_velocities!(state, grid; plans, params)
KE = flow_kinetic_energy(state.u, state.v)
```

For high-level models, `run!` computes the required derived fields before the
first output.
