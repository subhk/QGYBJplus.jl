# [Running Simulations](@id running)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl scripts use an Oceananigans-style flow:

1. Define a grid.
2. Define a model.
3. Initialize fields with `set!`.
4. Create a `Simulation`.
5. Call `run!`.

## Run Clock

```julia
simulation = Simulation(model;
                        Δt = 300.0,
                        stop_time = 5inertial_period(model),
                        output = NetCDFOutput(path = "output",
                                              schedule = TimeInterval(inertial_period(model)),
                                              fields = (:ψ, :waves)),
                        diagnostics = IterationInterval(20),
                        verbose = true)

run!(simulation)
finalize_simulation!(simulation)
```

You can stop by model time or by iteration count:

```julia
Simulation(model; stop_time = 10days)
Simulation(model; stop_iteration = 1000)
```

The current solver uses second-order exponential RK time stepping. There is no
runtime `timestepper` choice.

## Updating Runtime Options

The run clock and output can also be passed directly to `run!`:

```julia
run!(simulation;
     stop_time = 3inertial_period(model),
     output = NetCDFOutput(path = "output",
                           schedule = TimeInterval(inertial_period(model)),
                           fields = (:ψ, :waves)),
     progress = true)
```

## Output Streams

A simulation can write more than one NetCDF stream:

```julia
full_output = NetCDFOutput(path = "output/full",
                           schedule = TimeInterval(inertial_period(model)),
                           fields = (:ψ, :waves))

surface_output = NetCDFOutput(path = "output/surface",
                              schedule = TimeInterval(inertial_period(model)),
                              fields = (:ψ, :waves),
                              z = 0.0)

simulation = Simulation(model;
                        output = (full_output, surface_output),
                        stop_time = 5inertial_period(model))
```

This writes the full domain and the nearest surface level during the same run.

## Low-Level Development

For solver development, use the low-level state and stepper directly:

```julia
params = default_params(Lx = 500e3, Ly = 500e3, Lz = 4000.0,
                        nx = 64, ny = 64, nz = 32,
                        dt = 300.0, nt = 100)

grid, state, plans, a_ell = setup_model(params)
mask = dealias_mask(grid)
next_state = copy_state(state)

exp_rk2_step!(next_state, state, grid, params, plans;
              a = a_ell,
              dealias_mask = mask)
```

Most user scripts should use `Simulation` and `run!`; the low-level API is
mainly for testing numerical kernels or developing new operators.

## Cleanup

Call `finalize_simulation!` at the end of scripts that use the high-level API:

```julia
finalize_simulation!(simulation)
```

This clears MPI work buffers and finalizes MPI.
