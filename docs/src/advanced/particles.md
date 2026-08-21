# [Particle advection](@id particles)

```@meta
CurrentModule = QGYBJplus
```

Particle positions and tracker metadata are optional model-owned data.

## Install particles on a model

```julia
configuration = ParticleConfig{Float64}(
    x_max=model.grid.extent[1],
    y_max=model.grid.extent[2],
    z_level=-500.0,
    nx_particles=32,
    ny_particles=32,
    use_3d_advection=true,
    interpolation_method=TRICUBIC,
)
initialize_particles!(model, configuration)
```

The tracker is available as `model.particles`. Its geometry, FFT plans, MPI
communicator, and velocity source all refer back to the owning model.

## Three-dimensional distributions

```julia
layers = particles_in_layers(
    [-100.0, -500.0, -1000.0];
    x_max=model.grid.extent[1],
    y_max=model.grid.extent[2],
    nx=16,
    ny=16,
)
initialize_particles!(model, layers)
```

Other helpers create uniform volumes, circles, random clouds, and explicit
custom positions.

## Integration

Particles installed on a model are advanced automatically by
`run!(simulation)`. A direct update is also available:

```julia
compute_total_velocities!(model)
advect_particles!(model, 10.0)
```

Available interpolation schemes are `TRILINEAR`, `TRICUBIC`, `ADAPTIVE`, and
`QUINTIC`. The tracker configuration selects Euler, RK2, or RK4 trajectory
integration.

## Boundaries

Horizontal boundaries are periodic. The vertical boundary behavior follows
the particle configuration and the model's negative-depth coordinate system.
Under MPI, particles migrate to the rank that owns their updated horizontal
position after boundary normalization.

## Trajectories

Attach a manager to the simulation for automatic initial, scheduled, and final
particle output:

```julia
particle_output = ParticleOutputManager(
    "output";
    save_interval_iter=20,
    output_mode=:trajectory,
)
simulation = Simulation(model; particle_output=particle_output, ...)
```

The simulation initializes and finalizes the manager on every MPI rank, while
only the I/O rank writes the gathered trajectory. Snapshot and streaming modes
are also available. Root-side setup and write failures are propagated to every
rank and move the simulation to `Failed`. For one-off output, use
`write_particle_snapshot` or `write_particle_trajectories` directly.
