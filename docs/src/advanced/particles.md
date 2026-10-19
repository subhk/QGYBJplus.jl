# [Particle advection](@id particles)

```@meta
CurrentModule = QGYBJplus
```

Particle positions and tracker metadata are optional model-owned data.

## Install particles on a model

```julia
configuration = ParticleConfig{Float64}(
    x_min=first(model.grid.x_faces),
    x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces),
    y_max=last(model.grid.y_faces),
    z_level=-500.0,
    nx_particles=32,
    ny_particles=32,
    interpolation_method=TRICUBIC,
)
initialize_particles!(model, configuration)
```

The tracker is available as `model.particles`. Its geometry, FFT plans, and
velocity source all refer back to the owning model. Particles move with the
horizontal velocity and keep their depth.

Particle positions, the velocity that advects them, and the advection
itself live on the model's architecture: on a GPU, a particle step copies the
model's horizontal velocity within the device and advances every particle in
one kernel, with no transfer to the host. Copy the positions to use them on
the host, `Array(model.particles.particles.x)`; particle output and the
in-memory history do so when they record.

The `wave_motion` option of the particle configuration selects how particles
feel the near-inertial waves:

- `:averaged` (default): phase-averaged trajectories, advected by the
  Lagrangian-mean velocity ``\nabla^\perp\psi``. Because ``\psi`` is the
  Lagrangian-mean streamfunction, this already includes the waves' Stokes
  drift.
- `:instantaneous`: trajectories that trace the inertial oscillation, advected
  by the wave velocity ``e^{-if_0t}LA``, with the phase evaluated at every
  integration stage, plus the Lagrangian mean less the Stokes drift that the
  horizontal orbits make up themselves. Their drift averaged over an inertial
  period is therefore again the Lagrangian mean. The part of the Stokes drift
  that comes from vertical excursions in the vertical shear, which particles
  at a fixed depth do not make, stays in their mean velocity. The particle
  time step must resolve the inertial period, ``f_0\,\Delta t \lesssim 0.2``.

```julia
config = particles_in_box(-500.0; x_max=Lx, y_max=Ly, wave_motion=:instantaneous)
```

Both velocity components are evaluated at the cell centres and interpolated
to the particle positions with `interpolation_method`: `TRILINEAR`,
`TRICUBIC` (default), or `QUINTIC`. See
[Particle interpolation](@ref interpolation). `use_3d_advection` was removed:
setting it to `true` raises an error.

## Distributions over depth

Particles released at several depths each move at their own depth.

```julia
layers = particles_in_layers(
    [-100.0, -500.0, -1000.0];
    x_min=first(model.grid.x_faces),
    x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces),
    y_max=last(model.grid.y_faces),
    nx=16,
    ny=16,
)
initialize_particles!(model, layers)
```

Other helpers create uniform volumes, circles, random clouds, and explicit
custom positions.

## Integration

Particles installed on a model are advanced automatically by
`run!(simulation)`, after each model step. A direct update moves them from
`current_time` to `current_time + Δt` and takes the model's current state as
the velocity at the end of that interval, so advance the model over the same
interval first:

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
for n in 0:99
    step!(model, timestepper)                          # fields: t → t + Δt
    advect_particles!(model, 10.0; current_time=10.0n) # particles: t → t + Δt
end
```

`integration_method` selects `:euler`, `:rk2` (default), or `:rk4`. The
model's velocity is known only at the ends of a step, so the tracker keeps the
velocity from the end of the previous step, and every integration stage
interpolates linearly in time between the start and the end of the step. In
an evolving flow `:rk2` and `:rk4` are then second-order accurate in time,
like the model's ETD-RK2 flow, and `:euler` is first-order. `:rk4` reduces
only the error from trajectories that curve within a step, at twice the cost
of `:rk2`. The first manual step has no earlier velocity and uses the
end-of-step velocity throughout; `run!` records the initial velocity before
its first step.

## Boundaries

Horizontal boundaries are periodic (`periodic_x` and `periodic_y`, both on by
default; a non-periodic direction clamps particles into the domain).
Particles keep their depth. Vertically, the interpolation mirrors the
horizontal velocity at the lid and the bottom, where the rigid-lid boundary
conditions give it zero vertical derivative, so particles near either
boundary see a smooth velocity.

## Trajectories

Attach a manager to the simulation for automatic initial, scheduled, and final
particle output:

```julia
particle_output = ParticleOutputManager(
    "output";
    save_interval_iter=20,
    save_interval_time=0.0,
    output_mode=:trajectory,
)
simulation = Simulation(model; particle_output=particle_output, ...)
```

Set the unused interval to zero when selecting iteration- or time-based
scheduling. The output modes are:

- `:trajectory`: one file, written when the run ends;
- `:streaming`: one file, extended at every save, so a run that is stopped
  early keeps the trajectories saved so far, without holding them in memory;
- `:snapshots`: one file per save.

Each record holds the particle positions and their horizontal velocities,
`u` and `v`, at the recorded time. A run never silently replaces existing particle files of the
same mode and prefix: pass `overwrite=true` to the manager to replace them.
Particle positions are not part of the model's restart files, so particles
start new trajectories after a restart.

Without a manager, a tracker can keep its own in-memory history of the
positions, written with [`write_particle_trajectories`](@ref). It is off by
default; set `save_interval` in the particle configuration (0 for every
step) and, if needed, `max_save_points` (1000 by default; the tracker warns
when it is reached).
