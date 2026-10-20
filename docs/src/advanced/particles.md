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
horizontal velocity and keep their depth. `z` is a coordinate, negative below
a lid at z = 0: every particle must lie within the grid's vertical extent,
and a particle outside it is an error. A non-periodic direction checks the
horizontal extent too.

Positions and velocities have the configuration's precision, `Float64` by
default for the `particles_in_*` helpers (pass `Float32`, or
`precision=Float32`, to halve their memory). Keep `Float64` for slow
trajectories in large domains: a `Float32` position 1000 km from the origin
is rounded to about 6 cm, so a particle that moves a few cm per step drifts
by the rounding. Times, including `save_interval` and `particle_advec_time`,
are always `Float64`, like the simulation clock, so that the times of long
runs stay exact.

Particles installed through a simulation, `initialize_particles!(simulation,
config)`, start at its clock's time. Between runs the particles keep their
own time; each run puts them on its time axis. Particles that have not moved
since they were initialized just take the run's start time, as after
`restore!`. Particles that have moved continue from a new simulation's
clock, which starts at zero: once released they keep moving, their history
records their positions again at the new start time, and the inertial phase
of `:instantaneous` trajectories restarts there (the model warns). To keep
the original time axis, restore the last state file into the new
simulation.

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

The tracker's own `advect_particles!(tracker, fields, geometry, Δt, t)` uses
the coefficients of the model that owns it (f, N², formulation), unless they
are given as keywords.

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
    output_mode=:trajectory,
)
simulation = Simulation(model; particle_output=particle_output, ...)
```

Give a schedule: `save_interval_iter` iterations, or `save_interval_time` in
the model's time units (seconds for a dimensional model). There is no
default, and the iteration interval wins when both are given. The output
modes are:

- `:trajectory`: one file, written when the run ends;
- `:streaming`: one file, extended at every save, so a run that is stopped
  early keeps the trajectories saved so far, without holding them in memory;
- `:snapshots`: one file per save, named by the iteration padded to 8 digits,
  so that the names sort in time order.

Each record holds the particle positions and their horizontal velocities,
`u` and `v`, at the recorded time. A run never silently replaces existing
particle files of the same mode and prefix: pass `overwrite=true` to the
manager to replace them. The check comes before the run starts and removes
nothing, so a run that cannot start leaves the files as they were.

A manager serves one run: the simulation closes it when the run ends, and a
new simulation needs a new manager. Particle positions are not part of the
model's restart files, so particles start new trajectories after a restart,
and their files are not continued: write them to another directory or
prefix, or pass `overwrite=true` to replace the earlier ones.

Without a manager, a tracker can keep its own in-memory history of the
positions, written with [`write_particle_trajectories`](@ref), or one file
per initial level with [`write_particle_trajectories_by_zlevel`](@ref). It is
off by default; set `save_interval` in the particle configuration (0 for
every step) and, if needed, `max_save_points` (1000 by default; the tracker
warns when it is reached). The history records the positions when a run
starts and then at the multiples of `save_interval`.
