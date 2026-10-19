# [YBJ⁺ wave model](@id ybj-plus)

```@meta
CurrentModule = QGYBJplus
```

The Young–Ben Jelloul framework evolves a complex near-inertial-wave envelope
on the slow balanced-flow timescale.

## Wave equation

QGYBJ+.jl advances `B` according to

```math
\partial_t B + J(\psi,B)
= -\frac{i}{2}\zeta B
  -\frac{i f_0}{2}\nabla_h^2 A
  + \mathcal D_B,
```

where ``\zeta=\nabla_h^2\psi``. The terms represent advection,
refraction, dispersion, and configured dissipation.

## YBJ⁺ relation

For `YBJPlus()`,

```math
B=L^+A
=\partial_z\!\left(\frac{f_0^2}{N^2}\partial_z A\right)
 +\frac14\nabla_h^2A.
```

At horizontal wavenumber ``k_h`` this is

```math
\widehat B
=\partial_z\!\left(\frac{f_0^2}{N^2}\partial_z\widehat A\right)
 -\frac{k_h^2}{4}\widehat A.
```

The Helmholtz term regularizes short horizontal scales. Recovering `A`
requires one tridiagonal vertical solve per horizontal Fourier mode.

## Original YBJ relation

For `YBJ()`, the Helmholtz term is omitted:

```math
B=\partial_z\!\left(\frac{f_0^2}{N^2}\partial_z A\right).
```

The implementation recovers `A` by vertical integration with coefficient
``N^2/f_0^2`` and enforces the tendency solvability condition.

## Depth-mean part of the envelope

With ``A_z=0`` at the bottom and the lid, ``LA`` has zero depth mean, so the
wave velocity ``e^{-if_0t}LA`` has no barotropic part. In the horizontally
uniform column (``k_h=0``) both relations reduce to ``B=LA``, so that column
of `B` has zero depth mean; for `YBJ()` every column does. The model removes
any other depth mean from `B`: from an initial condition (including the
depth mean of a `SurfaceWave` profile, which is small for a surface layer
much thinner than the ocean) and at every time step, where refraction would
otherwise create it. A depth-uniform envelope is removed entirely, with a
warning.

```julia
invert_B_to_A!(model)
A = model.fields.A
```

## Wave vertical velocity

The wave velocity is ``u+iv=e^{-if_0t}LA``, and continuity gives its vertical
velocity (Asselin & Young 2019, equation 2.10),

```math
w = -\frac{f_0^2}{N^2}\,\partial_s A_z\,e^{-if_0t} + \text{c.c.},
\qquad
\partial_s = \tfrac12(\partial_x - i\partial_y).
```

`compute_ybj_vertical_velocity!(model; t)` evaluates it where ``A_z`` lives,
on the faces between levels, with ``N^2`` on the same faces. The face values
then satisfy discrete continuity with the model's ``LA`` exactly, and vanish
at the bottom and the lid, where ``A_z=0``. As for the QG vertical velocity,
the face values are averaged onto cell centres for storage.
`compute_total_velocities!(model; t)` adds it, at the same phase, to the
instantaneous velocity.

```julia
compute_ybj_vertical_velocity!(model; t=simulation.clock.time)
w = model.fields.w
```

## Select a formulation

```julia
plus_model = QGYBJModel(grid=grid, formulation=YBJPlus())
ybj_model = QGYBJModel(grid=grid, formulation=YBJ())
passive_model = QGYBJModel(grid=grid, formulation=PassiveWave())
```

`PassiveWave()` retains envelope advection but omits refraction and
dispersion. `NoDispersion()` disables dispersion while retaining the other
processes of the selected YBJ formulation.

## Initialize and diagnose waves

```julia
set!(model; waves=SurfaceWave(amplitude=0.1, scale=30.0))
B_energy, A_energy = wave_energy(model)
```

ETD-RK2 evaluates the explicit wave tendency at both stages and integrates
horizontal hyperdiffusion with its exponential factor.

## References

- Young, W. R. & Ben Jelloul, M. (1997), “Propagation of near-inertial
  oscillations through a geostrophic flow,” *Journal of Marine Research*, 55,
  735–766.
- Asselin, O. & Young, W. R. (2019),
  [“An improved model of near-inertial wave
  dynamics”](https://doi.org/10.1017/jfm.2019.557), *Journal of Fluid
  Mechanics*, 876, 428–448.
