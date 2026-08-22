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

```julia
invert_B_to_A!(model)
A = model.fields.A
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
