# [YBJ+ wave model](@id ybj-plus)

```@meta
CurrentModule = QGYBJplus
```

The Young–Ben Jelloul framework evolves a complex near-inertial wave envelope
on the slow balanced-flow timescale. QGYBJ+.jl supports the original relation
and the horizontally regularized YBJ+ relation of Asselin & Young (2019).

## Prognostic and diagnostic wave variables

The model advances `B`. The wave amplitude `A` is diagnosed from

```math
B = \partial_z\!\left(\frac{f_0^2}{N^2}\partial_z A\right)
    - \frac{k_h^2}{4}A.
```

The horizontal Helmholtz contribution regularizes short horizontal scales.
At each `(kₓ,kᵧ)`, recovering `A` is a tridiagonal vertical solve.

```julia
invert_B_to_A!(model)
A = model.fields.A
```

## Formulation choice

```julia
plus_model = QGYBJModel(grid=grid, formulation=YBJPlus())
normal_model = QGYBJModel(grid=grid, formulation=YBJ())
passive_model = QGYBJModel(grid=grid, formulation=PassiveWave())
```

`PassiveWave()` retains wave advection while disabling refraction and
dispersion. [`NoDispersion`](@ref) disables dispersion for the selected wave
formulation without changing the ownership model.

## Processes

The wave tendency combines:

- advection by the balanced horizontal velocity;
- refraction by balanced vorticity;
- YBJ/YBJ+ dispersion through the diagnosed amplitude;
- configured horizontal hyperdiffusion.

ETD-RK2 evaluates all explicit processes at both stages and integrates the
horizontal dissipative factor exactly.

## Initialization and energy

```julia
set!(model; waves=SurfaceWave(amplitude=0.1,
                              scale=30.0,
                              profile=:gaussian))
invert_B_to_A!(model)
B_energy, A_energy = wave_energy(model)
```

## References

- Young & Ben Jelloul (1997), *Journal of Marine Research*, 55, 735–766.
- Asselin & Young (2019), *Journal of Physical Oceanography*, 49, 1699–1717.
