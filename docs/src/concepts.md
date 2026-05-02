# [Key Concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

This page defines the main model variables and conventions.

## What Is Coupled

QGYBJplus.jl couples:

- **Balanced flow**, represented by quasi-geostrophic streamfunction `ψ`.
- **Near-inertial waves**, represented by the complex YBJ+ variable `B = L⁺A`.

The balanced flow advects and refracts the waves. When wave feedback is enabled,
the wave PV correction `qʷ` enters the streamfunction inversion through
`q* = q - qʷ`; the prognostic balanced-flow PV `q` itself is restored after the
inversion.

## Mean-Flow Variables

The streamfunction `ψ` defines geostrophic velocity:

```math
u = -\partial_y \psi, \qquad v = \partial_x \psi.
```

Relative vorticity is

```math
\zeta = \nabla_h^2 \psi.
```

The evolved mean-flow variable is the quasi-geostrophic potential vorticity `q`.
The streamfunction `ψ` is diagnosed from `q` by elliptic inversion.

## Wave Variables

The model evolves `B = L⁺A`. For each horizontal wavenumber,

```math
L^+A =
\partial_z\left(\frac{f_0^2}{N^2}\partial_z A\right)
- \frac{k_h^2}{4} A.
```

`A` is diagnosed by elliptic inversion. The wave velocity envelope is `LA`,
where `L = ∂z((f₀²/N²) ∂z)`. In spectral space,

```math
LA = L^+A + \frac{k_h^2}{4} A.
```

The NetCDF output writes the real and imaginary parts of `LA` as `LAr` and
`LAi`.

## Coordinates

- `x`, `y`: periodic horizontal coordinates in meters.
- `z`: vertical coordinate in meters, with `z = 0` at the surface and negative
  values below.
- The vertical grid is cell centered, so native saved `z` values lie between
  the surface and bottom boundaries.

## Time Stepping

The current solver uses second-order exponential RK time stepping. There is no
user-facing timestepper switch.

## Common Symbols

| Symbol | Meaning |
|:-------|:--------|
| `ψ` | quasi-geostrophic streamfunction |
| `q` | potential vorticity |
| `L⁺A` | evolved YBJ+ wave variable |
| `A` | diagnosed wave amplitude |
| `LA` | wave velocity envelope |
| `ζ` | relative vorticity |
| `f₀` | Coriolis parameter |
| `N²` | buoyancy frequency squared |

## Next

- [Quick Start](@ref quickstart)
- [Physics Overview](@ref physics-overview)
