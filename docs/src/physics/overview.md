# [Physics overview](@id physics-overview)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl couples quasigeostrophic balanced flow to a phase-averaged
near-inertial-wave envelope.

## State variables

| Field | Space | Meaning |
|:--|:--|:--|
| `q` | spectral | prognostic generalized potential vorticity |
| `B` | spectral | prognostic complex wave envelope |
| `psi` | spectral | streamfunction diagnosed from balanced PV |
| `A`, `C` | spectral | diagnosed wave amplitude and vertical derivative |
| `u`, `v`, `w` | physical | diagnosed velocity components |

With two-way feedback, `q` contains balanced and wave contributions:

```math
q = q^g + q^w,
\qquad
q^g = \nabla_h^2\psi
    + \partial_z\!\left(\frac{f_0^2}{N^2}\partial_z\psi\right).
```

The total PV evolves as

```math
\partial_t q + J(\psi,q) = \mathcal D_q.
```

The wave equation is

```math
\partial_t B + J(\psi,B)
= -\frac{i}{2}\zeta B
  -\frac{i f_0}{2}\nabla_h^2 A
  + \mathcal D_B,
\qquad \zeta=\nabla_h^2\psi.
```

See [QG equations](@ref qg-equations), [YBJ⁺ wave model](@ref ybj-plus), and
[wave–mean interaction](@ref wave-mean) for definitions and coupling details.

## Configurable dynamics

- `FixedFlow()` or `EvolvingFlow()` controls balanced-flow evolution.
- `YBJPlus()`, `YBJ()`, or `PassiveWave()` selects wave dynamics.
- `WaveMeanFeedback()` includes wave PV in balanced inversion;
  `NoWaveFeedback()` omits it.
- `NonlinearDynamics()` or `LinearDynamics()` controls nonlinear
  advection.
- `Dispersive()` or `NoDispersion()` controls wave dispersion.
- `Dissipative()` or `Inviscid()` controls configured closures.

## Domain and boundary conditions

Horizontal directions are periodic. Vertical nodes are cell centered between
a rigid bottom and surface, with `z=0` at the surface. The balanced
streamfunction uses homogeneous Neumann conditions at top and bottom; vertical
velocity vanishes at those boundaries.
