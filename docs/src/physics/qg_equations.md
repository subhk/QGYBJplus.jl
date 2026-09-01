# [QG equations](@id qg-equations)

```@meta
CurrentModule = QGYBJplus
```

## Generalized potential vorticity

The prognostic field is the total generalized PV,

```math
q = q^g + q^w,
```

with balanced component

```math
q^g = \nabla_h^2\psi
    + \partial_z\!\left(\frac{f_0^2}{N^2(z)}\partial_z\psi\right).
```

For constant stratification this becomes

```math
q^g = \nabla_h^2\psi
    + \frac{f_0^2}{N^2}\partial_z^2\psi.
```

When the flow evolves,

```math
\partial_t q + J(\psi,q) = \mathcal D_q,
\qquad
J(a,b)=\partial_xa\,\partial_yb-\partial_ya\,\partial_xb.
```

``\mathcal D_q`` denotes the configured horizontal and vertical
dissipation. A `FixedFlow()` model holds the balanced flow fixed instead of
advancing this equation.

## Wave contribution

With `WaveMeanFeedback()`, the dimensional wave envelope contributes

```math
q^w = \frac{i}{2f_0}J(B^*,B)
    + \frac{1}{4f_0}\nabla_h^2|B|^2.
```

The factors of ``1/f_0`` are part of the dimensional equation. With
`NoFeedback()` or `NoWaveFeedback()`, the solver uses ``q^w=0``.

This is the wave part of the "XV``^+``" potential vorticity of Asselin & Young
(2019), equation (3.5), whose argument is ``B=L^+A`` rather than the
backrotated velocity ``LA``. A&Y reach (3.5) from the Xie & Vanneste PV by the
substitution ``L\mapsto L^+``, and that substitution is what gives the coupled
system its "coupled energy" conservation law (their equations (3.6)–(3.7)).
Using ``LA`` here would recover the original XV potential vorticity instead.

Note the intended contrast with wave kinetic energy, which uses ``LA``: A&Y
equation (4.7) and the remark that "to define WKE for YBJ``^+`` we use ``L``,
not ``L^+``".

## Streamfunction inversion

The streamfunction is diagnosed from balanced PV:

```math
\nabla_h^2\psi
+ \partial_z\!\left(\frac{f_0^2}{N^2}\partial_z\psi\right)
= q-q^w.
```

Wave PV is subtracted only for inversion. The prognostic total `q` is
restored before each ETD-RK2 stage continues.

After horizontal Fourier transformation, every ``(k_x,k_y)`` mode is a
tridiagonal vertical problem,

```math
-k_h^2\widehat\psi
+ \partial_z\!\left(a(z)\partial_z\widehat\psi\right)
= \widehat{q-q^w},
\qquad a(z)=\frac{f_0^2}{N^2(z)}.
```

The public inversion entry point is:

```julia
invert_q_to_psi!(model)
```

## Velocity

Balanced horizontal velocity and relative vorticity follow from

```math
u=-\partial_y\psi,
\qquad
v=\partial_x\psi,
\qquad
\zeta=\nabla_h^2\psi.
```

Use model-level operators so transforms and distributed layouts are selected
from the owning runtime:

```julia
compute_velocities!(model)
compute_vertical_velocity!(model)
```

Horizontal derivatives and Jacobians are pseudo-spectral and use radial
two-thirds dealiasing. Vertical operators use second-order differences. See
[numerical methods](@ref numerical-methods).

## References

- Xie, J.-H. & Vanneste, J. (2015),
  [“A generalised-Lagrangian-mean model of the interactions between
  near-inertial waves and mean flow”](https://arxiv.org/abs/1411.3748),
  *Journal of Fluid Mechanics*, 774, 143–169.
- Vallis, G. K. (2017), *Atmospheric and Oceanic Fluid Dynamics*.
