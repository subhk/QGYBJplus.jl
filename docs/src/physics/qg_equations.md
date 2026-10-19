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

The prognostic ``q`` is then the total PV, but the flow's dissipation
``\mathcal D_q`` acts on the balanced PV ``q-q^w`` only: ``q^w`` belongs to the
waves, which have their own damping. Damping the total PV instead would create
a balanced flow from the wave PV and could add energy. The time stepper keeps
the exact damping factor on ``q`` and adds the damping of ``q^w`` to the
explicit part, so modes whose damping is very stiff relax to ``q = q^w``.

For `YBJPlus()`, this is the wave part of the "XV``^+``" potential vorticity
of Asselin & Young (2019), equation (3.5), with ``B=L^+A``. For the original
`YBJ()` formulation the prognostic envelope is instead ``B=LA``, recovering
the original Xie–Vanneste PV. A&Y reach (3.5) from that PV by the substitution
``L\mapsto L^+```, and that substitution is what gives the YBJ``^+`` coupled
system its "coupled energy" conservation law (their equations (3.6)–(3.7)).

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

Use model-level operators so transforms, work arrays, and the architecture are
selected from the owning runtime:

```julia
compute_velocities!(model)
compute_vertical_velocity!(model)
```

Horizontal derivatives and Jacobians are pseudo-spectral and use two-thirds
dealiasing. Vertical operators use second-order differences. See
[numerical methods](@ref numerical-methods).

## Vertical velocity

The balanced flow carries an ageostrophic vertical velocity. Eliminating the
time derivatives between the vorticity and buoyancy equations,

```math
\partial_t\zeta + J(\psi,\zeta) = f_0\,\partial_z w,
\qquad
\partial_t b + J(\psi,b) + N^2 w = 0,
\qquad
b = f_0\,\partial_z\psi,
```

gives the omega equation,

```math
N^2\nabla_h^2 w + f_0^2\,\partial_z^2 w
= 2\nabla\cdot\mathbf Q
= 2f_0\left[J(\psi_z,\zeta) - J(\psi_x,\psi_{xz}) - J(\psi_y,\psi_{yz})\right],
```

with ``w=0`` at the bottom and the rigid lid. `compute_vertical_velocity!`
solves it on the faces between levels, where the model's buoyancy
``f_0\,\delta_z\psi`` lives, with ``N^2`` sampled on the faces. The discrete
forcing is the one implied by the flux-form PV inversion, so the diagnosed
``w`` satisfies the model's own discrete vorticity and buoyancy equations
exactly.

With wave feedback the conserved PV is ``q^g+q^w``, so the vorticity equation
gains ``-D_t q^w`` and the omega equation the forcing
``f_0\,\partial_z(D_t q^w)``, where ``D_t q^w = \partial_t q^w + J(\psi,q^w)`` and
``\partial_t q^w`` follows from the inviscid wave tendency. The ``w`` of a coupled
run then closes the coupled buoyancy equation exactly. As ``\psi`` is the
Lagrangian-mean streamfunction (Xie & Vanneste 2015; Wagner & Young 2016),
``u``, ``v``, and ``w`` are Lagrangian-mean velocities; see
[Wave–mean interaction](@ref wave-mean) for the instantaneous velocity.

The vertical velocity is a diagnostic: QG advection of ``q`` is horizontal, so
``w`` does not feed back on the flow. It is computed on demand, by
`compute_velocities!(model)`, `compute_vertical_velocity!(model)`, and NetCDF
output with velocities; a time step leaves `w = 0`. Particles move
horizontally and do not use it.

The face values are averaged onto the cell centres for storage, so `u`, `v`,
and `w` share one grid in output files. The
wave-induced vertical velocity of `compute_ybj_vertical_velocity!` is stored
the same way (see [YBJ⁺ wave model](@ref ybj-plus)).

## References

- Xie, J.-H. & Vanneste, J. (2015),
  [“A generalised-Lagrangian-mean model of the interactions between
  near-inertial waves and mean flow”](https://arxiv.org/abs/1411.3748),
  *Journal of Fluid Mechanics*, 774, 143–169.
- Vallis, G. K. (2017), *Atmospheric and Oceanic Fluid Dynamics*.
