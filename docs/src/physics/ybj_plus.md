# [YBJ+ Wave Model](@id ybj-plus)

```@meta
CurrentModule = QGYBJplus
```

This page describes the Young-Ben Jelloul Plus (YBJ+) formulation for near-inertial wave evolution.

## Near-Inertial Waves

### Physical Background

Near-inertial waves (NIWs) are internal gravity waves with frequencies close to the local Coriolis frequency ``f``. They are:

- **Wind-generated**: Strong winds (storms, tropical cyclones) inject NIW energy
- **Ubiquitous**: Found throughout the world's oceans
- **Important for mixing**: NIWs break and drive turbulent mixing

### Wave Amplitude Representation

The NIW velocity field is written as:

```math
\mathbf{u}_{wave} = \text{Re}\left[ A(x,y,z,t) \, e^{-if_0 t} \, \hat{\mathbf{z}} \times \nabla_h \right] + \text{c.c.}
```

where ``A`` is the **slowly-varying complex wave amplitude**.

## The YBJ+ Equation

### Evolution Equation

The wave envelope ``L^+A`` evolves according to:

```math
\frac{\partial (L^+A)}{\partial t} + J(\psi, L^+A) = i\frac{k_h^2}{2 \cdot Bu \cdot Ro} A + \frac{1}{2}(L^+A) \times \zeta + \mathcal{D}_{L^+A}
```

!!! note "Sign convention"
    The sign in the dispersion term depends on the phase convention for the carrier wave (e.g., ``e^{-i f_0 t}`` vs ``e^{+i f_0 t}``). QGYBJ+.jl follows the ``e^{-i f_0 t}`` convention, yielding the ``+i`` sign shown here.

where:
- ``L^+A``: Evolved wave envelope (prognostic variable in code: `state.L⁺A`)
- ``\zeta = \nabla^2\psi``: Relative vorticity
- ``J(\psi, L^+A)``: Advection by geostrophic flow
- ``(1/2)(L^+A) \times \zeta``: Refraction term (wave focusing by vorticity)
- ``i k_h^2/(2 \cdot Bu \cdot Ro) A``: Dispersion (nondimensional form of ``i N^2 k_h^2/(2f_0) A``)

### Real/Imaginary Decomposition

Writing ``L^+A = (L^+A)_R + i (L^+A)_I`` and ``A = A_R + i A_I``, the equations become:

```math
\frac{\partial (L^+A)_R}{\partial t} = -J(\psi, (L^+A)_R) - \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_I + \frac{1}{2} (L^+A)_I \times \zeta - \mathcal{D}_{(L^+A)_R}
```

```math
\frac{\partial (L^+A)_I}{\partial t} = -J(\psi, (L^+A)_I) + \frac{k_h^2}{2 \cdot Bu \cdot Ro} A_R - \frac{1}{2} (L^+A)_R \times \zeta - \mathcal{D}_{(L^+A)_I}
```

### Physical Terms

| Term | Physics | Effect |
|:-----|:--------|:-------|
| ``J(\psi, L^+A)`` | Advection | Waves carried by eddies |
| ``(1/2)(L^+A) \times \zeta`` | Refraction | Focusing in anticyclones |
| ``k_h^2 A/(2 \cdot Bu \cdot Ro)`` | Dispersion | Horizontal spreading |
| ``\mathcal{D}_{L^+A}`` | Dissipation | Energy loss (hyperdiffusion) |

## The L⁺ Operator

### Definition

The YBJ+ operator ``L^+`` relates the evolved envelope ``L^+A`` and the wave amplitude ``A``:

```math
L^+ A = \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right) - \frac{k_h^2}{4}A
```

Note: The code uses the Unicode variable name `L⁺A` (Julia supports Unicode identifiers) for clarity.

### Inversion: L⁺A → A

To recover ``A`` from ``L^+A``, we solve:

```math
\frac{\partial}{\partial z}\left(a(z)\frac{\partial A}{\partial z}\right) - \frac{k_h^2}{4}A = L^+A
```

where ``a(z) = f_0^2/N^2(z)``.

### Tridiagonal System

In discretized form for each ``(k_x, k_y)``:

```math
a_k A_{k-1} + b_k A_k + c_k A_{k+1} = (L^+A)_k
```

with:
- ``a_k = a(z_{k-1/2})/\Delta z^2``
- ``c_k = a(z_{k+1/2})/\Delta z^2``
- ``b_k = -(a_k + c_k) - k_h^2/4``

### Boundary Conditions

- **Neumann**: ``\frac{\partial A}{\partial z} = 0`` at ``z = -H, 0``

## Wave Refraction

### Mechanism

Anticyclones (negative vorticity) **trap** waves:
- Effective frequency: ``f_{eff} = f_0 + \zeta/2``
- In anticyclones: ``\zeta < 0 \Rightarrow f_{eff} < f_0``
- Waves propagate toward regions of lower effective frequency

### Mathematical Form

The refraction term in the YBJ+ model:

```math
\text{Refraction} = \frac{1}{2} (L^+A) \times \zeta = \frac{1}{2} (L^+A) \times \nabla^2\psi
```

In terms of real/imaginary parts:
- ``r_{(L^+A)_R} = \frac{1}{2} (L^+A)_I \times \zeta`` contributes to ``\partial (L^+A)_R/\partial t``
- ``r_{(L^+A)_I} = -\frac{1}{2} (L^+A)_R \times \zeta`` contributes to ``\partial (L^+A)_I/\partial t``

This term represents focusing of wave energy by the background vorticity field.

### Code Implementation

```julia
# Compute refraction term: (1/2) * L⁺A × ζ where ζ = -kh²ψ (complex L⁺A form)
refraction_waqg_L⁺A!(rL⁺Ak, L⁺Ak, psik, grid, plans; Lmask=L)
```

## YBJ vs YBJ+

### Original YBJ (1997)

The original YBJ uses the ``L`` operator:

```math
LA = \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right)
```

- Simpler relation between ``LA`` and ``A``
- Recovery via vertical integration

### YBJ+ (Asselin & Young 2019)

The YBJ+ uses the ``L^+`` operator:

```math
L^+A = \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right) - \frac{k_h^2}{4}A = LA - \frac{k_h^2}{4}A
```

- Includes horizontal wavenumber dependence
- More accurate for high ``k_h`` modes
- Requires elliptic inversion (not just integration)

### When to Use Which

| Scenario | Recommendation |
|:---------|:---------------|
| Low-resolution | Normal YBJ is adequate |
| High-resolution | YBJ+ more accurate |
| Large-scale waves | Either works |
| Small-scale waves | YBJ+ essential |

Control in code:
```julia
params = QGParams(; ybj_plus=true)  # Use YBJ+ (default)
params = QGParams(; ybj_plus=false) # Use normal YBJ
```

## Dispersion Relation

### In the YBJ+ Framework

The dispersion relation for NIWs:

```math
\omega = f_0 + \frac{N^2 k_h^2}{2f_0 m^2}
```

where ``m`` is the vertical wavenumber.

### Physical Implications

- Frequency slightly above ``f_0``
- Higher ``k_h`` → faster frequency
- Lower ``m`` (longer vertical scale) → faster frequency

## Wave Energy

### Wave Kinetic Energy (Equation 4.7)

The wave kinetic energy (WKE) is defined per YBJ+ equation (4.7):

```math
\text{WKE} = \frac{1}{2} \int |LA|^2 \, dV
```

where ``L`` is the **original YBJ vertical operator** from equation (1.3):

```math
L = \partial_z \left( \frac{f_0^2}{N^2} \partial_z \right)
```

Note that ``LA`` (using ``L``) is different from ``L^+A`` (using ``L^+``). The relationship is:

```math
LA = L^+A + \frac{k_h^2}{4}A
```

In spectral space, this allows computing ``LA`` from the prognostic ``L^+A`` and diagnostic ``A``.

### Computation

```julia
# Detailed wave energy components (WKE, WPE, WCE)
WKE, WPE, WCE = compute_detailed_wave_energy(state, grid, params)

# Vertically-averaged wave energy using LA = L⁺A + (k_h²/4)A
WE = wave_energy_vavg(state.L⁺A, state.A, grid, plans)
```

!!! note "Physical interpretation"
    WKE represents the kinetic energy of the near-inertial wave field, computed using
    the original YBJ ``L`` operator (not ``L^+``). The code computes ``LA = L^+A + (k_h^2/4)A``
    in spectral space to get the wave velocity amplitude ``LA``.

## Implementation Details

### Key Functions

The YBJ+ implementation uses these core functions:
- `invert_L⁺A_to_A!` - Solve L⁺ operator for wave amplitude A from envelope L⁺A
- `refraction_waqg_L⁺A!` - Compute wave refraction by vorticity (complex L⁺A)
- `convol_waqg_L⁺A!` - Compute wave advection by geostrophic flow (complex L⁺A)

See the [Physics API Reference](../api/physics.md) for detailed documentation.

### Code Locations

- `elliptic.jl`: L⁺A → A inversion (`invert_L⁺A_to_A!`)
- `nonlinear.jl`: Refraction (`refraction_waqg_L⁺A!`), Advection (`convol_waqg_L⁺A!`)
- `ybj_normal.jl`: Normal YBJ (non-plus) operators (`sumL⁺A!`, `compute_A!`)

## References

- Young, W. R., & Ben Jelloul, M. (1997). Propagation of near-inertial oscillations through a geostrophic flow. *J. Mar. Res.*, 55, 735-766.
- Asselin, O., & Young, W. R. (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Phys. Oceanogr.*, 49, 1699-1717.
