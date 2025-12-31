# [Model Overview](@id physics-overview)

```@meta
CurrentModule = QGYBJplus
```

This page provides an overview of the physics implemented in QGYBJ+.jl. For detailed equations, see the individual sections on [QG dynamics](@ref qg-equations) and [YBJ+ waves](@ref ybj-plus).

## Physical Setting

### The Ocean's Two-Scale Problem

The ocean interior contains two dominant forms of variability:

1. **Mesoscale eddies** (balanced flow)
   - Horizontal scale: ~50-200 km
   - Vertical scale: ~1000 m (full depth)
   - Time scale: weeks to months
   - Energy: ~90% of oceanic kinetic energy

2. **Near-inertial waves** (NIWs)
   - Horizontal scale: ~10-50 km
   - Vertical scale: ~100-500 m
   - Time scale: hours (~ inertial period)
   - Energy: significant portion of internal wave energy

These two phenomena **interact**:
- Eddies **refract** NIWs, focusing energy in anticyclones
- NIWs can **feed energy back** to the balanced flow
- Combined dynamics drive **mixing** and **energy dissipation**

### Why QG-YBJ+?

The QG-YBJ+ model efficiently captures this interaction:

| Aspect | QG (Eddies) | YBJ+ (Waves) |
|:-------|:------------|:-------------|
| Variables | Streamfunction ψ | Wave envelope B |
| Dynamics | Advection of PV | Advection + refraction |
| Inversion | q → ψ (elliptic) | B → A (elliptic) |
| Coupling | Wave feedback qʷ | Refraction by ζ |

## Governing Equations

### Quasi-Geostrophic Potential Vorticity

The balanced flow evolves according to:

```math
\frac{\partial q}{\partial t} + J(\psi, q) + J(\psi, q^w) = \text{dissipation}
```

where:
- ``q = \nabla^2\psi + \frac{f_0^2}{N^2}\frac{\partial^2\psi}{\partial z^2}`` is potential vorticity
- ``\psi`` is the streamfunction (u = -∂ψ/∂y, v = ∂ψ/∂x)
- ``q^w`` is the wave feedback term
- ``J(a,b) = \frac{\partial a}{\partial x}\frac{\partial b}{\partial y} - \frac{\partial a}{\partial y}\frac{\partial b}{\partial x}`` is the Jacobian

### YBJ+ Wave Envelope

The near-inertial wave envelope evolves according to:

```math
\frac{\partial B}{\partial t} + J(\psi, B) = -i\frac{k_h^2}{2 \cdot Bu \cdot Ro} A + \frac{1}{2}B \times \zeta + \text{dissipation}
```

where:
- ``B = L^+ A`` is the evolved wave envelope
- ``A`` is the actual wave amplitude (recovered via elliptic inversion)
- ``\zeta = \nabla^2\psi`` is the relative vorticity
- ``L^+`` is an elliptic operator relating B and A
- ``Bu = (N_0 H / f_0 L)^2`` is the Burger number
- ``Ro = U / (f_0 L)`` is the Rossby number

The dispersion coefficient ``1/(2 \cdot Bu \cdot Ro)`` represents nondimensionalized ``N^2/(2f_0)``.

### Key Physical Processes

```
                    ┌─────────────────┐
                    │   Wind Forcing  │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Mean Flow   │◄───│  Near-Inertial  │───►│   Mixing &   │
│   (ψ, q)     │    │   Waves (B, A)  │    │  Dissipation │
└──────┬───────┘    └────────┬────────┘    └──────────────┘
       │                     │
       │    ┌────────────────┤
       │    │                │
       ▼    ▼                ▼
    Advection           Refraction
    J(ψ, q)             (1/2) B × ζ
       │                     │
       │    Wave Feedback    │
       └────────qʷ───────────┘
```

## Nondimensionalization

The model uses nondimensional variables scaled by:

| Quantity | Scale | Physical Value |
|:---------|:------|:---------------|
| Length | ``L`` | Rossby radius (~50 km) |
| Depth | ``H`` | Domain depth (~2000 m) |
| Time | ``1/f_0`` | Inertial period (~17 h at 45°N) |
| Velocity | ``U`` | Eddy velocity (~0.1 m/s) |
| Stratification | ``N_0`` | Reference buoyancy frequency |

Key nondimensional parameters:
- **Rossby number**: ``Ro = U/(f_0 L)`` ≈ 0.1
- **Burger number**: ``Bu = (N_0 H / f_0 L)^2`` ≈ 1

## Prognostic vs Diagnostic Variables

### Prognostic (Time-Stepped)
- ``q``: Potential vorticity (spectral)
- ``B``: Wave envelope (spectral)

### Diagnostic (Computed)
- ``\psi``: Streamfunction (from q via elliptic inversion)
- ``A``: Wave amplitude (from B via YBJ+ inversion)
- ``u, v``: Horizontal velocities (from ψ)
- ``w``: Vertical velocity (from omega equation or YBJ)

## Boundary Conditions

### Horizontal
- **Doubly periodic**: ``f(x+L_x, y) = f(x, y+L_y) = f(x, y)``

### Vertical
- **Rigid lid**: ``w = 0`` at ``z = 0`` and ``z = H``
- **No flux**: ``\frac{\partial\psi}{\partial z} = 0`` at boundaries (Neumann)

## Energy Budget

### Flow Energy
```math
E_{\text{flow}} = \frac{1}{2}\int \left( u^2 + v^2 + \frac{f_0^2}{N^2}\left(\frac{\partial\psi}{\partial z}\right)^2 \right) dV
```

### Wave Energy
```math
E_{\text{wave}} = \frac{1}{2}\int |A|^2 \, dV
```

### Energy Exchange
The wave feedback term ``q^w`` mediates energy transfer between waves and flow.

## Code Structure

The physics is implemented across several modules:

| Module | Physics |
|:-------|:--------|
| `elliptic.jl` | q → ψ and B → A inversions |
| `nonlinear.jl` | Jacobians, refraction, wave feedback |
| `operators.jl` | Velocity computation |
| `timestep.jl` | Time integration (Euler/Leapfrog) |
| `timestep_imex.jl` | IMEX-CNAB time integration for waves |
| `diagnostics.jl` | Energy and omega equation |

## Further Reading

- [QG Equations](@ref qg-equations): Detailed QG dynamics
- [YBJ+ Wave Model](@ref ybj-plus): Wave envelope formulation
- [Wave-Mean Interaction](@ref wave-mean): Energy exchange mechanism
- [Numerical Methods](@ref numerical-methods): Discretization and algorithms
