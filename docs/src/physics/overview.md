# [Model Overview](@id physics-overview)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl simulates the interaction between mesoscale eddies and near-inertial waves.

## The Two Components

| Component | Variable | Scale | Dynamics |
|:----------|:---------|:------|:---------|
| **Eddies** | ψ (streamfunction) | 50-200 km, weeks | Advection of PV |
| **Waves** | B (wave envelope) | 10-50 km, hours | Advection + refraction |

**Interactions**: Eddies refract waves (focusing in anticyclones); waves feed energy back to eddies.

## Governing Equations

### QG Potential Vorticity
```math
\frac{\partial q}{\partial t} + J(\psi, q) = \text{dissipation}
```
where the prognostic total PV is
``q = \nabla^2\psi + \frac{f_0^2}{N^2}\frac{\partial^2\psi}{\partial z^2} + q^w``.
The wave contribution is diagnosed from ``B`` and removed from ``q`` only for
streamfunction inversion.

### YBJ+ Wave Envelope
```math
\frac{\partial B}{\partial t} + J(\psi, B)
= -\frac{i}{2}\zeta B - \frac{if_0}{2}\nabla_h^2 A
+ \text{dissipation}
```
where ``B = L^+ A`` and A is recovered via elliptic inversion.

### Physical Processes

| Process | Term | Effect |
|:--------|:-----|:-------|
| Advection | ``J(\psi, B)`` | Waves carried by flow |
| Refraction | ``-(i/2)\zeta B`` | Waves focus in anticyclones |
| Dispersion | ``-(if_0/2)\nabla_h^2 A`` | Horizontal spreading |

!!! tip "Wave Trapping"
    Effective frequency ``f_{\text{eff}} = f_0 + \zeta/2``. In anticyclones (ζ < 0), waves slow and accumulate.

## Variables

| Type | Variables |
|:-----|:----------|
| **Prognostic** | q (potential vorticity), B (wave envelope) |
| **Diagnostic** | ψ (from q), A (from B), u, v (from ψ) |

## Boundary Conditions

- **Horizontal**: Doubly periodic
- **Vertical**: Rigid lid (w=0 at z=0, z=-Lz), no-flux (∂ψ/∂z=0)

!!! warning "Coordinate Convention"
    z = 0 at surface, z = -Lz at bottom.

## See Also

- [QG Equations](@ref qg-equations)
- [YBJ+ Wave Model](@ref ybj-plus)
- [Wave-Mean Interaction](@ref wave-mean)
- [Numerical Methods](@ref numerical-methods)
