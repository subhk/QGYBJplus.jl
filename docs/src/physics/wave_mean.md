# [Wave-Mean Interaction](@id wave-mean)

```@meta
CurrentModule = QGYBJ
```

This page describes the two-way coupling between near-inertial waves and the balanced mean flow.

## Physical Motivation

### Why Waves Affect the Mean Flow

Near-inertial waves carry momentum and energy. When they:
- **Break** or dissipate
- **Refract** through vorticity gradients
- **Interact nonlinearly** with the flow

...they can transfer energy and momentum to the balanced circulation.

### Observational Evidence

- Anticyclones contain **enhanced** NIW energy (chimney effect)
- Wave dissipation correlates with **mixing** in anticyclones
- Waves can **energize** the mesoscale eddy field

## The Wave Feedback Term

### Definition

The wave-induced potential vorticity is:

```math
q^w = \frac{1}{2}\nabla_h^2|A|^2 - \frac{1}{2}\frac{\partial^2|A|^2}{\partial z^2}
```

This enters the QG equation as:

```math
\frac{\partial q}{\partial t} + J(\psi, q) + J(\psi, q^w) = \mathcal{D}_q
```

### Physical Interpretation

| Term | Meaning |
|:-----|:--------|
| ``\nabla_h^2\|A\|^2`` | Horizontal wave intensity curvature |
| ``\partial_z^2\|A\|^2`` | Vertical wave intensity curvature |
| ``J(\psi, q^w)`` | Advection of wave-induced PV |

The wave feedback represents:
- **Radiation stress** from wave momentum flux
- **Form drag** from wave-induced pressure fluctuations

## Energy Exchange

### Wave-to-Flow Transfer

Energy flows from waves to mean flow when:
```math
\mathcal{E}_{w \to f} = -\int \psi \cdot J(\psi, q^w) \, dV
```

This can be positive or negative:
- **Positive**: Waves energize the flow
- **Negative**: Flow energizes waves (less common)

### Conservation

In the inviscid limit, total energy is conserved:
```math
\frac{d}{dt}(E_{flow} + E_{wave}) = 0
```

The wave feedback term merely **redistributes** energy.

## Refraction Mechanism

### How Eddies Focus Waves

Anticyclones (negative vorticity) trap waves:

1. **Effective frequency** is reduced: ``f_{eff} = f_0 + \zeta/2``
2. Waves propagate toward **lower** effective frequency
3. Energy **accumulates** in anticyclone cores

### The Chimney Effect

```
         Wind Forcing
              ↓
    ┌─────────────────────┐
    │   Surface Layer     │
    └─────────┬───────────┘
              │
    ┌─────────┼───────────┐
    │    ↙    ↓    ↘      │  ← Waves spread horizontally
    │   ↙     ↓     ↘     │
    └──↙──────┼──────↘────┘
       ↘      ↓      ↙
        ↘     ↓     ↙
         ↘    ↓    ↙        ← Anticyclone focuses waves
    ┌─────────┼───────────┐
    │         ↓           │
    │    Anticyclone      │  ← Enhanced dissipation
    │      (ζ < 0)        │
    └─────────────────────┘
```

Waves are funneled into anticyclones, enhancing deep mixing.

## Implementation

### Computing Wave Feedback

```julia
# In nonlinear.jl
function wavefb!(rqk, A, grid, params, plans)
    # Compute |A|²
    compute_wave_intensity!(A2, A, plans)

    # Compute ∇²|A|² (horizontal Laplacian)
    horizontal_laplacian!(lap_A2, A2, grid)

    # Compute ∂²|A|²/∂z² (vertical second derivative)
    vertical_second_deriv!(d2z_A2, A2, grid)

    # Wave-induced PV: qw = 0.5*(∇²|A|² - ∂²|A|²/∂z²)
    @. qw = 0.5 * (lap_A2 - d2z_A2)

    # Add J(ψ, qw) to tendency
    jacobian_spectral!(rqk, psi, qw, ...)
end
```

### Enabling/Disabling

```julia
# With wave feedback (default)
params = QGParams(; no_feedback=false)

# Without wave feedback
params = QGParams(; no_feedback=true)
```

Disabling is useful for:
- Studying one-way wave-flow interaction
- Isolating wave dynamics from flow effects
- Computational efficiency when feedback is weak

## Scaling Analysis

### When is Feedback Important?

The feedback strength scales as:

```math
\frac{|q^w|}{|q|} \sim \left(\frac{A_0}{U}\right)^2 \cdot \left(\frac{L_w}{L}\right)^2
```

where:
- ``A_0/U``: Wave-to-flow velocity ratio
- ``L_w/L``: Wave-to-eddy length ratio

Feedback matters when:
- Strong wind forcing (large ``A_0``)
- Compact wave packets (small ``L_w``)
- Weak background flow (small ``U``)

### Typical Values

| Scenario | Wave Feedback |
|:---------|:--------------|
| Weak winds, strong eddies | Negligible |
| Storm forcing | Moderate (1-10%) |
| Tropical cyclone | Strong (10-50%) |

## Coupled Dynamics

### Feedback Loop

```
┌──────────────┐         ┌──────────────┐
│              │         │              │
│   Eddies     │◄────────│    Waves     │
│   (ψ, q)     │         │   (A, B)     │
│              │         │              │
└──────┬───────┘         └──────┬───────┘
       │                        │
       │ Refraction             │ Feedback
       │ ∂ζ/∂t                  │ qw
       │                        │
       ▼                        ▼
┌──────────────────────────────────────┐
│                                      │
│       Wave-Mean Energy Exchange      │
│                                      │
└──────────────────────────────────────┘
```

### Equilibration

The coupled system can reach statistical equilibrium where:
- Wave generation (wind) balances dissipation
- Energy flux from waves to flow balances eddy dissipation
- Net energy is constant on average

## Diagnostics

### Monitoring Energy Exchange

```julia
# Compute wave feedback contribution
qw = compute_wave_pv(state.A, grid)

# Energy exchange rate
exchange_rate = compute_energy_exchange(state.psi, qw, grid, plans)
```

### Typical Analysis

1. Track ``E_{flow}`` and ``E_{wave}`` over time
2. Compute their time derivatives
3. Compare with wave feedback term to verify energy conservation

## References

- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
- Wagner, G. L., & Young, W. R. (2016). A three-component model for the coupled evolution of near-inertial waves, quasi-geostrophic flow and the near-inertial second harmonic. *J. Fluid Mech.*, 802, 806-837.
