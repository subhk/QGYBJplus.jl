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

Following Xie & Vanneste (2015), the wave-induced potential vorticity is:

```math
q^w = Ro \cdot W2F \cdot \left[ \frac{i}{2} J(B^*, B) - \frac{1}{4} \nabla_h^2 |B|^2 \right]
```

where ``W2F = (U_w/U)^2`` is the wave-to-flow velocity ratio squared.

### Decomposition in Real/Imaginary Parts

Writing ``B = B_R + i B_I``, the Jacobian term becomes:

```math
\frac{i}{2} J(B^*, B) = \frac{\partial B_R}{\partial y} \frac{\partial B_I}{\partial x} - \frac{\partial B_R}{\partial x} \frac{\partial B_I}{\partial y}
```

And the wave intensity:

```math
|B|^2 = B_R^2 + B_I^2
```

So the complete formula is:

```math
q^w = Ro \cdot W2F \cdot \left[ \left( \frac{\partial B_R}{\partial y} \frac{\partial B_I}{\partial x} - \frac{\partial B_R}{\partial x} \frac{\partial B_I}{\partial y} \right) + \frac{k_h^2}{4} (B_R^2 + B_I^2) \right]
```

Note: In spectral space, ``\nabla_h^2 \to -k_h^2``, so ``-\frac{1}{4}\nabla_h^2|B|^2 \to +\frac{k_h^2}{4}|B|^2``.

### How It Enters the QG Equation

The wave feedback modifies the effective PV used for streamfunction inversion:

```math
q^* = q - q^w
```

Then ``\psi`` is computed from ``q^*`` via the elliptic inversion:

```math
\nabla^2\psi + \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right) = q^*
```

### Physical Interpretation

| Term | Meaning |
|:-----|:--------|
| ``J(B^*, B)`` | Jacobian of complex wave field (wave momentum flux) |
| ``\nabla_h^2\|B\|^2`` | Horizontal curvature of wave energy density |
| ``Ro \cdot W2F`` | Scaling by Rossby number and wave amplitude |

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
# In nonlinear.jl: compute_qw!
function compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=nothing)
    # 1. Compute derivatives of BR and BI
    # BRx = ∂BR/∂x, BRy = ∂BR/∂y, etc.
    BRxk = im * kx .* BRk
    BRyk = im * ky .* BRk
    BIxk = im * kx .* BIk
    BIyk = im * ky .* BIk

    # 2. Transform to real space
    # ...

    # 3. Compute Jacobian term: BRy*BIx - BRx*BIy
    qwr = BRyr .* BIxr - BRxr .* BIyr

    # 4. Compute |B|² = BR² + BI²
    mag2 = BRr.^2 + BIr.^2

    # 5. Assemble in spectral space
    # qw = J_term - (1/4)*kh²*|B|²  (note: -∇² → +kh² in spectral)
    qwk = (fft(qwr) - 0.25 * kh2 .* fft(mag2)) / norm

    # 6. Scale by Ro * W2F
    qwk .*= (par.Ro * par.W2F)
end
```

### Usage in Time Stepping

The wave feedback enters via q* = q - qw:

```julia
# After computing q at new time step
if wave_feedback_enabled
    compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)
    q_arr .-= qwk_arr  # q* = q - qw
end

# Then invert q* to get ψ
invert_q_to_psi!(state, grid; a=a_vec)
```

### Enabling/Disabling

```julia
# With wave feedback (default)
params = QGParams(; no_feedback=false, no_wave_feedback=false)

# Without wave feedback
params = QGParams(; no_feedback=true)
# or
params = QGParams(; no_wave_feedback=true)
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
