# [Key Concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

Core concepts for understanding QGYBJ+.jl — no code, just ideas.

---

## What We Simulate

The ocean has two interacting components at very different scales:

```@raw html
<div class="feature-grid">
<div class="feature-card">
    <h3>Mesoscale Eddies</h3>
    <p><strong>Scale:</strong> ~100 km<br>
    <strong>Timescale:</strong> Weeks to months<br>
    <strong>Description:</strong> Slow spinning vortices that contain most of the ocean's kinetic energy</p>
</div>
<div class="feature-card">
    <h3>Near-Inertial Waves</h3>
    <p><strong>Scale:</strong> ~10 km<br>
    <strong>Timescale:</strong> ~17 hours<br>
    <strong>Description:</strong> Fast oscillations driven by wind, crucial for ocean mixing</p>
</div>
</div>
```

QGYBJ+.jl simulates both components and their interactions: **eddies refract waves, and waves feed energy back to the eddies**.

---

## The Two Main Variables

### Streamfunction (ψ) — Eddies

The streamfunction describes the balanced (geostrophic) eddy flow:

- High ψ = anticyclone (clockwise in Northern Hemisphere)
- Low ψ = cyclone (counter-clockwise in Northern Hemisphere)
- Velocities derived as: `u = -∂ψ/∂y`, `v = ∂ψ/∂x`
- Vorticity: `ζ = ∇²ψ` (positive = cyclonic, negative = anticyclonic)

### Wave Envelope (L⁺A) — Waves

The wave envelope captures wave energy without tracking fast oscillations:

- Complex-valued: `L⁺A = (L⁺A)ᵣ + i·(L⁺A)ᵢ`
- Related to physical wave amplitude via: `L⁺A = L⁺(A)` where `L⁺ = L - k_h²/4`
- Evolves on the slow (eddy) timescale
- The code uses the Unicode variable name `L⁺A` for mathematical clarity

---

## Wave-Eddy Interaction

Three key processes govern how waves and eddies interact:

| Process | What Happens | Physical Effect |
|:--------|:-------------|:----------------|
| **Advection** | `J(ψ, L⁺A)` | Waves are carried by the eddy velocity field |
| **Refraction** | `½ζ(L⁺A)` | Waves bend toward regions of negative vorticity |
| **Dispersion** | `ik²A` | Waves spread horizontally over time |

!!! tip "Wave Trapping"
    The effective wave frequency is `f_eff = f₀ + ζ/2`. In anticyclones where ζ < 0, waves slow down and get **trapped** — this is a key mechanism for wave energy concentration.

---

## L⁺A vs A: Why Two Wave Variables?

We evolve **L⁺A** (mathematically convenient) but diagnose **A** (physically meaningful):

```math
L^+A = \frac{\partial}{\partial z}\left[\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right] - \frac{k^2}{4}A
```

| Variable | Role | Why We Need It |
|:---------|:-----|:---------------|
| **L⁺A** | Prognostic (evolved) | Simpler time-stepping equations |
| **A** | Diagnostic (computed) | Represents physical wave amplitude |
| **LA** | Derived | Wave velocity amplitude: LA = L⁺A + (k_h²/4)A, used for wave kinetic energy |

---

## Coordinate System

### Spatial Coordinates
- **Horizontal**: x (east), y (north) — doubly periodic domain
- **Vertical**: z = 0 at surface, z = -Lz at bottom

### Spectral vs Physical Space
- **Derivatives** computed in spectral space (fast, accurate)
- **Nonlinear products** computed in physical space (avoid aliasing)
- Transform between spaces using FFT

---

## Time Stepping Options

| Method | Speed | Best For | How It Works |
|:-------|:------|:---------|:-------------|
| **Leapfrog** | Standard | General use | Explicit, simple, stable |
| **IMEX-CN** | 10× faster | Wave-dominated | Treats fast dispersion implicitly |

!!! note "Choosing a Method"
    Use IMEX-CN when wave dispersion limits your timestep. Leapfrog is simpler and sufficient when eddies dominate.

---

## Quick Glossary

| Symbol | Name | Meaning |
|:-------|:-----|:--------|
| ψ | Streamfunction | Describes eddy flow |
| q | Potential vorticity | Conserved quantity for eddies |
| L⁺A | Wave envelope | Evolved wave variable (YBJ+) |
| A | Wave amplitude | Physical wave amplitude (diagnostic) |
| LA | Wave velocity amplitude | LA = L⁺A + (k_h²/4)A, used for energy |
| ζ | Relative vorticity | ∇²ψ, measures rotation |
| f₀ | Coriolis parameter | Earth's rotation effect |
| N | Buoyancy frequency | Stratification strength |
| Lx, Ly | Domain size | Horizontal extent |
| Lz | Domain depth | Vertical extent |

---

## Next Steps

```@raw html
<div class="learning-path">
<div class="path-step">
    <div class="step-number">→</div>
    <div class="step-content">
        <strong><a href="../quickstart/">Quick Start</a></strong> — Run your first simulation
    </div>
</div>
<div class="path-step">
    <div class="step-number">→</div>
    <div class="step-content">
        <strong><a href="../physics/overview/">Physics Overview</a></strong> — See the full equations
    </div>
</div>
</div>
```
