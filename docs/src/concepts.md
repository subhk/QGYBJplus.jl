# [Key Concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

Core ideas behind QGYBJ+.jl, without code.

## The Two Main Variables

### Streamfunction (ψ) — Eddies

The streamfunction describes the balanced (geostrophic) eddy flow:

- High ψ = anticyclone (clockwise in Northern Hemisphere)
- Low ψ = cyclone (counter-clockwise in Northern Hemisphere)
- Velocities derived as: `u = -∂ψ/∂y`, `v = ∂ψ/∂x`
- Vorticity: `ζ = ∇²ψ` (positive = cyclonic, negative = anticyclonic)

### Wave Envelope (B) — Waves

The wave envelope captures wave energy without tracking fast oscillations:

- Complex-valued: `B = Bᵣ + i·Bᵢ`
- Magnitude `|B|` represents wave amplitude
- Phase `arg(B)` represents wave phase
- Evolves on the slow (eddy) timescale

## Wave-Eddy Interaction

Three key processes govern how waves and eddies interact:

| Process | What Happens | Physical Effect |
|:--------|:-------------|:----------------|
| **Advection** | `J(ψ, B)` | Waves are carried by the eddy velocity field |
| **Refraction** | `½ζB` | Waves bend toward regions of negative vorticity |
| **Dispersion** | `ik²A` | Waves spread horizontally over time |

!!! tip "Wave Trapping"
    The effective wave frequency is `f_eff = f₀ + ζ/2`. In anticyclones where ζ < 0, waves slow down and get **trapped** — this is a key mechanism for wave energy concentration.

## B vs A: Why Two Wave Variables?

We evolve **B** (mathematically convenient) but diagnose **A** (physically meaningful):

```math
B = L^+(A) = \frac{\partial}{\partial z}\left[\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right] - \frac{k^2}{4}A
```

| Variable | Role | Why We Need It |
|:---------|:-----|:---------------|
| **B** | Prognostic (evolved) | Simpler time-stepping equations |
| **A** | Diagnostic (computed) | Represents physical wave amplitude |

## Coordinate System

### Spatial Coordinates
- **Horizontal**: x (east), y (north) — doubly periodic domain
- **Vertical**: z = 0 at surface, z = -Lz at bottom

### Spectral vs Physical Space
- **Derivatives** computed in spectral space (fast, accurate)
- **Nonlinear products** computed in physical space (avoid aliasing)
- Transform between spaces using FFT

## Time Stepping

The model uses second-order ETD-RK2. Horizontal hyperdiffusion is integrated
exactly, while advection, refraction, dispersion, and vertical diffusion use
two explicit Runge–Kutta stages.

## Quick Glossary

| Symbol | Name | Meaning |
|:-------|:-----|:--------|
| ψ | Streamfunction | Describes eddy flow |
| q | Potential vorticity | Conserved quantity for eddies |
| B | Wave envelope | Evolved wave variable |
| A | Wave amplitude | Physical wave amplitude |
| ζ | Relative vorticity | ∇²ψ, measures rotation |
| f₀ | Coriolis parameter | Earth's rotation effect |
| N | Buoyancy frequency | Stratification strength |
| Lx, Ly | Domain size | Horizontal extent |
| Lz | Domain depth | Vertical extent |

## Next Steps

- [Quick Start](@ref quickstart) — run your first simulation
- [Physics Overview](@ref physics-overview) — the full equations
