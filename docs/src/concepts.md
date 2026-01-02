# [Key Concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

Core concepts for understanding QGYBJ+.jl - no code, just ideas.

## What We Simulate

The ocean has two interacting components:

| Component | Scale | Timescale | Description |
|:----------|:------|:----------|:------------|
| **Eddies** | ~100 km | weeks-months | Slow spinning vortices |
| **Waves** | ~10 km | ~17 hours | Fast near-inertial oscillations |

QGYBJ+.jl simulates both and their interactions: eddies refract waves, waves feed energy back.

## The Two Main Variables

### Streamfunction (ψ) - Eddies
- High ψ = anticyclone, low ψ = cyclone
- Velocities: `u = -∂ψ/∂y`, `v = ∂ψ/∂x`
- Vorticity: `ζ = ∇²ψ`

### Wave Envelope (B) - Waves
- Complex: `B = Br + i·Bi`
- `|B|` = amplitude, `arg(B)` = phase
- Captures wave energy distribution without tracking fast oscillations

## Wave-Eddy Interaction

| Process | Effect |
|:--------|:-------|
| Advection | Waves carried by eddy flow |
| Refraction | Waves bend toward anticyclones |
| Dispersion | Waves spread horizontally |

**Key insight**: Effective frequency `f_eff = f₀ + ζ/2`. In anticyclones (ζ < 0), waves slow and get **trapped**.

## B vs A

We evolve B (nicer math), diagnose A (physical amplitude):
```math
B = L^+(A) = \frac{\partial}{\partial z}\left[\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right] - \frac{k^2}{4}A
```

## Coordinates

- **Vertical**: z = 0 (surface), z = -Lz (bottom)
- **Spectral vs Physical**: Derivatives in spectral space, nonlinear products in physical space

## Time Stepping

| Method | Speed | Best For |
|:-------|:------|:---------|
| Leapfrog | Slower | General use |
| IMEX-CN | 10× faster | Wave-dominated (treats dispersion implicitly) |

## Glossary

| Symbol | Meaning |
|:-------|:--------|
| ψ | Streamfunction (eddies) |
| q | Potential vorticity |
| B | Wave envelope (evolved) |
| A | Wave amplitude (diagnosed) |
| ζ | Vorticity = ∇²ψ |
| f₀ | Coriolis parameter |
| N | Buoyancy frequency |

## Next Steps

- [Quick Start](@ref quickstart) - Run a simulation
- [Physics Overview](@ref physics-overview) - Full equations
