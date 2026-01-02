# [Key Concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

This page introduces the core concepts you need to understand QGYBJ+.jl. No code yet - just the ideas!

---

## The Big Picture

### What Are We Simulating?

The ocean has two overlapping "worlds" of motion:

**World 1: Eddies (Slow)**
- Giant spinning vortices ~100 km across
- Last weeks to months
- Move slowly (~10 cm/s)
- Think: weather systems in the ocean

**World 2: Waves (Fast)**
- Rippling internal waves ~10 km wavelength
- Oscillate every ~17 hours (inertial period)
- Propagate rapidly through the water column
- Think: ocean "breathing" after a storm

**QGYBJ+.jl simulates both worlds and how they interact.**

### Why Does This Matter?

These interactions control:
- **Energy transfer** in the ocean
- **Mixing** that affects climate
- **Where waves go** (they get trapped by eddies!)
- **Turbulent dissipation** at small scales

---

## The Two Main Variables

Everything in QGYBJ+.jl revolves around two key quantities:

### 1. Streamfunction (ψ) - "The Eddy Field"

The streamfunction describes the balanced (geostrophic) flow:
- **High ψ** = Anticyclone (clockwise in Northern Hemisphere)
- **Low ψ** = Cyclone (counter-clockwise)
- Flow follows contours of ψ (like elevation contours on a map)

**Key insight**: From ψ, we can compute:
- Horizontal velocities: `u = -∂ψ/∂y`, `v = +∂ψ/∂x`
- Vorticity: `ζ = ∂v/∂x - ∂u/∂y = ∇²ψ`

### 2. Wave Envelope (B) - "The Wave Field"

B represents the slowly-varying wave amplitude:
- B is **complex**: `B = Br + i·Bi`
- `|B|` = Wave amplitude (how strong)
- `arg(B)` = Wave phase (where in oscillation cycle)

**Key insight**: B captures how wave energy is distributed, without tracking every fast oscillation.

---

## How They Interact

The eddies and waves affect each other through three key processes:

| Process | What Happens | Mathematical Term |
|:--------|:-------------|:------------------|
| **Advection** | Waves carried by eddy flow | `J(ψ, B)` - Jacobian term |
| **Refraction** | Waves bend toward anticyclones | `(1/2)·ζ·B` - Vorticity coupling |
| **Dispersion** | Waves spread horizontally | `i·k²·A` - Phase propagation |

### Wave Refraction: The Key Phenomenon

Why waves concentrate in anticyclones:
- Effective frequency: `f_eff = f₀ + ζ/2`
- In anticyclones: `ζ < 0` → `f_eff < f₀` → Waves slow down and accumulate
- In cyclones: `ζ > 0` → `f_eff > f₀` → Waves speed up and leave

**Result**: Waves get **trapped** in anticyclones!

---

## The L⁺ Operator: B ↔ A Relationship

A key concept in YBJ+ is the relationship between B (what we evolve) and A (the actual wave amplitude):

```math
B = L^+(A) = \frac{\partial}{\partial z}\left[\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right] - \frac{k^2}{4}A
```

**Why two variables?**
- We **evolve** B (it has nicer mathematical properties)
- We **diagnose** A (it's what we physically measure)
- Converting B → A requires solving an elliptic equation (tridiagonal system)

---

## Coordinate System

### Vertical Coordinate

- `z = 0` at the **surface**
- `z = -Lz` at the **bottom**
- Depth from surface = `-z` (positive downward)

### Spectral vs Physical Space

| Physical Space | Spectral Space |
|:---------------|:---------------|
| `ψ(x, y, z)` | `ψ̂(kx, ky, z)` |
| Real values | Complex values |
| Grid points | Wavenumbers |
| Nonlinear terms (multiplication) | Linear terms (derivatives) |

**Key insight**: We work in BOTH spaces:
- Spectral for derivatives (multiply by `i·k`)
- Physical for nonlinear products (multiply fields together)

---

## Time Stepping Algorithms

### Two Options

| Method | Speed | Stability | Best For |
|:-------|:------|:----------|:---------|
| **Leapfrog** | Slower | CFL-limited by dispersion | Small dt (~2s) |
| **IMEX-CN** | 10x faster | Dispersion treated implicitly | Large dt (~20s) |

### The IMEX Advantage

- **Leapfrog**: Must resolve fastest waves → `dt ≤ 2f/N² ≈ 2s`
- **IMEX-CN**: Treats dispersion implicitly → Only limited by advection CFL → `dt` can be ~20s (10x speedup!)

---

## Glossary of Terms

| Term | Symbol | Meaning |
|:-----|:-------|:--------|
| Streamfunction | ψ | Defines the eddy velocity field |
| Potential Vorticity | q | Conserved quantity for balanced flow |
| Wave Envelope | B | Slowly-varying wave amplitude (evolved) |
| Wave Amplitude | A | Physical wave amplitude (from B via inversion) |
| Vorticity | ζ | Local rotation rate = ∇²ψ |
| Coriolis Parameter | f₀ | Earth rotation effect |
| Buoyancy Frequency | N | Stratification strength |
| Jacobian | J(a,b) | Advection operator |
| Spectral | k-space | Fourier transform domain |

---

## Mental Model: The Simulation Loop

Each time step follows this pattern:

1. **Inversions** (get diagnostic fields)
   - `q → ψ` (elliptic PV inversion)
   - `B → A` (elliptic YBJ+ inversion)
   - `ψ → u,v` (spectral derivatives)

2. **Tendencies** (compute time derivatives)
   - `∂q/∂t = -J(ψ,q) - J(ψ,qʷ) + dissipation`
   - `∂B/∂t = -J(ψ,B) + i·αdisp·k²·A - (i/2)·ζ·B + dissipation`

3. **Time step** (update prognostic fields)
   - Leapfrog: `q^{n+1} = q^{n-1} + 2·dt·(∂q/∂t)^n`
   - or IMEX-CN with operator splitting

4. **Output** (if save interval reached)
   - Write ψ, B, energies to NetCDF

---

## What's Next?

Now that you understand the concepts:

1. **[Quick Start](@ref quickstart)** - Run your first simulation (5 min)
2. **[Worked Example](@ref worked_example)** - Build a real simulation step-by-step
3. **[Physics Overview](@ref physics-overview)** - See the full equations

---

## FAQ for Newcomers

### Q: Why "quasi-geostrophic"?

The flow is approximately in geostrophic balance (pressure gradient ≈ Coriolis force), with small deviations that allow for evolution. "Quasi" = "almost".

### Q: Why is B complex?

Complex numbers naturally encode both amplitude and phase. The real and imaginary parts represent waves 90° out of phase.

### Q: Why spectral methods?

Derivatives become multiplications in spectral space, which is very accurate and efficient. The tradeoff is periodic boundary conditions.

### Q: What's the difference between State and Grid?

- **Grid** = The "stage" (coordinates, wavenumbers, domain size)
- **State** = The "actors" (field values that change in time)

### Q: When should I use MPI parallel?

When your grid is too large to fit in memory on one machine, or when you need faster results. Rule of thumb: 256³ grids and larger benefit from parallelization.
