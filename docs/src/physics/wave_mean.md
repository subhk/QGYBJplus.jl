# [Wave-Mean Interaction](@id wave-mean)

```@meta
CurrentModule = QGYBJplus
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
q^w = \frac{i}{2} J(B^*, B) + \frac{1}{4} \nabla_h^2 |B|^2
```

where ``B`` is the complex wave envelope with units of velocity (m/s).

!!! note "Dimensional Equations"
    The model solves dimensional equations where ``B`` has actual velocity amplitude.
    No additional scaling factors (like W2F) are needed.

### Decomposition in Real/Imaginary Parts

Writing ``B = B_R + i B_I``, the Jacobian term becomes:

```math
\frac{i}{2} J(B^*, B) = \frac{\partial B_R}{\partial y} \frac{\partial B_I}{\partial x} - \frac{\partial B_R}{\partial x} \frac{\partial B_I}{\partial y}
```

And the wave intensity:

```math
|B|^2 = B_R^2 + B_I^2
```

So the complete formula in spectral space is:

```math
q^w = \left( \frac{\partial B_R}{\partial y} \frac{\partial B_I}{\partial x} - \frac{\partial B_R}{\partial x} \frac{\partial B_I}{\partial y} \right) - \frac{k_h^2}{4} (B_R^2 + B_I^2)
```

Note: In spectral space, ``\nabla_h^2 \to -k_h^2``, so ``+\frac{1}{4}\nabla_h^2|B|^2 \to -\frac{k_h^2}{4}|B|^2``.

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
# In nonlinear.jl: compute_qw! (BR/BI form)
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
    # qw = J_term - (1/4)*kh²*|B|²  (note: ∇² → -kh² in spectral)
    qwk = (fft(qwr) - 0.25 * kh2 .* fft(mag2)) / norm

    # Note: No additional scaling needed - B has dimensional velocity units (m/s)
end
```

For the complex envelope form used in YBJ+ time stepping, use:

```julia
compute_qw_complex!(qwk, Bk, par, G, plans; Lmask=L)
```

### Usage in Time Stepping

The wave feedback enters via q* = q - qw:

```julia
# After computing q at new time step
if wave_feedback_enabled
    compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)
    q_arr .-= qwk_arr  # q* = q - qw
end

# Complex B form (YBJ+ path)
if wave_feedback_enabled
    compute_qw_complex!(qwk, Bk, par, G, plans; Lmask=L)
    q_arr .-= qwk_arr  # q* = q - qw
end

# Then invert q* to get ψ
invert_q_to_psi!(state, grid; a=a_vec)
```

### Enabling/Disabling

```julia
# With wave feedback (default)
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0; no_feedback=false, no_wave_feedback=false)

# Without wave feedback
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0; no_feedback=true)
# or
params = default_params(Lx=500e3, Ly=500e3, Lz=4000.0; no_wave_feedback=true)
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

## Complete Energy Budget

### Energy Components

The total energy of the QG-YBJ+ system consists of five components:

```math
E_{total} = \underbrace{E_{KE}^{flow} + E_{PE}^{flow}}_{\text{Mean flow energy}} + \underbrace{E_{KE}^{wave} + E_{PE}^{wave} + E_{CE}^{wave}}_{\text{Wave energy}}
```

#### Mean Flow Kinetic Energy

```math
E_{KE}^{flow} = \frac{1}{2} \int \int \int (u^2 + v^2) \, dx\, dy\, dz
```

In spectral space with dealiasing:
```math
E_{KE}^{flow} = \frac{1}{2} \sum_{k_x, k_y, z} L(k_x, k_y) \cdot k_h^2 |\hat{\psi}|^2 - \frac{1}{2}|\hat{\psi}(k_h=0)|^2
```

where ``L(k_x, k_y)`` is the dealiasing mask (2/3 rule) and the second term corrects for the zero-wavenumber mode.

#### Mean Flow Potential Energy

```math
E_{PE}^{flow} = \frac{1}{2} \int \int \int \frac{f_0^2}{N^2} \left(\frac{\partial \psi}{\partial z}\right)^2 dx\, dy\, dz
```

In spectral space:
```math
E_{PE}^{flow} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{f_0^2}{N^2(z)} |\hat{b}|^2
```

where ``b = \partial\psi/\partial z`` is the buoyancy from thermal wind balance.

#### Wave Kinetic Energy

```math
E_{KE}^{wave} = \frac{1}{2} \int \int \int |B|^2 \, dx\, dy\, dz
```

In spectral space with ``B = B_R + iB_I``:
```math
E_{KE}^{wave} = \frac{1}{2} \sum_{k_x, k_y, z} (|\hat{B}_R|^2 + |\hat{B}_I|^2) - \frac{1}{2}|\hat{B}(k_h=0)|^2
```

#### Wave Potential Energy

From the YBJ+ formulation, the wave potential energy involves ``C = \partial A/\partial z``:

```math
E_{PE}^{wave} = \frac{1}{2} \int \int \int \frac{N^2}{2f_0^2} k_h^2 |C|^2 \, dx\, dy\, dz
```

In spectral space:
```math
E_{PE}^{wave} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{k_h^2}{2 a_{ell}} (|\hat{C}_R|^2 + |\hat{C}_I|^2)
```

where ``a_{ell} = f_0^2/N^2`` is the elliptic coefficient.

#### Wave Correction Energy (YBJ+)

The YBJ+ equation introduces a higher-order correction:

```math
E_{CE}^{wave} = \frac{1}{8} \int \int \int \frac{N^4}{f_0^4} k_h^4 |A|^2 \, dx\, dy\, dz
```

In spectral space:
```math
E_{CE}^{wave} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{k_h^4}{8 a_{ell}^2} (|\hat{A}_R|^2 + |\hat{A}_I|^2)
```

This term accounts for horizontal wave dispersion and becomes significant at small scales.

### Energy Conservation Theorem

**Theorem**: In the inviscid limit (no dissipation), the total energy is conserved:

```math
\frac{dE_{total}}{dt} = 0
```

**Proof sketch**:
1. The QG PV equation conserves mean flow energy in the absence of wave feedback
2. The YBJ+ equation conserves wave energy in the absence of mean flow
3. The wave feedback term ``q^w`` transfers energy between waves and flow without dissipation
4. The refraction term ``\frac{1}{2}B \cdot \zeta`` exchanges energy via wave-vorticity interaction

### Energy Transfer Pathways

```
                    Wind Forcing
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │           Wave Energy                  │
    │   E_KE^wave + E_PE^wave + E_CE^wave    │
    └────────────────┬───────────────────────┘
                     │
           Refraction │ Wave Feedback
           (B·ζ term) │ (q^w term)
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │         Mean Flow Energy               │
    │        E_KE^flow + E_PE^flow           │
    └────────────────┬───────────────────────┘
                     │
                     ▼
              Viscous Dissipation
```

### Energy Exchange Rate

The rate of energy transfer from waves to mean flow is:

```math
\mathcal{P}_{w \to f} = -\int \psi \cdot J(\psi, q^w) \, dV
```

This can be computed diagnostically:
```julia
# Compute energy exchange rate
qw = compute_qw(state, grid, params, plans)
P_exchange = -sum(psi .* jacobian(psi, qw, grid, plans))
```

### Energy Scales

Using the characteristic scales:
- Velocity: ``U`` (mean flow), ``U_w`` (waves)
- Length: ``L`` (horizontal), ``H`` (vertical)
- Time: ``1/f_0``

The energy ratio scales as:
```math
\frac{E^{wave}}{E^{flow}} \sim \left(\frac{U_w}{U}\right)^2
```

Typical oceanic values for wave-to-flow energy ratio:
- Gulf Stream region: ``\sim 10^{-2}`` to ``10^{-1}``
- Open ocean: ``\sim 10^{-3}``
- After storm: ``\sim 1``

### Diagnostic Implementation

Energy diagnostics are automatically saved to separate files:

```julia
# Output files in diagnostic/ folder:
# - wave_KE.nc: E_KE^wave time series
# - wave_PE.nc: E_PE^wave time series
# - wave_CE.nc: E_CE^wave time series
# - mean_flow_KE.nc: E_KE^flow time series
# - mean_flow_PE.nc: E_PE^flow time series
# - total_energy.nc: All energies + totals

# Verify conservation
ds = NCDataset("output/diagnostic/total_energy.nc")
E_total = ds["total_energy"][:]
dE = (E_total[end] - E_total[1]) / E_total[1]
# Should be < 10^-6 for inviscid runs
```

See [Diagnostics Guide](../guide/diagnostics.md#energy-diagnostics-output-files) for detailed usage.

## References

- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
- Wagner, G. L., & Young, W. R. (2016). A three-component model for the coupled evolution of near-inertial waves, quasi-geostrophic flow and the near-inertial second harmonic. *J. Fluid Mech.*, 802, 806-837.
- Asselin, O., & Young, W. R. (2019). An improved model of near-inertial wave dynamics. *J. Fluid Mech.*, 876, 428-448.
