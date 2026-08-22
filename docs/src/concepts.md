# [Key concepts](@id concepts)

```@meta
CurrentModule = QGYBJplus
```

## Ownership

QGYBJ+.jl separates immutable geometry, model state, and run orchestration:

```text
RectilinearGrid → QGYBJModel → Simulation
```

- [`RectilinearGrid`](@ref) defines global coordinates and wavenumbers.
- [`QGYBJModel`](@ref) owns physical choices, numerical choices, distributed
  fields, FFT resources, and optional particles.
- [`Simulation`](@ref) owns time, ETD-RK2, stopping criteria, output, and
  diagnostics.

Finalize a simulation when the run ends. This closes its output services and
releases runtime resources owned by the model.

## Model fields

All model arrays use `(z, x, y)` ordering.

| Field | Role |
|:--|:--|
| `q` | prognostic generalized PV; with feedback it contains balanced and wave contributions |
| `B` | prognostic complex near-inertial-wave envelope |
| `psi` | balanced streamfunction diagnosed from PV |
| `A`, `C` | wave amplitude and its vertical derivative |
| `u`, `v`, `w` | physical-space velocity components |

The horizontal velocity follows
``u=-\partial_y\psi`` and ``v=\partial_x\psi``. The relative vorticity is
``\zeta=\nabla_h^2\psi``.

## Wave formulations

| Component | Meaning |
|:--|:--|
| [`YBJPlus`](@ref) | regularized YBJ⁺ wave dynamics |
| [`YBJ`](@ref) | original Young–Ben Jelloul relation |
| [`PassiveWave`](@ref) | wave-envelope advection without refraction or dispersion |

The model advances `B` and diagnoses `A`. See [YBJ⁺ wave model](@ref ybj-plus)
for the relation between them.

## Flow and feedback choices

The common configurations are:

| Use case | Flow | Feedback |
|:--|:--|:--|
| prescribed flow acting on waves | `FixedFlow()` | `NoFeedback()` |
| evolving flow acting on waves | `EvolvingFlow()` | `NoWaveFeedback()` |
| two-way wave–mean coupling | `EvolvingFlow()` | `WaveMeanFeedback()` |

With two-way coupling, wave PV modifies streamfunction inversion. In all
cases, advection and refraction depend on the selected flow and wave
formulation.

## Coordinates and transforms

- Horizontal boundaries are periodic.
- The vertical coordinate increases from the bottom to `z=0` at the surface.
- Vertical nodes are cell centered.
- Spectral fields are complex; diagnosed velocities are real.
- Horizontal derivatives are spectral, while nonlinear products are formed
  in physical space.

Continue with [configuration](@ref configuration) or the [physics
overview](@ref physics-overview).
