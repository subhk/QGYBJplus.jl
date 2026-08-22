# [Wave–mean interaction](@id wave-mean)

```@meta
CurrentModule = QGYBJplus
```

Balanced flow advects and refracts near-inertial waves. Two-way coupling adds
the wave contribution to generalized PV.

## Choose the coupling

| Use case | Components |
|:--|:--|
| prescribed flow acting on waves | `FixedFlow()`, `NoFeedback()` |
| evolving flow acting on waves | `EvolvingFlow()`, `NoWaveFeedback()` |
| two-way coupling | `EvolvingFlow()`, `WaveMeanFeedback()` |

For example:

```julia
model = QGYBJModel(
    grid=grid,
    flow=EvolvingFlow(),
    feedback=WaveMeanFeedback(),
    formulation=YBJPlus(),
)
```

`FixedFlow()` always prevents balanced-flow evolution. Both no-feedback
components omit wave PV from the inversion; `NoWaveFeedback()` makes the
one-way evolving-flow intent explicit.

## Wave PV

For two-way coupling,

```math
q^w = \frac{i}{2f_0}J(B^*,B)
    + \frac{1}{4f_0}\nabla_h^2|B|^2.
```

Each ETD-RK2 stage diagnoses ``q^w`` from its stage-local complex wave
field. Streamfunction inversion uses ``q-q^w``, after which the solver
restores the prognostic total `q`.

## Compare coupled and uncoupled runs

Construct separate models with identical geometry and initial conditions but
different feedback components. Each model owns independent arrays and runtime
resources, so finalize each simulation independently.

```julia
flow_energy = flow_kinetic_energy(model)
B_energy, A_energy = wave_energy(model)
```

In the inviscid continuous system, wave–mean exchange is internal to the
combined energy budget. Dissipation, timestep, and spatial resolution control
the corresponding discrete error.

See Xie & Vanneste (2015) for the generalized-Lagrangian-mean derivation and
the [QG equations](@ref qg-equations) for inversion details.
