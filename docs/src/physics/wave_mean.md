# [Wave–mean interaction](@id wave-mean)

```@meta
CurrentModule = QGYBJplus
```

Balanced flow advects and refracts near-inertial waves. In the coupled model,
wave potential vorticity also modifies the balanced inversion, providing a
two-way exchange pathway.

## Coupling modes

```julia
coupled = QGYBJModel(
    grid=grid,
    flow=EvolvingFlow(),
    feedback=WaveMeanFeedback(),
)

one_way = QGYBJModel(
    grid=grid,
    flow=EvolvingFlow(),
    feedback=NoWaveFeedback(),
)

uncoupled = QGYBJModel(
    grid=grid,
    feedback=NoFeedback(),
)
```

- `WaveMeanFeedback()` enables bidirectional coupling.
- `NoWaveFeedback()` lets the flow act on waves without the reverse term.
- `NoFeedback()` disables wave–mean coupling.
- `FixedFlow()` prevents balanced-flow evolution regardless of feedback mode.

## Effective potential vorticity

For bidirectional coupling, inversion uses the balanced component obtained by
removing the diagnosed wave contribution from total potential vorticity. The
ETD-RK2 kernel restores the prognostic quantity after inversion, so model
ownership does not change during a stage.

For the dimensional wave envelope ``B``, the diagnosed contribution is

```math
q^w = \frac{i}{2f_0}J(B^*, B)
    + \frac{1}{4f_0}\nabla_h^2 |B|^2.
```

Both ETD-RK2 stages diagnose this contribution from the stage-local complex
wave field.

## Energy pathway

In the inviscid continuous system, work exchanged through the feedback term is
internal to the combined wave–mean system. Dissipation and numerical
resolution determine how closely a discrete run preserves that budget.

```julia
flow_energy = flow_kinetic_energy(model)
wave_components = wave_energy(model)
```

For controlled comparisons, construct separate models with identical geometry
and initial conditions but distinct feedback components. Each model owns its
own runtime and arrays and must be finalized independently.

## Practical guidance

- Use `NoWaveFeedback()` to isolate refraction and wave capture by a prescribed
  or evolving flow.
- Use `WaveMeanFeedback()` for coupled energy-exchange studies.
- Use `FixedFlow()` for published imposed-flow experiments such as the Asselin
  dipole example.

See Xie & Vanneste (2015) for the generalized-Lagrangian-mean derivation.
