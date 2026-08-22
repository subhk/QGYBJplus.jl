# [Core types](@id api-types)

```@meta
CurrentModule = QGYBJplus
```

## Ownership

```@docs
RectilinearGrid
ModelFields
QGYBJModel
ModelPhysics
ModelNumerics
ModelRuntime
OperatorCoefficients
Simulation
Clock
SimulationState
```

`RectilinearGrid` is immutable. `QGYBJModel` owns model data and ephemeral
runtime resources. `Simulation` separately owns execution lifecycle.

## Physical components

```@docs
FPlane
ConstantStratification
FixedFlow
EvolvingFlow
NoFeedback
NoWaveFeedback
WaveMeanFeedback
YBJPlus
YBJ
PassiveWave
```

## Numerical components

```@docs
HorizontalHyperdiffusivity
VerticalDiffusivity
Dissipative
Inviscid
NonlinearDynamics
LinearDynamics
Dispersive
NoDispersion
```

## Initialization and scheduling

```@docs
set!
SurfaceWave
RandomStreamfunction
FieldArray
FieldFile
TimeInterval
IterationInterval
NetCDFOutput
EnergyDiagnosticsOutput
AnalyticalProfile
```
