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

## Architectures

```@docs
CPU
GPU
architecture
on_architecture
```

See [CPU and GPU execution](@ref architectures).

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
FlowHyperdiffusivity
WaveHyperdiffusivity
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
set_mean_flow!
set_surface_waves!
set_exponential_surface_waves!
set_wave_packet!
SurfaceWave
RandomStreamfunction
FieldArray
FieldFile
TimeInterval
IterationInterval
NetCDFOutput
EnergyDiagnosticsOutput
restore!
```

## Stratification

```@docs
StratificationProfile
ConstantN
SkewedGaussian
TanhProfile
ExponentialProfile
PiecewiseProfile
FileProfile
AnalyticalProfile
evaluate_N2
compute_stratification_profile
load_stratification_from_file
validate_stratification
```
