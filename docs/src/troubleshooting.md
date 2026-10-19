# Troubleshooting

```@meta
CurrentModule = QGYBJplus
```

## Construction fails

- All resolution and extent values must be positive.
- `FPlane(f=...)` requires a finite, nonzero Coriolis frequency.
- Stratification values must be finite and positive.

Start with a small model and `verbose=true` to see the runtime summary.

## `GPU()` throws an `ArgumentError`

`GPU()` needs a GPU package. Load CUDA.jl before building the grid:

```julia
using CUDA
using QGYBJplus
grid = RectilinearGrid(GPU(); size=(64, 64, 32), extent=(1e5, 1e5, 4e3))
```

If `CUDA.functional()` is `false`, CUDA.jl cannot reach a driver or device;
see the CUDA.jl documentation.

## "Scalar indexing is disallowed"

Model arrays on a GPU cannot be indexed element by element from the host.
Copy them first, or use whole-array operations:

```julia
q = Array(model.fields.q)     # host copy
q[1, 2, 3]
maximum(abs, model.fields.q)  # runs on the device
```

## Output is missing a field

Choose fields in [`NetCDFOutput`](@ref):

```julia
NetCDFOutput(path="output",
             fields=(:ψ, :waves),
             velocities=true)
```

The prognostic spectral arrays and vertical coefficients are always written.

## A run cannot be restarted

`restore!` requires a compatible model: the same resolution, domain extent,
wave formulation, and feedback convention. It restores `q` and `B`; restoring
a `Simulation` also sets its clock to the file's time and iteration. State
files written on a CPU restart on a GPU and vice versa.

## Output files already exist

A run never silently replaces earlier output. A restored simulation continues
the files in its output directory; any other existing state, energy, or
particle files raise an error. Choose a new directory, or pass
`overwrite=true` to `NetCDFOutput`, `EnergyDiagnosticsOutput`, or
`ParticleOutputManager` to replace them.

## A run is slow

- Let one `Simulation` own the complete run instead of constructing a
  timestepper for every step.
- Reduce output frequency and omit unused physical fields.
- Avoid copying fields to the host inside the time loop.
- On a CPU, start Julia with several threads; see [Performance](@ref
  performance).

## Report a reproducible issue

Include Julia and package versions, the architecture (CPU threads or GPU
model and CUDA version), global resolution, the focused component choices,
`Δt`, and the smallest script that reproduces the problem.
