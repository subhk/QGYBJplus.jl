# Troubleshooting

```@meta
CurrentModule = QGYBJplus
```

## Construction fails

- All resolution and extent values must be positive.
- `FPlane(f=...)` requires a finite, nonzero Coriolis frequency.
- Stratification values must be finite and positive.
- Explicit topology factors must divide the distributed dimensions.

Start with a small model and `verbose=true` to see MPI topology information.

## A second model fails after finalization

When a model initializes MPI itself, finalizing it closes that MPI session.
Applications that create multiple models should initialize MPI externally:

```julia
using MPI
MPI.Init()
try
    # construct and finalize any number of models
finally
    MPI.Finalize()
end
```

## Distributed array mismatch

Use `copy_fields(model.fields)` to preserve pencil layouts. Physical and
spectral local ranges differ; query them explicitly instead of reusing array
axes across transform spaces.

## Output is missing a field

Choose fields in [`NetCDFOutput`](@ref):

```julia
NetCDFOutput(path="output",
             fields=(:ψ, :waves),
             velocities=true)
```

The prognostic spectral arrays and vertical coefficients are always written.

## A run cannot be restarted

`restore!` requires a fresh compatible model and matching global dimensions.
It restores `q` and `B`; it does not rewind an existing simulation clock.

## A run is slow

- Let one `Simulation` own the complete run instead of constructing a
  timestepper for every step.
- Reduce output frequency and omit unused physical fields.
- Avoid gathers inside the time loop.
- Measure alternative MPI topologies.
- Ensure local domains remain large enough for particle halos.

## Particles disappear near a periodic seam

Run `test/test_mpi_particles_periodic.jl` under the same rank count. The test
checks global identifier conservation, wrapped coordinates, and agreement with
a serial interpolation reference.

## Report a reproducible issue

Include Julia and package versions, global resolution, topology, rank count,
the focused component choices, `Δt`, and the smallest script that reproduces
the problem.
