# [Parallel particle algorithm](@id parallel-particles)

```@meta
CurrentModule = QGYBJplus
```

Distributed particle advection follows the ownership of the model runtime.
Each rank stores only the particles currently inside its horizontal subdomain.

## Per-step sequence

1. Diagnose physical velocity arrays from model-owned spectral fields.
2. Exchange the halo width required by the selected interpolation scheme.
3. Interpolate velocity and advance local particles.
4. Canonicalize periodic horizontal coordinates.
5. Migrate particles whose destination lies on another rank.
6. Verify or record global particle counts when requested by the application.

Periodic normalization occurs before migration, so a particle crossing a
global seam is routed to the rank that owns its wrapped coordinate.

## Interpolation halo widths

| Method | Typical horizontal halo |
|:--|:--|
| `TRILINEAR` | one cell |
| `TRICUBIC` | two cells |
| `QUINTIC` | three cells |
| `ADAPTIVE` | three cells |

## Conservation check

```julia
using MPI

local_count = model.particles.particles.np
global_count = MPI.Allreduce(local_count, MPI.SUM,
                             model.runtime.mpi.comm)
```

Global identifiers remain unique across migration. The regression at
`test/test_mpi_particles_periodic.jl` checks conservation, seam crossings,
domain bounds, and agreement with a serial interpolation reference.

## Topology considerations

Particle ownership follows the model's two-dimensional topology. Choose a
topology that leaves enough local cells for the interpolation halo. Very small
local blocks can be correct but communication dominated.
