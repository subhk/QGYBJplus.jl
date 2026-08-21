# [MPI parallel execution](@id parallel)

```@meta
CurrentModule = QGYBJplus
```

QGYBJ+.jl uses PencilArrays.jl and PencilFFTs.jl for two-dimensional pencil
decomposition. Application code constructs the same model at every rank.

```bash
mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl
```

## Topology

Automatic topology selection is suitable for most runs. Supply an explicit
factorization when required:

```julia
model = QGYBJModel(
    grid=grid,
    topology=(2, 4),
    verbose=false,
)
```

Both factors must divide the relevant global dimensions. Local decomposition
metadata lives in `model.runtime.decomposition`; immutable global geometry
remains in `model.grid`.

## Local ranges

```julia
spectral_range = get_local_range_spectral(model)
physical_range = get_local_range_physical(model)
i_global = local_to_global(1, 2, model.fields.q)
```

Avoid assuming that physical and spectral pencils have the same local axes.

## Gather and scatter

```julia
global_q = gather_to_root(model.fields.q,
                          model.runtime.geometry,
                          model.runtime.mpi)

model.fields.q .= scatter_from_root(global_q,
                                    model.runtime.geometry,
                                    model.runtime.mpi;
                                    plans=model.runtime.plans)
```

Prefer scheduled NetCDF output for routine runs; explicit gather is primarily
for tests and custom analysis.

## MPI ownership

If MPI was already initialized, the model treats it as externally owned and
will not finalize it. Otherwise the first model initializes MPI and closes it
during `finalize_model!`. Test suites that create multiple models should
initialize MPI once around the suite.

## Parallel verification

The repository includes distributed regressions for transform round trips,
ETD-RK2 rank independence, and particle migration:

```bash
mpiexecjl -n 2 julia --project=. test/test_mpi_extension.jl
mpiexecjl -n 2 julia --project=. test/test_mpi_stepping_regression.jl
mpiexecjl -n 2 julia --project=. test/test_mpi_particles_periodic.jl
```
