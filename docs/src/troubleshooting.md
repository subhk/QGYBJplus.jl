## Troubleshooting

### Package Instantiation Errors

- Symptom: `Package <X> is a direct dependency, but does not appear in the manifest`
- Fix: run `Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()` in the project.

### Missing NCDatasets (NetCDF I/O)

- Symptom: `NCDatasets not available. Install NCDatasets.jl or skip NetCDF I/O.`
- Fix: `julia --project=. -e 'using Pkg; Pkg.add("NCDatasets")'`
  - Alternatively, disable NetCDF I/O by setting `save_*` to `false` in `OutputConfig`.

### FFT Issues on HPC/Clusters

- Install system FFTW or let FFTW.jl download binaries automatically.
- Ensure `using FFTW` works and that shared libraries are available in your environment.

### MPI/Pencil Setup

- Install MPI.jl and PencilArrays/PencilFFTs.
- Launch with `mpiexec -n <N> julia --project=. your_script.jl` and pass `use_mpi=true` to `setup_simulation`.

### Stability/Time Step

- If you see blow‑ups or NaNs:
  - Reduce `dt`
  - Increase resolution
  - Enable viscosity/hyperdiffusion (see parameters in `QGParams`)
  - Start with `linear=true` to isolate linear dynamics

