# KernelAbstractions CPU/GPU Architecture

## Status

Implemented on 2026-10-09. A breaking change: MPI decomposition is removed in
favour of single-device execution on a CPU or one GPU.

## Context

The model ran through MPI with PencilArrays.jl and PencilFFTs.jl even in a
single process, and every array loop was a plain host `for` loop over the
local pencil. That ruled out GPUs: PencilFFTs needs `permute_dims=Val(true)` to
transform on a GPU, which changes the memory layout every kernel indexed, and
the loops scalar-indexed their arrays.

## Decisions

1. **Architecture on the grid.** `RectilinearGrid(arch; ...)` with `CPU()`
   (default) or `GPU(backend)`, as in Oceananigans. `GPU()` takes its backend
   from a package extension (`QGYBJplusCUDAExt` defines
   `default_gpu_backend() = CUDABackend(always_inline=true)`); any
   KernelAbstractions backend can be passed explicitly.
2. **Plain device arrays.** Fields, scratch, ETD tables, coefficient profiles,
   and the dealiasing mask are `Array`/`CuArray` values of the grid's
   architecture. The grid's coordinate vectors stay on the host for setup and
   output; `RuntimeGeometry` carries device copies of `kx`, `ky`, `kh2`.
3. **Every index-dependent loop is a KernelAbstractions kernel**, launched with
   `launch!(kernel!, backend_or_array, ndrange, args...)` on the backend of the
   array it writes. Pointwise maps use `kernel_map!`; reductions use `sum`,
   `maximum`, and `mapreduce`, which run on the device for GPU arrays.
4. **Vertical operators are column kernels.** One work item per `(i, j)` mode
   loops over `z`: Thomas solves (`tridiagonal_row(system, r)` describes each
   operator's rows), cumulative integrations, energy column sums. The complex
   right-hand side is solved in one sweep, operation-for-operation equal to the
   former separate real/imaginary solves. The omega equation uses the same
   Thomas sweep instead of LAPACK `gtsv!`; its matrix is strictly diagonally
   dominant, so pivoting never triggers.
5. **Transforms.** One in-place AbstractFFTs plan over dimensions `(2, 3)` of
   the `(nz, nx, ny)` array transforms all levels at once: FFTW on the host,
   cuFFT on CUDA. Backends without AbstractFFTs plans add a
   `Transforms.horizontal_fft_plans` method.
6. **Masks as kernel rules.** `Lmask` keywords still accept Bool matrices or
   `nothing`; kernels receive `KeepAllModes()`, `RadialDealiasing(nx, ny)`, or
   a device mask, so the default rules cost no memory or transfers.
7. **Host-side work stays explicit.** User initial-condition functions are
   evaluated on the host and copied over. NetCDF output copies to the host.
   Particles keep positions and interpolation on the host, with velocities
   copied from the device once per particle step.

## Removed

`MPIConfig`, `PencilDecomp`, `MPIPlans`, z-pencil transposes (`with_z_local`),
`gather_to_root`, `scatter_from_root`, `get_local_range*`, `local_to_global`,
`get_kh2`, `is_root`, `nprocs`, the `topology`/`parallel_io` keywords, particle
halo exchange and migration, the loop macros, and the unused legacy
initializers (`init_analytical_psi!`, `init_random_waves!`, …).

## Verification

- Old (MPI) and new (KernelAbstractions) code run identical scenarios — every
  wave formulation, feedback mode, variable stratification, linear dynamics,
  total and YBJ vertical velocities, energies, seeded random flows, `nz = 1, 2`,
  and odd grid shapes — and agree to ≤ 7e-15 relative.
- `test/test_architectures.jl` runs the whole device code path on JLArrays
  (scalar indexing disabled) and requires it to reproduce the CPU model,
  including output, diagnostics, restart, and particles.
- The same scenarios on OpenCL.jl with the pocl CPU driver compile every
  kernel through GPUCompiler, the compiler CUDA.jl uses, and match as well.
- The suite passes on Julia 1.10, 1.11, and 1.13 and with four threads.

## Follow-ups

- Run on CUDA hardware: correctness is expected from the checks above, but
  performance has not been measured on a GPU.
- Column kernels walk `z`, the fastest-varying dimension, so neighbouring GPU
  threads access memory `nz` elements apart. An `(x, y, z)` layout would
  coalesce them.
- A step performs about ten small device-to-host reads (finite-state checks,
  the mean-PV compatibility warning, the scalar N² fallback).
