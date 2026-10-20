# [Performance](@id performance)

```@meta
CurrentModule = QGYBJplus
```

## Choose an architecture

A GPU is the fastest way to run large grids: build the grid with
`RectilinearGrid(GPU(); ...)` after `using CUDA`. On a CPU, start Julia with
several threads (`julia --threads=auto`) so that the KernelAbstractions
kernels run in parallel, and call `FFTW.set_num_threads` before building the
model to thread the transforms too. See [CPU and GPU execution](@ref
architectures).

Every model array lives in the GPU's memory. A running model holds about 16
complex `(nz, nx, ny)` arrays' worth of storage, roughly 260 bytes per grid
point, or about 2.2 GB for 256×256×128:

| Storage | Arrays |
|:--|:--|
| Fields: `q`, `B`, `ψ`, `A`, `A_z`, and the real `u`, `v`, `w` | 6.5 |
| ETD-RK2 workspace: stage `q` and `B`, and their first-stage tendencies | 4 |
| Scratch pool for transforms and products (high-water mark) | 5 |
| Tridiagonal solver scratch (real) | 0.5 |

Output, energy diagnostics, and particle velocities reuse the scratch pool.
The exceptions are cuFFT, whose plans need work space of their own, and
energy diagnostics with the original `YBJ()` formulation, which keep one copy
of the fields (6.5 arrays). Check that this fits the device before choosing a
resolution.

## Single precision

The grid's floating-point type sets the precision of every model array:

```julia
grid = RectilinearGrid(GPU(), Float32; size=(512, 512, 128), extent=(500e3, 500e3, 4000.0))
```

`Float32` halves the memory above, to about 130 bytes per grid point, and runs
much faster on GPUs with reduced double-precision throughput, which includes
most consumer cards. Single precision carries about seven significant digits.
In the test suite a coupled run tracks the `Float64` run to about `1e-4`
relative error over a few steps, and round-off grows over long integrations,
so keep `Float64` for conservation checks and comparisons of small
differences. Physical parameters are given as usual and converted, and output
files and energy series are written in `Float64` either way. Times stay in
`Float64` too: the simulation clock, `Δt`, output schedules, and particle
times, since a `Float32` clock no longer resolves the step after months of
simulated time. Particle positions have their own precision, `Float64` by
default (see [Particle advection](@ref particles)).

## Reuse the simulation timestepper

`Simulation` reuses one [`ExponentialRungeKutta2Workspace`](@ref) after the
first step. For manual field-only stepping, reuse the timestepper object:

```julia
timestepper = ExponentialRungeKutta2(Δt=10.0)
for _ in 1:100
    step!(model, timestepper)
end
```

Constructing a new timestepper inside the loop defeats workspace reuse.
Manual stepping does not advance particles, output, diagnostics, or a clock.

## Avoid host transfers in the time loop

- Avoid copying full fields to the host (`Array(model.fields.q)`) during the
  time loop.
- Write only the fields required for analysis.
- Measure with output disabled before diagnosing solver performance.
- Particles are advanced on the model's architecture, in one kernel per step
  over all particles, so they add no host transfer to the time loop; only
  particle output and the in-memory history copy positions to the host. The
  tracker keeps the horizontal velocity at both ends of the particle step:
  four grid-sized real arrays in the particles' precision on that
  architecture, or eight with `wave_motion = :instantaneous`.
- Call `finalize_simulation!` when a simulation is done. It drops the
  timestepper workspace and any diagnostics snapshot, so the garbage
  collector can return their device memory, and keeps the model for a later
  simulation. `finalize_model!(model)` releases the model's scratch buffers
  once it is no longer needed.

## Limit output

Writing physical `ψ`, waves, and velocities requires inverse transforms and a
copy to the host. Select only the fields required for analysis:

```julia
NetCDFOutput(path="output",
             schedule=IterationInterval(100),
             fields=(:waves,),
             velocities=false)
```

## Benchmark reproducibly

Warm up compilation before measuring: the first step compiles every kernel.
On a GPU, synchronize before reading the clock (`CUDA.@sync`) so that queued
kernels are included. Report resolution, architecture (CPU threads or GPU
model), Julia version, and output settings with every result.
