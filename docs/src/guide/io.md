# [I/O and restart](@id io-output)

```@meta
CurrentModule = QGYBJplus
```

## Scheduled NetCDF snapshots

```julia
output = NetCDFOutput(
    path="output",
    schedule=IterationInterval(100),
    fields=(:ψ, :waves),
    velocities=true,
)
simulation = Simulation(model; Δt=10.0, stop_iteration=1000,
                        output=output)
```

`TimeInterval(seconds)` provides time-based scheduling instead. The initial
condition is written when output is enabled, and finalization writes the last
iteration if it was not already scheduled.

Time schedules keep a nominal next deadline. If a step crosses a deadline,
one snapshot is written at that step and the next deadline advances from the
schedule rather than from the late write, avoiding cumulative drift.

A run never silently replaces files already in its output directory. A
simulation restored from a state file (see [Restart](@ref io-restart))
continues the numbering after the files up to its restart iteration. Any
other existing state files raise an error, unless the output is configured
with `NetCDFOutput(...; overwrite=true)`, which removes them first: all of
them for a run that starts at iteration 0, and only those after the restart
iteration for a restored run.

## Energy diagnostics

Passing a schedule as `diagnostics` records initial, scheduled, and final
energy values in `diagnostic/` beneath the Eulerian output directory:

```julia
simulation = Simulation(
    model;
    output=NetCDFOutput(path="output"),
    diagnostics=IterationInterval(10),
)
```

The directory contains `wave_KE.nc`, `wave_PE.nc`, `wave_CE.nc`,
`mean_flow_KE.nc`, `mean_flow_PE.nc`, and `total_energy.nc`. Configure an
independent location with
`EnergyDiagnosticsOutput(path="energies", schedule=TimeInterval(600.0))`.
The files are rewritten during the run, at most every ten seconds of wall
time, and once more when it ends, so a run that is stopped early keeps its
series. A restored simulation appends to the series already there; other
existing series need `overwrite=true` (which a schedule passed as
`diagnostics` takes from the `NetCDFOutput`).
The summary file preserves the component series together with
`total_wave_energy`, `total_flow_energy`, `coupled_energy`, and `total_energy`.
`coupled_energy = total_flow_energy + wave_PE + wave_CE` is the
wave--mean-feedback conservation diagnostic of Asselin & Young (2019), equation
(3.7), for the ideal inviscid coupled system; it excludes the informational
`wave_KE` series. The existing
`total_energy` remains the sum of every reported flow and wave component.

## Snapshot schema

Every file contains `x`, `y`, `z`, `z_face`, `time`, iteration metadata,
spectral `q_hat_real`, `q_hat_imag`, `B_hat_real`, and `B_hat_imag`, plus
`N2`, `N2_face`, and `a_ell`. The spectral fields are the unnormalised
horizontal FFTs of q and B, in FFTW order, on the dimensions
`(kx_index, ky_index, z)`; the variables `kx` and `ky` hold the wavenumbers.
Every variable has a `long_name`. The model has no fixed units, so the files
carry none: the `units_convention` attribute records that every quantity is
in the units of the model configuration. `N2` is sampled at cell centers on `z`;
`N2_face` and `a_ell = f0^2 / N2_face` are sampled at each cell's upper face
on `z_face = grid.z_faces[2:end]`. The `Lx`, `Ly`, and `Lz` attributes record
the domain extent.
`feedback_mode` records the configured feedback component, while
`generalized_pv` distinguishes balanced-only PV from total PV containing the
wave contribution.

Selecting `:ψ` adds physical `psi`. Selecting `:waves` adds physical
`A_real`, `A_imag`, `LA_real`, and `LA_imag`, where
``L A = \partial_z[(f_0^2/N^2)\partial_z A]``. `velocities=true` adds `u`,
`v`, and `w`, all at cell centres on `z`; `w` is the QG omega-equation
velocity, solved on the faces and averaged onto the centres.

Array dimensions in NetCDF are `(x, y, z)`; model arrays use `(z, x, y)`.
The `wave_formulation` attribute identifies the meaning of `B`: for
`YBJPlus`, ``B=L^+A``; for `YBJ`, ``B=LA``.

## Offline Fourier analysis

Each file stores the wavenumbers of its Fourier coefficients, in FFTW order:

```julia
using NCDatasets
using FFTW

NCDataset("output/state0001.nc", "r") do ds
    kx, ky = ds["kx"][:], ds["ky"][:]
    A = ds["A_real"][:, :, :] .+ im .* ds["A_imag"][:, :, :]
    LA = ds["LA_real"][:, :, :] .+ im .* ds["LA_imag"][:, :, :]

    forward = FFTW.plan_fft(A, (1, 2))
    backward = FFTW.plan_ifft(A, (1, 2))
    A_hat = forward * A
    LA_hat = forward * LA
    kh² = reshape(kx .^ 2, :, 1, 1) .+
          reshape(ky .^ 2, 1, :, 1)
    LplusA_hat = LA_hat .- 0.25 .* kh² .* A_hat
    LplusA = backward * LplusA_hat
end
```

For a YBJ⁺ snapshot, `LplusA_hat` agrees with
`B_hat_real + im * B_hat_imag`. The FFT dimensions are `(1, 2)` because
NetCDF fields use `(x, y, z)` order.

## [Restart](@id io-restart)

Build a compatible model and simulation, then restore the prognostic arrays:

```julia
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))
simulation = Simulation(model; Δt=10.0, stop_iteration=200,
                        output=NetCDFOutput(path="output",
                                            schedule=IterationInterval(10)))
restore!(simulation, "output/state0011.nc")   # iteration 100
run!(simulation)                               # iterations 101 to 200
```

`restore!(simulation, path)` also sets the simulation clock to the file's
time and iteration, so `stop_time` and `stop_iteration` count from the start
of the original run, and output, energy diagnostics, and particles continue
the original time axis. Continuing into the same directory appends
`state0012.nc` onward without writing the restart state again. Restarting
from an earlier file of a longer run needs `overwrite=true`, which replaces
the later files. `restore!(model, path)` restores the fields only.

The restart dimensions, domain extent, wave formulation, and generalized-PV
convention must match the receiving model; a different `f0` or `N²` profile
is allowed but warns, since it changes the flow inverted from q. Older
snapshots without the metadata remain readable. Diagnostic arrays are
reconstructed after the prognostic fields are copied onto the model's
architecture. Particle positions are not stored in state files, so particles
start new trajectories after a restart.

## Failure behavior

Output exceptions move the simulation to `Failed`, close the state,
diagnostics, and particle managers, and are rethrown. Use
`try`/`finally` to guarantee cleanup:

```julia
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```
