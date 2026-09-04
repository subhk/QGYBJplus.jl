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
`N2`, `N2_face`, and `a_ell`. `N2` is sampled at cell centers on `z`;
`N2_face` and `a_ell = f0^2 / N2_face` are sampled at each cell's upper face
on `z_face = grid.z_faces[2:end]`. The `Lx`, `Ly`, and `Lz` attributes record
the domain extent.
`feedback_mode` records the configured feedback component, while
`generalized_pv` distinguishes balanced-only PV from total PV containing the
wave contribution.

Selecting `:ψ` adds physical `psi`. Selecting `:waves` adds physical
`A_real`, `A_imag`, `LA_real`, and `LA_imag`, where
``L A = \partial_z[(f_0^2/N^2)\partial_z A]``. `velocities=true` adds `u`,
`v`, and `w`.

Array dimensions in NetCDF are `(x, y, z)`; model arrays use `(z, x, y)`.
The `wave_formulation` attribute identifies the meaning of `B`: for
`YBJPlus`, ``B=L^+A``; for `YBJ`, ``B=LA``.

## Offline Fourier analysis

Construct the same grid used by the simulation to obtain wavenumbers in
FFTW order:

```julia
using QGYBJplus
using NCDatasets
using FFTW

grid = RectilinearGrid(size=(256, 256, 128),
                       extent=(70e3, 70e3, 3e3), centered=true)
kx, ky = grid.kx, grid.ky

NCDataset("output/state0001.nc", "r") do ds
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

## Restart

Build a compatible model, then restore the prognostic arrays:

```julia
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5))
restore!(model, "output/state0011.nc")
simulation = Simulation(model; Δt=10.0, stop_iteration=100)
```

The restart dimensions, wave formulation, and generalized-PV convention must
match the receiving model. New snapshots validate these attributes before
loading; older snapshots without the metadata remain readable. Diagnostic
arrays are reconstructed after the distributed scatter.

## Failure behavior

Output exceptions move the simulation to `Failed`, close the state,
diagnostics, and particle managers, and are rethrown on every rank. Use
`try`/`finally` to guarantee cleanup:

```julia
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```
