# Composition-First Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `Grid`/`State`/`QGParams` core and alias-based declarative facade with a real composition-first `RectilinearGrid`/`ModelFields`/`QGYBJModel`/`Simulation` architecture while preserving all v1.0.24 features and ETD-RK2 behavior.

**Architecture:** `RectilinearGrid` owns immutable geometry; typed physics and numerics components replace `QGParams`; `QGYBJModel` owns fields plus a runtime containing MPI, FFT, coefficient, and workspace resources; and `Simulation` separately owns clock, ETD-RK2 stepping, schedules, output, and lifecycle state. The finished tree contains no compatibility aliases or duplicate data model.

**Tech Stack:** Julia 1.10-1.12, MPI.jl, PencilArrays.jl, PencilFFTs.jl, FFTW.jl, NCDatasets.jl, JLD2.jl, Test stdlib, Documenter.jl.

---

### Task 1: Add breaking-architecture contract tests

**Files:**
- Create: `test/test_core_architecture.jl`
- Modify: `test/runtests.jl`

**Step 1: Write the failing type-identity tests**

Add tests that express the desired ownership rather than the current facade:

```julia
using Test
using QGYBJplus

@testset "Composition-first type identities" begin
    grid = RectilinearGrid(size=(8, 8, 4),
                           x=(-pi, pi), y=(-pi, pi), z=(-1.0, 0.0))
    @test nameof(typeof(grid)) == :RectilinearGrid
    @test !isdefined(QGYBJplus, :RectilinearGridSpec)

    model = QGYBJModel(grid=grid,
                       coriolis=FPlane(f=1.0),
                       stratification=ConstantStratification(N²=1.0),
                       closure=HorizontalHyperdiffusivity(
                           flow=FlowHyperdiffusivity(coefficient=0),
                           wave=WaveHyperdiffusivity(coefficient=0)),
                       verbose=false)
    @test nameof(typeof(model)) == :QGYBJModel
    @test model.grid === grid
    @test hasproperty(model, :fields)
    @test hasproperty(model, :physics)
    @test hasproperty(model, :runtime)

    simulation = Simulation(model; Δt=0.01, stop_iteration=1, output=false)
    @test nameof(typeof(simulation)) == :Simulation
    @test simulation.model === model
    @test typeof(model) !== typeof(simulation)
end
```

Do not include this file in `test/runtests.jl` yet. It spans the grid, fields,
model, runtime, and simulation ownership tasks and becomes part of the main
suite in Task 4 when the complete contract can turn green.

**Step 2: Run the contract test to verify it fails**

Run: `julia --project=. test/test_core_architecture.jl`

Expected: FAIL because `RectilinearGrid` returns `RectilinearGridSpec`, `QGYBJModel` aliases `Simulation`, and the requested ownership fields do not exist.

**Step 3: Keep the verified-red contract uncommitted**

Keep the verified-red contract untracked through Tasks 2 and 3. Do not commit a
knowingly broken branch; Task 4 adds it to `test/runtests.jl` and commits it with
the implementation that makes the complete contract green.

---

### Task 2: Implement typed components and the real `RectilinearGrid`

**Files:**
- Create: `src/core/components.jl`
- Create: `src/core/grid.jl`
- Create: `test/test_core_components.jl`
- Modify: `src/QGYBJplus.jl`
- Modify: `src/simulation.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing component and geometry tests**

Test constructor validation, centered/explicit origins, horizontal Fourier
collocation coordinates, vertical cell-center coordinates, wavenumber
convention, and typed constructor conversion:

```julia
@testset "RectilinearGrid geometry" begin
    grid = RectilinearGrid(size=(8, 6, 4),
                           x=(-4.0, 4.0), y=(-3.0, 3.0), z=(-2.0, 0.0))
    @test grid.size == (8, 6, 4)
    @test grid.extent == (8.0, 6.0, 2.0)
    @test grid.origin == (-4.0, -3.0)
    @test grid.x[1] == -4.0
    @test grid.z == [-1.75, -1.25, -0.75, -0.25]
    @test grid.kx[1] == 0
    @test_throws ArgumentError RectilinearGrid(size=(0, 8, 4), extent=(1, 1, 1))
end

@testset "Typed physics components" begin
    @test FPlane(f=1) isa FPlane{Float64}
    @test_throws ArgumentError FPlane(f=0)
    @test_throws ArgumentError ConstantStratification(N²=0)
    @test FixedFlow() isa FlowEvolution
    @test NoFeedback() isa FeedbackMode
    @test YBJPlus() isa WaveFormulation
end
```

**Step 2: Run the focused test and confirm RED**

Run: `julia --project=. test/test_core_components.jl`

Expected: FAIL on the missing concrete types and geometry fields.

**Step 3: Implement immutable component types**

In `src/core/components.jl`, define abstract families and concrete components:

```julia
abstract type AbstractCoriolis end
abstract type AbstractStratification end
abstract type FlowEvolution end
abstract type FeedbackMode end
abstract type WaveFormulation end

struct FPlane{T} <: AbstractCoriolis
    f::T
end
struct ConstantStratification{T} <: AbstractStratification
    N²::T
end
struct FixedFlow <: FlowEvolution end
struct EvolvingFlow <: FlowEvolution end
struct NoFeedback <: FeedbackMode end
struct WaveMeanFeedback <: FeedbackMode end
struct NoWaveFeedback <: FeedbackMode end
struct YBJPlus <: WaveFormulation end
struct YBJ <: WaveFormulation end
```

Move and retain validated `HorizontalHyperdiffusivity`, `SurfaceWave`, schedule,
and output specifications here. Add focused types for equation options and
vertical diffusion; do not recreate a catch-all parameter bag.

**Step 4: Implement `RectilinearGrid`**

In `src/core/grid.jl`, replace `RectilinearGridSpec` with an immutable
`RectilinearGrid` that computes physical coordinates, spacings, faces, global
`kx`, `ky`, and two-dimensional `kh2`. Remove the duplicate declarative grid
function and spec from `src/simulation.jl`.

**Step 5: Run focused and existing geometry tests**

Run: `julia --project=. test/test_core_components.jl`

Expected: PASS.

Run: `julia --project=. test/runtests.jl`

Expected: PASS. Update constructor annotations and call sites that previously
expected `RectilinearGridSpec`; do not leave the main branch test suite red.

**Step 6: Commit**

```bash
git add src/core src/QGYBJplus.jl src/simulation.jl test/test_core_components.jl test/runtests.jl
git commit -m "feat: add typed components and computational grid"
```

---

### Task 3: Replace `State` with `ModelFields`

**Files:**
- Create: `src/core/fields.jl`
- Create: `test/test_model_fields.jl`
- Modify: `src/grid.jl`
- Modify: every source file returned by `rg -l '\bState\b' src`
- Modify: affected tests under `test/`
- Modify: `src/QGYBJplus.jl`

**Step 1: Write failing field-allocation and copy tests**

Specify `ModelFields` array layout and independence of copied fields:

```julia
@testset "ModelFields" begin
    fields = ModelFields(Float64, (4, 8, 8))
    @test size(fields.q) == (4, 8, 8)
    @test eltype(fields.q) == ComplexF64
    @test eltype(fields.u) == Float64
    copied = copy_fields(fields)
    copied.q[1] = 1
    @test fields.q[1] == 0
end
```

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_fields.jl`

Expected: FAIL because `ModelFields` and `copy_fields` are undefined.

**Step 3: Move the field struct and allocation logic**

Define `ModelFields{T,RT,CT}` in `src/core/fields.jl`. Rename `init_state` to
`allocate_fields` and `copy_state` to `copy_fields`. Preserve `(z,x,y)` layout,
PencilArray support, and the exact prognostic/diagnostic field set.

**Step 4: Migrate all field signatures**

Mechanically replace `State` annotations with `ModelFields`, then update names
in transforms, elliptic, operators, nonlinear, timestep, initialization,
diagnostics, I/O, and particles. Do not add `const State = ModelFields`.

**Step 5: Run focused and full tests**

Run: `julia --project=. test/test_model_fields.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS with no exported or defined `State`.

**Step 6: Commit**

```bash
git add src test
git commit -m "refactor: replace State with ModelFields"
```

---

### Task 4: Introduce model physics, numerics, and runtime ownership

**Files:**
- Create: `src/core/model.jl`
- Create: `src/core/runtime.jl`
- Create: `test/test_model_ownership.jl`
- Modify: `src/parallel_mpi.jl`
- Modify: `src/transforms.jl`
- Modify: `src/runtime.jl`
- Modify: `src/simulation.jl`
- Modify: `src/QGYBJplus.jl`

**Step 1: Write failing ownership and MPI-ownership tests**

Test that model construction preserves the public grid object, allocates fields,
creates runtime resources, and records whether it initialized MPI:

```julia
@testset "QGYBJModel ownership" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2pi, 2pi, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1),
                       stratification=ConstantStratification(N²=1),
                       closure=HorizontalHyperdiffusivity(
                           flow=FlowHyperdiffusivity(coefficient=0),
                           wave=WaveHyperdiffusivity(coefficient=0)),
                       verbose=false)
    @test model.grid === grid
    @test model.fields isa ModelFields
    @test model.runtime.plans !== nothing
    @test model.runtime.dealias_mask !== nothing
    @test model.physics.coriolis.f == 1
    @test !hasproperty(model, :params)
    finalize_model!(model)
end
```

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_ownership.jl`

Expected: FAIL because `QGYBJModel` is still an alias and owns legacy fields.

**Step 3: Define focused aggregates**

Implement `ModelPhysics`, `ModelNumerics`, `OperatorCoefficients`,
`ModelRuntime`, and mutable `QGYBJModel`. Constructor boundary helpers convert
`:fixed`, `:evolving`, feedback symbols, and `ybj_plus` into typed components.

`ModelRuntime` must own `MPIConfig`, decomposition/local spectral metadata,
plans, transform destinations, masks, elliptic coefficients, and reusable
workspaces. It records `owns_mpi::Bool`.

**Step 4: Refactor MPI/runtime builders**

Replace `init_mpi_grid`, `init_mpi_state`, `init_mpi_workspace`, and
`plan_mpi_transforms` orchestration with `build_runtime(grid, physics,
architecture)` and `allocate_fields(grid, runtime)`. Low-level decomposition
helpers may remain public only if they operate on runtime components.

**Step 5: Implement transactional cleanup**

If model construction fails after MPI initialization, release owned resources.
`finalize_model!` is idempotent and never finalizes externally initialized MPI.

**Step 6: Run tests**

Run: `julia --project=. test/test_model_ownership.jl`

Run: `julia --project=. test/test_mpi_extension.jl`

Expected: PASS.

**Step 7: Add the architecture contract to the main suite**

Include `test_core_architecture.jl` from `test/runtests.jl`, run it directly,
and then run the full suite. Both must pass now that grid, fields, model,
runtime, and simulation have distinct identities.

**Step 8: Commit**

```bash
git add src test/test_core_architecture.jl test/test_model_ownership.jl test/runtests.jl
git commit -m "feat: make QGYBJModel own runtime and fields"
```

---

### Task 5: Migrate grid and transform consumers to model/runtime components

**Files:**
- Modify: `src/grid.jl`
- Modify: `src/transforms.jl`
- Modify: `src/parallel_mpi.jl`
- Modify: `src/loop_macros.jl`
- Modify: `test/test_mpi_extension.jl`
- Modify: `test/test_mpi_stepping_regression.jl`

**Step 1: Add failing serial/distributed metadata-equivalence tests**

For the same `RectilinearGrid`, assert identical global coordinates and
wavenumbers at one and multiple ranks, correct local ranges, and round-trip FFT
normalization through `model.runtime`.

**Step 2: Verify RED under one and two ranks**

Run: `julia --project=. test/test_mpi_extension.jl`

Run: `julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=. test/test_mpi_extension.jl`)'`

Expected: FAIL where helpers still access `grid.decomp` or distributed
`grid.kh2`.

**Step 3: Move decomposition-dependent data into runtime**

Change `get_local_range*`, local/global mapping, `get_kh2`, allocation, and
transpose helpers to accept `ModelRuntime` or `QGYBJModel`. Keep only global
geometry on `RectilinearGrid`.

**Step 4: Migrate loop macros**

Make local spectral loops derive local ranges and masks from `model.runtime`.
Do not reintroduce decomposition fields on the grid.

**Step 5: Run serial and MPI tests**

Run the one- and two-rank commands above. Expected: PASS.

**Step 6: Commit**

```bash
git add src/grid.jl src/transforms.jl src/parallel_mpi.jl src/loop_macros.jl test
git commit -m "refactor: move distributed metadata into model runtime"
```

---

### Task 6: Remove `QGParams` from physics, elliptic, and velocity operators

**Files:**
- Modify: `src/physics.jl`
- Modify: `src/elliptic.jl`
- Modify: `src/operators.jl`
- Modify: `src/runtime.jl`
- Create: `test/test_model_operators.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing model-level operator regression tests**

Port the existing q-to-psi, B-to-A, velocity, variable-N², and vertical
velocity tests to model-level calls:

```julia
invert_q_to_psi!(model)
invert_B_to_A!(model)
compute_velocities!(model)
```

Assert the same deterministic values/tolerances as v1.0.24.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_operators.jl`

Expected: FAIL because only legacy multi-argument methods exist.

**Step 3: Refactor coefficient and solver APIs**

Move N² profiles, `a_ell`, density profiles, and solver coefficients into
typed stratification/runtime components. Implement model-level methods and
narrow internal methods that receive fields, grid, runtime, and relevant
physics components. Remove every `QGParams` signature in these files.

**Step 4: Remove grid/physical duplication**

Read dimensions and spacings from `model.grid`, Coriolis and stratification
from `model.physics`, and local spectral data/workspaces from `model.runtime`.

**Step 5: Run focused and full tests**

Run: `julia --project=. test/test_model_operators.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS.

**Step 6: Commit**

```bash
git add src test
git commit -m "refactor: migrate physical operators to QGYBJModel"
```

---

### Task 7: Migrate nonlinear dynamics and ETD-RK2

**Files:**
- Modify: `src/nonlinear.jl`
- Modify: `src/timestep.jl`
- Modify: `src/ybj_normal.jl`
- Create: `test/test_model_etdrk2.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing model-level ETD-RK2 tests**

Port the coefficient-near-zero, exact hyperdiffusion, feedback preservation,
YBJ/YBJ+, A-initialization, workspace reuse, and deterministic trajectory tests
to:

```julia
step!(model, ExponentialRungeKutta2(Δt=0.1))
```

The tests must assert no legacy timestepper symbols are defined.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_etdrk2.jl`

Expected: FAIL because `step!` and the typed timestepper are not wired.

**Step 3: Refactor RHS and diffusion access**

Make nonlinear and refraction routines read typed flow, feedback, formulation,
dispersion, passive, inviscid, linear, and closure components. Replace boolean
reads from `QGParams` with dispatch or focused predicates.

**Step 4: Move ETD workspace ownership**

Rename `ExpRK2Workspace` to `ExponentialRungeKutta2Workspace` internally and
store it in the simulation/model runtime as appropriate. Implement `step!`
without accepting independent state/grid/params/plans/mask/workspace arguments.

**Step 5: Run tests**

Run: `julia --project=. test/test_model_etdrk2.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS with unchanged numerical tolerances.

**Step 6: Commit**

```bash
git add src test
git commit -m "refactor: make ETD-RK2 advance QGYBJModel"
```

---

### Task 8: Migrate initialization and stratification

**Files:**
- Modify: `src/initconds.jl`
- Modify: `src/initialization.jl`
- Modify: `src/stratification.jl`
- Modify: `src/simulation.jl`
- Create: `test/test_model_initialization.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing public `set!` tests**

Cover analytical/random ψ, barotropic and QG PV construction, Gaussian and
exponential surface waves, vertical wave packets, direct/file-backed fields,
all stratification variants, and invalid profile arguments.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_initialization.jl`

Expected: FAIL where initialization still expects legacy state/grid/params.

**Step 3: Implement model-owned initialization**

Make `set!` dispatch on initial-condition objects, write local physical slabs
through `model.runtime`, transform into `model.fields`, and establish dependent
q/A/velocity consistency. Move computed N² and density profiles into the typed
stratification/runtime data.

**Step 4: Run tests**

Run: `julia --project=. test/test_model_initialization.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS.

**Step 5: Commit**

```bash
git add src test
git commit -m "refactor: initialize fields through QGYBJModel"
```

---

### Task 9: Separate `Simulation`, clock, schedules, and lifecycle

**Files:**
- Create: `src/core/simulation.jl`
- Create: `test/test_simulation_lifecycle.jl`
- Modify: `src/simulation.jl`
- Modify: `src/model_interface.jl`
- Modify: `src/QGYBJplus.jl`
- Modify: `test/runtests.jl`

**Step 1: Write failing separation and lifecycle tests**

Test distinct model/simulation identities, clock advancement, stop-time
crossing, iteration schedules, state transitions, reentrant-run rejection,
post-finalization rejection, and idempotent finalization.

```julia
simulation = Simulation(model; Δt=0.1, stop_iteration=2, output=false)
@test simulation.clock.time == 0
run!(simulation)
@test simulation.clock.iteration == 2
@test simulation.clock.time == 0.2
@test simulation.state == Stopped
finalize_simulation!(simulation)
finalize_simulation!(simulation)
@test_throws InvalidStateException run!(simulation)
```

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_simulation_lifecycle.jl`

Expected: FAIL because `Simulation(model)` mutates and returns the model.

**Step 3: Implement the orchestration types**

Define `Clock`, lifecycle markers/state, schedule bookkeeping, writer handles,
and a distinct mutable `Simulation{M,...}`. Move `Δt`, stop criteria,
diagnostics interval, output config, progress, and ETD workspace out of the
model.

**Step 4: Replace run drivers**

Consolidate `run!` and the useful parts of `run_simulation!` into the new
simulation loop. Remove `QGYBJSimulation`, `initialize_simulation`, and facade
mutation paths rather than aliasing them.

**Step 5: Run tests**

Run: `julia --project=. test/test_simulation_lifecycle.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS.

**Step 6: Commit**

```bash
git add src test
git commit -m "feat: separate Simulation orchestration from model"
```

---

### Task 10: Migrate NetCDF, restart, and diagnostics ownership

**Files:**
- Modify: `src/netcdf_io.jl`
- Modify: `src/diagnostics.jl`
- Modify: `src/energy_diagnostics.jl`
- Modify: `src/core/simulation.jl`
- Create: `test/test_model_io.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing I/O round-trip tests**

Use temporary directories to test initial/scheduled/final state files, selected
field sets, velocities, variable N² metadata, energy diagnostics, restart
round trips, and writer cleanup after a forced error.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_io.jl`

Expected: FAIL where managers require `QGParams` or legacy state/grid tuples.

**Step 3: Refactor I/O APIs**

Make writers consume `simulation.model` and `simulation.clock`. Derive schema
metadata from typed grid/physics components. Store manager/writer state on the
simulation and close it through guaranteed cleanup.

**Step 4: Refactor diagnostics**

Make energy and field diagnostics accept `QGYBJModel`; use model runtime for
collectives and transforms. Preserve file variable names and numerical
normalization.

**Step 5: Run tests**

Run: `julia --project=. test/test_model_io.jl`

Run: `julia --project=. test/runtests.jl`

Expected: PASS.

**Step 6: Commit**

```bash
git add src test
git commit -m "refactor: make simulation own output and diagnostics"
```

---

### Task 11: Migrate particle state and MPI migration

**Files:**
- Modify: `src/particles/particle_config.jl`
- Modify: `src/particles/halo_exchange.jl`
- Modify: `src/particles/interpolation_schemes.jl`
- Modify: `src/particles/particle_advection.jl`
- Modify: `src/particles/particle_io.jl`
- Create: `test/test_model_particles.jl`
- Modify: `test/test_mpi_particles_periodic.jl`
- Modify: `test/runtests.jl`

**Step 1: Add failing model-owned particle tests**

Test optional particle ownership on `QGYBJModel`, all initialization
distributions, interpolation methods, Euler/RK2/RK4 advection, periodic
migration, output scheduling, and serial/MPI equivalence.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_model_particles.jl`

Expected: FAIL because particle trackers still own legacy grids separately.

**Step 3: Refactor particle access**

Make particle code use `model.grid`, `model.fields`, and `model.runtime`.
Particle positions/state remain model state; particle-output managers and
schedules belong to `Simulation`.

**Step 4: Run serial and MPI tests**

Run: `julia --project=. test/test_model_particles.jl`

Run: `julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=. test/test_mpi_particles_periodic.jl`)'`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/particles test
git commit -m "refactor: make particles model-owned state"
```

---

### Task 12: Delete the legacy data model and clean exports

**Files:**
- Delete or empty after moving content: `src/parameters.jl`
- Delete or reduce to non-type helpers: `src/grid.jl`
- Delete: `src/config.jl`
- Delete or replace: `src/model_interface.jl`
- Modify: `src/QGYBJplus.jl`
- Modify: `src/pretty_printing.jl`
- Modify: all tests under `test/`

**Step 1: Add a failing legacy-symbol audit**

Create a test that asserts the module does not define/export `Grid`, `State`,
`QGParams`, `RectilinearGridSpec`, `QGYBJSimulation`, `default_params`,
`setup_model`, `initialize_simulation`, or configuration-builder APIs. Also
scan source for forbidden type annotations.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_core_architecture.jl`

Expected: FAIL until all old symbols and annotations are removed.

**Step 3: Remove legacy types and paths**

Move any still-useful helper into the owning core/operator module, delete old
config/builders, collapse exports to the new declarative API plus meaningful
model-level diagnostics/particle tools, and update pretty printing.

**Step 4: Verify the source audit**

Run: `rg -n '\b(Grid|State|QGParams|RectilinearGridSpec|QGYBJSimulation)\b' src`

Expected: no legacy definitions or annotations; documentation strings that
describe their removal should also be cleaned before completion.

**Step 5: Run the full suite**

Run: `julia --project=. test/runtests.jl`

Expected: PASS using only the new core.

**Step 6: Commit**

```bash
git add -A src test
git commit -m "refactor!: remove legacy core and configuration APIs"
```

---

### Task 13: Update documentation and examples

**Files:**
- Modify: `README.md`
- Modify: `examples/asselin_jpo2020.jl` only if required for corrected runtime semantics
- Modify: `examples/compute_energy.jl` if schema/API references changed
- Modify: all Markdown files under `docs/src/`
- Modify: `docs/make.jl`
- Create: `test/test_asselin_smoke.jl`
- Modify: `test/runtests.jl`

**Step 1: Add a failing reduced Asselin smoke test**

Refactor the example entry point only as needed to accept small keyword/env
overrides without changing its declarative core. Run 4x4x2 for one ETD-RK2
step in a temporary output directory and assert a state file is written.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_asselin_smoke.jl`

Expected: FAIL until the example is testable against the distinct model and
simulation types.

**Step 3: Rewrite documentation**

Remove legacy APIs from guides and API pages. Document ownership, typed
components, ETD-RK2-only stepping, lifecycle, MPI ownership, output, restart,
diagnostics, and particles. Keep the Asselin script as the primary worked API.

**Step 4: Run smoke and documentation build**

Run: `julia --project=. test/test_asselin_smoke.jl`

Run: `julia --project=docs docs/make.jl`

Expected: PASS with no missing docstrings or removed-symbol references.

**Step 5: Commit**

```bash
git add README.md examples docs test
git commit -m "docs: document composition-first model API"
```

---

### Task 14: Complete serial, MPI, allocation, and acceptance verification

**Files:**
- Modify only if a verification failure has a new failing regression test

**Step 1: Run formatting and source checks**

Run: `git diff --check`

Run: `rg -n '\b(Grid|State|QGParams|RectilinearGridSpec|QGYBJSimulation)\b' src docs/src README.md`

Expected: both clean, with no legacy references.

**Step 2: Run the complete serial test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: PASS on the current Julia version.

**Step 3: Run MPI stepping at one, two, and four ranks**

Run: `julia --project=. -e 'using MPI; for n in (1, 2, 4); run(`$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --project=. test/test_mpi_stepping_regression.jl`); end'`

Expected: PASS and serial-equivalent norms at every rank count.

**Step 4: Run MPI particles**

Run: `julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=. test/test_mpi_particles_periodic.jl`)'`

Expected: PASS with particle conservation and periodic migration.

**Step 5: Run allocation/workspace assertions**

Run the allocation testsets from `test/test_mpi_stepping_regression.jl` at the
CI-enabled settings. Expected: no regression beyond the documented budget.

**Step 6: Run the reduced Asselin acceptance test**

Run: `julia --project=. test/test_asselin_smoke.jl`

Expected: PASS, ETD-RK2 advances one step, and NetCDF output exists.

**Step 7: Review final exports and diff**

Inspect `src/QGYBJplus.jl`, `git diff --stat`, and `git diff --check`. Confirm
the public example contract, full-parity matrix, and clean-breaking acceptance
criteria from the design document.

**Step 8: Commit any verification-only fixes and request review**

Each fix must begin with a failing regression and receive its own commit. Then
use `superpowers:requesting-code-review` and
`superpowers:verification-before-completion` before integration.
