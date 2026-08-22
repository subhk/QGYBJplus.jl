# Composition-First Core Redesign

## Status

Approved on 2026-08-21. This design targets a clean breaking release based on
the fetched v1.0.24 codebase. The public workflow in
`examples/asselin_jpo2020.jl` is the primary API contract.

## Context

QGYBJplus currently exposes an Oceananigans-style declarative API, but the API
is a facade over the older core:

- `RectilinearGrid` constructs a `RectilinearGridSpec` rather than the
  computational grid.
- `QGYBJModel` is an alias for the current `Simulation` container.
- The model still owns a `Grid`, `State`, and `QGParams` parameter bag.
- Numerical kernels receive long argument lists containing the grid, state,
  parameters, FFT plans, masks, profiles, and workspaces separately.

The redesign makes the declarative objects the real internal architecture. It
does not retain legacy constructors, aliases, or compatibility wrappers.

The current production stepper is second-order exponential Runge--Kutta
(ETD-RK2). Leapfrog and IMEX-CN have already been removed and are not part of
this design.

## Goals

1. Replace `Grid` with `RectilinearGrid` throughout the implementation.
2. Replace `State` with model-owned `ModelFields`.
3. Replace `QGParams` with small typed physics and numerics components.
4. Make `QGYBJModel` the owner of fields, physics, transforms, MPI resources,
   coefficients, masks, and workspaces.
5. Make `Simulation` a distinct orchestration object that owns the clock,
   ETD-RK2 timestepper, stopping criteria, schedules, outputs, diagnostics, and
   callbacks.
6. Preserve all current numerical, MPI, I/O, diagnostic, stratification, and
   particle capabilities through the new core.
7. Keep the declarative Asselin example source-compatible and runnable.

## Non-goals

- Preserving `default_params`, `setup_model`, `initialize_simulation`,
  `QGYBJSimulation`, configuration builders, or direct low-level legacy APIs.
- Reintroducing leapfrog, Robert--Asselin filtering, or IMEX-CN.
- Adding new physical equations or discretizations during the migration.
- Running the production-sized Asselin case in CI.

## Public API contract

The intended workflow is:

```julia
grid = RectilinearGrid(size=(nx, ny, nz),
                       x=(-Lx / 2, Lx / 2),
                       y=(-Ly / 2, Ly / 2),
                       z=(-Lz, 0))

model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=f0),
                   stratification=ConstantStratification(N²=N2),
                   closure=HorizontalHyperdiffusivity(...),
                   flow=:fixed,
                   feedback=:none,
                   ybj_plus=true)

set!(model; ψ=streamfunction,
            pv_method=:barotropic,
            waves=SurfaceWave(...))

simulation = Simulation(model;
                        Δt=dt,
                        stop_time=duration,
                        output=NetCDFOutput(...),
                        diagnostics=IterationInterval(...))

run!(simulation)
finalize_simulation!(simulation)
```

Convenience symbols and booleans are accepted only at public constructor
boundaries. They are immediately converted into typed internal components.

## Ownership model

### `RectilinearGrid`

`RectilinearGrid` is the only grid type. It is immutable and owns global
geometry and discretization metadata:

- resolution, extents, origin, and topology;
- physical coordinates and grid spacing;
- vertical cell and face geometry;
- global horizontal wavenumbers.

It does not own MPI communicators, decompositions, FFT plans, fields, or model
physics. This keeps grid construction independent of runtime-resource lifetime.

### Physics and numerics components

The `QGParams` bag is replaced by focused immutable components:

- `FPlane` and future Coriolis models;
- `ConstantStratification`, `SkewedGaussianStratification`,
  `TanhStratification`, `AnalyticalStratification`, and
  `TabulatedStratification`;
- `HorizontalHyperdiffusivity`, composed from explicit
  `FlowHyperdiffusivity` and `WaveHyperdiffusivity` components, and vertical
  diffusion configuration;
- `FixedFlow` or `EvolvingFlow`;
- `NoFeedback`, `WaveMeanFeedback`, or `NoWaveFeedback`;
- `YBJPlus`, `YBJ`, `PassiveWave`, and dispersion/nonlinearity options;
- `ExponentialRungeKutta2` timestepper configuration.

Invalid combinations are rejected during model construction rather than
interpreted repeatedly through boolean branches in numerical kernels.

### `ModelFields`

`ModelFields` replaces `State` and owns prognostic and diagnostic arrays:

- prognostic `q` and `B`;
- diagnostic `ψ`, `A`, and `C`;
- physical velocities `u`, `v`, and `w`;
- any stage or auxiliary fields that are logically model state.

Fields may be ordinary arrays or distributed pencil arrays. Code dispatches on
the architecture/runtime, not on a second state abstraction.

### `ModelRuntime`

`ModelRuntime` owns ephemeral computational resources:

- serial or distributed architecture and explicit MPI ownership;
- pencil decompositions and local/global index mappings;
- FFT plans and reusable transform destinations;
- dealiasing masks and stratification/elliptic coefficients;
- nonlinear, elliptic, ETD-RK2, diagnostic, and particle workspaces.

The runtime is constructed transactionally by `QGYBJModel` after all component
validation succeeds.

### `QGYBJModel`

`QGYBJModel` owns the grid, typed physics/numerics components, model fields,
runtime, stratification data, and optional particle state. Numerical kernels
receive a model or the narrow component they operate on. They no longer accept
independent `Grid`, `State`, `QGParams`, plans, masks, profiles, and workspaces
as unrelated positional arguments.

### `Simulation`

`Simulation` is not an alias for the model. It owns:

- a `QGYBJModel` reference;
- a `Clock` containing time and iteration;
- the ETD-RK2 timestepper and step size;
- stop-time and stop-iteration criteria;
- output, diagnostics, particle-output, and callback schedules;
- writer/manager lifecycle state and progress configuration.

## Data flow

1. `RectilinearGrid` validates geometry and computes immutable metadata.
2. `QGYBJModel` validates typed components, establishes the architecture,
   creates MPI/FFT resources, allocates fields, and precomputes coefficients.
3. `set!` initializes physical or spectral fields and immediately establishes
   required diagnostic consistency (`ψ -> q`, `B -> A`, velocities, and
   particle placement).
4. `Simulation` creates the clock, schedules, writers, and ETD-RK2 workspace.
5. `run!` advances ETD-RK2 stages, updates the clock, evaluates collective
   termination conditions, and executes scheduled work.
6. `finalize_simulation!` closes writers and releases resources owned by the
   model/runtime.

## Lifecycle and errors

Simulation lifecycle states are:

```
constructed -> initialized -> running -> stopped -> finalized
```

- `set!` is rejected while running and after finalization.
- `run!` is rejected after finalization and while another run is active.
- Finalization is idempotent.
- MPI ownership is explicit. A runtime finalizes MPI only if it initialized
  MPI; externally initialized MPI remains externally owned.
- Constructor validation occurs before distributed allocation and collectives.
- Distributed non-finite and termination checks use reductions so all ranks
  make the same decision.
- Output errors are synchronized before subsequent collectives.
- Time schedules trigger when a clock step crosses a target; iteration
  schedules trigger on exact multiples.
- Fixed-step runs stop at the first step reaching or exceeding `stop_time` and
  report any sub-step overshoot.

## Feature parity

The breaking release must retain all capabilities present in v1.0.24:

- ETD-RK2 stepping and exact horizontal-hyperdiffusion integration;
- YBJ and YBJ+ branches;
- fixed and evolving balanced flow;
- feedback, no-feedback, and no-wave-feedback modes;
- passive, linear, inviscid, and no-dispersion configurations;
- constant, skewed-Gaussian, tanh, analytical, tabulated, and file-backed
  stratification;
- serial/single-rank and multi-rank pencil decompositions;
- NetCDF state output, diagnostics, energy output, and restart input;
- particle initialization, interpolation, migration, advection, and output;
- existing diagnostic and operator entry points where they remain meaningful
  in the new component model.

## Migration order

1. Add constructor-contract tests for the new ownership model.
2. Introduce the new grid, component, field, runtime, model, clock, schedule,
   and simulation types.
3. Migrate transforms, elliptic solvers, operators, nonlinear terms, and
   diagnostics to the new components.
4. Migrate initialization, stratification, and ETD-RK2 stepping.
5. Migrate MPI decomposition, collective utilities, and parallel I/O.
6. Migrate particles and particle output.
7. Remove `Grid`, `State`, `QGParams`, `QGYBJSimulation`, legacy configuration
   builders, old exports, and superseded source paths.
8. Rewrite remaining tests and documentation, preserving the declarative
   Asselin workflow.

Temporary migration code may exist inside the development branch, but the
finished tree contains one data model and no legacy compatibility facade.

## Verification

Every migration step follows red-green-refactor. Before replacing a numerical
subsystem, deterministic current behavior is captured by existing tests or a
small reference fixture. The final matrix includes:

- constructor, component-combination, lifecycle, and error-contract tests;
- serial operator and conservation regressions;
- ETD-RK2 coefficient, exact-diffusion, trajectory, feedback, and workspace
  reuse tests;
- all stratification variants and restart round trips;
- one-, two-, and four-rank numerical equivalence and decomposition tests;
- NetCDF schema/content and energy-diagnostic tests;
- particle interpolation, periodic migration, output, and serial/MPI agreement;
- allocation budgets for stepping and workspace reuse;
- documentation build and public-export audit;
- a reduced one-step execution of `examples/asselin_jpo2020.jl`.

The full 256 x 256 x 128, 15-inertial-period Asselin simulation is an
operational validation outside CI.

## Acceptance criteria

1. `Grid`, `State`, `QGParams`, `QGYBJSimulation`, and the alias-based facade
   are absent from source and exports.
2. Numerical kernels consume `QGYBJModel` or narrow owned components.
3. All v1.0.24 feature-parity tests pass through the new architecture.
4. Required MPI tests pass at one, two, and four ranks.
5. Documentation builds with no references to removed APIs.
6. The Asselin example retains its declarative structure and passes the reduced
   runtime smoke test.
