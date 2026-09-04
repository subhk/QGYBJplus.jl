# Oceananigans-Style Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Print Oceananigans-style detailed progress with human-readable time,
timestep, global speed maxima, and run wall time while retaining the existing
MPI/runtime initialization reports.

**Architecture:** Add a small internal seconds formatter and measure elapsed
wall time locally within each `run!` invocation. Pass the elapsed value to the
existing root-only detailed progress printer; do not change the public
`Simulation` data model. Keep the four startup `@info` reports intact and cover
them with a logging regression test.

**Tech Stack:** Julia, MPI.jl, PencilFFTs, `Printf`, Julia `Test` and `Logging`.

### Task 1: Specify the detailed progress format

**Files:**
- Modify: `test/test_simulation_lifecycle.jl`

**Step 1: Write the failing test**

Update the existing captured progress assertions to require lines shaped like:

```text
Iteration: 0001, time: 250 ms, Δt: 250 ms, max(|LA|) = 1.562e-02 m s⁻¹, max(|uₕ|) = 0.000e+00 m s⁻¹, wall time: 1.234 seconds
```

Require two scheduled lines, parse both maxima, and accept any valid
human-readable wall-time unit because real elapsed time is nondeterministic.
Add focused assertions for representative subsecond, seconds, minutes, hours,
and days formatting.

**Step 2: Run the lifecycle test and verify RED**

Run:

```sh
julia --startup-file=no --project=. test/test_simulation_lifecycle.jl
```

Expected: FAIL because progress still uses `iteration=... | time=...` and has
no timestep or wall time.

### Task 2: Implement human-readable progress and wall time

**Files:**
- Modify: `src/simulation.jl`

**Step 1: Implement the internal time formatter**

Format seconds using Oceananigans-style thresholds: nanoseconds,
microseconds, milliseconds, seconds, minutes, hours, and days, with three
decimal places only when the converted value is non-integral.

**Step 2: Measure the current run's wall time**

Record `time_ns()` when `run!` begins its execution lifecycle. At each detailed
progress event, convert the elapsed nanoseconds to seconds. Reset this timer on
each separate `run!` call; do not add mutable state to `Simulation`.

**Step 3: Print the new root-only line**

Retain the existing MPI reductions and emit:

```text
Iteration: %04d, time: %s, Δt: %s, max(|LA|) = %.3e m s⁻¹, max(|uₕ|) = %.3e m s⁻¹, wall time: %s
```

**Step 4: Run the lifecycle test and verify GREEN**

Run the Task 1 command. Expected: all lifecycle assertions pass.

### Task 3: Protect initialization reporting

**Files:**
- Modify: `test/test_core_architecture.jl`
- Verify: `src/parallel_mpi.jl`
- Verify: `src/core/runtime.jl`

**Step 1: Capture initialization logs**

Construct a verbose small model with a test logger and require messages for
MPI initialization, topology validation, pencil decompositions, and runtime
initialization. Verify the relevant process, topology, grid, decomposition,
and rank metadata.

**Step 2: Run the focused test**

Run:

```sh
julia --startup-file=no --project=. test/runtests.jl
```

Expected: initialization reporting and the full serial suite pass.

### Task 4: Update user documentation

**Files:**
- Modify: `docs/src/guide/simulation.md`

**Step 1: Replace the old sample**

Show the Oceananigans-style line and state that wall time measures the current
`run!` call. Briefly list the four root-only initialization reports that appear
during model construction.

**Step 2: Build the documentation**

Run:

```sh
julia --startup-file=no --project=docs docs/make.jl
```

Expected: Documenter completes without errors.

### Task 5: Verify the MPI path and repository state

**Files:**
- Verify all modified source, test, documentation, and plan files

**Step 1:** Run the full serial `Pkg.test()` suite.

**Step 2:** Run the four-rank stepping regression and confirm only rank zero
prints progress while all ranks participate in reductions.

**Step 3:** Run `git diff --check` and inspect the final diff.

**Step 4:** Commit the verified change on `feature/oceananigans-progress`.
