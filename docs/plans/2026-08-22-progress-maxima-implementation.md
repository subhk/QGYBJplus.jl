# Progress Maxima Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make an opt-in `run!` progress report print globally reduced maximum
wave speed and balanced-flow speed while retaining time-based stopping.

**Architecture:** An internal diagnostic computes physical `LA` in the
runtime-owned FFT buffers and reduces `max(abs(LA))` across MPI ranks. It also
reduces `max(hypot(u, v))` from the current physical velocity fields. Passing
`progress=true` selects the detailed one-line report at the existing
`diagnostics_interval`; ordinary runs keep the existing lightweight progress
message.

**Tech Stack:** Julia, MPI.jl, PencilFFTs, `Printf`, Julia `Test`.

### Task 1: Specify time-limited detailed progress

**Files:**
- Modify: `test/test_simulation_lifecycle.jl`

**Step 1: Write the failing test**

Create a small model and run:

```julia
simulation = Simulation(
    model;
    Δt=0.25,
    stop_time=0.5,
    output=false,
    diagnostics=false,
    verbose=false,
)
run!(simulation; progress=true, diagnostics_interval=1)
```

Capture standard output and require two lines containing `iteration=1` and
`iteration=2`, together with `max_wave_speed=` and `max_flow_speed=`. Assert
that the clock stops at `time == 0.5` and `stop_iteration === nothing`.

**Step 2: Run the test and verify RED**

Run:

```sh
julia --startup-file=no --project=. test/test_simulation_lifecycle.jl
```

Expected: FAIL because the current progress report contains neither maximum.

### Task 2: Implement allocation-free scheduled maxima

**Files:**
- Modify: `src/simulation.jl`

**Step 1: Compute physical wave speed**

Fill the runtime-owned FFT output buffer with spectral `LA`. For `YBJ()`,
`LA = B`; otherwise use `LA = B + k_h^2 A / 4`. Transform it into the
runtime-owned input buffer and compute the local maximum absolute value.

**Step 2: Compute horizontal flow speed**

Loop over physical `u` and `v` without allocating a temporary array and compute
the local maximum of `hypot(u, v)`.

**Step 3: Reduce and print**

Use `MPI.Allreduce(..., MPI.MAX, comm)` for both quantities and print only on
rank zero:

```text
iteration=100 | time=200.0 s | max_wave_speed=9.998e-02 m/s | max_flow_speed=3.349e-01 m/s
```

Detailed reporting is enabled only by an explicit `progress=true` argument.
The existing `diagnostics_interval` remains the iteration cadence.

**Step 4: Run the test and verify GREEN**

Run the Task 1 command. Expected: all lifecycle tests pass.

### Task 3: Update the Asselin example

**Files:**
- Modify: `examples/asselin_jpo2020.jl`
- Modify: `test/test_asselin_smoke.jl`

**Step 1: Preserve current user edits**

Keep the explicit coordinate bounds and explicit fourth-order flow/wave
closures already present in the working tree.

**Step 2: Configure the requested run**

Set `stop_time=10 * inertial_period` and call:

```julia
run!(simulation; progress=true, diagnostics_interval=1000)
```

Update the smoke assertions to require this time limit and progress call.

**Step 3: Run the Asselin smoke test**

Expected: all assertions and the one-step runtime smoke test pass.

### Task 4: Verify and publish

**Files:**
- Verify all changed source, test, example, and plan files

**Step 1:** Run the full serial `Pkg.test()` suite.

**Step 2:** Build the Documenter site.

**Step 3:** Run the four-rank stepping regression.

**Step 4:** Confirm `git diff --check` and obtain a read-only code review.

**Step 5:** Commit, push the existing `wave-operator-analysis` PR branch, and
update PR #20 with the new progress-report behavior and verification results.
