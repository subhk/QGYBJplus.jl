# Composed Hyperdiffusivity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose flow and wave hyperdiffusion as symmetric named components,
including an explicit zero flow component in the Asselin example.

**Architecture:** `HorizontalHyperdiffusivity` will own a
`FlowHyperdiffusivity` and a `WaveHyperdiffusivity`. Each field-specific
component stores validated `(coefficient, order)` terms, while its simple
keyword constructor creates one term. The integrating factor selects the
appropriate owned component and sums its terms; direct flow-only and wave-only
closures remain valid.

**Tech Stack:** Julia, QGYBJplus component API, `Test`, Documenter, MPI.

### Task 1: Specify the public component API

**Files:**
- Modify: `test/test_core_components.jl`

**Step 1: Write the failing test**

Add assertions for:

```julia
flow_closure = FlowHyperdiffusivity(coefficient=0)
wave_closure = WaveHyperdiffusivity(coefficient=1e5)
closure = HorizontalHyperdiffusivity(flow=flow_closure, wave=wave_closure)

@test closure.flow === flow_closure
@test closure.wave === wave_closure
```

Also test invalid flow coefficients and odd derivative orders.

**Step 2: Run the test to verify it fails**

Run:

```sh
julia --startup-file=no --project=. test/test_core_components.jl
```

Expected: `UndefVarError: FlowHyperdiffusivity not defined`.

### Task 2: Implement the composed closure

**Files:**
- Modify: `src/core/components.jl`
- Modify: `src/QGYBJplus.jl`
- Modify: `src/nonlinear.jl`

**Step 1: Add the minimal component implementation**

Define validated `FlowHyperdiffusivity` and `WaveHyperdiffusivity` components,
then define:

```julia
struct HorizontalHyperdiffusivity{F<:FlowHyperdiffusivity,
                                  W<:WaveHyperdiffusivity} <: AbstractClosure
    flow::F
    wave::W
end
```

The public single-term constructors accept `coefficient` and total derivative
`order`. Internal tuple constructors retain the existing two-term defaults
without retaining the ambiguous `flow2` or `waves2` names.

**Step 2: Update integrating-factor dispatch**

Select `closure.flow` for balanced-flow damping and `closure.wave` for wave
damping. Sum all configured terms as
`Δt * coefficient * (kx^2 + ky^2)^(order / 2)`.

**Step 3: Run the focused test to verify it passes**

Run the Task 1 command. Expected: all component tests pass.

### Task 3: Migrate repository callers

**Files:**
- Modify: `examples/asselin_jpo2020.jl`
- Modify: `test/test_asselin_smoke.jl`
- Modify: tests returned by `rg 'flow2|waves2|laplacian_order' test`
- Modify: `docs/src/guide/configuration.md`
- Modify: `docs/src/api/types.md`

**Step 1: Update the Asselin configuration**

Use the explicit composition:

```julia
closure=HorizontalHyperdiffusivity(
    flow=FlowHyperdiffusivity(coefficient=0),
    wave=WaveHyperdiffusivity(coefficient=1.0e5),
),
```

**Step 2: Replace old numbered constructor calls**

Convert remaining tests and docs to field-specific components. Confirm that
`flow2`, `waves2`, and the old Laplacian-order keywords no longer occur in
source, tests, examples, or web documentation.

**Step 3: Run focused tests**

Run the core component, ETD-RK2, and Asselin smoke tests. Expected: all pass.

### Task 4: Verify and publish

**Files:**
- Verify all changed files

**Step 1: Run serial verification**

Run `Pkg.test()` and the documentation consistency/build tests. Expected: all
tests pass, with only the repository's known internal-docstring warnings.

**Step 2: Run MPI verification**

Run the four-rank stepping regression. Expected: all MPI assertions pass.

**Step 3: Review the diff**

Confirm the public example is compact, the old names are absent, and the
integrating-factor behavior is unchanged for equivalent coefficients/orders.

**Step 4: Commit and push**

Commit the tested migration and push the current PR branch.
