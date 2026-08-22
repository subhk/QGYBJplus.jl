# [Asselin et al. dipole](@id worked_example)

```@meta
CurrentModule = QGYBJplus
```

The repository's primary worked example is
`examples/asselin_jpo2020.jl`. It follows the barotropic dipole setup of
Asselin, Thomas, Young, and Rainville (2020) and uses only the
composition-first API.

## Run the published-resolution setup

```bash
mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl
```

Defaults are 256×256×128 points, a 70 km square horizontal domain, 3 km depth,
and 15 inertial periods with `Δt = 2 s`.

## Run a small acceptance case

```bash
QGYBJ_ASSELIN_NX=32 \
QGYBJ_ASSELIN_NY=32 \
QGYBJ_ASSELIN_NZ=16 \
QGYBJ_ASSELIN_STEPS=2 \
QGYBJ_ASSELIN_OUTPUT=output_asselin_small \
julia --project=. examples/asselin_jpo2020.jl
```

The environment overrides change execution size only. They do not introduce a
second configuration system.

## Programmatic use

Including the script is side-effect free. Call its entry point with keyword
overrides:

```julia
include("examples/asselin_jpo2020.jl")

simulation = run_asselin_example(
    size=(32, 32, 16),
    stop_iteration=2,
    output_dir="output_asselin_small",
    output_schedule=IterationInterval(1),
    diagnostics=IterationInterval(1),
    verbose=false,
)
```

The function returns a finalized [`Simulation`](@ref). This makes it suitable
for tests and notebooks while guaranteeing runtime cleanup.

The example uses a fixed barotropic flow, YBJ⁺ waves, no wave feedback, and
explicit wave hyperdiffusion. `set!` initializes the dipole and Gaussian
surface wave; `Simulation` owns output and energy-diagnostic schedules.

## Output

Snapshots contain coordinates, time, iteration, physical `psi`, `LAr`,
`LAi`, `Ar`, and `Ai`, spectral real/imaginary parts of `q` and `B`, and the
vertical `N2` and `a_ell` profiles. Use `examples/compute_energy.jl` for the
post-processing workflow.

## Reference

Asselin, O., Thomas, L. N., Young, W. R., & Rainville, L. (2020),
[“Refraction and Straining of Near-Inertial Waves by Barotropic
Eddies”](https://doi.org/10.1175/JPO-D-20-0109.1), *Journal of Physical
Oceanography*, 50, 3439–3454.
