# QGYBJplus.jl

```@meta
CurrentModule = QGYBJplus
```

```@raw html
<p class="doc-lede">
QGYBJplus.jl is a dimensional Julia implementation of the QG-YBJ+ model for
near-inertial waves interacting with balanced quasi-geostrophic flow.
</p>
```

## Start Here

```@raw html
<div class="doc-grid">
<div class="doc-tile">
<strong><a href="quickstart/">Quick Start</a></strong>
<p>Build and run a minimal high-level simulation.</p>
</div>
<div class="doc-tile">
<strong><a href="guide/configuration/">Configuration</a></strong>
<p>Grid, model, initial conditions, and output options.</p>
</div>
<div class="doc-tile">
<strong><a href="guide/io/">I/O and Output</a></strong>
<p>NetCDF output, selected z levels, and reading results.</p>
</div>
<div class="doc-tile">
<strong><a href="worked_example/">Worked Example</a></strong>
<p>A complete script with full-domain and surface output.</p>
</div>
</div>
```

## Minimal Example

```julia
using QGYBJplus

grid = RectilinearGrid(size = (64, 64, 32),
                       x = (-250e3, 250e3),
                       y = (-250e3, 250e3),
                       z = (-4000.0, 0.0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1e-4),
                   stratification = ConstantStratification(N² = 1e-5),
                   flow = :fixed,
                   feedback = :none,
                   Δt = 300.0)

set!(model;
     ψ = (x, y, z) -> 1e3 * sin(2π * x / 500e3) * cos(2π * y / 500e3),
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = 0.05, scale = 500.0))

simulation = Simulation(model; stop_time = 2inertial_period(model))
run!(simulation)
finalize_simulation!(simulation)
```

## What Is Included

- Dimensional QG-YBJ+ equations.
- Constant, analytic, and file-backed stratification.
- Exponential RK2 time stepping.
- NetCDF output for full fields or selected nearest z levels.
- MPI pencil decomposition for larger runs.
- Optional Lagrangian particle tracking.

## Documentation Map

```@raw html
<div class="doc-grid">
<div class="doc-tile">
<strong>Physics</strong>
<p><a href="physics/overview/">Overview</a>, <a href="physics/qg_equations/">QG</a>, <a href="physics/ybj_plus/">YBJ+</a>, and numerical methods.</p>
</div>
<div class="doc-tile">
<strong>User Guide</strong>
<p><a href="guide/simulation/">Running</a>, <a href="guide/stratification/">stratification</a>, initial conditions, diagnostics, and output.</p>
</div>
<div class="doc-tile">
<strong>Advanced</strong>
<p><a href="advanced/parallel/">MPI</a>, performance, interpolation, and particle tracking.</p>
</div>
<div class="doc-tile">
<strong>API</strong>
<p>Reference pages for public types, solvers, diagnostics, and particles.</p>
</div>
</div>
```

## Citation

```bibtex
@software{qgybj_jl,
  author = {Kar, Subhajit},
  title = {QGYBJplus.jl: A Julia Implementation of the QG-YBJ+ Model},
  year = {2025},
  url = {https://github.com/subhk/QGYBJplus.jl}
}
```
