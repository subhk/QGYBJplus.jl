# [Configuration](@id configuration)

```@meta
CurrentModule = QGYBJplus
```

## Two APIs

| Use Case | API |
|:---------|:----|
| Quick start | `create_simple_config()` → `run_simple_simulation()` |
| Full control | `default_params()` → `setup_model()` → manual stepping |

!!! warning "Different Defaults"
    | Flag | Simple API | Full API |
    |:-----|:-----------|:---------|
    | `inviscid` | `true` | `false` |
    | `no_wave_feedback` | `false` | `true` |

## Simple API

```julia
config = create_simple_config(
    # Domain (REQUIRED)
    Lx=500e3, Ly=500e3, Lz=4000.0,
    # Grid
    nx=64, ny=64, nz=32,
    # Time
    dt=0.001, total_time=1.0,
    # Physics
    ybj_plus=true, inviscid=true, linear=false,
    # Output
    output_interval=100
)
result = run_simple_simulation(config)
```

## Full API

```julia
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,
    dt=0.001, nt=1000,
    f₀=1.0, N²=1.0,
    ybj_plus=true, inviscid=false
)

G, S, plans, a_ell = setup_model(par)
init_random_psi!(S, G; amplitude=0.1)
compute_q_from_psi!(S, G, plans, a_ell)

first_projection_step!(S, G, par, plans, a_ell)
for step = 2:par.nt
    leapfrog_step!(S, G, par, plans, a_ell)
end
```

!!! tip "IMEX"
    Use `imex_cn_step!()` instead of `leapfrog_step!()` for ~10× faster wave-dominated problems.

## Parameter Reference

### Domain (REQUIRED)

| Parameter | Description |
|:----------|:------------|
| `Lx`, `Ly` | Horizontal domain size [m] |
| `Lz` | Vertical depth [m] |
| `nx`, `ny`, `nz` | Grid points (default: 64) |

### Physics

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `f₀` | 1.0 | Coriolis parameter |
| `N²` | 1.0 | Buoyancy frequency squared |
| `γ` | 1e-3 | Robert-Asselin filter |

Unicode: type `f\_0<tab>` → `f₀`, `\nu<tab>` → `ν`

### Model Flags

| Flag | Default | Effect |
|:-----|:--------|:-------|
| `ybj_plus` | true | Use YBJ+ formulation |
| `inviscid` | false | Disable dissipation |
| `linear` | false | Disable nonlinear terms |
| `no_wave_feedback` | true | Disable qʷ term |
| `no_dispersion` | false | Disable wave dispersion |
| `fixed_flow` | false | Keep ψ constant |

### Dissipation

Two hyperdiffusion operators: `ν₁(-∇²)^ilap1 + ν₂(-∇²)^ilap2`

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `νₕ₁`, `νₕ₂` | 0.01, 10.0 | Hyperviscosity (flow) |
| `ilap1`, `ilap2` | 2, 6 | Laplacian power |
| `νₕ₁ʷ`, `νₕ₂ʷ` | 0.0, 10.0 | Hyperviscosity (waves) |

Higher `ilap` = more scale-selective: 1 (∇²), 2 (∇⁴), 4 (∇⁸)

## Examples

### QG-Only (No Waves)
```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, no_dispersion=true)
```

### Linear Wave Propagation
```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    linear=true, inviscid=true, no_wave_feedback=true
)
```

## Save/Load

```julia
using JLD2
@save "config.jld2" par
@load "config.jld2" par
```

## See Also

- [Stratification](@ref stratification)
- [I/O and Output](@ref io-output)
- [MPI Parallelization](@ref parallel)
