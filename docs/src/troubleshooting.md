# [Troubleshooting](@id troubleshooting)

```@meta
CurrentModule = QGYBJplus
```

## Quick Diagnostic

| Symptom | Check First |
|:--------|:------------|
| Won't run | Are `Lx`, `Ly`, `Lz` provided? (required) |
| Blows up (NaN) | Is `dt` small enough? Try `dt/2` |
| Wrong results | Check `ybj_plus=true` vs `false` |
| Too slow | Use IMEX-CN instead of leapfrog |
| Out of memory | Reduce grid size or use MPI |

## Installation

**Package not found**: Install from GitHub:
```julia
Pkg.add(url="https://github.com/subhk/QGYBJplus.jl")
```

**MPI fails**: Install system MPI first:
- macOS: `brew install open-mpi`
- Ubuntu: `sudo apt install libopenmpi-dev openmpi-bin`
- HPC: `module load openmpi`

Then: `Pkg.build("MPI")`

## Runtime Errors

### Missing Domain Size
```julia
# Wrong - MethodError
par = default_params(nx=64, ny=64, nz=32)

# Correct
par = default_params(nx=64, ny=64, nz=32, Lx=500e3, Ly=500e3, Lz=4000.0)
```

### Simulation Blows Up (NaN)

1. **Reduce time step**: `dt = dt / 2`
2. **Increase dissipation**: `νₕ₁ = 1e8, ilap1 = 2`
3. **Use IMEX** for wave-dominated problems: `imex_cn_step!()` instead of `leapfrog_step!()`
4. **Debug with linear mode**: `par = default_params(..., linear=true)`

!!! tip "Wave CFL"
    For YBJ+: `dt ≤ 2f₀/N²`. For ocean values: `dt ≤ 20s`.

### Out of Memory

- Reduce grid: `nx, ny, nz = 128, 128, 64`
- Use MPI: `mpiexec -n 16 julia script.jl`
- Use Float32: `par = default_params(..., T=Float32)`

Memory: 256³ complex array ≈ 1 GB. Full simulation needs 5-10× this.

## MPI Issues

**Pencil topology mismatch**: Use `copy_state(S)` not `deepcopy(S)`:
```julia
Snm1 = copy_state(S)  # Correct
```

**Deadlock**: Ensure all ranks call collective operations. Debug with:
```julia
MPI.Barrier(comm)
println("Rank $(MPI.Comm_rank(comm)) reached checkpoint")
```

**Segfaults**: Use actual array dimensions, not grid dimensions:
```julia
nz_phys, nx_phys, ny_phys = size(parent(phys_arr))
```

## Unicode Characters

Type LaTeX + Tab in Julia REPL:

| Type | Get |
|:-----|:----|
| `f\_0<tab>` | `f₀` |
| `\nu<tab>` | `ν` |
| `N\^2<tab>` | `N²` |

## Performance

1. Use IMEX time stepping (10× faster for waves)
2. Run with threads: `julia -t auto script.jl`
3. Use MPI for large grids
4. Reduce output frequency

## Stability: Hyperdiffusion

```julia
par = default_params(
    ...,
    νₕ₁ʷ = 1e7, ilap1w = 2,  # Biharmonic (recommended)
    γ = 0.01                  # Stronger Robert-Asselin filter
)
```

Higher `ilap` = more scale-selective: `ilap=1` (∇²), `ilap=2` (∇⁴), `ilap=4` (∇⁸)

## Still Stuck?

[Open an issue](https://github.com/subhk/QGYBJplus.jl/issues) with:
- Julia version, package versions
- Minimal reproducible example
- Full error message
