# [Stratification](@id stratification)

```@meta
CurrentModule = QGYBJplus
```

This page explains how to configure ocean stratification profiles in QGYBJ+.jl.

## Why Stratification Matters

The buoyancy frequency ``N(z)`` affects:

- **Wave propagation**: Dispersion depends on ``N^2``
- **Vertical structure**: Mode shapes vary with ``N(z)``
- **Refraction**: Waves bend toward regions of lower ``N``
- **Energy flux**: Vertical group velocity scales with ``N``

The vertical coordinate is `z ∈ [-Lz, 0]` with `z = 0` at the surface. Grid points are cell-centered at `z = -Lz+dz/2 ... -dz/2`. For stratification profiles we use depth `d = -z` (positive downward).

## Built-in Stratification Types

QGYBJ+.jl supports two stratification modes through the `default_params()` function:

### Constant N

Uniform stratification throughout the water column (default):

```julia
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    stratification=:constant_N,  # This is the default
    N²=1.0                       # Buoyancy frequency squared
)
```

Profile:
```math
N^2(z) = N_0^2 = \text{const}
```

Best for:
- Idealized studies
- Analytical comparisons
- Simple mode structure

### Skewed Gaussian (Pycnocline)

Sharp pycnocline with gradual decrease below:

```julia
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    stratification=:skewed_gaussian
)
```

Profile formula:
```math
N^2(d) = N_1^2 \exp\left(-\frac{(d-z_0)^2}{\sigma^2}\right) \left[1 + \text{erf}\left(\frac{\alpha(d-z_0)}{\sigma\sqrt{2}}\right)\right] + N_0^2
```

The skewed Gaussian parameters in `QGParams` are:
- `N₀²_sg`: Background N² value
- `N₁²_sg`: Peak N² amplitude
- `σ_sg`: Width of pycnocline
- `z₀_sg`: Center depth of pycnocline (positive below surface)
- `α_sg`: Skewness parameter

Profile (schematic):
```
N² →
│
│    ╱╲
│   ╱  ╲
│  ╱    ╲____
│ ╱           ╲____
│╱                  ──────
└─────────────────────────── depth d
     ↑
  pycnocline
```

Best for:
- Realistic subtropical ocean
- Strong near-surface trapping
- Wave focusing studies

## Setting Up with Stratification Profiles

### Using setup_model_with_profile

For non-constant stratification, use `setup_model_with_profile()` to get the N² profile:

```julia
using QGYBJplus

# Create parameters with skewed Gaussian stratification
par = default_params(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    stratification=:skewed_gaussian
)

# This returns the N² profile for use in physics
G, S, plans, a_ell, N2_profile = setup_model_with_profile(par)

# N2_profile is now available for vertical velocity computation, etc.
```

### Using the High-Level API

The `QGYBJSimulation` API handles stratification automatically:

```julia
using QGYBJplus

# Create configuration
domain = create_domain_config(
    nx=64, ny=64, nz=32,
    Lx=500e3, Ly=500e3, Lz=4000.0
)

strat = create_stratification_config(
    type=:skewed_gaussian
)

model = create_model_config(
    inviscid=false,
    ybj_plus=true
)

# Setup simulation - stratification is handled internally
sim = setup_simulation(domain, strat, model=model)
```

## Advanced: Custom Profiles

### StratificationProfile Types

QGYBJ+.jl provides several stratification profile types:

```julia
# Constant N²
profile = ConstantN{Float64}(1.0)  # N₀ = 1.0

# Tanh profile (pycnocline-like)
profile = TanhProfile{Float64}(
    0.01,    # N_upper
    0.025,   # N_lower
    2400.0,  # z_pycno depth (same units as Lz)
    200.0    # width (same units as Lz)
)

# Exponential profile
profile = ExponentialProfile{Float64}(
    0.02,    # N_surface
    1200.0,  # scale_height
    0.001    # N_deep
)

# Piecewise profile (two-layer)
profile = PiecewiseProfile{Float64}(
    [0.0, 2000.0, 4000.0],  # z_interfaces (depths below surface)
    [0.01, 0.03]            # N values in each layer
)
```

### Evaluating Profiles on the Grid

```julia
using QGYBJplus

# Create a profile
profile = TanhProfile{Float64}(0.01, 0.025, 2400.0, 200.0)

# Compute N² on the model grid
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)
G = init_grid(par)
N2_profile = compute_stratification_profile(profile, G)

# Compute elliptic coefficient from N² profile
a_ell = a_ell_from_N2(N2_profile, par)
```

### From Data File

Load stratification from a NetCDF file:

```julia
using QGYBJplus, NCDatasets

# Read N² profile from file
N2_profile = read_stratification_profile("N2_data.nc", G)

# Or read raw data for custom processing
z_data, N2_data = read_stratification_raw("N2_data.nc")
```

`z_data` should be depths (positive below the surface). The model grid uses `z ∈ [-Lz, 0]`, so file profiles are interpreted in depth coordinates.

## Effects on Dynamics

### Elliptic Coefficient

The elliptic coefficient `a = f²/N²` is computed from the N² profile:

```julia
# For constant N² (uses par.N²)
a_ell = a_ell_ut(par, G)

# For variable N² (from profile)
a_ell = a_ell_from_N2(N2_profile, par)
```

### Deformation Radius

The first baroclinic deformation radius can be computed:

```julia
using QGYBJplus: compute_deformation_radius

Ld = compute_deformation_radius(N2_profile, par.f₀, par.Lz)
```

### Wave Trapping

Strong surface stratification traps waves near the surface:

```
Strong N²    Weak N²
near surface everywhere

    │           │
  ──┴──       ──┴──
  Wave        Wave
  trapped     penetrates
  above       to depth
  pycnocline
```

## Visualization

### Profile Plot

```julia
using Plots

# Get grid depth coordinates (positive downward)
depth = -G.z

# Plot N² profile
plot(N2_profile, depth,
    xlabel = "N² (s⁻²)",
    ylabel = "Depth (m)",
    title = "Stratification Profile",
    legend = false
)
```

### Plotting Different Profiles

```julia
using Plots, QGYBJplus

# Create standard profiles for comparison
profiles = create_standard_profiles(4000.0)  # 4km domain

p = plot(title="Stratification Profiles", xlabel="N²", ylabel="Depth (m)")
for (name, profile) in profiles
    z_vals, N2_vals, _ = plot_stratification_profile(profile, 4000.0)
    depth_vals = -z_vals
    plot!(p, N2_vals, depth_vals, label=String(name))
end
display(p)
```

## Best Practices

### Resolution Guidelines

| Profile Type | Recommended ``n_z`` |
|:-------------|:--------------------|
| Constant N | 16-32 |
| Smooth exponential | 32-64 |
| Sharp pycnocline | 64-128 |
| Two-layer | 32-64 |

Sharp features need higher resolution to avoid Gibbs phenomena.

### Validation

Use `validate_stratification()` to check your profile:

```julia
errors, warnings = validate_stratification(N2_profile)

for err in errors
    @error err
end
for warn in warnings
    @warn warn
end
```

### Physical Constraints

- ``N^2 > 0`` everywhere (stable stratification)
- ``N^2`` should decrease with depth (typically)
- Avoid very small ``N^2`` (causes numerical issues in inversions)

```julia
# Ensure minimum N²
N2_min = 1e-6
N2_profile .= max.(N2_profile, N2_min)
```

## Stratification Types Reference

| Type | Symbol/Struct | Parameters |
|:-----|:--------------|:-----------|
| Constant | `:constant_N` | `N²` in `default_params()` |
| Skewed Gaussian | `:skewed_gaussian` | `N₀²_sg`, `N₁²_sg`, `σ_sg`, `z₀_sg`, `α_sg` |
| Tanh | `TanhProfile` | `N_upper`, `N_lower`, `z_pycno`, `width` |
| Exponential | `ExponentialProfile` | `N_surface`, `scale_height`, `N_deep` |
| Piecewise | `PiecewiseProfile` | `z_interfaces`, `N_values` |
| From File | `FileProfile` | `filename` |

## Related Topics

- [QG Equations](@ref qg-equations): How N² enters PV
- [YBJ+ Wave Model](@ref ybj-plus): Wave dispersion with N²
- [Configuration](@ref configuration): Setting up simulations
