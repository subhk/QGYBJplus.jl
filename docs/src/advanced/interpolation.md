# [Interpolation Methods](@id interpolation)

```@meta
CurrentModule = QGYBJplus
```

This page describes interpolation methods available in QGYBJ+.jl.

## Overview

Interpolation is needed for:
- **Particle advection**: Evaluating velocities at particle positions
- **Output**: Sampling fields at specific locations
- **Regridding**: Transferring data between grids

## Available Methods

### Nearest Neighbor

Fastest but lowest accuracy:

```julia
value = interpolate(field, x, y, z, grid; method=:nearest)
```

- **Order**: 0
- **Continuity**: None (discontinuous)
- **Best for**: Quick visualization, large particle counts

### Trilinear

Standard method for smooth fields:

```julia
value = interpolate(field, x, y, z, grid; method=:linear)
```

- **Order**: 1
- **Continuity**: C⁰ (continuous values, discontinuous derivatives)
- **Best for**: General-purpose interpolation

### Tricubic

High-accuracy interpolation:

```julia
value = interpolate(field, x, y, z, grid; method=:cubic)
```

- **Order**: 3
- **Continuity**: C¹ (continuous first derivatives)
- **Best for**: Accurate particle trajectories, smooth fields

### Spectral

Exact for band-limited fields:

```julia
value = interpolate_spectral(field_k, x, y, z, grid, plans)
```

- **Order**: Spectral (N)
- **Continuity**: C^∞
- **Best for**: Highest accuracy, single-point queries

## Batch Interpolation

For many points (e.g., particles):

```julia
# Pre-allocate output
values = zeros(nparticles)

# Batch interpolation
interpolate_batch!(values, field, xs, ys, zs, grid; method=:linear)
```

### Performance Comparison

| Method | Points/second | Memory |
|:-------|:--------------|:-------|
| Nearest | 10⁸ | Minimal |
| Linear | 10⁷ | Minimal |
| Cubic | 10⁶ | 64 coeff/point |
| Spectral | 10⁴ | Full field |

## Horizontal Interpolation

For 2D slices:

```julia
# At fixed depth
value_xy = interpolate_horizontal(field[:,:,k], x, y, grid; method=:linear)
```

## Vertical Interpolation

Along the vertical:

```julia
# At fixed horizontal position
value_z = interpolate_vertical(field[i,j,:], z, grid; method=:linear)
```

## Spectral Interpolation

### Theory

For a spectral field:
```math
f(x,y,z) = \sum_{k_x, k_y} \hat{f}(k_x, k_y, z) e^{i(k_x x + k_y y)}
```

### Implementation

```julia
function interpolate_spectral(field_k, x, y, z, grid, plans)
    # Compute phase factors
    phases = exp.(im .* (grid.kx .* x .+ grid.ky' .* y))

    # Find vertical indices
    k_lo, k_hi, wz = find_vertical_cell(z, grid)

    # Interpolate vertically
    field_z = (1-wz) .* field_k[:,:,k_lo] .+ wz .* field_k[:,:,k_hi]

    # Sum over wavenumbers
    return real(sum(field_z .* phases))
end
```

## Stencil Coefficients

### Linear (8-point)

```
      z₁ -------- z₁
      /|         /|
     / |        / |
   z₀ -------- z₀ |
    |  y₁ -----|-- y₁
    | /        | /
    |/         |/
   y₀ -------- y₀
   x₀         x₁
```

Weights: Product of 1D linear weights in each direction.

### Cubic (64-point)

Uses 4×4×4 stencil with Catmull-Rom spline weights.

## Boundary Handling

### Periodic

Default for horizontal directions:

```julia
x_wrapped = mod(x, grid.Lx)
y_wrapped = mod(y, grid.Ly)
```

### Extrapolation

For vertical boundaries:

```julia
# Clamp to domain
z_clamped = clamp(z, 0, grid.H)

# Or extrapolate linearly
if z > grid.H
    value = field[end] + (z - grid.H) * gradient[end]
end
```

## Interpolation for Derivatives

### Gradient Interpolation

```julia
# Interpolate gradient components
dudx = interpolate(dudx_field, x, y, z, grid)
dudy = interpolate(dudy_field, x, y, z, grid)
```

### Directly from Spectral

```julia
function interpolate_gradient_spectral(field_k, x, y, z, grid, plans)
    # Compute spectral derivatives
    dfdx_k = im .* grid.kx .* field_k
    dfdy_k = im .* grid.ky' .* field_k

    # Interpolate both
    dfdx = interpolate_spectral(dfdx_k, x, y, z, grid, plans)
    dfdy = interpolate_spectral(dfdy_k, x, y, z, grid, plans)

    return dfdx, dfdy
end
```

## Regridding

### To Finer Grid

```julia
function regrid_fine(field_coarse, grid_coarse, grid_fine)
    field_fine = zeros(grid_fine.nx, grid_fine.ny, grid_fine.nz)

    for k in 1:grid_fine.nz
        for j in 1:grid_fine.ny
            for i in 1:grid_fine.nx
                x = grid_fine.x[i]
                y = grid_fine.y[j]
                z = grid_fine.z[k]
                field_fine[i,j,k] = interpolate(field_coarse, x, y, z,
                                                 grid_coarse)
            end
        end
    end

    return field_fine
end
```

### Spectral Padding

For spectral fields, zero-pad in wavenumber space:

```julia
function spectral_refine(field_k_coarse, grid_coarse, grid_fine)
    field_k_fine = zeros(ComplexF64, grid_fine.nx÷2+1, grid_fine.ny, grid_fine.nz)

    # Copy low wavenumbers
    nkx = grid_coarse.nx÷2+1
    nky = grid_coarse.ny

    field_k_fine[1:nkx, 1:nky÷2, :] = field_k_coarse[:, 1:nky÷2, :]
    field_k_fine[1:nkx, end-nky÷2+1:end, :] = field_k_coarse[:, nky÷2+1:end, :]

    return field_k_fine
end
```

## API Reference

The interpolation functionality is provided through the particle advection system.
See [`interpolate_velocity_at_position`](@ref) in the particles module for the main
interpolation interface used for Lagrangian particle tracking.
