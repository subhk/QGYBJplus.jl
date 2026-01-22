# [Particle Advection](@id particles)

```@meta
CurrentModule = QGYBJplus
```

This page describes Lagrangian particle tracking in QGYBJ+.jl, including the physics, numerical algorithms, and parallel implementation.

## Overview

Particle tracking allows you to:

- Follow **fluid parcels** as they move with the flow
- Compute **Lagrangian statistics** (dispersion, diffusivity)
- Track **tracer concentrations** along trajectories
- Study **mixing** and **transport** in QG-YBJ+ dynamics

## Physics of Particle Advection

In QG-YBJ+ dynamics, Lagrangian particles experience velocities from both the balanced (geostrophic) flow and near-inertial waves. Understanding the complete velocity field is essential for accurate particle tracking.

### Complex Coordinate Notation

Before diving into the physics, we introduce the complex coordinate notation used throughout:

| Symbol | Definition | Description |
|:-------|:-----------|:------------|
| ``s`` | ``x + iy`` | Complex horizontal coordinate |
| ``s^*`` | ``x - iy`` | Complex conjugate coordinate |
| ``\partial_s`` | ``\frac{1}{2}(\partial_x - i\partial_y)`` | Complex derivative |
| ``\partial_{s^*}`` | ``\frac{1}{2}(\partial_x + i\partial_y)`` | Conjugate complex derivative |
| ``\tilde{z}`` | Stretched vertical coordinate | ``d\tilde{z} = (N/f_0) dz`` |

This notation simplifies the wave equations and makes the underlying structure more apparent.

### Total Velocity Field

Particles are advected by the **total velocity field**:

```math
\mathbf{u}_{total} = \mathbf{u}_{QG} + \mathbf{u}_{wave} + \mathbf{u}_{Stokes}
```

Each component has distinct physical origins:

| Component | Physical Origin | Typical Magnitude |
|:----------|:----------------|:------------------|
| ``\mathbf{u}_{QG}`` | Geostrophic balance | O(0.1-1 m/s) |
| ``\mathbf{u}_{wave}`` | Near-inertial wave orbital motion | O(0.01-0.1 m/s) |
| ``\mathbf{u}_{Stokes}`` | Wave-induced Lagrangian drift | O(0.001-0.01 m/s) |

---

### 1. Geostrophic Flow (QG Velocities)

**Physical interpretation**: The geostrophic flow arises from the balance between pressure gradient and Coriolis forces. It represents the large-scale, slowly-evolving background flow.

#### Horizontal Geostrophic Velocities

From geostrophic balance ``f\mathbf{u} = -\nabla p / \rho_0`` and the streamfunction definition:
```math
u_{QG} = -\frac{\partial \psi}{\partial y}, \quad v_{QG} = \frac{\partial \psi}{\partial x}
```

**Spectral space implementation**:
```math
\hat{u}_{QG} = -i k_y \hat{\psi}, \quad \hat{v}_{QG} = i k_x \hat{\psi}
```
where ``k_x, k_y`` are the horizontal wavenumbers.

#### Vertical QG Velocity (Omega Equation)

The ageostrophic vertical velocity maintains thermal wind balance as the flow evolves. It satisfies the **omega equation**:
```math
N^2 \nabla_H^2 w_{QG} + f_0^2 \frac{\partial^2 w_{QG}}{\partial z^2} = 2 f_0 \, J\left(\psi_z, \nabla_H^2 \psi\right)
```

or equivalently (dividing by ``N^2``):
```math
\nabla_H^2 w_{QG} + \frac{f_0^2}{N^2} \frac{\partial^2 w_{QG}}{\partial z^2} = \frac{2 f_0}{N^2} \, J\left(\psi_z, \nabla_H^2 \psi\right)
```

where ``J(a,b) = a_x b_y - a_y b_x`` is the Jacobian operator.

**Physical interpretation**: The RHS represents vorticity advection by the thermal wind shear. Where this is non-zero, vertical motion is required to maintain balance.

**Numerical solution**: In spectral space, this becomes a tridiagonal system at each ``(k_x, k_y)``:
```math
-k_h^2 \hat{w} + \frac{f_0^2}{N^2} \frac{\partial^2 \hat{w}}{\partial z^2} = \widehat{\text{RHS}}
```
solved with boundary conditions ``w = 0`` at ``z = 0`` (surface) and ``z = -H`` (bottom).

---

### 2. Wave Velocity (Near-Inertial Oscillations)

**Physical interpretation**: Near-inertial waves are oscillatory motions at frequencies close to the local Coriolis frequency ``f_0``. The wave velocity represents the direct orbital motion of fluid parcels due to these waves.

#### YBJ+ Wave Velocity Formulation

Following Asselin & Young (2019), the horizontal wave velocity is:
```math
u + iv = e^{-if_0 t} \, LA
```

where ``LA`` is the **backrotated wave velocity amplitude** (removing the inertial oscillation).

**The ``L`` operator**: The vertical operator ``L`` is defined as:
```math
L = \partial_z \left(\frac{f_0^2}{N^2}\right) \partial_z
```

For constant ``N^2``, this simplifies to ``L = (f_0^2/N^2) \partial_{zz}``.

**YBJ+ relation**: The evolved variable ``L^+A`` relates to the wave amplitude ``A`` via:
```math
L^+A = \left(L - \frac{k_h^2}{4}\right) A
```

where ``k_h^2 = k_x^2 + k_y^2`` is the horizontal wavenumber squared.

**Computing ``LA`` from ``L^+A`` and ``A``**:

From ``L^+A = LA - (k_h^2/4)A``, we get: ``LA = L^+A + (k_h^2/4)A``

In **spectral space**:
```math
\widehat{LA} = \widehat{L^+A} + \frac{k_h^2}{4} \hat{A}
```

**Wave velocity components**:
```math
u_{wave} = \text{Re}(LA), \quad v_{wave} = \text{Im}(LA)
```

**Physical interpretation**: ``LA`` represents the phase-averaged (backrotated) wave velocity. The real part gives the zonal component, the imaginary part gives the meridional component.

#### YBJ Vertical Velocity

The wave-induced vertical velocity follows from Asselin & Young (2019, eq. 2.10):
```math
w_0 = -\frac{f_0^2}{N^2} A_{zs} \, e^{-i f_0 t} + \text{c.c.}
```

**Expanded form** (separating oscillating components):
```math
w = -\frac{f_0^2}{N^2} \left[\cos(f_0 t) \cdot w_{cos} + \sin(f_0 t) \cdot w_{sin}\right]
```

where:
```math
w_{cos} = \text{Re}(\partial_x A_z) + \text{Im}(\partial_y A_z)
```
```math
w_{sin} = \text{Im}(\partial_x A_z) - \text{Re}(\partial_y A_z)
```

---

### 3. Wave-Induced Stokes Drift

**Physical interpretation**: Even though waves are oscillatory with zero Eulerian mean velocity, particles experience a net **Lagrangian drift** in the direction of wave propagation. This is the Stokes drift, arising from the correlation between particle displacement and velocity gradients.

#### Horizontal Stokes Drift (Wagner & Young 2016, eq. 3.16a-3.18)

The horizontal Stokes drift velocity is given by the complex velocity:
```math
U^S = u_S + i v_S
```

From eq. (3.18), this satisfies:
```math
i f_0 U^S = J_0
```

where ``J_0`` is the **horizontal Jacobian**:
```math
J_0 = \frac{\partial(M^*, M_{\tilde{z}})}{\partial(\tilde{z}, s^*)} = (LA)^* \partial_{s^*}(LA) - M^*_{s^*} \cdot (M_{\tilde{z}})_{\tilde{z}}
```

**Expanded form** using ``M = (f_0^2/N^2) A_z`` and ``M_{\tilde{z}} = LA``:
```math
J_0 = (LA)^* \partial_{s^*}(LA) - \frac{f_0^2}{N^2} (\partial_{s^*} A_z^*) \cdot \partial_z(LA)
```

**Extracting real velocities** from ``U^S = J_0 / (i f_0) = -i J_0 / f_0``:
```math
u_S = \frac{\text{Im}(J_0)}{f_0}, \quad v_S = -\frac{\text{Re}(J_0)}{f_0}
```

**Two-term structure**:
- **First term** ``(LA)^* \partial_{s^*}(LA)``: Primary Stokes drift from wave velocity gradients
- **Second term** ``-(f_0^2/N^2)(\partial_{s^*} A_z^*) \partial_z(LA)``: Correction from vertical structure of wave envelope

**Spectral space computation** of ``\partial_{s^*}``:
```math
\partial_{s^*} = \frac{1}{2}(\partial_x + i\partial_y) \quad \Rightarrow \quad \widehat{\partial_{s^*} f} = \frac{1}{2}(i k_x - k_y) \hat{f}
```

#### Vertical Stokes Drift (Wagner & Young 2016, eq. 3.19-3.20)

The vertical Stokes drift satisfies:
```math
i f_0 w^S = K_0^* - K_0 = -2i \, \text{Im}(K_0)
```

Therefore:
```math
w_S = -\frac{2 \, \text{Im}(K_0)}{f_0}
```

**The ``K_0`` Jacobian**:
```math
K_0 = \frac{\partial(M^*, M_s)}{\partial(\tilde{z}, s^*)} = M^*_z \cdot M_{ss^*} - M^*_{s^*} \cdot M_{sz}
```

where ``M = a \cdot A_z`` with ``a = f_0^2/N^2``.

**Detailed expansion of each term**:

| Term | Expression | Computation |
|:-----|:-----------|:------------|
| ``M^*_z`` | ``\partial_z(a A_z^*)`` | ``a_z A_z^* + a A_{zz}^*`` |
| ``M_{ss^*}`` | ``\partial_s \partial_{s^*}(a A_z)`` | ``\frac{a}{4} \nabla_H^2 A_z`` |
| ``M^*_{s^*}`` | ``\partial_{s^*}(a A_z^*)`` | ``a (\partial_{s^*} A_z)^* = a (A_{zs})^*`` |
| ``M_{sz}`` | ``\partial_s \partial_z(a A_z)`` | ``a_z A_{zs} + a A_{zzs}`` |

where:
- ``a_z = \partial_z(f_0^2/N^2)`` captures stratification variations
- ``A_{zs} = \partial_s(A_z) = \frac{1}{2}(\partial_x - i\partial_y) A_z``
- ``A_{zzs} = \partial_s(A_{zz})``

**Spectral space computation**:
- ``\nabla_H^2 A_z \to -k_h^2 \hat{A}_z``
- ``A_{zs} \to \frac{1}{2}(i k_x + k_y) \hat{A}_z``
- Vertical derivatives use finite differences

---

### Total Velocity Summary

The complete velocity used for particle advection is:

```math
\boxed{
\begin{aligned}
u_{total} &= u_{QG} + u_{wave} + u_S \\
v_{total} &= v_{QG} + v_{wave} + v_S \\
w_{total} &= w_{QG} + w_S
\end{aligned}
}
```

| Component | Horizontal | Vertical |
|:----------|:-----------|:---------|
| **QG** | ``-\psi_y, +\psi_x`` | Omega equation |
| **Wave** | ``\text{Re}(LA), \text{Im}(LA)`` | (included in QG or YBJ) |
| **Stokes** | ``\text{Im}(J_0)/f_0, -\text{Re}(J_0)/f_0`` | ``-2\text{Im}(K_0)/f_0`` |

---

### Advection Options

The particle advection behavior can be controlled via `ParticleConfig` options:

#### 2D vs 3D Advection (`use_3d_advection`)

| Setting | Behavior |
|:--------|:---------|
| `use_3d_advection = true` (default) | Full 3D advection with vertical velocity |
| `use_3d_advection = false` | **Horizontal-only advection at constant z** |

When `use_3d_advection = false`:
- The vertical velocity ``w`` is **not computed** (skipped in `compute_total_velocities!`)
- The ``dz/dt = w`` time stepping is **skipped** in the advection loop
- Particles remain at their initial depth levels

This provides a **performance benefit** since vertical velocity computation (omega equation, vertical Stokes drift) is expensive. Useful for:
- Tracking particles on specific isopycnal surfaces
- Studying horizontal dispersion without vertical mixing
- Comparing with drifter observations at fixed depths

**Example:**
```julia
# Create configuration for 2D horizontal advection only
config = ParticleConfig(
    z_level = -500.0,           # Initial depth (particles stay here)
    use_3d_advection = false,   # Disable vertical advection
    # ... other options
)
```

#### Vertical Velocity Source (`use_ybj_w`)

| Setting | Vertical velocity source |
|:--------|:-------------------------|
| `use_ybj_w = false` (default) | QG omega equation: ``w_{QG}`` |
| `use_ybj_w = true` | YBJ wave-induced: ``w_{YBJ}`` |

This option only affects `w_{QG}` in the total velocity; Stokes drift ``w_S`` is always included when `use_3d_advection = true`.

---

### Implementation Notes

1. **Order of operations**: QG velocities are computed first, then wave velocity and Stokes drift are **added** in-place
2. **Spectral vs physical space**: Derivatives are computed in spectral space; products (Jacobians) are computed in physical space
3. **Vertical derivatives**: Use second-order finite differences with one-sided stencils at boundaries
4. **Stratification profile**: The code supports both constant ``N^2`` and depth-varying ``N^2(z)``

### References

- **Asselin, O. & Young, W. R.** (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Fluid Mech.*, 876, 428-448.
- **Wagner, G. L. & Young, W. R.** (2016). A three-component model for the coupled evolution of near-inertial waves, quasi-geostrophic flow and the near-inertial second harmonic. *J. Fluid Mech.*, 802, 806-837.
- **Xie, J.-H. & Vanneste, J.** (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.

## Quick Start

### Co-Evolution with Fluid (Recommended)

Particles can be passed directly to the timestep functions to co-evolve with the wave and mean flow equations. This ensures particles use the same `dt` as the fluid simulation:

```julia
using QGYBJplus

# Setup model first
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(par)

# Create particle configuration (100 particles, default Euler integration)
# NOTE: x_max, y_max are REQUIRED - use G.Lx, G.Ly from grid
particle_config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Create and initialize particle tracker
tracker = ParticleTracker(particle_config, G)
initialize_particles!(tracker, particle_config)

# Particles co-evolve automatically with the fluid
# Option 1: Leapfrog time stepping
first_projection_step!(S, G, par, plans; a=a, particle_tracker=tracker, current_time=0.0)
for step in 1:nsteps
    current_time = step * par.dt
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a,
                   particle_tracker=tracker, current_time=current_time)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end

# Option 2: IMEX time stepping
first_imex_step!(S, G, par, plans, imex_ws; a=a, particle_tracker=tracker, current_time=0.0)
for step in 1:nsteps
    current_time = step * par.dt
    imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a=a,
                  particle_tracker=tracker, current_time=current_time)
    Sn, Snp1 = Snp1, Sn
end

# Save trajectories
write_particle_trajectories("particles.nc", tracker)
```

### Manual Advection (Alternative)

For more control, particles can be advected manually:

```julia
# Create particle configuration
# NOTE: x_max, y_max are REQUIRED - use G.Lx, G.Ly from grid
particle_config = particles_in_box(-2000.0;
    x_max=G.Lx, y_max=G.Ly,  # REQUIRED
    nx=10, ny=10,
    save_interval=0.1
)

# Create particle tracker
tracker = ParticleTracker(particle_config, sim.grid)
initialize_particles!(tracker, particle_config)

# Advect particles manually after each timestep
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, par.dt, sim.current_time)
end

# Save trajectories
write_particle_trajectories("particles.nc", tracker)
```

## Generalized Lagrangian Mean (GLM) Framework

QGYBJ+.jl implements particle advection using the **Generalized Lagrangian Mean (GLM)** framework, which cleanly separates the slowly-evolving mean trajectory from fast wave oscillations.

### GLM Position Decomposition

The physical particle position ``\mathbf{x}(t)`` is decomposed into:

```math
\mathbf{x}(t) = \mathbf{X}(t) + \boldsymbol{\xi}(\mathbf{X}(t), t)
```

where:
- ``\mathbf{X}(t)`` is the **mean position** (Lagrangian-mean trajectory)
- ``\boldsymbol{\xi}`` is the **wave displacement** (fast oscillatory component)

### Mean Position Advection

The mean position evolves according to the **QG velocity** (which is the Lagrangian-mean flow):

```math
\frac{d\mathbf{X}}{dt} = \mathbf{u}^L_{QG}(\mathbf{X}, t)
```

This is time-stepped using **Euler method**:

```math
\mathbf{X}^{n+1} = \mathbf{X}^n + \Delta t \cdot \mathbf{u}_{QG}(\mathbf{X}^n, t^n)
```

### Wave Displacement Reconstruction

The wave displacement is reconstructed from the wave velocity amplitude ``LA``:

```math
\xi_x + i\xi_y = \text{Re}\left\{\frac{LA(\mathbf{X}, t)}{-if_0} e^{-if_0 t}\right\}
```

This captures the oscillatory motion of particles due to near-inertial waves. The wave displacement:
- Has zero time-mean (pure oscillation at frequency ``f_0``)
- Is computed **diagnostically** from the current wave field
- Does not require time-stepping (instantaneous reconstruction)

### Physical Position

The full physical position at each output time is:

```math
\mathbf{x}^{n+1} = \mathbf{X}^{n+1} + \boldsymbol{\xi}^{n+1}
```

### Advantages of GLM Approach

| Aspect | Benefit |
|:-------|:--------|
| **Numerical stability** | Mean trajectory uses larger ``\Delta t`` (not constrained by wave period) |
| **Physical clarity** | Separates slow (QG) and fast (wave) dynamics |
| **Efficiency** | Wave displacement is diagnostic, not prognostic |
| **Accuracy** | No accumulation of phase errors in wave oscillations |

## Time Integration

Particle mean positions are advected using **Euler method**:

```math
\mathbf{X}^{n+1} = \mathbf{X}^n + \Delta t \cdot \mathbf{u}_{QG}(\mathbf{X}^n, t^n)
```

```julia
# Create particle configuration (Euler is the only method)
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)
```

The Euler method is well-suited for GLM advection because:
- QG velocities vary slowly compared to the timestep
- Wave effects are captured through displacement reconstruction, not velocity integration
- Simplicity and efficiency for co-evolution with the fluid solver

## Interpolation Methods

Velocity must be interpolated from the grid to particle positions.

### Trilinear (Default)
- **Stencil**: 2×2×2 = 8 points
- **Order**: O(h²)
- **Smoothness**: C⁰ continuous

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=TRILINEAR)
```

### Tricubic
- **Stencil**: 4×4×4 = 64 points (Catmull-Rom splines)
- **Order**: O(h⁴)
- **Smoothness**: C¹ continuous

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=TRICUBIC)
```

### Quintic
- **Stencil**: 6×6×6 = 216 points (B-splines)
- **Order**: O(h⁶)
- **Smoothness**: C⁴ continuous

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=QUINTIC)
```

### Adaptive
Automatically selects trilinear or tricubic based on local field smoothness.

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, interpolation_method=ADAPTIVE)
```

| Method | Points | Error | Best For |
|:-------|:-------|:------|:---------|
| `TRILINEAR` | 8 | O(h²) | Speed, rough fields |
| `TRICUBIC` | 64 | O(h⁴) | Accuracy, smooth fields |
| `QUINTIC` | 216 | O(h⁶) | Highest accuracy |
| `ADAPTIVE` | 8-64 | Variable | Mixed conditions |

## Particle Initialization

QGYBJ+.jl provides simple, intuitive constructors for initializing particles:

| Constructor | Description |
|:------------|:------------|
| `particles_in_box(z; ...)` | Uniform grid in a 2D rectangular box at fixed z |
| `particles_in_circle(z; ...)` | Circular disk at fixed z (sunflower/rings/random) |
| `particles_in_grid_3d(; ...)` | Uniform 3D rectangular grid |
| `particles_in_layers(z_levels; ...)` | Multiple 2D grids at different z-levels |
| `particles_random_3d(n; ...)` | Random distribution in 3D volume |
| `particles_custom(positions; ...)` | User-specified positions |

Note: the vertical coordinate is `z ∈ [-Lz, 0]` with `z = 0` at the surface. Use negative `z` for a positive depth (e.g., depth 2000 m → `z = -2000.0`).

### Particles in a Box (2D at fixed z)

```julia
# 100 particles (10×10) in a box at z = -2000 m (depth 2000 m)
# NOTE: x_max, y_max are REQUIRED
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Custom subdomain
config = particles_in_box(-2000.0;
    x_min=100e3, x_max=400e3,  # subset of domain
    y_min=100e3, y_max=400e3,
    nx=20, ny=20               # 400 particles
)
```

### Particles in a Circle (2D at fixed z)

```julia
# 100 particles in a circle of radius 1.0 at z = -π/2
config = particles_in_circle(-π/2; radius=1.0, n=100)

# Custom center and pattern
config = particles_in_circle(-1.0;
    center=(2.0, 2.0),        # Circle center
    radius=1.5,
    n=200,
    pattern=:sunflower        # :sunflower, :rings, or :random
)
```

**Available patterns:**
- `:sunflower` - Fibonacci spiral (very uniform, recommended)
- `:rings` - Concentric rings
- `:random` - Uniform random within disk

Single-level distributions (like `particles_in_circle` and `particles_custom`) can use `z_min == z_max`.

### Particles in a 3D Grid

```julia
# 500 particles in a 10×10×5 grid
# NOTE: x_max, y_max, z_max are REQUIRED
config = particles_in_grid_3d(; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, nx=10, ny=10, nz=5)

# Custom subdomain
config = particles_in_grid_3d(;
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    z_min=-2500.0, z_max=-500.0,
    nx=8, ny=8, nz=4
)
```

### Particles in Layers (multiple z-levels)

```julia
# 300 particles at 3 z-levels (10×10 per level)
# NOTE: x_max, y_max are REQUIRED
config = particles_in_layers([-1000.0, -2000.0, -3000.0]; x_max=G.Lx, y_max=G.Ly, nx=10, ny=10)

# Custom horizontal subdomain
config = particles_in_layers([-500.0, -1000.0, -1500.0, -2000.0];
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    nx=5, ny=5
)
```

### Random 3D Distribution

```julia
# 500 random particles in full domain
# NOTE: x_max, y_max, z_max are REQUIRED
config = particles_random_3d(500; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz)

# Custom subdomain with seed
config = particles_random_3d(1000;
    x_min=100e3, x_max=400e3,
    y_min=100e3, y_max=400e3,
    z_min=-2500.0, z_max=-500.0,
    seed=42
)
```

### Custom Positions

```julia
# Particles at specific (x, y, z) locations
config = particles_custom([
    (1.0, 1.0, -0.5),
    (2.0, 2.0, -1.0),
    (3.0, 1.5, -0.75),
    (1.5, 3.0, -1.25)
])
```

## Boundary Conditions

### Horizontal (Periodic)
Particles wrap around domain edges:
```julia
x_new = mod(x, Lx)
y_new = mod(y, Ly)
```

### Vertical (Reflective)
Particles bounce off top and bottom:
```julia
if z > 0
    z = -z
    w = -w  # Reverse vertical velocity
elseif z < -Lz
    z = -2*Lz - z
    w = -w
end
```

Configure via:
```julia
config = particles_in_box(-2000.0;
    x_max=G.Lx, y_max=G.Ly,
    periodic_x=true,
    periodic_y=true,
    reflect_z=true      # Reflective vertical BCs
)
```

## Delayed Particle Release

Start advecting particles after the flow has developed:

```julia
config = particles_in_box(-2000.0; x_max=G.Lx, y_max=G.Ly, particle_advec_time=100.0)  # Start at t=100.0
```

Particles remain stationary until `current_time >= particle_advec_time`.

## Trajectory Output

### Save Interval

Control how often positions are recorded:
```julia
config = particles_in_box(-2000.0;
    x_max=G.Lx, y_max=G.Ly,
    save_interval=10.0,       # Save every 10.0 time units
    max_save_points=1000      # Max points per file
)
```

### Automatic File Splitting

For long simulations:
```julia
tracker = ParticleTracker(config, grid)
enable_auto_file_splitting!(tracker, "long_run", max_points_per_file=500)

# Files created: long_run.nc, long_run_part1.nc, long_run_part2.nc, ...
```

### Writing Trajectories

```julia
# Standard output
write_particle_trajectories("particles.nc", tracker)

# With metadata
write_particle_trajectories("particles.nc", tracker;
    metadata = Dict("experiment" => "test1", "description" => "...")
)

# By z-level (for layered distributions)
write_particle_trajectories_by_zlevel("particles", tracker)
# Creates: particles_z0.nc, particles_z1.nc, ...
```

## Parallel Algorithm

When running with MPI, particle advection uses domain decomposition.

### Domain Decomposition

The domain is split in both x and y according to the MPI process grid (px × py).
Each rank owns a tile in the horizontal plane and the full z-range.

Example for a 2×2 topology:

```
┌─────────────────────────┬─────────────────────────┐
│ Rank (0,1)               │ Rank (1,1)              │
│ x∈[0, Lx/2), y∈[Ly/2, Ly) │ x∈[Lx/2, Lx), y∈[Ly/2, Ly) │
├─────────────────────────┼─────────────────────────┤
│ Rank (0,0)               │ Rank (1,0)              │
│ x∈[0, Lx/2), y∈[0, Ly/2)  │ x∈[Lx/2, Lx), y∈[0, Ly/2)  │
└─────────────────────────┴─────────────────────────┘
```

Particles belong to the rank that owns their (x, y) position.

### Halo Exchange

For interpolation near domain boundaries, velocity data is exchanged between neighbors:

```
┌─────────────────────────────────────────────────────────────┐
│                     HALO EXCHANGE                           │
│                                                             │
│   Rank 0                        Rank 1                      │
│   ┌─────────────────┐          ┌─────────────────┐          │
│   │ Local │  Right  │          │ Left  │ Local   │          │
│   │ Data  │  Halo   │  ←────→  │ Halo  │ Data    │          │
│   │       │ (ghost) │          │(ghost)│         │          │
│   └───────┴─────────┘          └───────┴─────────┘          │
│                                                             │
│   • Rank 0 sends RIGHT edge → Rank 1's LEFT halo            │
│   • Rank 1 sends LEFT edge  → Rank 0's RIGHT halo           │
│                                                             │
│   Halo width depends on interpolation: 1 (trilinear), 2     │
│   (tricubic), 3 (quintic/adaptive)                          │
└─────────────────────────────────────────────────────────────┘
```

For 2D topologies (py > 1), halos are exchanged in both x and y directions,
including corner halos needed by wider stencils.

### Particle Migration

When particles cross domain boundaries, they are transferred:

```
┌─────────────────────────────────────────────────────────────┐
│                   PARTICLE MIGRATION                        │
│                                                             │
│   1. After advection, check each particle's position        │
│   2. If (x,y) outside local domain → pack into send buffer  │
│   3. MPI.Alltoall to exchange particle counts               │
│   4. MPI.Send/Recv to transfer particle data                │
│   5. Unpack received particles into local collection        │
│                                                             │
│   Particle data transferred: [x, y, z, u, v, w]             │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Timestep Workflow

```
┌───────────────────────────────────────────────────────────────┐
│              PARALLEL GLM ADVECTION TIMESTEP                  │
│                                                               │
│  1. UPDATE VELOCITY FIELDS                                    │
│     • Compute QG velocities (distributed FFT)                 │
│     • Solve omega equation (tridiagonal in z)                 │
│     • Compute wave amplitude LA = L⁺A + (k_h²/4)A             │
│     • Exchange velocity halos in x/y (and corners for 2D)     │
│                              ↓                                │
│  2. ADVECT MEAN POSITIONS (each rank processes local parts)   │
│     • Interpolate QG velocity (use halo for boundary parts)   │
│     • Euler step: X^{n+1} = X^n + Δt × u_QG                   │
│                              ↓                                │
│  3. RECONSTRUCT WAVE DISPLACEMENT                             │
│     • Interpolate LA at mean positions                        │
│     • ξ = Re{(LA/(-if)) × e^{-ift}} (diagnostic)              │
│     • Physical position: x = X + ξ                            │
│                              ↓                                │
│  4. MIGRATE PARTICLES                                         │
│     • Identify particles that left local domain               │
│     • Exchange particle data between ranks (MPI)              │
│                              ↓                                │
│  5. APPLY BOUNDARY CONDITIONS                                 │
│     • Periodic wrap in x, y                                   │
│     • Reflective bounce in z                                  │
│                              ↓                                │
│  6. SAVE TRAJECTORIES (if save_interval reached)              │
│     • Each rank saves local particles, or                     │
│     • Gather to rank 0 for unified output                     │
└───────────────────────────────────────────────────────────────┘
```

### Using Parallel Particles

```julia
using MPI
using QGYBJplus

MPI.Init()

# Set up parallel configuration
parallel_config = setup_mpi_environment()

# Create particle tracker with parallel support
tracker = ParticleTracker(particle_config, sim.grid, parallel_config)
initialize_particles!(tracker, particle_config)

# Advection automatically handles:
# - Halo exchange for boundary interpolation
# - Particle migration between ranks
for step in 1:nsteps
    timestep!(sim)
    advect_particles!(tracker, sim.state, sim.grid, dt, sim.current_time)
end

MPI.Finalize()
```

## Key Data Structures

### ParticleConfig

```julia
struct ParticleConfig{T}
    # Spatial domain
    x_min, x_max, y_min, y_max::T
    z_level::T

    # Particle count
    nx_particles, ny_particles::Int

    # Physics
    use_ybj_w::Bool           # YBJ vs QG vertical velocity
    use_3d_advection::Bool    # Include vertical advection

    # Timing
    particle_advec_time::T    # Delayed start time

    # Numerics
    interpolation_method::InterpolationMethod  # TRILINEAR, etc.

    # Boundaries
    periodic_x, periodic_y::Bool
    reflect_z::Bool

    # I/O
    save_interval::T
    max_save_points::Int
    auto_split_files::Bool
end
```

### ParticleTracker

```julia
mutable struct ParticleTracker{T}  # Simplified view (omits I/O bookkeeping)
    config::ParticleConfig{T}
    particles::ParticleState{T}   # x, y, z, id, u, v, w arrays

    # Grid info
    nx, ny, nz::Int
    Lx, Ly, Lz, dx, dy, dz::T

    # Velocity workspace
    u_field, v_field, w_field::Array{T,3}

    # MPI info (for parallel)
    comm, rank, nprocs, is_parallel
    local_domain::NamedTuple  # x/y bounds, local sizes, topology info
    halo_info::HaloInfo{T}
    send_buffers, recv_buffers::Vector{Vector{T}}
    is_io_rank::Bool
    gather_for_io::Bool
end
```

## Performance Considerations

| Aspect | Serial | Parallel |
|:-------|:-------|:---------|
| Velocity computation | O(N) | O(N/P) per rank |
| Interpolation | O(Np × stencil) | O(Np/P × stencil) |
| Halo exchange | N/A | O((nx_local + ny_local) × nz × halo_width) |
| Migration | N/A | O(Np_crossing) |

**Tips:**
- Use `TRILINEAR` for speed, `TRICUBIC` for accuracy
- GLM framework with Euler is efficient: wave effects are diagnostic, not integrated
- Halo exchange overhead is small for typical particle counts
- Migration cost depends on flow strength near boundaries

## Visualization

### Plot Particle Positions

```julia
using Plots

# 2D scatter plot
scatter(tracker.particles.x, tracker.particles.y,
    markersize=2, alpha=0.6,
    xlabel="x", ylabel="y",
    title="Particle Distribution"
)
```

### Plot Trajectories

```julia
# Load saved trajectories
using NCDatasets
ds = NCDataset("particles.nc")
x_hist = ds["x"][:]  # (np, ntime)
y_hist = ds["y"][:]
close(ds)

# Plot first 50 particle tracks
p = plot(legend=false)
for i in 1:50
    plot!(p, x_hist[i,:], y_hist[i,:], alpha=0.3)
end
display(p)
```

### Animation

```julia
anim = @animate for t in 1:10:size(x_hist, 2)
    scatter(x_hist[:,t], y_hist[:,t],
        markersize=2, xlim=(0,2π), ylim=(0,2π),
        title="t = $(t)")
end
gif(anim, "particles.gif", fps=20)
```

## API Reference

See the [Particle API Reference](../api/particles.md) for complete documentation of:

**Types:**
- [`ParticleConfig`](@ref) - Configuration options
- [`ParticleState`](@ref) - Particle positions, IDs, and velocities
- [`ParticleTracker`](@ref) - Main tracking object

**Initialization Constructors:**
- [`particles_in_box`](@ref) - 2D box at fixed z-level
- [`particles_in_circle`](@ref) - Circular disk at fixed z-level
- [`particles_in_grid_3d`](@ref) - Uniform 3D grid
- [`particles_in_layers`](@ref) - Multiple z-levels
- [`particles_random_3d`](@ref) - Random 3D distribution
- [`particles_custom`](@ref) - User-specified positions

**Core Functions:**
- [`initialize_particles!`](@ref) - Initialize particle positions
- [`advect_particles!`](@ref) - Advect particles one timestep
- [`interpolate_velocity_at_position`](@ref) - Velocity interpolation
- [`write_particle_trajectories`](@ref) - Save to NetCDF
