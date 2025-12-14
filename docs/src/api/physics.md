# [Physics Functions](@id api-physics)

```@meta
CurrentModule = QGYBJ
```

This page documents the physics functions in QGYBJ.jl.

## Elliptic Inversions

### Streamfunction Inversion

```@docs
invert_q_to_psi!
```

**Solves:** ``\nabla^2\psi + \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right) = q``

**Usage:**
```julia
invert_q_to_psi!(state, grid, params, a_ell)
# Updates state.psi from state.q
```

### Wave Amplitude Inversion

```@docs
invert_B_to_A!
```

**Solves:** ``\frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right) - \frac{k_h^2}{4}A = B``

**Usage:**
```julia
invert_B_to_A!(state, grid, params, a_ell)
# Updates state.A from state.B
```

## Nonlinear Terms

### Jacobian

```@docs
jacobian_spectral!
```

**Computes:** ``J(a, b) = \frac{\partial a}{\partial x}\frac{\partial b}{\partial y} - \frac{\partial a}{\partial y}\frac{\partial b}{\partial x}``

**Usage:**
```julia
jacobian_spectral!(Jab, a, b, grid, work, plans)
# Jab contains spectral Jacobian
```

### Advection

```@docs
convol_qg!
convol_waqg!
```

**QG Advection:** ``J(\psi, q)`` - Advection of PV by streamfunction

**Wave Advection:** ``J(\psi, B)`` - Advection of wave envelope

### Refraction

```@docs
refraction_waqg!
```

**Computes:** ``B \frac{\partial\zeta}{\partial t}`` - Wave refraction by vorticity changes

### Wave Feedback

```@docs
wavefb!
```

**Computes:** ``q^w = \frac{1}{2}\nabla_h^2|A|^2 - \frac{1}{2}\frac{\partial^2|A|^2}{\partial z^2}``

## Velocity Computation

```@docs
compute_velocities!
```

**Computes:**
- ``u = -\partial\psi/\partial y``
- ``v = \partial\psi/\partial x``

**Usage:**
```julia
compute_velocities!(state, grid, plans)
# Updates state.u and state.v from state.psi
```

## Dispersion

### Wave Dispersion

```@docs
wave_dispersion!
```

**Computes:** ``-i\frac{N^2}{2f_0}\nabla_h^2 A``

This term causes horizontal spreading of wave packets.

## Dissipation

### Hyperdiffusion

```@docs
apply_dissipation!
```

**Applies:**
```math
\mathcal{D} = -\nu_{h1}(-\nabla^2)^{p_1} - \nu_{h2}(-\nabla^2)^{p_2} - \nu_z\frac{\partial^2}{\partial z^2}
```

### Integrating Factors

```@docs
compute_integrating_factors
apply_integrating_factor!
```

For handling stiff diffusion terms efficiently.

## Diagnostics Functions

### Energy

```@docs
flow_kinetic_energy
flow_potential_energy
flow_total_energy
wave_energy
```

### Enstrophy

```@docs
relative_enstrophy
potential_enstrophy
```

### Spectra

```@docs
horizontal_energy_spectrum
vertical_energy_spectrum
```

### Omega Equation

```@docs
compute_omega
```

**Solves:**
```math
\nabla^2 w + \frac{N^2}{f_0^2}\frac{\partial^2 w}{\partial z^2} = 2J\left(\frac{\partial\psi}{\partial z}, \nabla^2\psi\right)
```

## Transform Functions

### Forward Transforms

```@docs
fft_forward!
```

Real space → Spectral space

### Backward Transforms

```@docs
fft_backward!
```

Spectral space → Real space

### Dealiasing

```@docs
apply_dealias!
```

Removes aliased modes using 2/3 rule.

## Initialization Functions

### Flow Initialization

```@docs
initialize_random_flow!
initialize_vortex!
initialize_from_spectrum!
```

### Wave Initialization

```@docs
initialize_random_waves!
initialize_plane_wave!
initialize_wave_packet!
```

## Utility Functions

### Gradients

```@docs
compute_gradient_x!
compute_gradient_y!
compute_gradient_z!
horizontal_laplacian!
```

### Interpolation

```@docs
interpolate_to_point
interpolate_to_particles!
```

### Statistics

```@docs
domain_average
horizontal_mean
vertical_mean
compute_rms
```

## Function Signatures Summary

| Function | Input | Output | In-place |
|:---------|:------|:-------|:---------|
| `invert_q_to_psi!` | q | psi | Yes |
| `invert_B_to_A!` | B | A | Yes |
| `jacobian_spectral!` | a, b | J(a,b) | Yes |
| `compute_velocities!` | psi | u, v | Yes |
| `flow_kinetic_energy` | u, v | scalar | No |
| `wave_energy` | B, A | (E_B, E_A) | No |

## Performance Notes

- All physics functions are **in-place** to avoid allocations
- FFT plans are **pre-computed** for efficiency
- Tridiagonal systems use **Thomas algorithm** (O(n))
- Functions are **type-stable** for optimal JIT compilation
