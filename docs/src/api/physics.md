# [Physics Functions](@id api-physics)

```@meta
CurrentModule = QGYBJplus
```

This page documents the physics functions in QGYBJ+.jl.

## Elliptic Inversions

### Streamfunction Inversion

```@docs
invert_q_to_psi!
```

**Solves:** ``\nabla^2\psi + \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right) = q``

**Usage:**
```julia
# Serial mode
invert_q_to_psi!(state, grid; a=a_ell)

# Parallel mode (with workspace for 2D decomposition)
invert_q_to_psi!(state, grid; a=a_ell, workspace=workspace)
# Updates state.psi from state.q
```

### Wave Amplitude Inversion

```@docs
invert_L⁺A_to_A!
```

**Solves:** ``\frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial A}{\partial z}\right) - \frac{k_h^2}{4}A = L^+A``

**Usage:**
```julia
# Serial mode
invert_L⁺A_to_A!(state, grid, params, a_ell)

# Parallel mode (with workspace for 2D decomposition)
invert_L⁺A_to_A!(state, grid, params, a_ell; workspace=workspace)
# Updates state.A from state.L⁺A
```

### Helmholtz Solver

```@docs
invert_helmholtz!
```

## Nonlinear Terms

### Jacobian

```@docs
jacobian_spectral!
```

**Computes:** ``J(a, b) = \frac{\partial a}{\partial x}\frac{\partial b}{\partial y} - \frac{\partial a}{\partial y}\frac{\partial b}{\partial x}``

### Wave Advection and Refraction

The wave nonlinear terms are documented in the [Time Stepping API](timestepping.md):
- `convol_waqg_L⁺A!` / `refraction_waqg_L⁺A!` / `compute_qw_complex!` - Complex L⁺A (YBJ+) operators
- `convol_waqg!` / `refraction_waqg!` / `compute_qw!` - BR/BI-decomposed operators

## Velocity Computation

```@docs
compute_velocities!
```

**Computes:**
- ``u = -\partial\psi/\partial y``
- ``v = \partial\psi/\partial x``

### Vertical Velocity

```@docs
compute_vertical_velocity!
```

### Total Velocities

```@docs
compute_total_velocities!
```

## Dissipation

### Vertical Diffusion

Dissipation functions are documented in the [Time Stepping API](timestepping.md):
- `dissipation_q_nv!` - Applies vertical diffusion ``\nu_z \partial^2 q / \partial z^2``
- `int_factor` - Integrating factor for stiff hyperdiffusion terms

## Diagnostics Functions

### Energy

**Flow Kinetic Energy:**
```@docs
flow_kinetic_energy
```

**Wave Energy:**
```@docs
wave_energy
```

### Spectral Energy Functions

The following spectral energy functions compute energy with proper dealiasing and density weighting:

```@docs
flow_kinetic_energy_spectral
flow_potential_energy_spectral
wave_energy_spectral
```

### Global Energy Functions (MPI-aware)

For parallel simulations, use these MPI-aware versions that reduce across all processes:

**Physical-space energy (simple sum):**
```@docs
flow_kinetic_energy_global
wave_energy_global
```

**Spectral energy (with dealiasing):**
```@docs
flow_kinetic_energy_spectral_global
flow_potential_energy_spectral_global
wave_energy_spectral_global
```

### Energy Diagnostics Manager

The `EnergyDiagnosticsManager` provides automatic saving of energy time series to separate NetCDF files:

```@docs
EnergyDiagnosticsManager
record_energies!
write_all_energy_files!
```

**Output Files:**
- `diagnostic/wave_KE.nc` - Wave kinetic energy
- `diagnostic/wave_PE.nc` - Wave potential energy
- `diagnostic/wave_CE.nc` - Wave correction energy (YBJ+)
- `diagnostic/mean_flow_KE.nc` - Mean flow kinetic energy
- `diagnostic/mean_flow_PE.nc` - Mean flow potential energy
- `diagnostic/total_energy.nc` - Summary with all energies

**Usage:**
```julia
# Automatic (created during setup_simulation)
sim = setup_simulation(config)
run_simulation!(sim)  # Energies saved automatically

# Manual
manager = EnergyDiagnosticsManager("output_dir"; output_interval=1.0)
record_energies!(manager, time, wke, wpe, wce, mke, mpe)
write_all_energy_files!(manager)
```

### Omega Equation

```@docs
omega_eqn_rhs!
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
dealias_mask
```

Creates a radial dealiasing mask using the 2/3 rule: modes with ``k_h^2 > k_{max}^2`` where ``k_{max} = \min(n_x, n_y) / 3`` are zeroed.

### Hyperdiffusion Parameters

Helper functions for computing scale-selective hyperdiffusion coefficients:

```@docs
compute_hyperdiff_coeff
compute_hyperdiff_params
```

**4th Order Hyperdiffusion (Biharmonic):**

The model supports 4th order horizontal hyperdiffusion (∇⁴ operator) for scale-selective damping of grid-scale noise while preserving large scales:

```julia
# Compute coefficient for 10-step e-folding at grid scale
hd = compute_hyperdiff_params(
    nx=128, ny=128, Lx=70e3, Ly=70e3, dt=10.0,
    order=4, efold_steps=10
)

# Use in parameters
par = default_params(
    nx=128, ny=128, nz=64,
    Lx=70e3, Ly=70e3, Lz=3000.0,
    νₕ₁=hd.ν, ilap1=hd.ilap,  # 4th order hyperdiffusion
    νₕ₂=0.0                    # Disable 2nd hyperviscosity slot
)
```

**Damping Rate:**

The damping rate at wavenumber ``k`` is:
- 2nd order (∇²): ``\lambda = \nu_2 k^2``
- 4th order (∇⁴): ``\lambda = \nu_4 k^4``
- 8th order (∇⁸): ``\lambda = \nu_8 k^8``

Higher orders provide more scale-selective damping, concentrating dissipation at the smallest scales.

## YBJ Normal Mode Functions

```@docs
sumL⁺A!
compute_sigma
compute_A!
```

## Function Signatures Summary

| Function | Input | Output | In-place |
|:---------|:------|:-------|:---------|
| `invert_q_to_psi!` | q | psi | Yes |
| `invert_L⁺A_to_A!` | L⁺A | A, C | Yes |
| `jacobian_spectral!` | a, b | J(a,b) | Yes |
| `compute_velocities!` | psi | u, v | Yes |
| `flow_kinetic_energy` | u, v | scalar | No |
| `flow_kinetic_energy_global` | u, v, mpi_config | scalar | No |
| `wave_energy_vavg` | L⁺A, A | WKE | No |
| `wave_energy_global` | L⁺A, A, mpi_config | (E_L⁺A, E_A) | No |

## Performance Notes

- All physics functions are **in-place** to avoid allocations
- FFT plans are **pre-computed** for efficiency
- Tridiagonal systems use **Thomas algorithm** (O(n))
- Functions are **type-stable** for optimal JIT compilation

## 2D Decomposition Notes

Functions requiring vertical operations automatically detect 2D decomposition and use the appropriate method:

| Function | Serial | Parallel (2D) |
|:---------|:-------|:--------------|
| `invert_q_to_psi!` | Direct solve | Transpose → solve → transpose |
| `invert_L⁺A_to_A!` | Direct solve | Transpose → solve → transpose |
| `compute_vertical_velocity!` | Direct solve | Transpose → solve → transpose |
| `dissipation_q_nv!` | Direct | Transpose if needed |

**Pattern:**
```julia
need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)
if need_transpose
    _function_2d!(...)   # Uses transpose operations
else
    _function_direct!(...)  # Direct vertical access
end
```

**Workspace requirement:** Pass `workspace` argument for parallel mode to avoid repeated allocation of z-pencil arrays.
