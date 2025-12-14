# [API Reference](@id api-index)

```@meta
CurrentModule = QGYBJ
```

Complete API reference for QGYBJ.jl.

## Quick Links

- [Core Types](@ref api-types): `QGParams`, `Grid`, `State`
- [Physics Functions](@ref api-physics): Inversions, operators, diagnostics

## All Exported Functions

### Setup and Initialization

```@docs
create_simple_config
run_simple_simulation
Grid
QGParams
create_state
create_work_arrays
plan_transforms!
setup_elliptic_matrices
```

### Time Stepping

```@docs
timestep!
ab3_step!
euler_step!
```

### Elliptic Solvers

```@docs
invert_q_to_psi!
invert_B_to_A!
solve_tridiagonal!
```

### Nonlinear Operators

```@docs
jacobian_spectral!
convol_qg!
convol_waqg!
refraction_waqg!
wavefb!
```

### Transforms

```@docs
fft_forward!
fft_backward!
apply_dealias!
```

### Velocities and Gradients

```@docs
compute_velocities!
compute_vorticity
compute_gradient_x!
compute_gradient_y!
compute_gradient_z!
horizontal_laplacian!
```

### Diagnostics

```@docs
flow_kinetic_energy
flow_potential_energy
flow_total_energy
wave_energy
relative_enstrophy
potential_enstrophy
horizontal_energy_spectrum
vertical_energy_spectrum
compute_omega
```

### Stratification

```@docs
setup_stratification!
set_stratification!
get_stratification
compute_deformation_radius
compute_vertical_modes
```

### Initialization

```@docs
initialize_random_flow!
initialize_random_waves!
initialize_vortex!
initialize_plane_wave!
initialize_wave_packet!
initialize_from_spectrum!
```

### Particles

```@docs
create_particles
advect_particles!
advect_particles_2d!
interpolate_to_particles!
particle_dispersion
lagrangian_diffusivity
```

### I/O

```@docs
init_output!
write_output!
close_output!
save_checkpoint
load_checkpoint
```

### MPI Functions

Available when MPI extension is loaded:

```@docs
setup_mpi_environment
init_mpi_grid
init_mpi_state
plan_mpi_transforms!
gather_to_root!
scatter_from_root!
mpi_reduce_sum
mpi_barrier
```

## Module Structure

```
QGYBJ
├── Core Types
│   ├── QGParams
│   ├── Grid
│   └── State
├── Physics
│   ├── elliptic.jl      # Inversions
│   ├── nonlinear.jl     # Jacobians, refraction
│   ├── operators.jl     # Velocities, gradients
│   └── transforms.jl    # FFT wrappers
├── Diagnostics
│   ├── diagnostics.jl   # Energy, enstrophy
│   └── omega.jl         # Omega equation
├── Time Stepping
│   └── timestep.jl      # AB3 integrator
├── Initialization
│   ├── initial_conditions.jl
│   └── stratification.jl
├── Particles
│   ├── particles.jl     # Core particle types
│   ├── advection.jl     # Particle advection
│   └── interpolation.jl # Field interpolation
└── I/O
    └── io.jl            # NetCDF, checkpoints
```

## Naming Conventions

| Suffix | Meaning | Example |
|:-------|:--------|:--------|
| `!` | In-place modification | `compute_velocities!` |
| `_spectral` | Operates in spectral space | `jacobian_spectral!` |
| `_qg` | QG-specific | `convol_qg!` |
| `_waqg` | Wave-related | `convol_waqg!` |
| `_mpi` | MPI-enabled version | `init_mpi_grid` |

## Type Parameters

Many functions accept type parameters:

```julia
# Default (Float64)
state = create_state(grid)

# Single precision
state = create_state(grid; T=Float32)
```

## Error Handling

Functions throw informative errors:

```julia
try
    grid = Grid(nx=-64, ny=64, nz=32)
catch e
    println(e)  # "nx must be positive"
end
```

## Thread Safety

- Core computation functions are **thread-safe**
- I/O functions should be called from **single thread**
- MPI functions follow **collective semantics**
