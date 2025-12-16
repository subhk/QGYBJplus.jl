## Configuration

QGYBJ.jl provides a high‑level configuration system for domains,
stratification, initial conditions, and output.

### Domain

```julia
domain = create_domain_config(
    nx=128, ny=128, nz=64,
    Lx=4π, Ly=4π, Lz=2π,
)
```

### Stratification

```julia
# Constant N²
strat = create_stratification_config(:constant_N, N0=1.0)

# Skewed Gaussian (Fortran test case parameters)
strat = create_stratification_config(:skewed_gaussian)

# Tanh profile
strat = create_stratification_config(:tanh_profile,
    N_upper=0.01, N_lower=0.025, z_pycno=0.6, width=0.05)

# From NetCDF file containing an N²(z) variable
strat = create_stratification_config(:from_file, filename="N2_profile.nc")
```

### Initial Conditions

```julia
init = create_initial_condition_config(
    psi_type=:random,     # :analytical, :from_file, :random
    wave_type=:random,    # :zero, :analytical, :from_file, :random
    wave_amplitude=1e-3,
    random_seed=1234,
)
```

### Output

```julia
output = create_output_config(
    output_dir="./output",
    psi_interval=1.0,
    wave_interval=1.0,
    diagnostics_interval=0.5,
    save_psi=true,
    save_waves=true,
    save_velocities=true,
    save_vertical_velocity=false,
)
```

### Combine into a ModelConfig

```julia
config = create_model_config(domain, strat, init, output;
    dt=1e-3, total_time=10.0,
    f₀=1.0,                    # Coriolis parameter (type: f\_0<tab>)
    # Model switches
    ybj_plus=true,
    no_wave_feedback=false,
    fixed_mean_flow=false,
)

# Validate and print a summary
errors, warnings = validate_config(config)
print_config_summary(config)
```

