# QG-YBJ Model User Interface Guide

This guide describes the new user-friendly interface for the QG-YBJ model, designed to make it easy to set up simulations with various configurations, initial conditions, and output options.

## Quick Start

The simplest way to run a simulation:

```julia
using QGYBJ

# Run a basic simulation with default parameters
sim = run_simple_simulation(
    nx=64, ny=64, nz=32,           # Grid resolution
    total_time=10.0,               # Integration time
    output_dir="./my_simulation"   # Output directory
)
```

## Configuration System

The interface is built around a configuration system with four main components:

### 1. Domain Configuration

Set up the computational domain:

```julia
domain = create_domain_config(
    nx=128, ny=128, nz=64,         # Grid points
    Lx=4π, Ly=4π, Lz=2π,          # Domain size (nondimensional)
    dom_x_m=314159.0,              # Physical size in meters (optional)
    dom_y_m=314159.0,              # Physical size in meters (optional)  
    dom_z_m=4000.0                 # Physical size in meters (optional)
)
```

### 2. Stratification Configuration

Choose from several stratification profiles:

#### Constant N²
```julia
stratification = create_stratification_config(
    :constant_N,
    N0=1.0  # Buoyancy frequency
)
```

#### Tropopause-like Profile
```julia
stratification = create_stratification_config(
    :tanh_profile,
    N_trop=0.01,      # Tropospheric N
    N_strat=0.04,     # Stratospheric N  
    z_trop=0.6,       # Tropopause height (fraction)
    width=0.05        # Transition width
)
```

#### Skewed Gaussian (from Fortran test cases)
```julia
stratification = create_stratification_config(:skewed_gaussian)
```

#### From NetCDF File
```julia
stratification = create_stratification_config(
    :from_file,
    filename="N2_profile.nc"
)
```

### 3. Initial Conditions Configuration

Set up initial fields:

#### Random Fields
```julia
initial_conditions = create_initial_condition_config(
    psi_type=:random,              # Random stream function
    wave_type=:random,             # Random wave field
    psi_amplitude=0.1,             # Stream function amplitude
    wave_amplitude=0.01,           # Wave field amplitude
    random_seed=1234               # For reproducibility
)
```

#### Analytical Fields
```julia
initial_conditions = create_initial_condition_config(
    psi_type=:analytical,          # Analytical stream function
    wave_type=:analytical,         # Analytical wave field
    psi_amplitude=0.2,
    wave_amplitude=0.005
)
```

#### From NetCDF Files
```julia
initial_conditions = create_initial_condition_config(
    psi_type=:from_file,
    psi_filename="psi_initial.nc",
    wave_type=:from_file,
    wave_filename="wave_initial.nc"
)
```

### 4. Output Configuration

Control what gets saved and when:

```julia
output = create_output_config(
    output_dir="./my_run",
    psi_interval=1.0,              # Save stream function every 1 time unit
    wave_interval=1.0,             # Save wave fields every 1 time unit
    diagnostics_interval=0.5,      # Save diagnostics every 0.5 time units
    state_file_pattern="state%04d.nc",  # Files: state0001.nc, state0002.nc, ...
    save_psi=true,                 # Save stream function
    save_waves=true,               # Save wave fields (L+A real and imaginary parts)
    save_velocities=true,          # Save velocity fields
    save_vorticity=false,          # Save vorticity field
    save_diagnostics=true          # Save diagnostic quantities
)
```

## Complete Simulation Setup

Combine all configurations:

```julia
using QGYBJ

# 1. Create configurations
domain = create_domain_config(nx=128, ny=128, nz=64, Lx=6π, Ly=6π, Lz=2π)

stratification = create_stratification_config(
    :tanh_profile,
    N_trop=0.01, N_strat=0.03, z_trop=0.6, width=0.05
)

initial_conditions = create_initial_condition_config(
    psi_type=:random,
    wave_type=:random,
    psi_amplitude=0.1,
    wave_amplitude=0.01
)

output = create_output_config(
    output_dir="./stratified_run",
    psi_interval=0.5,
    wave_interval=0.5,
    diagnostics_interval=0.1
)

# 2. Create complete model configuration
config = create_model_config(
    domain, stratification, initial_conditions, output,
    Ro=0.1,                        # Rossby number
    Fr=0.1,                        # Froude number  
    f0=1.0,                        # Coriolis parameter
    dt=1e-3,                       # Time step
    total_time=20.0,               # Total simulation time
    linear=false,                  # Include nonlinear terms
    inviscid=true,                 # No viscosity
    ybj_plus=true,                 # Use YBJ+ formulation
    no_feedback=false              # Include wave-mean flow feedback
)

# 3. Set up and run simulation
sim = setup_simulation(config)
run_simulation!(sim)
```

## NetCDF File Formats

### State Files (state0001.nc, state0002.nc, ...)

Each state file contains:

**Dimensions:**
- `x`: Zonal coordinate (nx points)
- `y`: Meridional coordinate (ny points) 
- `z`: Vertical coordinate (nz points)
- `time`: Time coordinate (1 point per file)

**Variables:**
- `psi(x,y,z)`: Stream function [m²/s]
- `LAr(x,y,z)`: Wave field L+A real part [wave amplitude]
- `LAi(x,y,z)`: Wave field L+A imaginary part [wave amplitude]
- `u(x,y,z)`: Zonal velocity [m/s] (if enabled)
- `v(x,y,z)`: Meridional velocity [m/s] (if enabled)
- `time`: Model time [time units]

### Initial Condition Files

For initializing from files, use the same format as state files:

**Stream function file (psi_initial.nc):**
```
dimensions: x, y, z, time
variables: psi(x,y,z), time
```

**Wave field file (wave_initial.nc):**
```
dimensions: x, y, z, time  
variables: LAr(x,y,z), LAi(x,y,z), time
```

### Stratification Profile Files

**Stratification file (N2_profile.nc):**
```
dimensions: z
variables: 
  - z(z): Height coordinate [m]
  - N2(z): Buoyancy frequency squared [s⁻²]
```

## Model Physics Options

Control various physical aspects:

```julia
config = create_model_config(
    domain, stratification, initial_conditions, output,
    
    # Physical parameters
    Ro=0.1,                        # Rossby number (U/fL)
    Fr=0.1,                        # Froude number (U/NH)  
    f0=1.0,                        # Coriolis parameter
    
    # Time stepping
    dt=1e-3,                       # Time step
    total_time=50.0,               # Total integration time
    
    # Numerical parameters
    nu_h=0.0,                      # Horizontal viscosity
    nu_v=0.0,                      # Vertical viscosity
    
    # Model physics switches
    linear=false,                  # true: Linear dynamics only
    inviscid=true,                 # true: No dissipation
    no_dispersion=false,           # true: No wave dispersion
    passive_scalar=false,          # true: Waves as passive scalars
    ybj_plus=true,                 # true: Use YBJ+ formulation
    
    # Wave-mean flow interaction controls
    no_wave_feedback=false,        # true: Waves don't affect mean flow (qw = 0)
    fixed_mean_flow=false,         # true: Mean flow doesn't evolve in time
    
    # Legacy compatibility
    no_feedback=false              # Deprecated: use no_wave_feedback instead
)
```

## Wave-Mean Flow Interaction Controls

The QG-YBJ model includes sophisticated controls for wave-mean flow interactions, allowing you to isolate different physical processes:

### **Full Coupling (Default)**
```julia
config = create_model_config(...,
    no_wave_feedback=false,        # Waves affect mean flow
    fixed_mean_flow=false          # Mean flow evolves
)
```
- Complete wave-mean flow interaction
- Waves generate potential vorticity forcing (qw ≠ 0)
- Mean flow evolves due to its own nonlinearity AND wave feedback
- Most physically realistic for studying wave-mean flow dynamics

### **Fixed Mean Flow**
```julia
config = create_model_config(...,
    no_wave_feedback=true,         # Waves don't affect mean flow
    fixed_mean_flow=true           # Mean flow doesn't evolve
)
```
- Mean flow remains constant in time
- Waves evolve in a prescribed, static background
- Useful for studying wave propagation and breaking
- Equivalent to solving wave equation in fixed background state

### **No Wave Feedback**
```julia
config = create_model_config(...,
    no_wave_feedback=true,         # Waves don't affect mean flow
    fixed_mean_flow=false          # Mean flow can still evolve
)
```
- Mean flow evolves due to its own nonlinear dynamics only
- Waves evolve independently (no feedback qw = 0)
- Allows studying separate evolution of waves and mean flow
- Useful for process isolation studies

### **Wave-Only Dynamics**
```julia
config = create_model_config(...,
    no_wave_feedback=true,         # Waves don't affect mean flow
    fixed_mean_flow=true,          # Mean flow doesn't evolve
    # Initialize with zero or minimal mean flow
    psi_amplitude=0.0
)
```
- Pure wave dynamics in quiescent background
- No mean flow present or evolving
- Ideal for studying wave-wave interactions
- Equivalent to solving just the wave equation

### **Linear Wave Evolution**
```julia
config = create_model_config(...,
    no_wave_feedback=true,         # Waves don't affect mean flow
    fixed_mean_flow=true,          # Mean flow doesn't evolve
    linear=true                    # Linear wave dynamics only
)
```
- Linear wave equation in prescribed flow
- No wave-wave interactions
- Useful for studying wave refraction, dispersion
- Allows analytical comparison and validation

### **Practical Examples**

**Studying Wave Breaking:**
```julia
# Strong jet with small wave perturbations
initial_conditions = create_initial_condition_config(
    psi_type=:analytical,          # Prescribed jet
    wave_type=:random,
    psi_amplitude=0.3,             # Strong mean flow
    wave_amplitude=0.01            # Small wave amplitude
)

config = create_model_config(...,
    fixed_mean_flow=true,          # Keep jet fixed
    no_wave_feedback=true          # Waves don't affect jet
)
```

**Mean Flow Adjustment:**
```julia
# Start with out-of-balance initial condition
config = create_model_config(...,
    no_wave_feedback=false,        # Allow wave feedback
    fixed_mean_flow=false,         # Allow mean flow evolution
    wave_amplitude=0.1             # Significant wave field
)
```

**Process Isolation:**
```julia
# Study mean flow turbulence without waves
config = create_model_config(...,
    no_wave_feedback=true,         # No wave effects
    fixed_mean_flow=false,         # Mean flow evolves
    wave_amplitude=0.0             # No waves
)
```

## Diagnostics and Monitoring

The simulation automatically computes and saves diagnostic quantities:

- **Energy diagnostics**: Kinetic energy, potential energy, wave energy
- **Domain integrals**: Total enstrophy
- **Field statistics**: Min, max, RMS values
- **Time series**: All diagnostics vs time

Access diagnostics from the simulation object:
```julia
# After running simulation
energy_data = sim.diagnostics["step_1000"]
println("Kinetic energy: ", energy_data["kinetic_energy"])
```

## Progress Monitoring

Add a custom progress callback:

```julia
function my_progress_callback(sim)
    if sim.time_step % 1000 == 0
        println("Step $(sim.time_step), t=$(sim.current_time)")
        
        # Custom analysis or early stopping conditions
        if some_condition
            return :stop  # Request early termination
        end
    end
end

run_simulation!(sim; progress_callback=my_progress_callback)
```

## Examples

See `examples/demo_user_interface.jl` for comprehensive examples including:

1. **Basic simulation** with random initial conditions
2. **Stratified case** with tropopause-like profile
3. **File-based initialization** from NetCDF files
4. **Parameter sweep** over Rossby numbers
5. **Custom stratification** profiles
6. **Simple interface** for quick tests

## Error Handling and Validation

The interface includes comprehensive validation:

- **Configuration validation**: Checks for consistent parameters
- **File validation**: Verifies input files exist and have correct format
- **Physical validation**: Ensures parameters are physically reasonable  
- **Stability checks**: Monitors for numerical blow-up during integration

Validation errors and warnings are reported during setup:

```julia
sim = setup_simulation(config)
# Will report any configuration issues before starting
```

## Advanced Usage

### Custom Initial Conditions

Create custom initial condition functions:

```julia
function my_custom_init!(psi, B, grid)
    # Custom initialization logic
    # psi: stream function (spectral)
    # B: wave field (spectral)  
    # grid: Grid object with dimensions
end

# Apply during setup
initialize_from_config(config, grid, state, plans)
my_custom_init!(state.psi, state.B, grid)
```

### Custom Stratification Profiles

Implement new stratification types by extending the system:

```julia
struct MyCustomProfile{T} <: StratificationProfile{T}
    # Custom parameters
end

function evaluate_N2(profile::MyCustomProfile, z::Real)
    # Return N²(z) for your profile
end
```

This interface provides a complete, user-friendly way to set up and run QG-YBJ simulations with all the features you requested: domain configuration, file-based initialization, various stratification options, and standardized NetCDF output with time intervals.