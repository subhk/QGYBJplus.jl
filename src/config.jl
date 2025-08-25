"""
Configuration and user interface for QG-YBJ model.

This module provides a user-friendly interface for setting up domains, 
initial conditions, stratification, and output configurations.
"""

using ..QGYBJ: QGParams, Grid, State
using NCDatasets

"""
    DomainConfig

Configuration for computational domain.
"""
Base.@kwdef struct DomainConfig{T}
    # Grid resolution
    nx::Int = 64
    ny::Int = 64
    nz::Int = 64
    
    # Domain size
    Lx::T = 2π
    Ly::T = 2π
    Lz::T = 2π
    
    # Physical domain size (optional, for dimensional analysis)
    dom_x_m::Union{T,Nothing} = nothing  # meters
    dom_y_m::Union{T,Nothing} = nothing  # meters  
    dom_z_m::Union{T,Nothing} = nothing  # meters
end

"""
    StratificationConfig

Configuration for background stratification.
"""
Base.@kwdef struct StratificationConfig{T}
    type::Symbol = :constant_N  # :constant_N, :skewed_gaussian, :from_file, :tanh_profile
    
    # For constant N
    N0::T = 1.0
    
    # For skewed Gaussian (from Fortran parameters)
    N02_sg::T = 0.537713935783168
    N12_sg::T = 2.684198470106461
    sigma_sg::T = 0.648457170048730
    z0_sg::T = 6.121537923499139
    alpha_sg::T = -5.338431587899242
    
    # For tanh profile
    N_upper::T = 0.01   # Upper ocean N
    N_lower::T = 0.02  # Deep ocean N
    z_pycno::T = 0.5    # Pycnocline depth (fraction of domain)
    width::T = 0.1     # Transition width
    
    # From file
    filename::Union{String,Nothing} = nothing
end

"""
    InitialConditionConfig

Configuration for initial conditions.
"""
Base.@kwdef struct InitialConditionConfig{T}
    # Stream function initialization
    psi_type::Symbol = :analytical  # :analytical, :from_file, :random
    psi_filename::Union{String,Nothing} = nothing
    psi_amplitude::T = 1.0
    
    # Wave field (L+A) initialization  
    wave_type::Symbol = :zero  # :zero, :analytical, :from_file, :random
    wave_filename::Union{String,Nothing} = nothing
    wave_amplitude::T = 1e-3
    
    # Random seed for reproducibility
    random_seed::Int = 1234
end

"""
    OutputConfig

Configuration for model output.
"""
Base.@kwdef struct OutputConfig{T}
    # Output directory
    output_dir::String = "./output"
    
    # Time intervals for output (in model time units)
    psi_interval::T = 1.0
    wave_interval::T = 1.0
    diagnostics_interval::T = 0.1
    
    # Output format
    output_format::Symbol = :netcdf  # :netcdf, :hdf5
    
    # State file naming pattern
    state_file_pattern::String = "state%04d.nc"  # e.g., state0001.nc, state0002.nc
    
    # Variables to save
    save_psi::Bool = true
    save_waves::Bool = true
    save_velocities::Bool = true
    save_vertical_velocity::Bool = false
    save_vorticity::Bool = false
    save_diagnostics::Bool = true
end

"""
    ModelConfig

Complete model configuration combining all components.
"""
Base.@kwdef struct ModelConfig{T}
    domain::DomainConfig{T}
    stratification::StratificationConfig{T}
    initial_conditions::InitialConditionConfig{T}
    output::OutputConfig{T}
    
    # Physical parameters
    f0::T = 1.0        # Coriolis parameter
    
    # Time stepping
    dt::T = 1e-3
    total_time::T = 10.0
    
    # Numerical parameters
    nu_h::T = 0.0      # Horizontal viscosity
    nu_v::T = 0.0      # Vertical viscosity
    
    # Model switches
    linear::Bool = false
    inviscid::Bool = true
    no_dispersion::Bool = false
    passive_scalar::Bool = false
    ybj_plus::Bool = true
    
    # Wave-mean flow interaction controls
    no_wave_feedback::Bool = false     # true: waves don't affect mean flow (qw = 0)
    fixed_mean_flow::Bool = false      # true: mean flow doesn't evolve in time
    
    # Legacy compatibility
    no_feedback::Bool = false          # Deprecated: use no_wave_feedback instead
end

"""
    create_domain_config(; kwargs...)

Create a domain configuration with user-friendly parameters.

# Examples
```julia
# Simple cubic domain
domain = create_domain_config(nx=128, ny=128, nz=64, Lx=4π, Ly=4π, Lz=2π)

# With physical dimensions
domain = create_domain_config(
    nx=256, ny=256, nz=128,
    dom_x_m=314159.0,  # ~314 km
    dom_y_m=314159.0,  # ~314 km  
    dom_z_m=4000.0     # 4 km depth
)
```
"""
function create_domain_config(; kwargs...)
    T = Float64
    return DomainConfig{T}(; kwargs...)
end

"""
    create_stratification_config(type::Symbol; kwargs...)

Create stratification configuration.

# Examples
```julia
# Constant N
strat = create_stratification_config(:constant_N, N0=2.0)

# Skewed Gaussian (using default parameters from Fortran code)
strat = create_stratification_config(:skewed_gaussian)

# Tanh profile (pycnocline-like)
strat = create_stratification_config(:tanh_profile, 
    N_upper=0.01, N_lower=0.025, z_pycno=0.6, width=0.05)

# From NetCDF file
strat = create_stratification_config(:from_file, filename="N2_profile.nc")
```
"""
function create_stratification_config(type::Symbol; kwargs...)
    T = Float64
    return StratificationConfig{T}(; type=type, kwargs...)
end

"""
    create_initial_condition_config(; kwargs...)

Create initial condition configuration.

# Examples
```julia
# Analytical initialization
init = create_initial_condition_config(
    psi_type=:analytical, 
    wave_type=:random,
    wave_amplitude=1e-4
)

# From files
init = create_initial_condition_config(
    psi_type=:from_file,
    psi_filename="psi_initial.nc",
    wave_type=:from_file, 
    wave_filename="wave_initial.nc"
)
```
"""
function create_initial_condition_config(; kwargs...)
    T = Float64
    return InitialConditionConfig{T}(; kwargs...)
end

"""
    create_output_config(; kwargs...)

Create output configuration.

# Examples
```julia
# Basic output every model time unit
output = create_output_config(
    output_dir="./my_run",
    psi_interval=0.5,
    wave_interval=0.5
)

# Minimal output for long runs
output = create_output_config(
    psi_interval=10.0,
    wave_interval=10.0,
    save_velocities=false,
    save_diagnostics=false
)
```
"""
function create_output_config(; kwargs...)
    T = Float64
    return OutputConfig{T}(; kwargs...)
end

"""
    create_model_config(domain, stratification, initial_conditions, output; kwargs...)

Create complete model configuration.
"""
function create_model_config(domain, stratification, initial_conditions, output; kwargs...)
    T = Float64
    return ModelConfig{T}(;
        domain=domain,
        stratification=stratification, 
        initial_conditions=initial_conditions,
        output=output,
        kwargs...
    )
end

"""
    validate_config(config::ModelConfig)

Validate model configuration and check for consistency.
"""
function validate_config(config::ModelConfig)
    errors = String[]
    warnings = String[]
    
    # Domain validation
    if config.domain.nx < 8 || config.domain.ny < 8 || config.domain.nz < 8
        push!(errors, "Grid resolution too small (minimum 8 points per dimension)")
    end
    
    if config.domain.Lx <= 0 || config.domain.Ly <= 0 || config.domain.Lz <= 0
        push!(errors, "Domain sizes must be positive")
    end
    
    # Check if grid resolution is reasonable for domain size
    dx = config.domain.Lx / config.domain.nx
    dy = config.domain.Ly / config.domain.ny  
    dz = config.domain.Lz / config.domain.nz
    
    if dx > config.domain.Lx / 8
        push!(warnings, "Horizontal resolution may be too coarse (< 8 points per domain)")
    end
    
    # Time stepping validation
    if config.dt <= 0
        push!(errors, "Time step must be positive")
    end
    
    if config.total_time <= 0
        push!(errors, "Total integration time must be positive")
    end
    
    # CFL-like check (rough estimate)
    if config.dt > min(dx, dy, dz) / 10
        push!(warnings, "Time step may be too large for stability")
    end
    
    # Physical parameter validation
    if config.f0 <= 0
        push!(errors, "Coriolis parameter f0 must be positive")
    end
    
    # File existence checks
    if config.initial_conditions.psi_type == :from_file
        if isnothing(config.initial_conditions.psi_filename)
            push!(errors, "Psi filename required when psi_type=:from_file")
        elseif !isfile(config.initial_conditions.psi_filename)
            push!(errors, "Psi file not found: $(config.initial_conditions.psi_filename)")
        end
    end
    
    if config.initial_conditions.wave_type == :from_file
        if isnothing(config.initial_conditions.wave_filename)
            push!(errors, "Wave filename required when wave_type=:from_file")
        elseif !isfile(config.initial_conditions.wave_filename)
            push!(errors, "Wave file not found: $(config.initial_conditions.wave_filename)")
        end
    end
    
    if config.stratification.type == :from_file
        if isnothing(config.stratification.filename)
            push!(errors, "Stratification filename required when type=:from_file")
        elseif !isfile(config.stratification.filename)
            push!(errors, "Stratification file not found: $(config.stratification.filename)")
        end
    end
    
    # Output directory
    if !isdir(dirname(config.output.output_dir))
        push!(warnings, "Output directory parent does not exist: $(dirname(config.output.output_dir))")
    end
    
    return errors, warnings
end

"""
    print_config_summary(config::ModelConfig)

Print a summary of the model configuration.
"""
function print_config_summary(config::ModelConfig)
    println("=== QG-YBJ Model Configuration Summary ===")
    println()
    
    println("Domain:")
    println("  Grid: $(config.domain.nx) × $(config.domain.ny) × $(config.domain.nz)")
    println("  Size: $(config.domain.Lx) × $(config.domain.Ly) × $(config.domain.Lz)")
    if !isnothing(config.domain.dom_x_m)
        println("  Physical: $(config.domain.dom_x_m/1000) × $(config.domain.dom_y_m/1000) × $(config.domain.dom_z_m/1000) km")
    end
    println()
    
    println("Stratification:")
    println("  Type: $(config.stratification.type)")
    if config.stratification.type == :constant_N
        println("  N₀: $(config.stratification.N0)")
    elseif config.stratification.type == :from_file
        println("  File: $(config.stratification.filename)")
    end
    println()
    
    println("Initial Conditions:")
    println("  Psi: $(config.initial_conditions.psi_type)")
    println("  Waves: $(config.initial_conditions.wave_type)")
    println()
    
    println("Parameters:")
    println("  f0 = $(config.f0)")
    println("  dt = $(config.dt), T_total = $(config.total_time)")
    println()
    
    println("Output:")
    println("  Directory: $(config.output.output_dir)")
    println("  Psi interval: $(config.output.psi_interval)")
    println("  Wave interval: $(config.output.wave_interval)")
    println()
    
    println("Model switches:")
    println("  Linear: $(config.linear)")
    println("  Inviscid: $(config.inviscid)")
    println("  YBJ+: $(config.ybj_plus)")
    println("  No wave feedback: $(config.no_wave_feedback)")
    println("  Fixed mean flow: $(config.fixed_mean_flow)")
    if config.no_feedback != config.no_wave_feedback
        println("  Warning: Legacy no_feedback differs from no_wave_feedback")
    end
end