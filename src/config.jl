"""
Configuration and user interface for QG-YBJ model.

This module provides a user-friendly interface for setting up domains, 
initial conditions, stratification, and output configurations.
"""

using ..QGYBJplus: QGParams, Grid, State

"""
    DomainConfig

Configuration for computational domain.

# Fields
- `nx, ny, nz`: Grid resolution (default: 64)
- `Lx, Ly, Lz`: Domain size in meters (REQUIRED - no default)
- `x0, y0`: Domain origin (default: 0.0)

# Example
```julia
domain = DomainConfig{Float64}(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km
```
"""
Base.@kwdef struct DomainConfig{T}
    # Grid resolution
    nx::Int = 64
    ny::Int = 64
    nz::Int = 64

    # Domain size (REQUIRED - no defaults)
    Lx::T
    Ly::T
    Lz::T

    # Domain origin (default: 0, use -Lx/2 or -Ly/2 for centered domains)
    x0::T = 0.0
    y0::T = 0.0
end

"""
    StratificationConfig

Configuration for background stratification.

# Supported stratification types
- `:constant_N` - Uniform buoyancy frequency N throughout the domain (default)
- `:skewed_gaussian` - Realistic pycnocline with skewed Gaussian N² profile
- `:tanh_profile` - Tanh transition between upper and lower N values
- `:from_file` - Load N² profile from NetCDF file
- `:analytical` (alias `:function`) - User-specified analytic N(z) or N²(z)
"""
Base.@kwdef struct StratificationConfig{T}
    type::Symbol = :constant_N  # Supported: :constant_N, :skewed_gaussian, :tanh_profile, :from_file, :analytical
    
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
    z_pycno::T = 0.5    # Pycnocline depth (positive below surface, same units as Lz)
    width::T = 0.1     # Transition width (same units as z)
    
    # From file
    filename::Union{String,Nothing} = nothing

    # Analytical profile
    N_func::Union{Nothing, Function} = nothing   # N(z) in s^-1
    N2_func::Union{Nothing, Function} = nothing  # N²(z) in s^-2
end

"""
    InitialConditionConfig

Configuration for initial conditions.

# Wave types
- `:zero` - No waves (default)
- `:analytical` - Analytic wave pattern
- `:random` - Random wave spectrum
- `:from_file` - Read waves from NetCDF
- `:surface_waves` - Surface-confined waves with configurable vertical profile
- `:surface_exponential` - Exponentially decaying surface waves (convenience alias)
- `:surface_gaussian` - Gaussian-decaying surface waves (convenience alias)
"""
Base.@kwdef struct InitialConditionConfig{T}
    # Stream function initialization
    psi_type::Symbol = :analytical  # :analytical, :from_file, :random
    psi_filename::Union{String,Nothing} = nothing
    psi_amplitude::T = 1.0
    
    # Wave field (L+A) initialization  
    wave_type::Symbol = :zero  # :zero, :analytical, :from_file, :random, :surface_waves
    wave_filename::Union{String,Nothing} = nothing
    wave_amplitude::T = 1e-3

    # Surface wave initialization (used when wave_type is surface waves)
    wave_surface_depth::T = 30.0      # e-folding depth [m]
    wave_profile::Symbol = :gaussian  # :gaussian or :exponential
    wave_uniform::Bool = true         # Horizontally uniform waves

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
    f0::T = 1.0        # Coriolis parameter (can be negative for southern hemisphere)

    # Time stepping
    dt::T = 1e-3
    total_time::T = 10.0

    # Numerical parameters - basic viscosity
    nu_h::T = 0.0      # Horizontal viscosity (legacy, prefer hyperdiffusion)
    nu_v::T = 0.0      # Vertical viscosity

    # Hyperdiffusion for mean flow: ν₁(-∇²)^ilap1 + ν₂(-∇²)^ilap2
    nu_h1::T = 0.01    # First hyperviscosity coefficient for mean flow
    nu_h2::T = 10.0    # Second hyperviscosity coefficient for mean flow
    ilap1::Int = 2     # First Laplacian power (2 = biharmonic)
    ilap2::Int = 6     # Second Laplacian power (6 = hyper-6)

    # Hyperdiffusion for waves
    nu_h1_wave::T = 0.0   # First hyperviscosity coefficient for waves
    nu_h2_wave::T = 10.0  # Second hyperviscosity coefficient for waves
    ilap1_wave::Int = 2   # First Laplacian power for waves
    ilap2_wave::Int = 6   # Second Laplacian power for waves

    # Model switches
    # NOTE: These defaults differ from default_params() for historical reasons:
    #   - ModelConfig: inviscid=true, no_wave_feedback=false (idealized runs)
    #   - default_params: inviscid=false, no_wave_feedback=true (production runs)
    # When using setup_model_with_config(), these ModelConfig values are used.
    # When using default_params() directly, that function's defaults apply.
    linear::Bool = false
    inviscid::Bool = true              # NOTE: default_params() uses inviscid=false
    no_dispersion::Bool = false
    passive_scalar::Bool = false
    ybj_plus::Bool = true

    # Wave-mean flow interaction controls
    no_wave_feedback::Bool = false     # NOTE: default_params() uses no_wave_feedback=true
    fixed_mean_flow::Bool = false      # true: mean flow doesn't evolve in time

    # Legacy compatibility
    no_feedback::Bool = false          # Deprecated: use no_wave_feedback instead
end

"""
    create_domain_config(; Lx, Ly, Lz, kwargs...)

Create a domain configuration with user-friendly parameters.

# Arguments
- `Lx, Ly, Lz`: Domain size in meters (REQUIRED - no defaults)
- `nx, ny, nz`: Grid resolution (default: 64)
- `x0, y0`: Domain origin (default: 0.0)

# Examples
```julia
# Mesoscale ocean domain (500km × 500km × 4km depth)
domain = create_domain_config(Lx=500e3, Ly=500e3, Lz=4000.0, nx=128, ny=128, nz=64)

# Large domain (1000km × 1000km × 5km depth)
domain = create_domain_config(
    nx=256, ny=256, nz=128,
    Lx=1000e3, Ly=1000e3, Lz=5000.0
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
# Constant N (SUPPORTED)
strat = create_stratification_config(:constant_N, N0=2.0)

# Skewed Gaussian (SUPPORTED - uses default parameters from Fortran code)
strat = create_stratification_config(:skewed_gaussian)

# Tanh profile
# strat = create_stratification_config(:tanh_profile,
#     N_upper=0.01, N_lower=0.025, z_pycno=0.6, width=0.05)

# From file
# strat = create_stratification_config(:from_file, filename="N2_profile.nc")

# Analytical N(z)
# N_func = z -> 0.01 - 2e-6 * z
# strat = create_stratification_config(:analytical, N_func=N_func)
```

See [`StratificationConfig`](@ref) for details on supported types.
"""
function create_stratification_config(type::Symbol; kwargs...)
    T = Float64
    return StratificationConfig{T}(; type=type, kwargs...)
end

function create_stratification_config(; type::Symbol=:constant_N, kwargs...)
    return create_stratification_config(type; kwargs...)
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

# Exponentially decaying surface waves
init = create_initial_condition_config(
    wave_type=:surface_exponential,
    wave_amplitude=0.1,
    wave_surface_depth=50.0
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

    if dx > config.domain.Lx / 8 || dy > config.domain.Ly / 8
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
    # Note: f0 can be negative for southern hemisphere simulations
    if config.f0 == 0
        push!(errors, "Coriolis parameter f0 cannot be zero (use negative values for southern hemisphere)")
    end

    # Stratification validation
    supported_stratifications = (:constant_N, :skewed_gaussian, :tanh_profile, :from_file, :analytical, :function)
    if config.stratification.type ∉ supported_stratifications
        push!(errors, "Unknown stratification type :$(config.stratification.type). " *
                     "Supported types: $(supported_stratifications)")
    end

    # Wave type validation
    supported_wave_types = (:zero, :analytical, :random, :from_file,
                            :surface_waves, :surface_exponential, :surface_gaussian)
    if config.initial_conditions.wave_type ∉ supported_wave_types
        push!(errors, "Unknown wave_type :$(config.initial_conditions.wave_type). " *
                      "Supported types: $(supported_wave_types)")
    end

    if config.initial_conditions.wave_type in (:surface_waves, :surface_exponential, :surface_gaussian)
        if config.initial_conditions.wave_surface_depth <= 0
            push!(errors, "wave_surface_depth must be positive for surface waves " *
                          "(got $(config.initial_conditions.wave_surface_depth))")
        end
        if config.initial_conditions.wave_type == :surface_waves
            if config.initial_conditions.wave_profile ∉ (:gaussian, :exponential)
                push!(errors, "wave_profile must be :gaussian or :exponential for surface_waves " *
                              "(got $(config.initial_conditions.wave_profile))")
            end
        end
    end

    # N0 validation for constant_N
    if config.stratification.type == :constant_N && config.stratification.N0 <= 0
        push!(errors, "Stratification N0 must be positive for constant_N (got N0=$(config.stratification.N0))")
    end

    if config.stratification.type in (:analytical, :function)
        if isnothing(config.stratification.N_func) && isnothing(config.stratification.N2_func)
            push!(errors, "Provide N_func or N2_func for stratification type :$(config.stratification.type)")
        end
        if !isnothing(config.stratification.N_func) && !isnothing(config.stratification.N2_func)
            push!(warnings, "Both N_func and N2_func are set; N2_func will take precedence")
        end
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
    
    # Output directory - check if parent directory exists (for nested paths)
    # Note: dirname("./output") returns "." which always exists, so also check for absolute paths
    parent_dir = dirname(config.output.output_dir)
    if !isempty(parent_dir) && parent_dir != "." && !isdir(parent_dir)
        push!(warnings, "Output directory parent does not exist: $parent_dir")
    end

    # Viscosity validation (must be non-negative)
    if config.nu_h < 0
        push!(errors, "Horizontal viscosity nu_h must be non-negative (got nu_h=$(config.nu_h))")
    end
    if config.nu_v < 0
        push!(errors, "Vertical viscosity nu_v must be non-negative (got nu_v=$(config.nu_v))")
    end

    # Hyperdiffusion coefficient validation (must be non-negative)
    if config.nu_h1 < 0
        push!(errors, "Hyperviscosity nu_h1 must be non-negative (got nu_h1=$(config.nu_h1))")
    end
    if config.nu_h2 < 0
        push!(errors, "Hyperviscosity nu_h2 must be non-negative (got nu_h2=$(config.nu_h2))")
    end
    if config.nu_h1_wave < 0
        push!(errors, "Wave hyperviscosity nu_h1_wave must be non-negative (got nu_h1_wave=$(config.nu_h1_wave))")
    end
    if config.nu_h2_wave < 0
        push!(errors, "Wave hyperviscosity nu_h2_wave must be non-negative (got nu_h2_wave=$(config.nu_h2_wave))")
    end

    # Laplacian power validation (must be positive integers)
    if config.ilap1 < 1
        push!(errors, "Laplacian power ilap1 must be >= 1 (got ilap1=$(config.ilap1))")
    end
    if config.ilap2 < 1
        push!(errors, "Laplacian power ilap2 must be >= 1 (got ilap2=$(config.ilap2))")
    end
    if config.ilap1_wave < 1
        push!(errors, "Wave Laplacian power ilap1_wave must be >= 1 (got ilap1_wave=$(config.ilap1_wave))")
    end
    if config.ilap2_wave < 1
        push!(errors, "Wave Laplacian power ilap2_wave must be >= 1 (got ilap2_wave=$(config.ilap2_wave))")
    end

    # FFT efficiency warning
    if !ispow2(config.domain.nx) || !ispow2(config.domain.ny)
        push!(warnings, "Grid dimensions (nx=$(config.domain.nx), ny=$(config.domain.ny)) are not powers of 2 - FFTs may be slower")
    end

    # Deprecation warning for no_feedback
    if config.no_feedback
        push!(warnings, "Field 'no_feedback' is deprecated, use 'no_wave_feedback' instead. " *
                       "Currently no_feedback=$(config.no_feedback), no_wave_feedback=$(config.no_wave_feedback)")
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
    println("  Origin: ($(config.domain.x0), $(config.domain.y0))")
    # Print physical size in km if values are large (likely in meters)
    if config.domain.Lx > 1000 || config.domain.Ly > 1000 || config.domain.Lz > 100
        println("  Physical: $(config.domain.Lx/1000) × $(config.domain.Ly/1000) × $(config.domain.Lz/1000) km")
    end
    println()
    
    println("Stratification:")
    println("  Type: $(config.stratification.type)")
    if config.stratification.type == :constant_N
        println("  N₀: $(config.stratification.N0)")
    elseif config.stratification.type in (:analytical, :function)
        if !isnothing(config.stratification.N2_func)
            println("  Analytical N²(z): provided")
        else
            println("  Analytical N(z): provided")
        end
    elseif config.stratification.type == :from_file
        println("  File: $(config.stratification.filename)")
    end
    println()
    
    println("Initial Conditions:")
    println("  Psi: $(config.initial_conditions.psi_type)")
    println("  Waves: $(config.initial_conditions.wave_type)")
    if config.initial_conditions.wave_type in (:surface_waves, :surface_exponential, :surface_gaussian)
        effective_profile = config.initial_conditions.wave_type == :surface_exponential ? :exponential :
                           config.initial_conditions.wave_type == :surface_gaussian ? :gaussian :
                           config.initial_conditions.wave_profile
        println("  Wave profile: $(effective_profile)")
        println("  Wave surface depth: $(config.initial_conditions.wave_surface_depth)")
        println("  Wave uniform: $(config.initial_conditions.wave_uniform)")
    end
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
