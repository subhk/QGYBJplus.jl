"""
Stratification profile module for QG-YBJ model.

Provides various stratification profiles including:
- Constant N²
- Skewed Gaussian (from Fortran test cases)
- Tanh profiles (pycnocline-like)
- Exponential profiles
- Piecewise profiles
- Custom profiles from files
"""

using ..QGYBJ: Grid, QGParams

"""
    StratificationProfile{T}

Abstract type for stratification profiles.
"""
abstract type StratificationProfile{T} end

"""
    ConstantN{T} <: StratificationProfile{T}

Constant buoyancy frequency profile.
"""
struct ConstantN{T} <: StratificationProfile{T}
    N0::T
end

"""
    SkewedGaussian{T} <: StratificationProfile{T}

Skewed Gaussian profile from Fortran test cases.
"""
struct SkewedGaussian{T} <: StratificationProfile{T}
    N02::T    # Background N²
    N12::T    # Peak amplitude
    sigma::T  # Width parameter
    z0::T     # Center position
    alpha::T  # Skewness parameter
end

"""
    TanhProfile{T} <: StratificationProfile{T}

Hyperbolic tangent profile (pycnocline-like).
"""
struct TanhProfile{T} <: StratificationProfile{T}
    N_upper::T    # Upper ocean N
    N_lower::T   # Deep ocean N
    z_pycno::T    # Pycnocline depth
    width::T     # Transition width
end

"""
    ExponentialProfile{T} <: StratificationProfile{T}

Exponential stratification profile.
"""
struct ExponentialProfile{T} <: StratificationProfile{T}
    N_surface::T   # Surface N
    scale_height::T # e-folding scale
    N_deep::T      # Deep value
end

"""
    PiecewiseProfile{T} <: StratificationProfile{T}

Piecewise constant stratification.
"""
struct PiecewiseProfile{T} <: StratificationProfile{T}
    z_interfaces::Vector{T}  # Interface depths
    N_values::Vector{T}      # N values in each layer
end

"""
    FileProfile{T} <: StratificationProfile{T}

Stratification profile from file.
"""
struct FileProfile{T} <: StratificationProfile{T}
    filename::String
    z_data::Vector{T}
    N2_data::Vector{T}
end

"""
    create_stratification_profile(config::StratificationConfig)

Create stratification profile from configuration.
"""
function create_stratification_profile(config)
    T = Float64
    
    if config.type == :constant_N
        return ConstantN{T}(config.N0)
        
    elseif config.type == :skewed_gaussian
        return SkewedGaussian{T}(
            config.N02_sg,
            config.N12_sg, 
            config.sigma_sg,
            config.z0_sg,
            config.alpha_sg
        )
        
    elseif config.type == :tanh_profile
        return TanhProfile{T}(
            config.N_upper,
            config.N_lower,
            config.z_pycno,
            config.width
        )
        
    elseif config.type == :from_file
        if isnothing(config.filename)
            error("Filename required for stratification type :from_file")
        end
        return load_stratification_from_file(config.filename)
        
    else
        error("Unknown stratification type: $(config.type)")
    end
end

"""
    evaluate_N2(profile::ConstantN, z::Real)

Evaluate N² for constant stratification.
"""
function evaluate_N2(profile::ConstantN{T}, z::Real) where T
    return profile.N0^2
end

"""
    evaluate_N2(profile::SkewedGaussian, z::Real)

Evaluate N² for skewed Gaussian profile.
"""
function evaluate_N2(profile::SkewedGaussian{T}, z::Real) where T
    # Normalized coordinate
    ζ = (z - profile.z0) / profile.sigma
    
    # Skewed Gaussian function
    gaussian = exp(-ζ^2 / 2)
    skewness = 1 + profile.alpha * ζ
    
    N2 = profile.N02 + profile.N12 * gaussian * skewness
    
    return max(N2, T(0.01) * profile.N02)  # Ensure N² remains positive
end

"""
    evaluate_N2(profile::TanhProfile, z::Real)

Evaluate N² for tanh profile.
"""
function evaluate_N2(profile::TanhProfile{T}, z::Real) where T
    # Normalized height
    ζ = (z - profile.z_pycno) / profile.width
    
    # Smooth transition between upper and deep ocean
    N_interp = profile.N_upper + (profile.N_lower - profile.N_upper) * 
               (1 + tanh(ζ)) / 2
    
    return N_interp^2
end

"""
    evaluate_N2(profile::ExponentialProfile, z::Real)

Evaluate N² for exponential profile.
"""  
function evaluate_N2(profile::ExponentialProfile{T}, z::Real) where T
    N_val = profile.N_deep + (profile.N_surface - profile.N_deep) * 
            exp(-z / profile.scale_height)
    return N_val^2
end

"""
    evaluate_N2(profile::PiecewiseProfile, z::Real)

Evaluate N² for piecewise profile.
"""
function evaluate_N2(profile::PiecewiseProfile{T}, z::Real) where T
    # Find which layer z belongs to
    for i in 1:(length(profile.z_interfaces)-1)
        if profile.z_interfaces[i] <= z < profile.z_interfaces[i+1]
            return profile.N_values[i]^2
        end
    end
    
    # Default to last layer if outside bounds
    return profile.N_values[end]^2
end

"""
    evaluate_N2(profile::FileProfile, z::Real)

Evaluate N² for file-based profile using interpolation.
"""
function evaluate_N2(profile::FileProfile{T}, z::Real) where T
    # Linear interpolation
    if z <= profile.z_data[1]
        return profile.N2_data[1]
    elseif z >= profile.z_data[end]
        return profile.N2_data[end]
    else
        # Find surrounding points
        for i in 1:(length(profile.z_data)-1)
            if profile.z_data[i] <= z <= profile.z_data[i+1]
                # Linear interpolation
                α = (z - profile.z_data[i]) / (profile.z_data[i+1] - profile.z_data[i])
                return (1-α) * profile.N2_data[i] + α * profile.N2_data[i+1]
            end
        end
    end
    
    return T(0)  # Fallback
end

"""
    compute_stratification_profile(profile::StratificationProfile, G::Grid)

Compute N² profile on model grid.
"""
function compute_stratification_profile(profile::StratificationProfile{T}, G::Grid) where T
    N2_profile = zeros(T, G.nz)
    dz = 2π / G.nz  # Assuming domain [0, 2π] - adjust as needed
    
    for k in 1:G.nz
        z = (k - 1) * dz
        N2_profile[k] = evaluate_N2(profile, z)
    end
    
    return N2_profile
end

"""
    compute_stratification_coefficients(N2_profile::Vector, G::Grid; Bu::Real=1.0)

Compute stratification-dependent coefficients for the model.

Following the Fortran init_base_state routine:
- r_1 = 1.0 (unity for Boussinesq)
- r_2 = N² (buoyancy frequency squared)
- r_3 = 0.0 (not used in standard QG formulation)
- a_ell = Bu/N² (elliptic coefficient for PV inversion)
- rho = 1.0 (Boussinesq approximation)

# Arguments
- `N2_profile::Vector`: Buoyancy frequency squared N²(z) at each vertical level
- `G::Grid`: Grid structure
- `Bu::Real`: Burger number (default 1.0)

# Returns
Named tuple with coefficients:
- `r_1`: Unity array (length nz)
- `r_2`: N² profile (length nz)
- `r_3`: Zero array (length nz)
- `a_ell_u`: Bu/N² at unstaggered points (length nz)
- `a_ell_s`: Bu/N² at staggered points (length nz)
- `rho_u`: Unity density at unstaggered points (length nz)
- `rho_s`: Unity density at staggered points (length nz)
"""
function compute_stratification_coefficients(N2_profile::Vector{T}, G::Grid; Bu::Real=T(1.0)) where T
    nz = G.nz
    dz = nz > 1 ? (G.z[2] - G.z[1]) : T(2π / nz)

    # Initialize coefficient arrays (following Fortran init_base_state)
    # Unstaggered (at cell centers)
    r_1_u = ones(T, nz)         # r₁ = 1.0 (Boussinesq)
    r_2_u = zeros(T, nz)        # r₂ = N² at unstaggered points
    r_3_u = zeros(T, nz)        # r₃ = 0.0 (not used in standard QG)
    a_ell_u = zeros(T, nz)      # a_ell = Bu/N² at unstaggered points
    rho_u = ones(T, nz)         # ρ = 1.0 (Boussinesq)

    # Staggered (at cell faces)
    r_1_s = ones(T, nz)         # r₁ = 1.0 at staggered points
    r_2_s = zeros(T, nz)        # r₂ = N² at staggered points
    r_3_s = zeros(T, nz)        # r₃ = 0.0 at staggered points
    a_ell_s = zeros(T, nz)      # a_ell = Bu/N² at staggered points
    rho_s = ones(T, nz)         # ρ = 1.0 at staggered points

    # Compute coefficients at each vertical level
    for k in 1:nz
        # Unstaggered values (cell centers)
        N2_u = N2_profile[k]
        r_2_u[k] = N2_u
        a_ell_u[k] = Bu / max(N2_u, eps(T))  # Avoid division by zero

        # Staggered values (cell faces at z + dz/2)
        # Interpolate N² to staggered grid
        if k < nz
            N2_s = T(0.5) * (N2_profile[k] + N2_profile[k+1])
        else
            N2_s = N2_profile[k]  # At top boundary, use cell center value
        end
        r_2_s[k] = N2_s
        a_ell_s[k] = Bu / max(N2_s, eps(T))
    end

    return (
        # Unstaggered coefficients
        r_1_u = r_1_u,
        r_2_u = r_2_u,
        r_3_u = r_3_u,
        a_ell_u = a_ell_u,
        rho_u = rho_u,
        # Staggered coefficients
        r_1_s = r_1_s,
        r_2_s = r_2_s,
        r_3_s = r_3_s,
        a_ell_s = a_ell_s,
        rho_s = rho_s,
        # Legacy aliases for backward compatibility
        r_1 = r_1_u,
        r_2 = r_2_u,
        r_3 = r_3_u,
        a_ell = a_ell_u,
        b_ell = zeros(T, nz)  # Not used in standard formulation
    )
end

"""
    load_stratification_from_file(filename::String)

Load stratification profile from NetCDF file.
"""
function load_stratification_from_file(filename::String)
    T = Float64
    
    # This would use the NetCDF reading functions
    z_data, N2_data = read_stratification_profile(filename, -1)  # -1 to read all points
    
    return FileProfile{T}(filename, z_data, N2_data)
end

"""
    create_standard_profiles()

Create some standard stratification profiles for testing.
"""
function create_standard_profiles()
    T = Float64
    profiles = Dict{Symbol, StratificationProfile{T}}()
    
    # Constant stratification
    profiles[:constant] = ConstantN{T}(1.0)
    
    # Strong stratification  
    profiles[:strong] = ConstantN{T}(5.0)
    
    # Weak stratification
    profiles[:weak] = ConstantN{T}(0.2)
    
    # Tropopause-like
    profiles[:pycnocline] = TanhProfile{T}(0.01, 0.025, 0.6, 0.05)
    
    # Ocean-like (exponential)
    profiles[:ocean] = ExponentialProfile{T}(0.02, 0.3, 0.001)
    
    # Two-layer
    profiles[:two_layer] = PiecewiseProfile{T}(
        [0.0, 0.5, 2π],  # interfaces
        [0.01, 0.03]     # N values
    )
    
    # Skewed Gaussian (test1 parameters)
    profiles[:test1] = SkewedGaussian{T}(
        0.537713935783168,
        2.684198470106461, 
        0.648457170048730,
        6.121537923499139,
        -5.338431587899242
    )
    
    return profiles
end

"""
    plot_stratification_profile(profile::StratificationProfile, nz::Int=100)

Generate data for plotting stratification profile.
"""
function plot_stratification_profile(profile::StratificationProfile{T}, nz::Int=100) where T
    z_vals = LinRange(0, 2π, nz)
    N2_vals = [evaluate_N2(profile, z) for z in z_vals]
    N_vals = sqrt.(N2_vals)
    
    return collect(z_vals), N2_vals, N_vals
end

"""
    validate_stratification(N2_profile::Vector)

Check that stratification profile is physically reasonable.
"""
function validate_stratification(N2_profile::Vector{T}) where T
    warnings = String[]
    errors = String[]
    
    # Check for negative N²
    if any(N2_profile .<= 0)
        push!(errors, "Negative or zero N² values detected - stratification unstable")
    end
    
    # Check for extreme values
    N2_max = maximum(N2_profile)
    N2_min = minimum(N2_profile)
    
    if N2_max > 100.0
        push!(warnings, "Very large N² values (max = $N2_max) - check units")
    end
    
    if N2_min < 1e-6
        push!(warnings, "Very small N² values (min = $N2_min) - may cause numerical issues")
    end
    
    # Check for sharp gradients
    gradients = diff(N2_profile)
    max_gradient = maximum(abs.(gradients))
    if max_gradient > 10.0 * (N2_max - N2_min) / length(N2_profile)
        push!(warnings, "Sharp N² gradients detected - may require high vertical resolution")
    end
    
    return errors, warnings
end

"""
    compute_deformation_radius(N2_profile::Vector, f0::Real, H::Real)

Compute first baroclinic deformation radius.
"""
function compute_deformation_radius(N2_profile::Vector{T}, f0::Real, H::Real) where T
    # Compute N_avg as depth-weighted average
    N_avg = sqrt(sum(N2_profile) / length(N2_profile))
    
    # First baroclinic deformation radius
    Ld = N_avg * H / (π * abs(f0))
    
    return Ld
end

"""
    adjust_stratification_for_domain!(N2_profile::Vector, target_Ld::Real, f0::Real, H::Real)

Adjust stratification to achieve target deformation radius.
"""
function adjust_stratification_for_domain!(N2_profile::Vector{T}, target_Ld::Real, f0::Real, H::Real) where T
    current_Ld = compute_deformation_radius(N2_profile, f0, H)
    
    if current_Ld > 0
        scale_factor = (target_Ld / current_Ld)^2
        N2_profile .*= scale_factor
        
        @info "Stratification adjusted: Ld = $current_Ld → $target_Ld (scale = $scale_factor)"
    end
    
    return N2_profile
end