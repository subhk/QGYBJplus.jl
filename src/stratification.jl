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

using SpecialFunctions: erf
using ..QGYBJplus: Grid, read_stratification_raw

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
    z0::T     # Center depth (positive below surface)
    alpha::T  # Skewness parameter
end

"""
    TanhProfile{T} <: StratificationProfile{T}

Hyperbolic tangent profile (pycnocline-like).
"""
struct TanhProfile{T} <: StratificationProfile{T}
    N_upper::T    # Upper ocean N
    N_lower::T   # Deep ocean N
    z_pycno::T    # Pycnocline depth (positive below surface)
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
    z_interfaces::Vector{T}  # Interface depths (positive below surface)
    N_values::Vector{T}      # N values in each layer
end

"""
    FileProfile{T} <: StratificationProfile{T}

Stratification profile from file.
"""
struct FileProfile{T} <: StratificationProfile{T}
    filename::String
    z_data::Vector{T}    # Depths (positive below surface)
    N2_data::Vector{T}
end

"""
    create_stratification_profile(config)

Create stratification profile from configuration.

# Arguments
- `config`: Configuration object with fields:
  - `type::Symbol`: One of `:constant_N`, `:skewed_gaussian`, `:tanh_profile`, `:from_file`
  - For `:constant_N`: `N0` (buoyancy frequency)
  - For `:skewed_gaussian`: `N02_sg`, `N12_sg`, `sigma_sg`, `z0_sg`, `alpha_sg`
  - For `:tanh_profile`: `N_upper`, `N_lower`, `z_pycno`, `width`
  - For `:from_file`: `filename` (path to NetCDF file)

# Returns
A `StratificationProfile` subtype appropriate for the configuration.
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

Uses the formula matching physics.jl:a_ell_ut():
    N²(z) = N₁² exp(-(z-z₀)²/σ²) [1 + erf(α(z-z₀)/(σ√2))] + N₀²

This creates a realistic pycnocline with:
- Enhanced stratification near z₀
- Asymmetric shape controlled by α (via error function)
- Background N₀² in deep ocean
"""
function evaluate_N2(profile::SkewedGaussian{T}, z::Real) where T
    # Use the same formula as physics.jl:a_ell_ut() for consistency
    # N²(z) = N₁² exp(-(z-z₀)²/σ²) [1 + erf(α(z-z₀)/(σ√2))] + N₀²
    z0 = profile.z0
    σ = profile.sigma
    α = profile.alpha
    N02 = profile.N02
    N12 = profile.N12

    depth = -z
    N2 = N12 * exp(-((depth - z0)^2) / (σ^2)) * (1 + erf(α * (depth - z0) / (σ * sqrt(2.0)))) + N02

    return max(N2, T(0.01) * N02)  # Ensure N² remains positive
end

"""
    evaluate_N2(profile::TanhProfile, z::Real)

Evaluate N² for tanh profile.
"""
function evaluate_N2(profile::TanhProfile{T}, z::Real) where T
    # Normalized height (depth-based)
    depth = -z
    ζ = (depth - profile.z_pycno) / profile.width
    
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
    depth = -z
    N_val = profile.N_deep + (profile.N_surface - profile.N_deep) *
            exp(-depth / profile.scale_height)
    return N_val^2
end

"""
    evaluate_N2(profile::PiecewiseProfile, z::Real)

Evaluate N² for piecewise profile.
"""
function evaluate_N2(profile::PiecewiseProfile{T}, z::Real) where T
    # Find which layer depth belongs to
    depth = -z
    for i in 1:(length(profile.z_interfaces)-1)
        if profile.z_interfaces[i] <= depth < profile.z_interfaces[i+1]
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
    depth = -z
    # Linear interpolation
    if depth <= profile.z_data[1]
        return profile.N2_data[1]
    elseif depth >= profile.z_data[end]
        return profile.N2_data[end]
    else
        # Find surrounding points
        for i in 1:(length(profile.z_data)-1)
            if profile.z_data[i] <= depth <= profile.z_data[i+1]
                # Linear interpolation
                α = (depth - profile.z_data[i]) / (profile.z_data[i+1] - profile.z_data[i])
                return (1-α) * profile.N2_data[i] + α * profile.N2_data[i+1]
            end
        end
    end

    # Should never reach here - indicates a logic error or malformed profile data
    error("evaluate_N2: Failed to interpolate N² at depth=$depth. " *
          "z_data range: [$(profile.z_data[1]), $(profile.z_data[end])]")
end

"""
    compute_stratification_profile(profile::StratificationProfile, G::Grid)

Compute N² profile on model grid.

The profile is evaluated on unstaggered (face) levels at `z = G.z - dz/2`
to match the Fortran coefficient grid.
"""
function compute_stratification_profile(profile::StratificationProfile{T}, G::Grid) where T
    N2_profile = zeros(T, G.nz)
    dz = G.Lz / G.nz

    for k in 1:G.nz
        z_unstag = G.z[k] - dz / 2
        N2_profile[k] = evaluate_N2(profile, z_unstag)
    end

    return N2_profile
end

"""
    compute_stratification_coefficients(N2_profile::Vector, G::Grid; f0_sq::Real=1.0)

Compute stratification-dependent coefficients for the model.

Following the Fortran init_base_state routine:
- r_1 = 1.0 (unity for Boussinesq)
- r_2 = N² (buoyancy frequency squared)
- r_3 = 0.0 (not used in standard QG formulation)
- a_ell = f²/N² (elliptic coefficient for PV inversion)
- rho = 1.0 (Boussinesq approximation)

# Arguments
- `N2_profile::Vector`: Buoyancy frequency squared N²(z) at each vertical level
- `G::Grid`: Grid structure
- `f0_sq::Real`: Coriolis parameter squared f² (default 1.0)

# Returns
Named tuple with coefficients:
- `r_1_u`, `r_1_s`: Unity arrays at unstaggered/staggered points (length nz)
- `r_2_u`, `r_2_s`: N² at unstaggered/staggered points (length nz)
- `r_3_u`, `r_3_s`: Zero arrays (length nz, not used in standard QG)
- `a_ell_u`, `a_ell_s`: f²/N² at unstaggered/staggered points (length nz)
- `rho_u`, `rho_s`: Unity density at unstaggered/staggered points (length nz)
- `r_1`, `r_2`, `r_3`, `a_ell`: Legacy aliases (equal to `*_u` versions)
- `b_ell`: Zero array (length nz, reserved for future use)
"""
function compute_stratification_coefficients(N2_profile::Vector{T}, G::Grid; f0_sq::Real=T(1.0)) where T
    # Check for empty profile
    length(N2_profile) > 0 || error("Empty N2_profile: cannot compute stratification coefficients")

    nz = G.nz
    dz = nz > 1 ? (G.z[2] - G.z[1]) : T(G.Lz / nz)

    # Initialize coefficient arrays (following Fortran init_base_state)
    # Unstaggered (cell faces at z = G.z - dz/2)
    r_1_u = ones(T, nz)         # r₁ = 1.0 (Boussinesq)
    r_2_u = zeros(T, nz)        # r₂ = N² at unstaggered points
    r_3_u = zeros(T, nz)        # r₃ = 0.0 (not used in standard QG)
    a_ell_u = zeros(T, nz)      # a_ell = f²/N² at unstaggered points
    rho_u = ones(T, nz)         # ρ = 1.0 (Boussinesq)

    # Staggered (cell centers at z = G.z)
    r_1_s = ones(T, nz)         # r₁ = 1.0 at staggered points
    r_2_s = zeros(T, nz)        # r₂ = N² at staggered points
    r_3_s = zeros(T, nz)        # r₃ = 0.0 at staggered points
    a_ell_s = zeros(T, nz)      # a_ell = f²/N² at staggered points
    rho_s = ones(T, nz)         # ρ = 1.0 at staggered points

    # Compute coefficients at each vertical level
    for k in 1:nz
        # Unstaggered values (cell faces)
        N2_u = N2_profile[k]
        r_2_u[k] = N2_u
        a_ell_u[k] = f0_sq / max(N2_u, eps(T))  # Avoid division by zero

        # Staggered values (cell centers from adjacent unstaggered levels)
        if k < nz
            N2_s = T(0.5) * (N2_profile[k] + N2_profile[k+1])
        else
            N2_s = N2_profile[k]  # At top boundary, use cell center value
        end
        r_2_s[k] = N2_s
        a_ell_s[k] = f0_sq / max(N2_s, eps(T))
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

Uses `read_stratification_raw` to read both z coordinates and N² values
without interpolation. The resulting FileProfile can then be evaluated
at any z (depth = -z) using linear interpolation.
"""
function load_stratification_from_file(filename::String)
    T = Float64

    # Use read_stratification_raw which returns (z_data, N2_data) tuple
    z_data, N2_data = read_stratification_raw(filename)

    return FileProfile{T}(filename, Vector{T}(z_data), Vector{T}(N2_data))
end

"""
    create_standard_profiles(Lz::Real)

Create some standard stratification profiles for testing.

# Arguments
- `Lz`: Domain depth in meters (REQUIRED, z ∈ [-Lz, 0])

# Returns
Dictionary of standard stratification profiles with keys:
- `:constant`: Constant N² = 1.0
- `:strong`: Strong stratification N² = 5.0
- `:weak`: Weak stratification N² = 0.2
- `:pycnocline`: Tanh profile (pycnocline-like)
- `:ocean`: Exponential profile (ocean-like)
- `:two_layer`: Piecewise two-layer profile
"""
function create_standard_profiles(Lz::Real)
    T = Float64

    profiles = Dict{Symbol, StratificationProfile{T}}()

    # Constant stratification
    profiles[:constant] = ConstantN{T}(1.0)

    # Strong stratification
    profiles[:strong] = ConstantN{T}(5.0)

    # Weak stratification
    profiles[:weak] = ConstantN{T}(0.2)

    # Pycnocline-like (depths scaled by Lz)
    profiles[:pycnocline] = TanhProfile{T}(0.01, 0.025, 0.6 * Lz, 0.05 * Lz)

    # Ocean-like (exponential, scale height relative to Lz)
    profiles[:ocean] = ExponentialProfile{T}(0.02, 0.3 * Lz, 0.001)

    # Two-layer (interfaces scaled by Lz)
    profiles[:two_layer] = PiecewiseProfile{T}(
        T[0.0, 0.5 * Lz, Lz],  # interfaces at surface, mid-depth, bottom
        T[0.01, 0.03]          # N values
    )

    return profiles
end

"""
    plot_stratification_profile(profile::StratificationProfile, Lz::Real; nz::Int=100)

Generate data for plotting stratification profile.

# Arguments
- `profile`: Stratification profile to plot
- `Lz`: Domain depth in meters (REQUIRED, z ∈ [-Lz, 0])
- `nz`: Number of points for plotting (default: 100)

# Returns
Tuple of (z_vals, N2_vals, N_vals) for plotting
"""
function plot_stratification_profile(profile::StratificationProfile{T}, Lz::Real; nz::Int=100) where T
    z_vals = LinRange(-Lz, 0, nz)
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

    # Check for empty profile
    if length(N2_profile) == 0
        push!(errors, "Empty N² profile - no stratification data provided")
        return errors, warnings
    end

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
    # Check for empty profile
    length(N2_profile) > 0 || error("Empty N2_profile: cannot compute deformation radius")

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
