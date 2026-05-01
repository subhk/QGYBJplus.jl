"""
Stratification profile module for QG-YBJ model.

Provides various stratification profiles including:
- Constant N²
- Skewed Gaussian (from Fortran test cases)
- Tanh profiles (pycnocline-like)
- Exponential profiles
- Piecewise profiles
- Custom profiles from files
- Analytical N(z) or N²(z) profiles
"""

using JLD2
using NCDatasets
using SpecialFunctions: erf
using ..QGYBJplus: Grid

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

Stratification profile loaded from file.
"""
struct FileProfile{T} <: StratificationProfile{T}
    filename::String
    z_data::Vector{T}     # Physical z coordinate [m], sorted bottom to surface.
    data::Vector{T}       # N [s⁻¹] or N² [s⁻²].
    is_N2::Bool
end

function FileProfile{T}(filename::AbstractString,
                        z_data::AbstractVector,
                        N2_data::AbstractVector) where T
    z_model, values = _normalize_file_stratification(T.(z_data), T.(N2_data), T)
    return FileProfile{T}(String(filename), z_model, values, true)
end

"""
    FileStratification(filename; z="z", N="N", N2=nothing, N²=nothing)
    FileProfile(filename; z="z", N="N", N2=nothing, N²=nothing)

Load a dimensional stratification profile from a NetCDF file.

By default the file is expected to contain vertical coordinate `z` and
buoyancy frequency `N`. If the file stores buoyancy frequency squared, pass
`N2="variable_name"` (or `N²="variable_name"`). Positive `z` coordinates are
interpreted as depth below the surface and converted to physical `z <= 0`.
"""
function FileProfile(filename::AbstractString;
                     z::AbstractString = "z",
                     N::Union{Nothing, AbstractString} = "N",
                     N2::Union{Nothing, AbstractString} = nothing,
                     N²::Union{Nothing, AbstractString} = nothing,
                     T::Type{<:AbstractFloat} = Float64)

    z_data, data, is_N2 = _read_file_stratification(filename; z, N, N2, N², T)
    z_model, values = _normalize_file_stratification(z_data, data, T)

    return FileProfile{T}(String(filename), z_model, values, is_N2)
end

FileStratification(args...; kwargs...) = FileProfile(args...; kwargs...)

"""
    AnalyticalProfile{T} <: StratificationProfile{T}

Analytical stratification profile from a user-provided function.
"""
struct AnalyticalProfile{T, F} <: StratificationProfile{T}
    N_func::F
    is_N2::Bool
end

"""
    create_stratification_profile(config)

Create stratification profile from configuration.

# Arguments
- `config`: Configuration object with fields:
  - `type::Symbol`: One of `:constant_N`, `:skewed_gaussian`, `:tanh_profile`, `:from_file`, `:analytical`
  - For `:constant_N`: `N0` (buoyancy frequency)
  - For `:skewed_gaussian`: `N02_sg`, `N12_sg`, `sigma_sg`, `z0_sg`, `alpha_sg`
  - For `:tanh_profile`: `N_upper`, `N_lower`, `z_pycno`, `width`
  - For `:from_file`: `filename` (path to NetCDF file)
  - For `:analytical`: `N_func` (N(z)) or `N2_func` (N²(z))

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

    elseif config.type == :analytical || config.type == :function
        if !isnothing(config.N2_func)
            return AnalyticalProfile{T, typeof(config.N2_func)}(config.N2_func, true)
        elseif !isnothing(config.N_func)
            return AnalyticalProfile{T, typeof(config.N_func)}(config.N_func, false)
        else
            error("Provide N_func or N2_func for stratification type :$(config.type)")
        end
        
    else
        error("Unknown stratification type: $(config.type)")
    end
end

function _read_file_stratification(filename::AbstractString;
                                   z::AbstractString,
                                   N::Union{Nothing, AbstractString},
                                   N2::Union{Nothing, AbstractString},
                                   N²::Union{Nothing, AbstractString},
                                   T::Type{<:AbstractFloat})
    extension = lowercase(splitext(String(filename))[2])
    if extension == ".jld2"
        return _read_jld2_stratification(filename; z, N, N2, N², T)
    else
        return _read_netcdf_stratification(filename; z, N, N2, N², T)
    end
end

function _read_netcdf_stratification(filename::AbstractString;
                                     z::AbstractString,
                                     N::Union{Nothing, AbstractString},
                                     N2::Union{Nothing, AbstractString},
                                     N²::Union{Nothing, AbstractString},
                                     T::Type{<:AbstractFloat})
    z_data = T[]
    values = T[]
    is_N2 = false

    N2_name = N2 === nothing ? N² : N2

    NCDataset(filename, "r") do ds
        haskey(ds, z) || throw(ArgumentError("No vertical coordinate '$z' found in $filename."))
        z_data = vec(T.(Array(ds[z][:])))

        if N2_name !== nothing
            haskey(ds, N2_name) || throw(ArgumentError("No N² variable '$N2_name' found in $filename."))
            values = vec(T.(Array(ds[N2_name][:])))
            is_N2 = true
        elseif N !== nothing && haskey(ds, N)
            values = vec(T.(Array(ds[N][:])))
            is_N2 = false
        else
            found_N2 = findfirst(name -> haskey(ds, name), _common_N2_names())
            if found_N2 !== nothing
                name = _common_N2_names()[found_N2]
                values = vec(T.(Array(ds[name][:])))
                is_N2 = true
            else
                expected = _expected_stratification_variable_message(N)
                throw(ArgumentError("No stratification variable found in $filename. Expected $expected."))
            end
        end
    end

    return z_data, values, is_N2
end

function _read_jld2_stratification(filename::AbstractString;
                                   z::AbstractString,
                                   N::Union{Nothing, AbstractString},
                                   N2::Union{Nothing, AbstractString},
                                   N²::Union{Nothing, AbstractString},
                                   T::Type{<:AbstractFloat})
    data = JLD2.load(filename)
    haskey(data, z) || throw(ArgumentError("No vertical coordinate '$z' found in $filename."))

    z_data = vec(T.(data[z]))
    N2_name = N2 === nothing ? N² : N2

    if N2_name !== nothing
        haskey(data, N2_name) || throw(ArgumentError("No N² variable '$N2_name' found in $filename."))
        return z_data, vec(T.(data[N2_name])), true
    elseif N !== nothing && haskey(data, N)
        return z_data, vec(T.(data[N])), false
    else
        found_N2 = findfirst(name -> haskey(data, name), _common_N2_names())
        if found_N2 !== nothing
            name = _common_N2_names()[found_N2]
            return z_data, vec(T.(data[name])), true
        end
    end

    expected = _expected_stratification_variable_message(N)
    throw(ArgumentError("No stratification variable found in $filename. Expected $expected."))
end

_common_N2_names() = ("N2", "N²", "N_squared", "buoyancy_frequency_squared",
                      "brunt_vaisala_frequency_squared")

_expected_stratification_variable_message(N) =
    N === nothing ? join(_common_N2_names(), ", ") :
    string("'", N, "' or one of ", join(_common_N2_names(), ", "))

function _normalize_file_stratification(z_data::AbstractVector,
                                        values::AbstractVector,
                                        ::Type{T}) where T
    length(z_data) == length(values) ||
        throw(ArgumentError("Stratification coordinate and data lengths differ: " *
                            "$(length(z_data)) z values, $(length(values)) data values."))
    length(z_data) >= 2 ||
        throw(ArgumentError("File-backed stratification requires at least two vertical levels."))
    all(isfinite, z_data) || throw(ArgumentError("Stratification z values must be finite."))
    all(isfinite, values) || throw(ArgumentError("Stratification values must be finite."))

    z_model = minimum(z_data) >= zero(T) ? -collect(z_data) : collect(z_data)
    data = collect(values)
    permutation = sortperm(z_model)
    z_sorted = z_model[permutation]
    values_sorted = data[permutation]

    all(diff(z_sorted) .> zero(T)) ||
        throw(ArgumentError("Stratification z coordinates must be unique after sorting."))

    return z_sorted, values_sorted
end

function _linear_interpolate(z_data::AbstractVector{T}, values::AbstractVector{T}, z::Real) where T
    zT = T(z)
    if zT <= z_data[1]
        return values[1]
    elseif zT >= z_data[end]
        return values[end]
    end

    lower = searchsortedlast(z_data, zT)
    lower = clamp(lower, 1, length(z_data) - 1)
    upper = lower + 1
    weight = (zT - z_data[lower]) / (z_data[upper] - z_data[lower])

    return (one(T) - weight) * values[lower] + weight * values[upper]
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
    evaluate_N2(profile::AnalyticalProfile, z::Real)

Evaluate N² for an analytical profile.
"""
function evaluate_N2(profile::AnalyticalProfile{T}, z::Real) where T
    val = profile.N_func(z)
    if profile.is_N2
        return max(T(val), zero(T))
    else
        return max(T(val)^2, zero(T))
    end
end

"""
    evaluate_N2(profile::FileProfile, z::Real)

Evaluate N² for file-based profile using physical-z interpolation.
"""
function evaluate_N2(profile::FileProfile{T}, z::Real) where T
    value = _linear_interpolate(profile.z_data, profile.data, z)
    if profile.is_N2
        return max(value, zero(T))
    else
        return max(value, zero(T))^2
    end
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
- `N2_profile::Vector`: Buoyancy frequency squared N²(z) on unstaggered (face) levels
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

This is a legacy alias for `FileProfile(filename; kwargs...)`. By default it
reads variables `z` and `N`, and it also supports `N2`/`N²` variables.
"""
load_stratification_from_file(filename::String; kwargs...) = FileProfile(filename; kwargs...)

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
