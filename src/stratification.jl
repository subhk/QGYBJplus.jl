"""
Stratification profile module for QG-YBJ model.

Provides various stratification profiles including:
- Constant N²
- Skewed Gaussian (from Fortran test cases)
- Tanh profiles (pycnocline-like)
- Exponential profiles
- Piecewise profiles
- Tabulated profiles from NetCDF or JLD2 files
- Analytical N(z) or N²(z) profiles
"""

using SpecialFunctions: erf
using NCDatasets

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

Tabulated N² values `N2_data` at depths `z_data` (positive downward, strictly
increasing), as returned by `load_stratification_from_file`. `evaluate_N2`
interpolates linearly in depth and holds the end values beyond the data.
"""
struct FileProfile{T} <: StratificationProfile{T}
    filename::String
    z_data::Vector{T}    # Depths (positive below surface)
    N2_data::Vector{T}
end

"""
    AnalyticalProfile{T} <: StratificationProfile{T}

Analytical stratification profile from a user-provided function.
"""
struct AnalyticalProfile{T, F} <: StratificationProfile{T}
    N_func::F
    is_N2::Bool
end

"""
    AnalyticalProfile(func; returns=:N², precision=Float64)

Create a stratification profile from a function of vertical coordinate `z`.
Set `returns=:N²` when `func(z)` returns buoyancy frequency squared, or
`returns=:N` when it returns buoyancy frequency.
"""
function AnalyticalProfile(func::F; returns::Symbol=:N²,
                           precision::Type{T}=Float64) where {F, T<:AbstractFloat}
    returns in (:N, :N²) ||
        throw(ArgumentError("returns must be :N or :N²"))
    return AnalyticalProfile{T, F}(func, returns === :N²)
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
  - For `:from_file`: `filename` (path to a NetCDF or JLD2 file)
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
    compute_stratification_profile(profile::StratificationProfile, grid::RectilinearGrid)

Compute the pointwise N² profile at the model's cell centers (`grid.z`).

Flux-form vertical operators use a separately sampled upper-face profile; see
`_compute_stratification_face_profile`.
"""
function compute_stratification_profile(
    profile::StratificationProfile{T}, geometry::RectilinearGrid) where T

    return T[evaluate_N2(profile, z) for z in geometry.z]
end

"""
    _compute_stratification_face_profile(profile, grid)

Sample N² at the upper face of each cell, `grid.z_faces[2:end]`. The final
entry is the top boundary; homogeneous Neumann flux operators retain it for a
uniform indexing convention even though its boundary flux is zero.
"""
function _compute_stratification_face_profile(
    profile::StratificationProfile{T}, geometry::RectilinearGrid) where T

    return T[evaluate_N2(profile, z) for z in @view geometry.z_faces[2:end]]
end

"""
    load_stratification_from_file(filename::String; group=nothing)

Load a tabulated N² profile from a NetCDF file, or from a JLD2 file when
`filename` ends in `.jld2`.

The file must hold a vertical coordinate named `z` or `depth` and N² named
`N2`, `N²`, or `n2`, as real arrays of equal length. `depth` is positive
downward and `z` positive upward; a `z` without negative values is read as
depth. `group` names the NetCDF group, or the JLD2 group, NamedTuple, or Dict,
that holds the arrays, e.g. `group="ocean/cast12"`. JLD2 arrays may be ranges.

The data need not be sorted or lie on the model's levels: the model
interpolates the profile linearly in depth to its own cell centers and faces,
holds the end values beyond the data, and warns when that leaves more than one
grid spacing uncovered at the top or bottom of the domain.
"""
function load_stratification_from_file(filename::String;
                                       group::Union{Nothing, AbstractString}=nothing)
    T = Float64
    source = _source_description(filename, group)
    read_profile = _is_jld2(filename) ? _read_jld2_profile : _read_netcdf_profile
    coordinate_name, coordinate_data, N2_data = read_profile(filename, group)

    depths = _file_depths(coordinate_name,
        _real_values(coordinate_data, coordinate_name, source))
    values = _real_values(N2_data, "N²", source)

    isempty(depths) &&
        throw(ArgumentError("$source contains no coordinate data"))
    length(depths) == length(values) ||
        throw(ArgumentError(
            "the coordinate and N² arrays in $source must have equal lengths"))
    all(isfinite, depths) ||
        throw(ArgumentError("the coordinates in $source must be finite"))
    all(isfinite, values) ||
        throw(ArgumentError("the N² values in $source must be finite"))

    permutation = sortperm(depths)
    depths = depths[permutation]
    values = values[permutation]
    all(>(zero(T)), diff(depths)) ||
        throw(ArgumentError("the coordinates in $source must be unique"))

    return FileProfile{T}(filename, depths, values)
end

const _PROFILE_COORDINATE_NAMES = ("z", "depth")
const _PROFILE_N2_NAMES = ("N2", "N²", "n2")

function _read_netcdf_profile(filename, group)
    return NCDataset(filename, "r") do dataset
        container = _netcdf_container(dataset, filename, group)
        z_name, N2_name = _profile_names(keys(container), filename, group)
        (z_name, container[z_name][:], container[N2_name][:])
    end
end

function _read_jld2_profile(filename, group)
    return jldopen(filename, "r") do file
        container = _jld2_container(file, filename, group)
        z_name, N2_name = _profile_names(_entry_names(container), filename, group)
        (z_name, _entry(container, z_name), _entry(container, N2_name))
    end
end

function _profile_names(available, filename, group)
    first_present(names) = (index = findfirst(in(available), names);
                            index === nothing ? nothing : names[index])
    z_name = first_present(_PROFILE_COORDINATE_NAMES)
    N2_name = first_present(_PROFILE_N2_NAMES)
    if z_name === nothing || N2_name === nothing
        wanted = z_name === nothing ? "coordinate named z or depth" :
                                      "N² named N2, N², or n2"
        found = isempty(available) ? "nothing" :
                join(sort!(String[string(name) for name in available]), ", ")
        throw(ArgumentError("$(_source_description(filename, group)) has no " *
                            "$wanted; it contains $found"))
    end
    return z_name, N2_name
end

"""
Warn when the tabulated depths leave more than one grid spacing uncovered at
the top or bottom of the domain, where `evaluate_N2` holds the end values.
"""
function _warn_uncovered_depths(profile::FileProfile, grid::RectilinearGrid)
    top, bottom = _depth(grid.z_faces[end]), _depth(grid.z_faces[1])
    shallowest, deepest = first(profile.z_data), last(profile.z_data)
    if shallowest - top > grid.dz || bottom - deepest > grid.dz
        @warn "The stratification data in $(profile.filename) cover depths " *
              "$shallowest to $deepest, but the domain spans depths $top to " *
              "$bottom; outside that range N² is held at the nearest data value."
    end
    return nothing
end

_warn_uncovered_depths(profile::StratificationProfile, grid) = nothing

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
    
    # Check for sharp gradients when at least one interval exists.
    if length(N2_profile) > 1
        gradients = diff(N2_profile)
        max_gradient = maximum(abs.(gradients))
        if max_gradient > 10.0 * (N2_max - N2_min) / length(N2_profile)
            push!(warnings, "Sharp N² gradients detected - may require high vertical resolution")
        end
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
