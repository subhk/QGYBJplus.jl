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

Abstract type for stratification profiles. Every profile but
[`AnalyticalProfile`](@ref) is defined by depth below the surface:
`evaluate_N2(profile, z)` takes the depth as `-z`, for a lid at z = 0, and a
model whose grid has its top at `z_top` samples the profile at `z - z_top`,
so that depths are always measured below its lid. An `AnalyticalProfile` is
evaluated at the grid's own z.
"""
abstract type StratificationProfile{T} end

"""
    ConstantN{T}(N0) <: StratificationProfile{T}

Constant buoyancy frequency `N0`: N² = N0² at every depth.
"""
struct ConstantN{T} <: StratificationProfile{T}
    N0::T
end

"""
    SkewedGaussian{T}(N02, N12, sigma, z0, alpha) <: StratificationProfile{T}

The stratification of the original Fortran test cases: a pycnocline of peak
N² `N12` above the background `N02`, centred at the depth `z0` below the
surface, with width `sigma` and skewness `alpha` (see [`evaluate_N2`](@ref)).
"""
struct SkewedGaussian{T} <: StratificationProfile{T}
    N02::T    # Background N²
    N12::T    # Peak amplitude
    sigma::T  # Width parameter
    z0::T     # Center depth (positive below surface)
    alpha::T  # Skewness parameter
end

"""
    TanhProfile{T}(N_upper, N_lower, z_pycno, width) <: StratificationProfile{T}

A pycnocline: N changes from `N_upper` near the surface to `N_lower` at depth
across the depth `z_pycno` below the surface, over `width`,
`N(d) = N_upper + (N_lower - N_upper)(1 + tanh((d - z_pycno)/width))/2`.
"""
struct TanhProfile{T} <: StratificationProfile{T}
    N_upper::T    # Upper ocean N
    N_lower::T   # Deep ocean N
    z_pycno::T    # Pycnocline depth (positive below surface)
    width::T     # Transition width
end

"""
    ExponentialProfile{T}(N_surface, scale_height, N_deep) <: StratificationProfile{T}

N decaying from `N_surface` at the surface toward `N_deep` over the depth
scale `scale_height`, `N(d) = N_deep + (N_surface - N_deep) exp(-d/scale_height)`.
"""
struct ExponentialProfile{T} <: StratificationProfile{T}
    N_surface::T   # Surface N
    scale_height::T # e-folding scale
    N_deep::T      # Deep value
end

"""
    PiecewiseProfile(z_interfaces, N_values)

Piecewise constant stratification: `N = N_values[i]` for depths from
`z_interfaces[i]` to `z_interfaces[i + 1]` (positive below the surface,
strictly increasing), so there is one value fewer than interfaces. Depths
above the first interface take the top layer's value, and depths below the
last one the bottom layer's.
"""
struct PiecewiseProfile{T} <: StratificationProfile{T}
    z_interfaces::Vector{T}  # Interface depths (positive below surface)
    N_values::Vector{T}      # N values in each layer

    function PiecewiseProfile{T}(z_interfaces::AbstractVector,
                                 N_values::AbstractVector) where T
        length(N_values) == length(z_interfaces) - 1 >= 1 || throw(ArgumentError(
            "a piecewise profile needs one N value fewer than its interfaces " *
            "and at least one layer (got $(length(z_interfaces)) interfaces " *
            "and $(length(N_values)) values)"))
        all(>(0), diff(z_interfaces)) || throw(ArgumentError(
            "the interface depths of a piecewise profile must increase strictly"))
        return new{T}(z_interfaces, N_values)
    end
end

PiecewiseProfile(z_interfaces::AbstractVector, N_values::AbstractVector) =
    PiecewiseProfile{float(promote_type(eltype(z_interfaces), eltype(N_values)))}(
        z_interfaces, N_values)

"""
    FileProfile{T}(filename, z_data, N2_data) <: StratificationProfile{T}

Tabulated N² values `N2_data` at depths `z_data` (positive downward, strictly
increasing, finite, and as many as the N² values), as returned by
`load_stratification_from_file`. `evaluate_N2` interpolates linearly in depth
and holds the end values beyond the data.
"""
struct FileProfile{T} <: StratificationProfile{T}
    filename::String
    z_data::Vector{T}    # Depths (positive below surface)
    N2_data::Vector{T}

    function FileProfile{T}(filename::AbstractString, z_data::AbstractVector,
                            N2_data::AbstractVector) where T
        isempty(z_data) && throw(ArgumentError("a file profile needs at least one depth"))
        length(z_data) == length(N2_data) || throw(ArgumentError(
            "a file profile needs as many N² values as depths (got " *
            "$(length(N2_data)) and $(length(z_data)))"))
        all(isfinite, z_data) && all(isfinite, N2_data) ||
            throw(ArgumentError("the depths and N² values of a file profile must be finite"))
        all(>(0), diff(z_data)) || throw(ArgumentError(
            "the depths of a file profile must increase strictly (positive downward)"))
        return new{T}(String(filename), z_data, N2_data)
    end
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
    evaluate_N2(profile, z)

N² of `profile` at the height `z` relative to a lid at z = 0, that is at the
depth `-z`; an [`AnalyticalProfile`](@ref) is evaluated at `z` itself. A model
samples its profile at its own levels measured from its lid (see
[`StratificationProfile`](@ref)).
"""
function evaluate_N2(profile::ConstantN{T}, z::Real) where T
    return profile.N0^2
end

"""
    evaluate_N2(profile::SkewedGaussian, z::Real)

Evaluate N² for skewed Gaussian profile, the stratification of the Fortran
test cases, at the depth `d = -z`:
    N²(d) = N₁² exp(-(d-z₀)²/σ²) [1 + erf(α(d-z₀)/(σ√2))] + N₀²

This creates a realistic pycnocline with:
- Enhanced stratification near z₀
- Asymmetric shape controlled by α (via error function)
- Background N₀² in deep ocean
"""
function evaluate_N2(profile::SkewedGaussian{T}, z::Real) where T
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
    depth = -z
    interfaces = profile.z_interfaces
    # Above the first interface lies the top layer.
    depth < first(interfaces) && return first(profile.N_values)^2
    for i in 1:(length(interfaces) - 1)
        if interfaces[i] <= depth < interfaces[i + 1]
            return profile.N_values[i]^2
        end
    end
    # Below the last interface lies the bottom layer.
    return last(profile.N_values)^2
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

Compute the pointwise N² profile at the model's cell centers (`grid.z`),
with depths measured below the top of the grid (see
[`StratificationProfile`](@ref)).

Flux-form vertical operators use a separately sampled upper-face profile; see
`_compute_stratification_face_profile`.
"""
function compute_stratification_profile(
    profile::StratificationProfile{T}, geometry::RectilinearGrid) where T

    return T[evaluate_N2(profile, _profile_coordinate(profile, z, geometry))
             for z in geometry.z]
end

"""
The coordinate at which a model on `grid` evaluates `profile` at its height
`z`: relative to its lid for a profile defined by depth, and `z` itself for an
`AnalyticalProfile`, a function of the grid's own coordinate.
"""
_profile_coordinate(profile::StratificationProfile, z, grid::RectilinearGrid) =
    z - last(grid.z_faces)
_profile_coordinate(profile::AnalyticalProfile, z, grid::RectilinearGrid) = z

"""
    _compute_stratification_face_profile(profile, grid)

Sample N² at the upper face of each cell, `grid.z_faces[2:end]`. The final
entry is the top boundary; homogeneous Neumann flux operators retain it for a
uniform indexing convention even though its boundary flux is zero.
"""
function _compute_stratification_face_profile(
    profile::StratificationProfile{T}, geometry::RectilinearGrid) where T

    return T[evaluate_N2(profile, _profile_coordinate(profile, z, geometry))
             for z in @view geometry.z_faces[2:end]]
end

"""
    load_stratification_from_file(filename::String; group=nothing)

Load a tabulated N² profile from a NetCDF file, or from a JLD2 file when
`filename` ends in `.jld2`.

The file must hold a vertical coordinate named `z` or `depth` and N² named
`N2`, `N²`, or `n2`, as real arrays of equal length. `depth` is positive
downward and `z` positive upward, both relative to the surface; a `z` without
negative values is read as depth, as in earlier versions. `group` names the
NetCDF group, or the JLD2 group, NamedTuple, or Dict, that holds the arrays,
e.g. `group="ocean/cast12"`. JLD2 arrays may be ranges.

The data need not be sorted or lie on the model's levels: the model
interpolates the profile linearly in depth below its lid to its own cell
centers and faces, holds the end values beyond the data, and warns when that
leaves more than one grid spacing uncovered at the top or bottom of the
domain.
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
    # Depths below the lid, which need not be at z = 0.
    top, bottom = zero(eltype(grid.z_faces)), grid.z_faces[end] - grid.z_faces[1]
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
    validate_stratification(N2_profile::Vector) -> (errors, warnings)

Check a sampled N² profile, returning lists of messages. Errors: an empty
profile, non-finite values, and N² ≤ 0 (unstable or neutral). Warnings:
N² above 100 or below 1e-6, which in s⁻² usually signals a units mistake
(though abyssal N² can fall below 1e-6), and jumps between neighbouring
values much larger than the profile's average change.
"""
function validate_stratification(N2_profile::Vector{T}) where T
    warnings = String[]
    errors = String[]

    # Check for empty profile
    if length(N2_profile) == 0
        push!(errors, "Empty N² profile - no stratification data provided")
        return errors, warnings
    end

    if !all(isfinite, N2_profile)
        push!(errors, "Non-finite N² values (NaN or Inf) detected")
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
