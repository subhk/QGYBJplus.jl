#=
================================================================================
                    physics.jl - Stratification and Vertical Coefficients
================================================================================

This file provides the physical stratification profiles and derived coefficients
used in the QG-YBJ+ model's vertical operators.

PHYSICAL BACKGROUND:
--------------------
In the ocean, the buoyancy frequency N(z) (also called Brunt-Väisälä frequency)
characterizes the strength of density stratification:

    N² = -(g/ρ₀) ∂ρ/∂z

where:
- g is gravitational acceleration
- ρ₀ is reference density
- ρ is background density

N² controls:
1. The stretching term in QG PV: q = ∇²ψ + f²/N² ∂²ψ/∂z²
2. The vertical propagation of near-inertial waves
3. The YBJ+ elliptic operator structure

STRATIFICATION PROFILES:
------------------------
Two profiles are implemented (matching Fortran):

1. CONSTANT N (constant_N):
   - Uniform stratification: N² = 1 (nondimensional)
   - Simplest case, good for testing

2. SKEWED GAUSSIAN (skewed_gaussian):
   - Realistic ocean profile with surface pycnocline
   - Formula: N² = N₁² exp(-(z-z₀)²/σ²)[1 + erf(α(z-z₀)/σ√2)] + N₀²
   - Parameters:
     * N₀² = deep ocean value (background)
     * N₁² = pycnocline amplitude
     * z₀  = pycnocline center depth
     * σ   = pycnocline width
     * α   = asymmetry (skewness) parameter

DERIVED COEFFICIENTS:
---------------------
From N²(z), we derive:

- a_ell = 1/N²: Coefficient in elliptic operators
  Used in: ∂/∂z(a_ell ∂ψ/∂z) for streamfunction inversion

- rho_ut, rho_st: Density weights on unstaggered/staggered grids
  Used in: Mass-weighted vertical operators (currently unity for Boussinesq)

DEALIASING:
-----------
The dealias_mask implements the 2/3 rule for spectral dealiasing:
- Quadratic nonlinearities (e.g., u·∇q) generate spurious high-k modes
- Truncating to 2/3 of Nyquist prevents aliasing errors
- Uses radial cutoff: keep modes with |k| ≤ k_max/3

FORTRAN CORRESPONDENCE:
-----------------------
- a_ell_ut → a_ell(k) in init.f90
- N2_ut → n2(k) in init.f90
- dealias_mask → LL(i,j) in init.f90

================================================================================
=#

#=
================================================================================
                        ELLIPTIC OPERATOR COEFFICIENTS
================================================================================
These coefficients appear in the vertical elliptic operators used to invert
PV to streamfunction and wave envelope B to amplitude A.
================================================================================
=#

"""
    a_ell_ut(par, G) -> Vector

Compute the vertical elliptic coefficient a(z) = 1/N²(z) on unstaggered levels.

# Physical Meaning
This coefficient appears in the stretching term of the QG elliptic operator:

    L_ψ[ψ] = ∇²ψ + ∂/∂z(a(z) ∂ψ/∂z)

where a(z) = f²/(N²(z)) in dimensional form, normalized to 1/N² in nondimensional.

For the YBJ+ wave operator, a(z) also appears in the L⁺ operator that relates
the wave envelope B to wave amplitude A.

# Stratification Options
- `:constant_N`: Returns a(z) = 1 everywhere (uniform stratification)
- `:skewed_gaussian`: Returns a(z) = 1/N²(z) with skewed Gaussian N² profile

# Arguments
- `par::QGParams`: Parameters including stratification choice and coefficients
- `G::Grid`: Grid with vertical levels z

# Returns
Vector of length nz with a(z_k) values.

# Example
```julia
a = a_ell_ut(par, G)  # Use in tridiagonal solver
```

# Fortran Correspondence
Matches `a_ell(k)` computed in `init_base_state` (init.f90).
"""
function a_ell_ut(par::QGParams, G::Grid)
    nz = G.nz
    a = similar(G.z)
    f₀_sq = par.f₀^2

    if par.stratification === :constant_N
        #= Constant N²: a = f²/N² =#
        T = eltype(a)
        # Warn once if division guard activates (N² ≈ 0)
        if par.N² < sqrt(eps(T))
            @warn "N² ≈ 0 in constant_N stratification (N²=$(par.N²)), using eps for numerical stability" maxlog=1
        end
        @inbounds for k in 1:nz
            a[k] = f₀_sq / max(par.N², eps(T))
        end

    elseif par.stratification === :skewed_gaussian
        #= Skewed Gaussian N² profile:
        N²(z) = N₁² exp(-(z-z₀)²/σ²)[1 + erf(α(z-z₀)/σ√2)] + N₀²

        This creates a realistic pycnocline with:
        - Enhanced stratification near z₀
        - Asymmetric shape controlled by α
        - Background N₀² in deep ocean =#
        T = eltype(a)
        N02 = par.N₀²_sg; N12 = par.N₁²_sg; σ = par.σ_sg; z0 = par.z₀_sg; α = par.α_sg
        division_guard_warned = false
        @inbounds for k in 1:nz
            z = G.z[k]
            N2_z = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
            # Warn once if division guard activates at any level
            if N2_z < sqrt(eps(T)) && !division_guard_warned
                @warn "N²(z) ≈ 0 at z=$(z) in skewed_gaussian stratification, using eps for numerical stability" maxlog=1
                division_guard_warned = true
            end
            a[k] = f₀_sq / max(N2_z, eps(T))  # a = f²/N²(z), protected against N²≈0
        end

    else
        error("Unsupported stratification: $(par.stratification)")
    end
    return a
end

"""
    a_ell_from_N2(N2_profile::AbstractVector, par::QGParams) -> Vector

Compute the vertical elliptic coefficient a(z) = f²/N²(z) from a given N² profile.

This function allows custom N² profiles (from files, tanh profiles, etc.) to be
used in the core elliptic operators (q→ψ and B→A inversions), ensuring consistency
between the stratification used in vertical velocity calculations and the main
dynamics.

# Arguments
- `N2_profile::AbstractVector`: N²(z) values at each vertical level (length nz)
- `par::QGParams`: Parameters (used for f₀)

# Returns
Vector of length nz with a(z) = f²/N²(z) values.

# Example
```julia
# Use custom N² profile for elliptic operators
N2_profile = [compute_N2(z) for z in G.z]
a_ell = a_ell_from_N2(N2_profile, par)
invert_q_to_psi!(S, G; a=a_ell)
```

# Note
If `N2_profile` is `nothing` or empty, falls back to `a_ell_ut(par, G)`.
This function ensures that custom stratification profiles provided by the user
are actually used in the main dynamics, not just in vertical velocity calculations.
"""
function a_ell_from_N2(N2_profile::AbstractVector, par::QGParams)
    nz = length(N2_profile)
    T = eltype(N2_profile)
    a = similar(N2_profile)
    f₀_sq = par.f₀^2

    division_guard_warned = false
    @inbounds for k in 1:nz
        N2_z = N2_profile[k]
        # Warn once if division guard activates (N² ≈ 0)
        if N2_z < sqrt(eps(T)) && !division_guard_warned
            @warn "N²(z) ≈ 0 at level k=$k in custom N² profile, using eps for numerical stability" maxlog=1
            division_guard_warned = true
        end
        a[k] = f₀_sq / max(N2_z, eps(T))  # a = f²/N²(z), protected against N²≈0
    end
    return a
end

#=
================================================================================
                        DENSITY WEIGHTS FOR VERTICAL OPERATORS
================================================================================
In non-Boussinesq formulations, vertical operators can be weighted by the
background density ρ(z). The current implementation uses the Boussinesq
approximation (ρ = const), so these return unity weights.
================================================================================
=#

"""
    rho_ut(par, G) -> Vector

Background density weight on unstaggered vertical levels.

# Physical Context
In the general (non-Boussinesq) formulation, the vertical elliptic operators
include density weights:

    L[ψ] = ∇²ψ + (1/ρ) ∂/∂z(ρ a(z) ∂ψ/∂z)

For the Boussinesq approximation used in QG-YBJ+, ρ is constant and these
weights reduce to unity.

# Current Implementation
- Returns user-provided profile if `par.ρ_ut_profile` is set
- Otherwise returns ones(nz) for Boussinesq dynamics

# Arguments
- `par::QGParams`: May contain custom ρ_ut_profile
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with density weights ρ(z_k).

# Fortran Correspondence
Matches `rho_ut(k)` in the Fortran implementation.
"""
function rho_ut(par::QGParams, G::Grid)
    # Check for user-provided custom profile
    if par.ρ_ut_profile !== nothing
        @assert length(par.ρ_ut_profile) == G.nz
        return copy(par.ρ_ut_profile)
    end
    # Default: unity weights (Boussinesq approximation)
    w = similar(G.z)
    @inbounds fill!(w, 1.0)
    return w
end

"""
    rho_st(par, G) -> Vector

Background density weight on staggered vertical levels (z + dz/2).

# Physical Context
Staggered density values are used in finite-difference approximations of
vertical derivatives. In the vertical discretization:
- Unstaggered: function values at z[k]
- Staggered: derivatives at z[k] + dz/2

The staggered density is typically interpolated from unstaggered values
or defined at half-levels.

# Current Implementation
- Returns user-provided profile if `par.ρ_st_profile` is set
- Otherwise returns ones(nz) for Boussinesq dynamics

# Arguments
- `par::QGParams`: May contain custom ρ_st_profile
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with staggered density weights.

# Fortran Correspondence
Matches `rho_st(k)` in the Fortran implementation.
"""
function rho_st(par::QGParams, G::Grid)
    # Check for user-provided custom profile
    if par.ρ_st_profile !== nothing
        @assert length(par.ρ_st_profile) == G.nz
        return copy(par.ρ_st_profile)
    end
    # Default: unity weights (Boussinesq approximation)
    w = similar(G.z)
    @inbounds fill!(w, 1.0)
    return w
end

"""
    b_ell_ut(par, G) -> Vector

First-derivative coefficient b(z) for generalized vertical elliptic operators.

# Physical Context
A general vertical elliptic operator can include a first-derivative term:

    L[ψ] = a(z) ∂²ψ/∂z² + b(z) ∂ψ/∂z + c(z) ψ

The b(z) coefficient arises in:
- Non-Boussinesq formulations with density gradients
- Certain coordinate transformations
- Extended wave dispersion relations

# Current Implementation
The QG-YBJ+ model currently uses b = 0 (no first-derivative term in vertical).
This coefficient is provided for:
- Completeness and extensibility
- Alternative Helmholtz problems
- Future non-Boussinesq extensions

# Arguments
- `par::QGParams`: May contain custom b_ell_profile
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with b(z_k) values (default: zeros).
"""
function b_ell_ut(par::QGParams, G::Grid)
    # Check for user-provided custom profile
    if par.b_ell_profile !== nothing
        @assert length(par.b_ell_profile) == G.nz
        return copy(par.b_ell_profile)
    end
    # Default: no first-derivative term
    b = zeros(eltype(G.z), G.nz)
    return b
end

#=
================================================================================
                        STRATIFICATION PROFILE (N²)
================================================================================
The Brunt-Väisälä frequency squared N²(z) is the fundamental physical
quantity characterizing ocean stratification.
================================================================================
=#

"""
    N2_ut(par, G) -> Vector

Compute the Brunt-Väisälä frequency squared N²(z) on unstaggered vertical levels.

# Physical Meaning
N² (buoyancy frequency squared) measures the strength of density stratification:

    N² = -(g/ρ₀) ∂ρ/∂z

where ρ is the background density profile. N² controls:
- Vertical wave propagation speed: c_g ∝ N
- Internal wave frequency range: f < ω < N
- Stretching in QG dynamics: f²/N² term
- YBJ+ wave dispersion relation

# Typical Ocean Values
- Surface mixed layer: N² ≈ 0 (well-mixed)
- Pycnocline (100-500m): N² ≈ 10⁻⁴ s⁻² (strong stratification)
- Deep ocean: N² ≈ 10⁻⁶ s⁻² (weak stratification)

# Stratification Profiles
1. `:constant_N`: N² = 1 everywhere (nondimensional)
   - Simplest case, constant vertical wave speed

2. `:skewed_gaussian`: Realistic ocean profile
   - N²(z) = N₁² exp(-(z-z₀)²/σ²)[1 + erf(α(z-z₀)/σ√2)] + N₀²
   - Enhanced stratification at pycnocline depth z₀
   - Asymmetry parameter α creates realistic upper-ocean structure

# Arguments
- `par::QGParams`: Stratification type and coefficients
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with N²(z_k) values.

# Example
```julia
N2 = N2_ut(par, G)
# For plotting: plot(G.z, N2) shows pycnocline structure
```

# Fortran Correspondence
Matches `n2(k)` computed in `init_base_state` (init.f90).
"""
function N2_ut(par::QGParams, G::Grid)
    nz = G.nz
    N2 = similar(G.z)

    if par.stratification === :constant_N
        #= Uniform stratification: N² = 1 (nondimensional)
        Corresponds to constant vertical group velocity for internal waves =#
        @inbounds fill!(N2, 1.0)

    elseif par.stratification === :skewed_gaussian
        #= Skewed Gaussian profile for realistic pycnocline:
        - N₀² provides background deep-ocean stratification
        - N₁² is the pycnocline enhancement
        - z₀ is the pycnocline center depth
        - σ is the pycnocline width
        - α controls asymmetry (positive = sharper above z₀) =#
        N02 = par.N₀²_sg; N12 = par.N₁²_sg; σ = par.σ_sg; z0 = par.z₀_sg; α = par.α_sg
        @inbounds for k in 1:nz
            z = G.z[k]
            N2[k] = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
        end

    else
        error("Unsupported stratification: $(par.stratification)")
    end
    return N2
end

"""
    derive_density_profiles(par, G; N2_profile=nothing) -> (rho_ut, rho_st)

Derive background density profiles from stratification N²(z).

# Physical Background
In the ocean, density ρ(z) and stratification N²(z) are related by:

    N² = -(g/ρ₀) ∂ρ/∂z

Integrating: ρ(z) = ρ₀ - (ρ₀/g) ∫ N²(z') dz'

This function could derive ρ(z) from N²(z) for non-Boussinesq dynamics.

# Algorithm (if implemented)
1. Get N²(z) from `N2_ut(par, G)` or provided profile
2. Integrate: dρ/dz = -N² (nondimensional with g = ρ₀ = 1)
3. Normalize to unit mean for numerical stability
4. Interpolate to staggered grid for rho_st

# Current Implementation
Returns unity profiles (Boussinesq approximation). The QG-YBJ+ model
assumes constant background density, with stratification effects entering
only through a_ell = 1/N² in the elliptic operators.

# Arguments
- `par::QGParams`: Model parameters
- `G::Grid`: Grid structure
- `N2_profile`: Optional custom N² profile (uses N2_ut if nothing)

# Returns
- `rho_ut`: Density on unstaggered levels (length nz)
- `rho_st`: Density on staggered levels (length nz)

# Fortran Correspondence
The Fortran test1 case also uses unity weights (Boussinesq).
"""
function derive_density_profiles(par::QGParams, G::Grid; N2_profile=nothing)
    #= For the QG-YBJ+ model with Boussinesq approximation:
    - Background density ρ = const = 1 (nondimensional)
    - Stratification enters through a_ell = 1/N² in elliptic operators
    - No density weighting in vertical derivatives

    This matches the Fortran reference implementation (test1). =#
    nz = G.nz
    rho_ut = ones(eltype(G.z), nz)
    rho_st = ones(eltype(G.z), nz)
    return rho_ut, rho_st
end

#=
================================================================================
                        SPECTRAL DEALIASING
================================================================================
Pseudo-spectral methods require dealiasing to prevent errors from the
nonlinear terms. The 2/3 rule is the standard approach.
================================================================================
=#

"""
    dealias_mask(G) -> Matrix{Bool}

Compute the 2/3-rule dealiasing mask for spectral space.

# Physical Background
In pseudo-spectral methods, nonlinear terms are computed by:
1. Transform fields to physical space (inverse FFT)
2. Compute products in physical space
3. Transform back to spectral space (forward FFT)

The problem: A product of two fields with max wavenumber k_max produces
wavenumbers up to 2×k_max. With finite resolution, these high-k components
"fold back" (alias) onto resolved wavenumbers, causing errors.

# The 2/3 Rule
To prevent aliasing from quadratic nonlinearities (e.g., u·∇q):
- Keep only wavenumbers |k| ≤ (2/3) × k_Nyquist
- Truncated modes: set to zero before computing nonlinear products
- Result: product wavenumbers stay within (2/3)×2 = (4/3) < k_Nyquist

This rule is exact for quadratic nonlinearities in 1D. For 2D with
radial cutoff, it provides effective dealiasing.

# Algorithm
Uses radial (isotropic) cutoff:
- k_max = min(nx, ny) / 3
- Keep mode (i,j) if sqrt(kx² + ky²) ≤ k_max
- More isotropic than rectangular truncation

# Arguments
- `G::Grid`: Grid with dimensions nx, ny

# Returns
Matrix{Bool} of size (nx, ny):
- true = keep this wavenumber
- false = truncate (set to zero)

# Usage
```julia
mask = dealias_mask(G)
q_hat .*= mask  # Zero out aliased modes
```

# Fortran Correspondence
Matches `LL(i,j)` mask in the Fortran implementation.
"""
function dealias_mask(G::Grid)
    nx, ny = G.nx, G.ny
    keep = falses(nx, ny)

    #= Radial 2/3 cutoff in wavenumber index space
    k_max = N/3 where N = min(nx, ny)
    This ensures isotropic dealiasing =#
    kmax = floor(Int, min(nx, ny) / 3)

    for j in 1:ny, i in 1:nx
        #= Convert array index to wavenumber index (FFTW convention):
        - Indices 1 to N/2: wavenumber 0 to N/2-1 (positive)
        - Indices N/2+1 to N: wavenumber -N/2 to -1 (negative) =#
        ix = i-1; ix = ix <= nx÷2 ? ix : ix - nx
        jy = j-1; jy = jy <= ny÷2 ? jy : jy - ny

        # Radial distance in wavenumber space
        r = sqrt(ix^2 + jy^2)

        # Keep mode if within dealiasing radius
        keep[i,j] = (r <= kmax)
    end
    return keep
end

"""
    is_dealiased(i_global::Int, j_global::Int, nx::Int, ny::Int) -> Bool

Efficiently check if a mode at global indices (i_global, j_global) should be kept
after 2/3-rule dealiasing. This is more memory-efficient than storing a full mask
for distributed computing.

# Arguments
- `i_global`: Global x-index (1-based)
- `j_global`: Global y-index (1-based)
- `nx, ny`: Grid dimensions

# Returns
- `true` if the mode should be kept (inside dealiasing circle)
- `false` if the mode should be set to zero (outside dealiasing circle)

# Example
```julia
# In a distributed loop:
for j_local in 1:ny_local, i_local in 1:nx_local
    i_global = local_to_global(i_local, 1, G)
    j_global = local_to_global(j_local, 2, G)
    if is_dealiased(i_global, j_global, G.nx, G.ny)
        # Process this mode
    else
        # Set to zero
    end
end
```

# Note
This function is `@inline` for performance in tight loops.
"""
@inline function is_dealiased(i_global::Int, j_global::Int, nx::Int, ny::Int)
    # Compute 2/3 cutoff radius
    kmax = floor(Int, min(nx, ny) / 3)

    # Convert global array index to wavenumber index (FFTW convention)
    ix = i_global - 1
    ix = ix <= nx ÷ 2 ? ix : ix - nx

    jy = j_global - 1
    jy = jy <= ny ÷ 2 ? jy : jy - ny

    # Radial distance in wavenumber space
    r2 = ix^2 + jy^2

    # Keep mode if within dealiasing radius (squared comparison for efficiency)
    return r2 <= kmax^2
end

"""
    is_dealiased(i_global::Int, j_global::Int, G::Grid) -> Bool

Overload for Grid argument - extracts nx, ny from Grid.
"""
@inline is_dealiased(i_global::Int, j_global::Int, G::Grid) = is_dealiased(i_global, j_global, G.nx, G.ny)
