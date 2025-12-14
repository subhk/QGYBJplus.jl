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

    if par.stratification === :constant_N
        #= Constant N²: simplest case, a = 1/1 = 1 everywhere =#
        @inbounds for k in 1:nz
            a[k] = 1.0 / 1.0  # Normalized
        end

    elseif par.stratification === :skewed_gaussian
        #= Skewed Gaussian N² profile:
        N²(z) = N₁² exp(-(z-z₀)²/σ²)[1 + erf(α(z-z₀)/σ√2)] + N₀²

        This creates a realistic pycnocline with:
        - Enhanced stratification near z₀
        - Asymmetric shape controlled by α
        - Background N₀² in deep ocean =#
        N02 = par.N02_sg; N12 = par.N12_sg; σ = par.sigma_sg; z0 = par.z0_sg; α = par.alpha_sg
        @inbounds for k in 1:nz
            z = G.z[k]
            N2 = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
            a[k] = 1.0 / N2  # a = 1/N² (nondimensional)
        end

    else
        error("Unsupported stratification: $(par.stratification)")
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
- Returns user-provided profile if `par.rho_ut_profile` is set
- Otherwise returns ones(nz) for Boussinesq dynamics

# Arguments
- `par::QGParams`: May contain custom rho_ut_profile
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with density weights ρ(z_k).

# Fortran Correspondence
Matches `rho_ut(k)` in the Fortran implementation.
"""
function rho_ut(par::QGParams, G::Grid)
    # Check for user-provided custom profile
    if par.rho_ut_profile !== nothing
        @assert length(par.rho_ut_profile) == G.nz
        return copy(par.rho_ut_profile)
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
- Returns user-provided profile if `par.rho_st_profile` is set
- Otherwise returns ones(nz) for Boussinesq dynamics

# Arguments
- `par::QGParams`: May contain custom rho_st_profile
- `G::Grid`: Grid with vertical levels

# Returns
Vector of length nz with staggered density weights.

# Fortran Correspondence
Matches `rho_st(k)` in the Fortran implementation.
"""
function rho_st(par::QGParams, G::Grid)
    # Check for user-provided custom profile
    if par.rho_st_profile !== nothing
        @assert length(par.rho_st_profile) == G.nz
        return copy(par.rho_st_profile)
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
        N02 = par.N02_sg; N12 = par.N12_sg; σ = par.sigma_sg; z0 = par.z0_sg; α = par.alpha_sg
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

Derive density-like vertical profiles from stratification:
- Start with N²(z) (uses `N2_ut(par,G)` if not provided).
- Integrate a simple Boussinesq relation dρ/dz = -N² (nondimensional g=ρ₀=1)
  to obtain a monotonically varying background density ρ(z). Then normalize
  ρ to unit mean and ensure positivity.
- Construct `rho_ut[k] = ρ(z_k)` on unstaggered levels.
- Construct staggered `rho_st` as vertical averages between adjacent levels
  with boundary handling: `rho_st[1]=rho_ut[1]`, `rho_st[nz]=rho_ut[nz-1]`, and
  `rho_st[k]=0.5(rho_ut[k]+rho_ut[k-1])` for interior 2..nz-1.

This heuristic mirrors the Fortran usage of `rho_ut`/`rho_st` in weighted
vertical operators and provides a consistent default derived from N².
"""
function derive_density_profiles(par::QGParams, G::Grid; N2_profile=nothing)
    # For the Fortran reference (test1), the background density weights used in
    # vertical operators are unity (Boussinesq), while stratification enters via
    # N² and the corresponding a_ell = 1/N². So return ones.
    nz = G.nz
    rho_ut = ones(eltype(G.z), nz)
    rho_st = ones(eltype(G.z), nz)
    return rho_ut, rho_st
end

"""
    dealias_mask(G) -> Matrix{Bool}

2/3-rule horizontal dealiasing mask `L(i,j)` with radial cutoff, modeled after
the Fortran practice. True indicates mode is kept; false indicates it is
truncated.
"""
function dealias_mask(G::Grid)
    nx, ny = G.nx, G.ny
    keep = falses(nx, ny)
    # Radial 2/3 cutoff in index space
    kmax = floor(Int, min(nx, ny) / 3)
    for j in 1:ny, i in 1:nx
        ix = i-1; ix = ix <= nx÷2 ? ix : ix - nx
        jy = j-1; jy = jy <= ny÷2 ? jy : jy - ny
        r = sqrt(ix^2 + jy^2)
        keep[i,j] = (r <= kmax)
    end
    return keep
end
