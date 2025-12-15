#=
================================================================================
                        parameters.jl - Model Parameters
================================================================================

This file defines the QGParams struct containing all physical and numerical
parameters for the QG-YBJ+ model. Parameters are organized into categories:

1. DOMAIN PARAMETERS: Grid resolution and physical domain size
2. TIME STEPPING: Time step, number of steps
3. PHYSICAL PARAMETERS: Coriolis f0, stratification N²
4. VISCOSITY/HYPERVISCOSITY: Dissipation parameters
5. PHYSICS SWITCHES: Control different physics modes
6. STRATIFICATION PROFILES: Parameters for N²(z) profiles

DIMENSIONAL PARAMETERS:
-----------------------
The model uses dimensional-like parameters directly:
- f0: Coriolis parameter (default 1.0)
- N2: Buoyancy frequency squared N² (default 1.0)

Key coefficients derived from these:
- Wave dispersion coefficient: N²/(2f)
- Elliptic coefficient: a = f²/N² = 1/N² when f=1
- Inertial period: T = 2π/f

With defaults f0=1, N2=1:
- Dispersion coefficient = 0.5
- Inertial period = 2π

FORTRAN CORRESPONDENCE:
----------------------
This struct corresponds to parameters_test1.f90 and parameters_test2.f90
in the original QG_YBJp Fortran code.

================================================================================
=#

"""
    QGParams{T}

Container for all physical and numerical parameters of the QG-YBJ+ model.

# Domain Parameters
- `nx, ny, nz`: Grid resolution in x, y, z directions
- `Lx, Ly`: Horizontal domain size (typically 2π for nondimensional)

# Time Stepping
- `dt`: Time step size (nondimensional)
- `nt`: Total number of time steps

# Physical Parameters
- `f0`: Coriolis parameter (typically 1.0 for nondimensional)

# Viscosity/Hyperviscosity
The model uses two hyperdiffusion operators for stability:
- `nuh1, ilap1`: First hyperviscosity coefficient and Laplacian power for mean flow
- `nuh2, ilap2`: Second hyperviscosity coefficient and Laplacian power for mean flow
- `nuh1w, ilap1w`: First hyperviscosity for waves
- `nuh2w, ilap2w`: Second hyperviscosity for waves
- `nuz`: Vertical diffusion coefficient

The hyperdiffusion term is: ν₁(-∇²)^ilap1 + ν₂(-∇²)^ilap2

# Physical Parameters
- `N2`: Buoyancy frequency squared N² (default 1.0)
- `W2F`: (Uw/U)² - ratio of wave to flow velocity squared
- `gamma`: Robert-Asselin filter coefficient (typically 10⁻³)

# Physics Switches
These boolean flags control different physics modes:
- `inviscid`: If true, disable all dissipation
- `linear`: If true, disable nonlinear advection terms
- `no_dispersion`: If true, disable wave dispersion (A=0)
- `passive_scalar`: If true, waves are passive (no dispersion, no refraction)
- `ybj_plus`: If true, use YBJ+ formulation; if false, use normal YBJ
- `no_feedback`: If true, disable wave feedback on mean flow
- `fixed_flow`: If true, mean flow doesn't evolve (ψ constant in time)
- `no_wave_feedback`: If true, waves don't affect mean flow (qʷ = 0)

# Stratification Parameters (Skewed Gaussian profile)
For the skewed Gaussian N²(z) profile:
    N²(z) = N₁² exp(-(z-z₀)²/σ²) [1 + erf(α(z-z₀)/(σ√2))] + N₀²

- `N02_sg`: Background N² (N₀²)
- `N12_sg`: Peak N² amplitude (N₁²)
- `sigma_sg`: Width parameter (σ)
- `z0_sg`: Center depth (z₀)
- `alpha_sg`: Skewness parameter (α)

# Example
```julia
par = default_params(nx=128, ny=128, nz=64, dt=0.001, nt=10000)
```

See also: [`default_params`](@ref), [`with_density_profiles`](@ref)
"""
Base.@kwdef mutable struct QGParams{T}
    #= ====================================================================
                            DOMAIN PARAMETERS
    ==================================================================== =#
    nx::Int                    # Number of grid points in x (horizontal)
    ny::Int                    # Number of grid points in y (horizontal)
    nz::Int                    # Number of grid points in z (vertical)
    Lx::T                      # Domain size in x (typically 2π)
    Ly::T                      # Domain size in y (typically 2π)

    #= ====================================================================
                            TIME STEPPING
    ==================================================================== =#
    dt::T                      # Time step (nondimensional: dt = Ro/20 typical)
    nt::Int                    # Total number of time steps

    #= ====================================================================
                            PHYSICAL PARAMETERS
    ==================================================================== =#
    f0::T                      # Coriolis parameter (1.0 for nondimensional)

    #= ====================================================================
                        VISCOSITY / HYPERVISCOSITY
    ====================================================================
    The model uses TWO hyperdiffusion operators for numerical stability:

    Dissipation = -ν₁(-1)^ilap1 ∇^(2*ilap1) - ν₂(-1)^ilap2 ∇^(2*ilap2)

    Typical values from Fortran test1:
    - ilap1=2, ilap2=6 (biharmonic + hyper-6)
    - nuh1~0.01, nuh2~10.0 for 256³ resolution
    ==================================================================== =#
    nu_h::T                    # Generic horizontal viscosity (legacy)
    nu_v::T                    # Generic vertical viscosity (legacy)

    # Mean flow hyperdiffusion
    nuh1::T                    # First hyperviscosity coefficient (flow)
    nuh2::T                    # Second hyperviscosity coefficient (flow)
    ilap1::Int                 # First Laplacian power (e.g., 2 = biharmonic)
    ilap2::Int                 # Second Laplacian power (e.g., 6 = hyper-6)

    # Wave field hyperdiffusion
    nuh1w::T                   # First hyperviscosity coefficient (waves)
    nuh2w::T                   # Second hyperviscosity coefficient (waves)
    ilap1w::Int                # First Laplacian power for waves
    ilap2w::Int                # Second Laplacian power for waves

    # Vertical diffusion
    nuz::T                     # Vertical diffusion coefficient for q

    #= ====================================================================
                        PHYSICAL PARAMETERS
    ====================================================================
    These are the key physical parameters that control the wave dynamics:
    - f0: Coriolis parameter (default 1.0)
    - N2: Buoyancy frequency squared N² (default 1.0 for constant_N)

    The wave dispersion coefficient is: N²/(2f)
    The elliptic coefficient is: a = f²/N² (= 1/N² when f=1)
    ==================================================================== =#
    N2::T                      # Buoyancy frequency squared N² (default 1.0)
    W2F::T                     # (Uw/U)² = wave-to-flow velocity ratio squared
    gamma::T                   # Robert-Asselin filter parameter (typ. 10⁻³)

    #= ====================================================================
                            FLAGS AND SWITCHES
    ====================================================================
    These boolean flags allow running the model in various limiting cases
    for testing and understanding different physical regimes.
    ==================================================================== =#
    linear_vert_structure::Int # Mapping from Fortran (0 or 1)

    stratification::Symbol     # :constant_N or :skewed_gaussian

    # Dissipation control
    inviscid::Bool             # true = disable ALL dissipation

    # Nonlinearity control
    linear::Bool               # true = disable nonlinear advection (J terms)

    # Wave physics control
    no_dispersion::Bool        # true = disable wave dispersion (set A=0)
    passive_scalar::Bool       # true = waves are passive tracers (no dispersion, no refraction)
    ybj_plus::Bool             # true = use YBJ+ formulation; false = normal YBJ

    # Wave-mean flow interaction control
    no_feedback::Bool          # true = disable wave feedback completely
    fixed_flow::Bool           # true = mean flow ψ doesn't evolve in time
    no_wave_feedback::Bool     # true = waves don't affect mean flow (qʷ = 0)

    #= ====================================================================
                    SKEWED GAUSSIAN STRATIFICATION PARAMETERS
    ====================================================================
    The skewed Gaussian N² profile is:

        N²(z) = N₁² exp(-(z-z₀)²/σ²) [1 + erf(α(z-z₀)/(σ√2))] + N₀²

    This allows modeling realistic ocean stratification with:
    - A pycnocline (region of strong N²) at depth z₀
    - Asymmetric profile controlled by skewness α
    - Background stratification N₀² above/below pycnocline

    Default values are from Fortran test1 (nondimensional, L3=2π domain).
    ==================================================================== =#
    N02_sg::T                  # Background N² value (N₀²)
    N12_sg::T                  # Peak N² amplitude (N₁²)
    sigma_sg::T                # Width of pycnocline (σ)
    z0_sg::T                   # Center depth of pycnocline (z₀)
    alpha_sg::T                # Skewness parameter (α)

    #= ====================================================================
                    OPTIONAL VERTICAL PROFILES (Advanced)
    ====================================================================
    These allow overriding default density/stratification profiles with
    custom user-provided profiles. If nothing, defaults are computed.
    ==================================================================== =#
    rho_ut_profile::Union{Nothing,Vector{T}} = nothing   # Unstaggered density weights
    rho_st_profile::Union{Nothing,Vector{T}} = nothing   # Staggered density weights
    b_ell_profile::Union{Nothing,Vector{T}} = nothing    # b_ell coefficient profile
end

"""
    with_density_profiles(par; rho_ut, rho_st, b_ell=nothing)

Return a new `QGParams` with user-provided vertical density profiles.

This is useful for implementing custom stratification that doesn't fit
the standard profiles (constant N², skewed Gaussian).

# Arguments
- `par`: Existing QGParams to copy
- `rho_ut`: Unstaggered density weights (length nz)
- `rho_st`: Staggered density weights (length nz)
- `b_ell`: Optional b_ell coefficient profile (length nz)

# Returns
New QGParams with profiles populated.

# Example
```julia
par = default_params(nz=64)
rho_ut = ones(64)
rho_st = ones(64)
par_custom = with_density_profiles(par; rho_ut=rho_ut, rho_st=rho_st)
```
"""
function with_density_profiles(par::QGParams{T};
                               rho_ut::AbstractVector{T},
                               rho_st::AbstractVector{T},
                               b_ell::Union{Nothing,AbstractVector{T}}=nothing) where T
    @assert length(rho_ut) == par.nz "rho_ut must have length nz=$(par.nz)"
    @assert length(rho_st) == par.nz "rho_st must have length nz=$(par.nz)"
    bprof = b_ell
    # Rebuild parameter struct with all existing fields + new profiles
    return QGParams{T}(;
        (name => getfield(par, name) for name in fieldnames(typeof(par)) if !(name in (:rho_ut_profile, :rho_st_profile, :b_ell_profile)))...,
        rho_ut_profile = collect(rho_ut),
        rho_st_profile = collect(rho_st),
        b_ell_profile = bprof === nothing ? nothing : collect(bprof),
    )
end

"""
    default_params(; kwargs...) -> QGParams

Construct a default parameter set using dimensional parameters f0 and N2.

With f0=1, N2=1 (constant_N stratification):
- Dispersion coefficient = N²/(2f) = 0.5
- Inertial period = 2π/f = 2π

# Keyword Arguments

**Domain and Time:**
- `nx, ny, nz`: Grid resolution (default: 64)
- `Lx, Ly`: Domain size (default: 2π)
- `dt`: Time step (default: 0.001)
- `nt`: Number of steps (default: 10000)

**Physical Parameters:**
- `f0`: Coriolis parameter f (default: 1.0)
- `N2`: Buoyancy frequency squared N² (default: 1.0)
- `W2F`: Wave-to-flow velocity ratio squared (default: 0.01)
- `stratification`: :constant_N or :skewed_gaussian (default: :constant_N)

**Hyperdiffusion:**
- `nuh1, nuh2`: Flow hyperviscosity coefficients (default: 0.01, 10.0)
- `ilap1, ilap2`: Laplacian powers (default: 2, 6)
- `nuh1w, nuh2w`: Wave hyperviscosity coefficients (default: 0.0, 10.0)
- `gamma`: Robert-Asselin filter (default: 1e-3)

**Physics Switches:**
- `ybj_plus`: Use YBJ+ formulation (default: true)
- `fixed_flow`: Keep mean flow constant (default: false)
- `no_wave_feedback`: Disable wave feedback on flow (default: true)
- `inviscid`: Disable dissipation (default: false)
- `linear`: Disable nonlinear advection (default: false)
- `no_dispersion`: Disable wave dispersion (default: false)
- `passive_scalar`: Waves as passive tracers (default: false)

# Example
```julia
# Basic usage - f0=1, N2=1 by default
par = default_params()

# Custom resolution with steady flow
par = default_params(nx=128, ny=128, nz=64, fixed_flow=true)

# Custom wave amplitude (30% of flow velocity)
par = default_params(W2F=0.09)  # W2F = (0.3)² = 0.09

# Custom stratification (stronger N²)
par = default_params(N2=4.0)  # Dispersion = 4/(2×1) = 2.0
```

See also: [`QGParams`](@ref)
"""
function default_params(; nx=64, ny=64, nz=64, Lx=2π, Ly=2π,
                           dt=1e-3, nt=10_000, f0=1.0, N2=1.0,
                           W2F=0.01, gamma=1e-3,
                           nu_h=0.0, nu_v=0.0,
                           nuh1=0.01, nuh2=10.0, ilap1=2, ilap2=6,
                           nuh1w=0.0, nuh2w=10.0, ilap1w=2, ilap2w=6,
                           nuz=0.0,
                           linear_vert_structure=0,
                           stratification::Symbol=:constant_N,
                           inviscid=false, linear=false,
                           no_dispersion=false, passive_scalar=false,
                           ybj_plus=true, no_feedback=true,
                           fixed_flow=false, no_wave_feedback=true)
    T = Float64

    #= Dimensional parameters f0 and N2:
    - Dispersion coefficient = N²/(2f)
    - Elliptic coefficient a = f²/N² = 1/N² when f=1
    - Inertial period T = 2π/f
    =#

    #= Skewed Gaussian stratification parameters (Fortran test1 values)
    These are nondimensionalized for L3 = 2π domain =#
    N02_sg = T(0.537713935783168)     # Background N²
    N12_sg = T(2.684198470106461)     # Peak N² amplitude
    sigma_sg = T(0.648457170048730)   # Pycnocline width
    z0_sg = T(6.121537923499139)      # Pycnocline depth
    alpha_sg = T(-5.338431587899242)  # Skewness

    return QGParams{T}(; nx, ny, nz, Lx=T(Lx), Ly=T(Ly), dt=T(dt), nt,
                         f0=T(f0), nu_h=T(nu_h), nu_v=T(nu_v),
                         linear_vert_structure, stratification,
                         N2=T(N2), W2F=T(W2F), gamma=T(gamma),
                         nuh1=T(nuh1), nuh2=T(nuh2), ilap1, ilap2,
                         nuh1w=T(nuh1w), nuh2w=T(nuh2w), ilap1w, ilap2w,
                         nuz=T(nuz), inviscid, linear, no_dispersion, passive_scalar,
                         ybj_plus, no_feedback, fixed_flow, no_wave_feedback,
                         N02_sg, N12_sg, sigma_sg, z0_sg, alpha_sg)
end
