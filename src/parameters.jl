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
- `Lx, Ly, Lz`: Domain size in x, y, z in meters (REQUIRED - no default)
- `x0, y0`: Domain origin in x, y (default: 0, use -Lx/2,-Ly/2 for centered domain)

# Time Stepping
- `dt`: Time step size
- `nt`: Total number of time steps

# Physical Parameters
- `f₀`: Coriolis parameter (typically 1.0 for nondimensional)
- `N²`: Buoyancy frequency squared (default 1.0)
- `W2F`: DEPRECATED - no longer used (kept for backward compatibility)
- `γ`: Robert-Asselin filter coefficient (typically 10⁻³)
- `linear_vert_structure`: Legacy Fortran flag (0 or 1), typically 0

# Viscosity/Hyperviscosity
The model uses two hyperdiffusion operators for stability:
- `νₕ, νᵥ`: Legacy generic viscosity coefficients (prefer specific coefficients below)
- `νₕ₁, ilap1`: First hyperviscosity coefficient and Laplacian power for mean flow
- `νₕ₂, ilap2`: Second hyperviscosity coefficient and Laplacian power for mean flow
- `νₕ₁ʷ, ilap1w`: First hyperviscosity for waves
- `νₕ₂ʷ, ilap2w`: Second hyperviscosity for waves
- `νz`: Vertical diffusion coefficient

The hyperdiffusion term is: ν₁(-∇²)^ilap1 + ν₂(-∇²)^ilap2

# Physics Switches
These boolean flags control different physics modes:
- `inviscid`: If true, disable all dissipation
- `linear`: If true, disable nonlinear advection terms
- `no_dispersion`: If true, disable wave dispersion (A=0)
- `passive_scalar`: If true, waves are passive (no dispersion, no refraction)
- `ybj_plus`: If true, use YBJ+ formulation; if false, use normal YBJ
- `no_feedback`: If true, disable ALL wave-mean flow coupling (master switch)
- `fixed_flow`: If true, mean flow doesn't evolve (ψ constant in time)
- `no_wave_feedback`: If true, disable qʷ term specifically (waves don't modify PV)

Note: Wave feedback is enabled only when BOTH `no_feedback=false` AND `no_wave_feedback=false`.

# Stratification Parameters (Skewed Gaussian profile)
For the skewed Gaussian N²(z) profile:
    N²(z) = N₁² exp(-(z-z₀)²/σ²) [1 + erf(α(z-z₀)/(σ√2))] + N₀²

- `N₀²_sg`: Background N² (N₀²)
- `N₁²_sg`: Peak N² amplitude (N₁²)
- `σ_sg`: Width parameter (σ)
- `z₀_sg`: Center depth (z₀)
- `α_sg`: Skewness parameter (α)

# Example
```julia
par = default_params(nx=128, ny=128, nz=64, Lx=500e3, Ly=500e3, Lz=4000.0, dt=0.001, nt=10000)
```

See also: `default_params`, `with_b_ell_profile`
"""
Base.@kwdef mutable struct QGParams{T}
    #= ====================================================================
                            DOMAIN PARAMETERS
    ==================================================================== =#
    nx::Int                    # Number of grid points in x (horizontal)
    ny::Int                    # Number of grid points in y (horizontal)
    nz::Int                    # Number of grid points in z (vertical)
    Lx::T                      # Domain size in x [m] (REQUIRED)
    Ly::T                      # Domain size in y [m] (REQUIRED)
    Lz::T                      # Domain size in z [m] (REQUIRED)
    x0::T                      # Domain origin in x [m] (default: 0, use -Lx/2 for centered)
    y0::T                      # Domain origin in y [m] (default: 0, use -Ly/2 for centered)

    #= ====================================================================
                            TIME STEPPING
    ==================================================================== =#
    dt::T                      # Time step
    nt::Int                    # Total number of time steps

    #= ====================================================================
                            PHYSICAL PARAMETERS
    ====================================================================
    Key physical parameters that control the wave-mean flow dynamics:
    - f₀: Coriolis parameter (default 1.0)
    - N²: Buoyancy frequency squared (default 1.0 for constant_N)

    The wave dispersion coefficient is: N²/(2f)
    The elliptic coefficient is: a = f²/N² (= 1/N² when f=1)
    ==================================================================== =#
    f₀::T                      # Coriolis parameter (1.0 for nondimensional)
    N²::T                      # Buoyancy frequency squared (default 1.0)
    W2F::T                     # DEPRECATED: not used (dimensional equations have B with actual amplitude)
    γ::T                       # Robert-Asselin filter parameter (typ. 10⁻³)

    #= ====================================================================
                        VISCOSITY / HYPERVISCOSITY
    ====================================================================
    The model uses TWO hyperdiffusion operators for numerical stability:

    Dissipation = -ν₁(-1)^ilap1 ∇^(2*ilap1) - ν₂(-1)^ilap2 ∇^(2*ilap2)

    Typical values from Fortran test1:
    - ilap1=2, ilap2=6 (biharmonic + hyper-6)
    - νₕ₁~0.01, νₕ₂~10.0 for 256³ resolution
    ==================================================================== =#
    νₕ::T                      # Generic horizontal viscosity (legacy)
    νᵥ::T                      # Generic vertical viscosity (legacy)

    # Mean flow hyperdiffusion
    νₕ₁::T                     # First hyperviscosity coefficient (flow)
    νₕ₂::T                     # Second hyperviscosity coefficient (flow)
    ilap1::Int                 # First Laplacian power (e.g., 2 = biharmonic)
    ilap2::Int                 # Second Laplacian power (e.g., 6 = hyper-6)

    # Wave field hyperdiffusion
    νₕ₁ʷ::T                    # First hyperviscosity coefficient (waves)
    νₕ₂ʷ::T                    # Second hyperviscosity coefficient (waves)
    ilap1w::Int                # First Laplacian power for waves
    ilap2w::Int                # Second Laplacian power for waves

    # Vertical diffusion
    νz::T                      # Vertical diffusion coefficient for q

    #= ====================================================================
                            FLAGS AND SWITCHES
    ====================================================================
    These boolean flags allow running the model in various limiting cases
    for testing and understanding different physical regimes.
    ==================================================================== =#
    linear_vert_structure::Int # Mapping from Fortran (0 or 1)

    stratification::Symbol     # :constant_N, :skewed_gaussian, or profile-based (:tanh_profile/:from_file)

    # Dissipation control
    inviscid::Bool             # true = disable ALL dissipation

    # Nonlinearity control
    linear::Bool               # true = disable nonlinear advection (J terms)

    # Wave physics control
    no_dispersion::Bool        # true = disable wave dispersion (set A=0)
    passive_scalar::Bool       # true = waves are passive tracers (no dispersion, no refraction)
    ybj_plus::Bool             # true = use YBJ+ formulation; false = normal YBJ

    # Wave-mean flow interaction control
    no_feedback::Bool          # true = disable ALL wave-mean flow coupling (master switch)
    fixed_flow::Bool           # true = mean flow ψ doesn't evolve in time
    no_wave_feedback::Bool     # true = disable qʷ term specifically (waves don't modify PV)

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
    N₀²_sg::T                  # Background N² value (N₀²)
    N₁²_sg::T                  # Peak N² amplitude (N₁²)
    σ_sg::T                    # Width of pycnocline (σ)
    z₀_sg::T                   # Center depth of pycnocline (positive below surface)
    α_sg::T                    # Skewness parameter (α)

    #= ====================================================================
                    OPTIONAL VERTICAL PROFILES (Advanced)
    ====================================================================
    These allow overriding default stratification profiles with
    custom user-provided profiles. If nothing, defaults are computed.
    ==================================================================== =#
    b_ell_profile::Union{Nothing,Vector{T}} = nothing  # b_ell coefficient profile (f₀²/N²)
end

"""
    with_b_ell_profile(par; b_ell)

Return a new `QGParams` with a user-provided b_ell (f₀²/N²) profile.

This is useful for implementing custom stratification that doesn't fit
the standard profiles (constant N², skewed Gaussian).

# Arguments
- `par`: Existing QGParams to copy
- `b_ell`: b_ell coefficient profile (length nz), where b_ell = f₀²/N²

# Returns
New QGParams with b_ell_profile populated.

# Example
```julia
par = default_params(nz=64)
b_ell = ones(64)  # constant N² case
par_custom = with_b_ell_profile(par; b_ell=b_ell)
```
"""
function with_b_ell_profile(par::QGParams{T};
                            b_ell::AbstractVector{T}) where T
    @assert length(b_ell) == par.nz "b_ell must have length nz=$(par.nz)"
    # Rebuild parameter struct with all existing fields + new profile
    return QGParams{T}(;
        (name => getfield(par, name) for name in fieldnames(typeof(par)) if name != :b_ell_profile)...,
        b_ell_profile = collect(b_ell),
    )
end

"""
    default_params(; kwargs...) -> QGParams

Construct a default parameter set using dimensional parameters f₀ and N².

With f₀=1, N²=1 (constant_N stratification):
- Dispersion coefficient = N²/(2f) = 0.5
- Inertial period = 2π/f = 2π

# Keyword Arguments

**Domain and Time:**
- `nx, ny, nz`: Grid resolution (default: 64)
- `Lx, Ly, Lz`: Domain size in meters (REQUIRED - no default)
- `centered`: If true, center domain at origin: x ∈ [-Lx/2, Lx/2) (default: false)
- `x0, y0`: Domain origin (default: 0, or -Lx/2,-Ly/2 if centered=true)
- `dt`: Time step (default: 0.001)
- `nt`: Number of steps (default: 10000)

**Physical Parameters:**
- `f₀`: Coriolis parameter f (default: 1.0)
- `N²`: Buoyancy frequency squared (default: 1.0)
- `stratification`: :constant_N, :skewed_gaussian, :tanh_profile, or :from_file (default: :constant_N)
  - Note: :tanh_profile and :from_file require a supplied N² profile at runtime.

**Hyperdiffusion:**
- `νₕ₁, νₕ₂`: Flow hyperviscosity coefficients (default: 0.01, 10.0)
- `ilap1, ilap2`: Laplacian powers (default: 2, 6)
- `νₕ₁ʷ, νₕ₂ʷ`: Wave hyperviscosity coefficients (default: 0.0, 10.0)
- `γ`: Robert-Asselin filter (default: 1e-3)

**Physics Switches:**
- `ybj_plus`: Use YBJ+ formulation (default: true)
- `fixed_flow`: Keep mean flow constant (default: false)
- `no_feedback`: Master switch to disable all wave-mean coupling (default: true)
- `no_wave_feedback`: Disable qʷ term specifically (default: true)
- `inviscid`: Disable dissipation (default: false)
- `linear`: Disable nonlinear advection (default: false)
- `no_dispersion`: Disable wave dispersion (default: false)
- `passive_scalar`: Waves as passive tracers (default: false)

Note: Wave feedback is enabled only when BOTH `no_feedback=false` AND `no_wave_feedback=false`.

# Example
```julia
# Basic dimensional setup - domain size is REQUIRED
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km

# Centered domain: x,y ∈ [-Lx/2, Lx/2) - useful for dipole simulations
par = default_params(Lx=70e3, Ly=70e3, Lz=2000.0, centered=true)  # x,y ∈ [-35km, 35km)

# Custom resolution with steady flow
par = default_params(nx=128, ny=128, nz=64, Lx=500e3, Ly=500e3, Lz=4000.0, fixed_flow=true)

# Custom stratification (stronger N²)
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, N²=4.0)

# Enable wave feedback on mean flow
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, no_wave_feedback=false)
```

!!! note "Differences from ModelConfig"
    The defaults here differ from `ModelConfig` (used by `create_simple_config`):
    - `default_params`: `inviscid=false`, `no_wave_feedback=true` (production runs with dissipation)
    - `ModelConfig`: `inviscid=true`, `no_wave_feedback=false` (idealized inviscid runs)

See also: [`QGParams`](@ref)
"""
function default_params(; nx=64, ny=64, nz=64,
                           Lx::Real, Ly::Real, Lz::Real,  # REQUIRED - no defaults
                           x0::Union{Real,Nothing}=nothing,  # Domain origin in x (nothing = use centered flag)
                           y0::Union{Real,Nothing}=nothing,  # Domain origin in y (nothing = use centered flag)
                           centered::Bool=false,          # If true, center domain at origin: x ∈ [-Lx/2, Lx/2)
                           dt=1e-3, nt=10_000, f₀=1.0, N²=1.0,
                           W2F=nothing, γ=1e-3,  # W2F is deprecated
                           νₕ=0.0, νᵥ=0.0,
                           νₕ₁=0.01, νₕ₂=10.0, ilap1=2, ilap2=6,
                           νₕ₁ʷ=0.0, νₕ₂ʷ=10.0, ilap1w=2, ilap2w=6,
                           νz=0.0,
                           linear_vert_structure=0,
                           stratification::Symbol=:constant_N,
                           inviscid=false, linear=false,
                           no_dispersion=false, passive_scalar=false,
                           ybj_plus=true, no_feedback=true,
                           fixed_flow=false, no_wave_feedback=true)

    #= ============== PARAMETER VALIDATION ============== =#

    # W2F deprecation warning
    if W2F !== nothing
        @warn "W2F parameter is deprecated and no longer used. It will be ignored." maxlog=1
    end
    W2F_val = 0.01  # Default value for struct (unused)

    # Grid dimensions must be positive
    nx > 0 || throw(ArgumentError("nx must be positive (got nx=$nx)"))
    ny > 0 || throw(ArgumentError("ny must be positive (got ny=$ny)"))
    nz > 0 || throw(ArgumentError("nz must be positive (got nz=$nz)"))

    # Domain sizes must be positive
    Lx > 0 || throw(ArgumentError("Lx must be positive (got Lx=$Lx)"))
    Ly > 0 || throw(ArgumentError("Ly must be positive (got Ly=$Ly)"))
    Lz > 0 || throw(ArgumentError("Lz must be positive (got Lz=$Lz)"))

    # Time stepping parameters
    dt > 0 || throw(ArgumentError("dt must be positive (got dt=$dt)"))
    nt >= 1 || throw(ArgumentError("nt must be at least 1 (got nt=$nt)"))

    # Physical parameters
    N² > 0 || throw(ArgumentError("N² (buoyancy frequency squared) must be positive (got N²=$N²)"))
    # Note: f₀ can be negative for southern hemisphere simulations (f₀ < 0 when latitude < 0)
    f₀ != 0 || throw(ArgumentError("f₀ (Coriolis parameter) cannot be zero (use negative values for southern hemisphere)"))

    # Robert-Asselin filter coefficient
    0 <= γ <= 1 || throw(ArgumentError("γ (Robert-Asselin coefficient) must be in [0,1] (got γ=$γ)"))

    # Hyperviscosity coefficients must be non-negative
    νₕ₁ >= 0 || throw(ArgumentError("νₕ₁ must be non-negative (got νₕ₁=$νₕ₁)"))
    νₕ₂ >= 0 || throw(ArgumentError("νₕ₂ must be non-negative (got νₕ₂=$νₕ₂)"))
    νₕ₁ʷ >= 0 || throw(ArgumentError("νₕ₁ʷ must be non-negative (got νₕ₁ʷ=$νₕ₁ʷ)"))
    νₕ₂ʷ >= 0 || throw(ArgumentError("νₕ₂ʷ must be non-negative (got νₕ₂ʷ=$νₕ₂ʷ)"))
    νz >= 0 || throw(ArgumentError("νz must be non-negative (got νz=$νz)"))

    # Laplacian powers must be positive integers
    ilap1 > 0 || throw(ArgumentError("ilap1 must be positive (got ilap1=$ilap1)"))
    ilap2 > 0 || throw(ArgumentError("ilap2 must be positive (got ilap2=$ilap2)"))
    ilap1w > 0 || throw(ArgumentError("ilap1w must be positive (got ilap1w=$ilap1w)"))
    ilap2w > 0 || throw(ArgumentError("ilap2w must be positive (got ilap2w=$ilap2w)"))

    # Stratification type
    stratification in (:constant_N, :skewed_gaussian, :tanh_profile, :from_file, :analytical, :function) ||
        throw(ArgumentError("stratification must be :constant_N, :skewed_gaussian, :tanh_profile, :from_file, or :analytical (got :$stratification)"))

    if stratification in (:tanh_profile, :from_file, :analytical, :function)
        @warn "default_params uses only constant or skewed_gaussian profiles. For stratification=$stratification, " *
              "provide N2_profile at runtime or use ModelConfig/StratificationConfig." maxlog=1
    end

    # Warnings for non-optimal settings
    if !ispow2(nx) || !ispow2(ny)
        @warn "Grid dimensions (nx=$nx, ny=$ny) are not powers of 2 - FFTs may be slower"
    end

    T = Float64

    # Compute domain origin from centered flag if not explicitly provided
    x0_val = if x0 !== nothing
        T(x0)
    elseif centered
        T(-Lx/2)  # Center at origin: x ∈ [-Lx/2, Lx/2)
    else
        T(0)      # Standard: x ∈ [0, Lx)
    end

    y0_val = if y0 !== nothing
        T(y0)
    elseif centered
        T(-Ly/2)  # Center at origin: y ∈ [-Ly/2, Ly/2)
    else
        T(0)      # Standard: y ∈ [0, Ly)
    end

    #= Dimensional parameters f₀ and N²:
    - Dispersion coefficient = N²/(2f)
    - Elliptic coefficient a = f²/N² = 1/N² when f=1
    - Inertial period T = 2π/f
    =#

    #= Skewed Gaussian stratification parameters (Fortran test1 values)
    These are nondimensionalized for Lz = 2π domain =#
    N₀²_sg = T(0.537713935783168)     # Background N²
    N₁²_sg = T(2.684198470106461)     # Peak N² amplitude
    σ_sg = T(0.648457170048730)       # Pycnocline width
    z₀_sg = T(6.121537923499139)      # Pycnocline depth
    α_sg = T(-5.338431587899242)      # Skewness

    return QGParams{T}(; nx, ny, nz, Lx=T(Lx), Ly=T(Ly), Lz=T(Lz),
                         x0=x0_val, y0=y0_val, dt=T(dt), nt,
                         f₀=T(f₀), νₕ=T(νₕ), νᵥ=T(νᵥ),
                         linear_vert_structure, stratification,
                         N²=T(N²), W2F=T(W2F_val), γ=T(γ),
                         νₕ₁=T(νₕ₁), νₕ₂=T(νₕ₂), ilap1, ilap2,
                         νₕ₁ʷ=T(νₕ₁ʷ), νₕ₂ʷ=T(νₕ₂ʷ), ilap1w, ilap2w,
                         νz=T(νz), inviscid, linear, no_dispersion, passive_scalar,
                         ybj_plus, no_feedback, fixed_flow, no_wave_feedback,
                         N₀²_sg, N₁²_sg, σ_sg, z₀_sg, α_sg)
end
