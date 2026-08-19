"""
High-level Simulation API for QG-YBJ model.

Provides a simplified interface that hides MPI complexity from users:
- `Simulation` struct wraps all components (grid, state, plans, etc.)
- `QGYBJModel()` constructs a model from small, composable user objects
- `set!()` initializes model fields
- `Simulation(model; Δt, stop_time, output, diagnostics)` configures a run
- `initialize_simulation()` handles all MPI setup automatically
- `set_mean_flow!()`, `set_surface_waves!()` for common initial conditions
- `set_exponential_surface_waves!()` for exponential vertical decay
- `run!()` for time integration

# Example
```julia
using QGYBJplus

grid = RectilinearGrid(size=(256, 256, 128),
                       x=(-35e3, 35e3), y=(-35e3, 35e3), z=(-2e3, 0))

model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1.24e-4),
                   stratification=ConstantStratification(N²=1e-5))

# Set initial conditions
κ = sqrt(2) * π / 70e3
U = 0.335
dipole = (x, y, z) -> begin
    x_rot = (x - y) / sqrt(2)
    y_rot = (x + y) / sqrt(2)
    (U / κ) * sin(κ * x_rot) * cos(κ * y_rot)
end
set!(model; ψ=dipole, waves=SurfaceWave(amplitude=0.10, scale=30.0))

simulation = Simulation(model; Δt=20.0, stop_iteration=10_000,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(inertial_period(model))))
run!(simulation) # ETD-RK2

# Cleanup
finalize_simulation!(simulation)
```
"""

using MPI
using Printf

#=
================================================================================
                     DECLARATIVE USER INTERFACE
================================================================================
=#

"""Schedule output or diagnostics at a time interval measured in seconds."""
struct TimeInterval{T}
    interval::T
end

function TimeInterval(interval::Real)
    interval > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
    value = float(interval)
    return TimeInterval{typeof(value)}(value)
end

"""Schedule output or diagnostics every `interval` model iterations."""
struct IterationInterval
    interval::Int
end

function IterationInterval(interval::Integer)
    interval > 0 || throw(ArgumentError("interval must be positive (got $interval)"))
    return IterationInterval(Int(interval))
end

"""
    NetCDFOutput(; path="output", schedule=nothing, fields=(:ψ, :waves), velocities=false)

Declarative NetCDF output settings for [`Simulation`](@ref). `schedule` may be
a [`TimeInterval`](@ref) or [`IterationInterval`](@ref).
"""
struct NetCDFOutput{S}
    path::String
    schedule::S
    fields::Tuple
    velocities::Bool
end

function NetCDFOutput(; path::AbstractString="output", schedule=nothing,
    fields=(:ψ, :waves), velocities::Bool=false)

    schedule === nothing || schedule isa TimeInterval || schedule isa IterationInterval ||
        throw(ArgumentError("schedule must be a TimeInterval or IterationInterval"))
    isempty(path) && throw(ArgumentError("output path cannot be empty"))
    output_fields = fields isa Symbol ? (fields,) : Tuple(fields)
    return NetCDFOutput(String(path), schedule, output_fields, velocities)
end

mutable struct SimulationRunOptions{T}
    output_dir::String
    save_interval::Union{Nothing, T}
    diagnostics_interval::Int
    verbose::Bool
    save_psi::Bool
    save_waves::Bool
    save_velocities::Bool
    output
end

default_run_options(::Type{T}) where T = SimulationRunOptions{T}(
    "output", nothing, 10, true, true, true, false, nothing)

"""
    RectilinearGrid(; size, extent=nothing, x=nothing, y=nothing, z=nothing,
                      centered=false)

Describe a regular periodic-horizontal grid. `size` is `(nx, ny, nz)` and
`extent` is `(Lx, Ly, Lz)`. Alternatively, pass coordinate ranges such as
`x=(-35e3, 35e3)`, `y=(-35e3, 35e3)`, and `z=(-3e3, 0)`.
"""
struct RectilinearGridSpec{T}
    size::NTuple{3, Int}
    extent::NTuple{3, T}
    origin::NTuple{2, T}
end

function RectilinearGrid(; size::NTuple{3, Int},
    extent::Union{Nothing, Tuple{<:Real, <:Real, <:Real}}=nothing,
    x::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    y::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    z::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    centered::Bool=false)

    all(>(0), size) || throw(ArgumentError("all grid dimensions must be positive"))
    if extent === nothing
        x === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or x=(x₁, x₂)"))
        y === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or y=(y₁, y₂)"))
        z === nothing && throw(ArgumentError("provide extent=(Lx, Ly, Lz) or z=(z₁, z₂)"))
        extent = (x[2] - x[1], y[2] - y[1], abs(z[2] - z[1]))
    end

    extent_values = float.(extent)
    T = promote_type(map(typeof, extent_values)...)
    Lx, Ly, Lz = T.(extent_values)
    all(>(zero(T)), (Lx, Ly, Lz)) || throw(ArgumentError("all grid extents must be positive"))

    _check_extent(range, length, name) = begin
        range === nothing && return
        actual = T(abs(range[2] - range[1]))
        isapprox(actual, length; rtol=10eps(T), atol=10eps(T) * max(length, one(T))) ||
            throw(ArgumentError("$name range length $actual does not match extent $length"))
    end
    _check_extent(x, Lx, "x")
    _check_extent(y, Ly, "y")
    _check_extent(z, Lz, "z")

    if centered && (x !== nothing || y !== nothing)
        throw(ArgumentError("centered=true cannot be combined with explicit x or y ranges"))
    end
    x0 = x === nothing ? (centered ? -Lx / 2 : zero(T)) : T(x[1])
    y0 = y === nothing ? (centered ? -Ly / 2 : zero(T)) : T(y[1])
    return RectilinearGridSpec{T}(size, (Lx, Ly, Lz), (x0, y0))
end

"""Constant Coriolis parameter for an f-plane model."""
struct FPlane{T}
    f::T
end

FPlane(; f::Real) = FPlane(f)
FPlane(f::Real) = FPlane{typeof(float(f))}(float(f))

"""Constant buoyancy frequency squared `N²` in s⁻²."""
struct ConstantStratification{T}
    N²::T
end

ConstantStratification(; N²::Real) = ConstantStratification(N²)
function ConstantStratification(N²::Real)
    N² > 0 || throw(ArgumentError("N² must be positive (got $N²)"))
    return ConstantStratification{typeof(float(N²))}(float(N²))
end

"""Horizontal hyperdiffusion coefficients for the balanced flow and waves."""
struct HorizontalHyperdiffusivity{T}
    flow::T
    flow2::T
    flow_laplacian_order::Int
    flow_laplacian_order2::Int
    waves::T
    waves2::T
    wave_laplacian_order::Int
    wave_laplacian_order2::Int
end

function HorizontalHyperdiffusivity(; flow::Real=0.01, flow2::Real=10.0,
    flow_laplacian_order::Int=2, flow_laplacian_order2::Int=6,
    waves::Real=0.0, waves2::Real=10.0,
    wave_laplacian_order::Int=2, wave_laplacian_order2::Int=6)

    coefficients = (flow, flow2, waves, waves2)
    all(>=(0), coefficients) || throw(ArgumentError("hyperdiffusion coefficients must be non-negative"))
    orders = (flow_laplacian_order, flow_laplacian_order2,
              wave_laplacian_order, wave_laplacian_order2)
    all(>(0), orders) || throw(ArgumentError("Laplacian orders must be positive"))
    values = float.(coefficients)
    T = promote_type(map(typeof, values)...)
    return HorizontalHyperdiffusivity{T}(
        T(flow), T(flow2), flow_laplacian_order, flow_laplacian_order2,
        T(waves), T(waves2), wave_laplacian_order, wave_laplacian_order2)
end

"""Horizontally uniform, surface-confined wave initial condition."""
struct SurfaceWave{T}
    amplitude::T
    scale::T
    profile::Symbol
end

function SurfaceWave(; amplitude::Real, scale::Real, profile::Symbol=:gaussian)
    scale > 0 || throw(ArgumentError("wave scale must be positive (got $scale)"))
    profile in (:gaussian, :exponential) ||
        throw(ArgumentError("profile must be :gaussian or :exponential"))
    values = float.((amplitude, scale))
    T = promote_type(map(typeof, values)...)
    return SurfaceWave{T}(T(amplitude), T(scale), profile)
end

#=
================================================================================
                        SIMULATION STRUCT
================================================================================
=#

"""
    Simulation{T, G, S, P, M, W, R}

High-level container for all simulation components.

# Fields
- `grid`: Grid structure with MPI decomposition
- `state`: Model state (q, ψ, B, etc.)
- `params`: Model parameters
- `plans`: FFT plans for spectral transforms
- `mpi_config`: MPI configuration
- `workspace`: Pre-allocated workspace arrays
- `N2_profile`: Stratification profile N²(z) on unstaggered (face) levels
"""
mutable struct Simulation{T, G<:Grid, S<:State, P, M<:MPIConfig, W, R}
    grid::G
    state::S
    params::QGParams{T}
    plans::P
    mpi_config::M
    workspace::W
    N2_profile::Vector{T}
    run_options::R
end

const QGYBJModel = Simulation

# Convenience accessors
is_root(sim::Simulation) = sim.mpi_config.is_root
nprocs(sim::Simulation) = sim.mpi_config.nprocs

#=
================================================================================
                        INITIALIZATION
================================================================================
=#

"""
    initialize_simulation(; kwargs...) -> Simulation

Initialize a complete simulation with all MPI components set up automatically.

This is the main entry point for the high-level API. It handles:
- MPI initialization and environment setup
- Grid creation with domain decomposition
- FFT plan creation
- State allocation
- Workspace allocation
- Stratification profile computation

# Keyword Arguments

## Grid parameters (required)
- `nx`, `ny`, `nz`: Grid resolution
- `Lx`, `Ly`, `Lz`: Domain size [m]

## Physical parameters
- `f₀`: Coriolis parameter [s⁻¹] (default: 1e-4)
- `N²`: Buoyancy frequency squared [s⁻²] (default: 1e-5)

## Time stepping
- `dt`: Time step [s] (default: 1.0)
- `nt`: Number of time steps (default: 1000)

## Model options
- `ybj_plus`: Enable YBJ⁺ wave-wave interactions (default: true)
- `fixed_flow`: Use fixed background flow (default: false)
- `no_wave_feedback`: Disable wave feedback on flow (default: false)

## Diffusion
- `νₕ₁ʷ`: Horizontal hyperdiffusion for waves [m⁴/s] (default: 0)
- `ilap1w`: Hyperdiffusion order (default: 2 for ∇⁴)

## MPI options
- `topology`: Process grid (px, py), auto-computed if not specified
- `parallel_io`: Enable parallel I/O (default: false)

# Returns
A `Simulation` object ready for initial conditions and time integration.

# Example
```julia
sim = initialize_simulation(
    nx=256, ny=256, nz=128,
    Lx=70e3, Ly=70e3, Lz=2000.0,
    f₀=1.24e-4, N²=1e-5,
    dt=20.0, nt=10000,
    ybj_plus=true, fixed_flow=true
)
```
"""
function initialize_simulation(;
    # Grid parameters
    nx::Int, ny::Int, nz::Int,
    Lx::Real, Ly::Real, Lz::Real,
    centered::Bool = false,  # Center domain at origin: x,y ∈ [-Lx/2, Lx/2)
    x0::Union{Real, Nothing} = nothing,
    y0::Union{Real, Nothing} = nothing,
    # Physical parameters
    f₀::Real = 1e-4,
    N²::Real = 1e-5,
    stratification_profile = nothing,
    # Time stepping
    dt::Real = 1.0,
    nt::Int = 1000,
    # Model options
    ybj_plus::Bool = true,
    fixed_flow::Bool = false,
    no_feedback::Union{Bool, Nothing} = nothing,
    no_wave_feedback::Bool = false,
    # Diffusion
    νₕ₁::Real = 0.01,
    νₕ₂::Real = 10.0,
    ilap1::Int = 2,
    ilap2::Int = 6,
    νₕ₁ʷ::Real = 0.0,
    νₕ₂ʷ::Real = 10.0,
    ilap1w::Int = 2,
    ilap2w::Int = 6,
    # MPI options
    topology = nothing,
    parallel_io::Bool = false,
    # Output verbosity
    verbose::Bool = true)

    T = Float64

    # Initialize MPI
    if !MPI.Initialized()
        MPI.Init()
    end
    mpi_config = setup_mpi_environment(; topology=topology, parallel_io=parallel_io)

    if mpi_config.is_root && verbose
        println("="^70)
        println("QGYBJplus Simulation Initialization")
        println("="^70)
        println("MPI processes: $(mpi_config.nprocs), Topology: $(mpi_config.topology)")
        @printf("Resolution: %d × %d × %d\n", nx, ny, nz)
        @printf("Domain: %.1f km × %.1f km × %.1f m\n", Lx/1e3, Ly/1e3, Lz)
    end

    # Create parameters
    params = default_params(
        nx = nx, ny = ny, nz = nz,
        Lx = T(Lx), Ly = T(Ly), Lz = T(Lz),
        centered = centered,  # Center domain at origin if true
        x0 = x0,
        y0 = y0,
        dt = T(dt), nt = nt,
        f₀ = T(f₀), N² = T(N²),
        ybj_plus = ybj_plus,
        fixed_flow = fixed_flow,
        # Preserve the original simplified API: its wave-feedback switch also
        # controls the master coupling switch unless explicitly overridden.
        no_feedback = no_feedback === nothing ? no_wave_feedback : no_feedback,
        no_wave_feedback = no_wave_feedback,
        νₕ₁ = T(νₕ₁),
        νₕ₂ = T(νₕ₂),
        ilap1 = ilap1,
        ilap2 = ilap2,
        νₕ₁ʷ = T(νₕ₁ʷ),
        νₕ₂ʷ = T(νₕ₂ʷ),
        ilap1w = ilap1w,
        ilap2w = ilap2w
    )

    # Initialize grid, plans, state, workspace
    grid = init_mpi_grid(params, mpi_config)
    plans = plan_mpi_transforms(grid, mpi_config)
    state = init_mpi_state(grid, plans, mpi_config)
    workspace = init_mpi_workspace(grid, mpi_config)

    # Compute stratification profile
    profile = stratification_profile === nothing ? ConstantN{T}(sqrt(T(N²))) : stratification_profile
    N2_profile = T.(compute_stratification_profile(profile, grid))
    params.N² = sum(N2_profile) / length(N2_profile)

    if mpi_config.is_root && verbose
        println("Initialization complete.")
        println("="^70)
    end

    MPI.Barrier(mpi_config.comm)

    run_options = default_run_options(T)
    return Simulation{T, typeof(grid), typeof(state), typeof(plans),
                      typeof(mpi_config), typeof(workspace), typeof(run_options)}(
        grid, state, params, plans, mpi_config, workspace, N2_profile, run_options
    )
end

_coriolis_frequency(f::Real) = f
_coriolis_frequency(coriolis::FPlane) = coriolis.f
_stratification_N²(N²::Real) = N²
_stratification_N²(stratification::ConstantStratification) = stratification.N²
_stratification_N²(stratification::StratificationProfile) = evaluate_N2(stratification, 0.0)
_stratification_profile(::Real) = nothing
_stratification_profile(::ConstantStratification) = nothing
_stratification_profile(stratification::StratificationProfile) = stratification

function _feedback_flags(feedback)
    if feedback === false || feedback in (:none, :off)
        return true, true
    elseif feedback === true || feedback in (:wave_mean, :on)
        return false, false
    elseif feedback == :no_wave_feedback
        return false, true
    end
    throw(ArgumentError("feedback must be :none, :wave_mean, or :no_wave_feedback"))
end

function _fixed_flow(flow)
    (flow === true || flow == :fixed) && return true
    (flow === false || flow == :evolving) && return false
    throw(ArgumentError("flow must be :fixed or :evolving"))
end

"""
    QGYBJModel(; grid, coriolis=FPlane(f=1e-4),
                 stratification=ConstantStratification(N²=1e-5), kwargs...)

Construct a complete QG-YBJ+ model while hiding MPI decomposition, FFT plans,
workspaces, and parameter bookkeeping. Time integration is always ETD-RK2.
"""
function QGYBJModel(; grid::RectilinearGridSpec,
    coriolis=FPlane(f=1e-4),
    stratification=ConstantStratification(N²=1e-5),
    closure::HorizontalHyperdiffusivity=HorizontalHyperdiffusivity(),
    flow=:evolving,
    feedback=:none,
    ybj_plus::Bool=true,
    Δt::Real=1.0,
    stop_iteration::Int=1000,
    topology=nothing,
    parallel_io::Bool=false,
    verbose::Bool=true)

    no_feedback, no_wave_feedback = _feedback_flags(feedback)
    nx, ny, nz = grid.size
    Lx, Ly, Lz = grid.extent
    x0, y0 = grid.origin
    return initialize_simulation(
        nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, x0=x0, y0=y0,
        f₀=_coriolis_frequency(coriolis),
        N²=_stratification_N²(stratification),
        stratification_profile=_stratification_profile(stratification),
        dt=Δt, nt=stop_iteration,
        ybj_plus=ybj_plus, fixed_flow=_fixed_flow(flow),
        no_feedback=no_feedback, no_wave_feedback=no_wave_feedback,
        νₕ₁=closure.flow, νₕ₂=closure.flow2,
        ilap1=closure.flow_laplacian_order, ilap2=closure.flow_laplacian_order2,
        νₕ₁ʷ=closure.waves, νₕ₂ʷ=closure.waves2,
        ilap1w=closure.wave_laplacian_order, ilap2w=closure.wave_laplacian_order2,
        topology=topology, parallel_io=parallel_io, verbose=verbose)
end

function Base.show(io::IO, grid::RectilinearGridSpec)
    print(io, "RectilinearGrid(size=$(grid.size), extent=$(grid.extent))")
end

#=
================================================================================
                        INITIAL CONDITIONS
================================================================================
=#

"""
    set_mean_flow!(sim::Simulation; psi_func, method=:function, amplitude=1.0,
                   spectral_slope=-3.0, seed=0)

Set up the balanced mean flow from an analytical streamfunction or random noise.

For `method=:function`, `psi_func(x, y, z)` should return ψ in m²/s at the
cell centers. Coordinates respect the grid origin (`x0`, `y0`) and use `G.z`
for vertical levels. This works in MPI because each rank fills only its local
slab before the FFT.

For `method=:random`, a deterministic MPI-safe random spectrum is generated.

# Arguments
- `sim`: Simulation object
- `psi_func`: Function returning ψ(x, y, z) when `method=:function`
- `method`: `:function` or `:random` (alias `:analytical` for `:function`)
- `amplitude`: Random-field amplitude (used for `:random`)
- `spectral_slope`: Spectral slope for random field (default: -3)
- `seed`: Random seed (default: 0)

# Example
```julia
κ = sqrt(2) * π / Lx
U = 0.335
dipole = (x, y, z) -> begin
    x_rot = (x - y) / sqrt(2)
    y_rot = (x + y) / sqrt(2)
    (U / κ) * sin(κ * x_rot) * cos(κ * y_rot)
end
set_mean_flow!(sim; psi_func=dipole)
set_mean_flow!(sim; method=:random, amplitude=0.1, spectral_slope=-3.0, seed=42)
```
"""
function set_mean_flow!(sim::Simulation;
    psi_func = nothing,
    method::Symbol = :function,
    pv_method::Symbol = :qg,
    amplitude::Real = 1.0,
    spectral_slope::Real = -3.0,
    seed::Int = 0,
    verbose::Bool = true)

    G = sim.grid
    S = sim.state
    plans = sim.plans

    if method == :function || method == :analytical
        psi_func === nothing && throw(ArgumentError("psi_func must be provided when method=:function"))

        if sim.mpi_config.is_root && verbose
            println("Setting mean flow from analytical ψ(x, y, z)")
        end

        local_range = get_local_range_physical(plans)
        psi_phys = allocate_fft_backward_dst(S.psi, plans)
        psi_arr = parent(psi_phys)
        T = eltype(psi_arr)

        for k_local in axes(psi_arr, 1)
            k_global = local_range[1][k_local]
            z = G.z[k_global]
            for j_local in axes(psi_arr, 3)
                j_global = local_range[3][j_local]
                y = G.y0 + (j_global - 1) * G.dy
                for i_local in axes(psi_arr, 2)
                    i_global = local_range[2][i_local]
                    x = G.x0 + (i_global - 1) * G.dx
                    psi_arr[k_local, i_local, j_local] = T(psi_func(x, y, z))
                end
            end
        end

        fft_forward!(S.psi, psi_phys, plans)
    elseif method == :random
        if sim.mpi_config.is_root && verbose
            println("Setting random mean flow: amplitude = $(amplitude), slope = $(spectral_slope), seed = $(seed)")
        end
        init_mpi_random_psi!(S.psi, G, amplitude; slope=spectral_slope, seed=seed, seed_offset=0)
    else
        throw(ArgumentError("Unknown method=$method. Use :function or :random."))
    end

    if pv_method in (:qg, :balanced)
        add_balanced_component!(S, G, sim.params, sim.plans; N2_profile=sim.N2_profile)
    elseif pv_method in (:barotropic, :asselin)
        compute_barotropic_q_from_psi!(S.q, S.psi, G)
    elseif pv_method != :none
        throw(ArgumentError("pv_method must be :qg, :barotropic, or :none"))
    end

    return sim
end

"""
    set_surface_waves!(sim::Simulation; amplitude, surface_depth, uniform=true, profile=:gaussian)

Set up surface-confined near-inertial waves.

The wave initial condition follows Asselin et al. (2020):
    u(t=0) = u₀ exp(-d²/s²), v(t=0) = 0

where d = -z is depth below the surface, u₀ is the wave velocity amplitude,
and s is the surface layer depth.

# Arguments
- `sim`: Simulation object
- `amplitude`: Wave velocity amplitude u₀ [m/s]
- `surface_depth`: Surface layer depth s [m] (used as e-folding depth for :exponential)
- `uniform`: Horizontally uniform waves (default: true)
- `profile`: Vertical decay profile (:gaussian or :exponential, default: :gaussian)

# Example
```julia
set_surface_waves!(sim; amplitude=0.10, surface_depth=30.0)  # u₀ = 10 cm/s
set_surface_waves!(sim; amplitude=0.10, surface_depth=50.0, profile=:exponential)
```
"""
function set_surface_waves!(sim::Simulation;
    amplitude::Real,
    surface_depth::Real,
    uniform::Bool = true,
    profile::Symbol = :gaussian,
    verbose::Bool = true)

    G = sim.grid
    S = sim.state
    plans = sim.plans

    if sim.mpi_config.is_root && verbose
        println("Setting surface waves: u₀ = $(amplitude) m/s, s = $(surface_depth) m, profile=$(profile)")
    end
    surface_depth > 0 || throw(ArgumentError("surface_depth must be positive (got $surface_depth)"))

    # Get local ranges
    local_range = get_local_range_physical(plans)

    # Allocate physical-space array
    B_phys = allocate_fft_backward_dst(S.B, plans)
    B_arr = parent(B_phys)
    T = typeof(real(zero(eltype(B_arr))))

    dz = G.Lz / G.nz
    for k_local in axes(B_arr, 1)
        k_global = local_range[1][k_local]
        # Depth from surface (z=0 is surface, z=-Lz is bottom).
        # Use a dz/2 shift so the top cell center corresponds to z=0.
        depth = max(zero(T), -G.z[k_global] - dz / 2)
        wave_profile = if profile == :gaussian
            exp(-(depth^2) / (surface_depth^2))
        elseif profile == :exponential
            exp(-depth / surface_depth)
        else
            throw(ArgumentError("Unknown profile=$profile. Use :gaussian or :exponential."))
        end

        if uniform
            # Horizontally uniform waves
            B_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        else
            # Could add horizontal structure here
            B_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        end
    end

    # Transform to spectral space
    fft_forward!(S.B, B_phys, plans)

    return sim
end

"""
    set_exponential_surface_waves!(sim::Simulation; amplitude, efold_depth, uniform=true)

Convenience wrapper for exponentially decaying, horizontally uniform surface waves.
Uses `profile=:exponential` in `set_surface_waves!`.
"""
function set_exponential_surface_waves!(sim::Simulation;
    amplitude::Real,
    efold_depth::Real,
    uniform::Bool = true,
    verbose::Bool = true)
    return set_surface_waves!(sim;
        amplitude=amplitude,
        surface_depth=efold_depth,
        uniform=uniform,
        profile=:exponential,
        verbose=verbose)
end

_first_notnothing(values...) = begin
    for value in values
        value !== nothing && return value
    end
    return nothing
end

"""
    set!(model; ψ=nothing, waves=nothing, pv_method=:qg)

Set model initial conditions. The mean flow is a streamfunction function
`ψ(x, y, z)`; `waves` may be a [`SurfaceWave`](@ref).
"""
function set!(model::Simulation;
    ψ=nothing, psi=nothing, mean_flow=nothing,
    pv_method::Symbol=:qg,
    waves=nothing, B=nothing,
    verbose::Bool=false)

    flow = _first_notnothing(mean_flow, ψ, psi)
    wave = _first_notnothing(waves, B)

    if flow !== nothing
        flow isa Function || throw(ArgumentError("mean_flow/ψ must be a function of (x, y, z)"))
        set_mean_flow!(model; psi_func=flow, pv_method=pv_method, verbose=verbose)
    end
    if wave !== nothing
        wave isa SurfaceWave || throw(ArgumentError("waves/B must be a SurfaceWave"))
        set_surface_waves!(model; amplitude=wave.amplitude, surface_depth=wave.scale,
                           profile=wave.profile, verbose=verbose)
    end
    return model
end

"""
    set_wave_packet!(sim::Simulation; amplitude, kx, ky, sigma_k, z_center=nothing, z_width=nothing)

Set up a localized wave packet in wavenumber space.

# Arguments
- `sim`: Simulation object
- `amplitude`: Wave amplitude
- `kx`, `ky`: Central wavenumbers
- `sigma_k`: Wavenumber spread
- `z_center`: Vertical center depth below surface (default: Lz/2)
- `z_width`: Vertical width in depth units (default: Lz/4)
"""
function set_wave_packet!(sim::Simulation;
    amplitude::Real,
    kx::Int,
    ky::Int,
    sigma_k::Real,
    z_center::Union{Real, Nothing} = nothing,
    z_width::Union{Real, Nothing} = nothing)

    G = sim.grid
    S = sim.state

    z_c = z_center === nothing ? G.Lz / 2 : z_center
    z_w = z_width === nothing ? G.Lz / 4 : z_width

    0 <= z_c <= G.Lz || throw(ArgumentError("z_center must lie between 0 and Lz=$(G.Lz)"))
    z_w > 0 || throw(ArgumentError("z_width must be positive"))

    if sim.mpi_config.is_root
        println("Setting wave packet: kx=$kx, ky=$ky, σ_k=$sigma_k")
    end

    # Use the existing create_wave_packet function
    packet = create_wave_packet(G, kx, ky, sigma_k, amplitude;
                                z_center=z_c, z_width=z_w)

    # Copy to state (handling MPI distribution)
    S.B .= scatter_from_root(packet, G, sim.mpi_config; plans=sim.plans)

    return sim
end

#=
================================================================================
                        RUNNING SIMULATIONS
================================================================================
=#

function _configure_time_stepping!(sim::Simulation;
    Δt=nothing, stop_time=nothing, stop_iteration=nothing)

    if Δt !== nothing
        Δt > 0 || throw(ArgumentError("Δt must be positive (got $Δt)"))
        sim.params.dt = typeof(sim.params.dt)(Δt)
    end
    if stop_iteration !== nothing
        stop_iteration > 0 || throw(ArgumentError("stop_iteration must be positive"))
        sim.params.nt = Int(stop_iteration)
    elseif stop_time !== nothing
        stop_time > 0 || throw(ArgumentError("stop_time must be positive"))
        sim.params.nt = max(1, round(Int, stop_time / sim.params.dt))
    end
    return sim
end

_saves_psi(fields) = any(field -> field in (:ψ, :psi, :q, :flow), fields)
_saves_waves(fields) = any(field -> field in (:waves, :wave, :B, :A), fields)

function _schedule_in_seconds(schedule, dt)
    schedule === nothing && return nothing
    schedule isa TimeInterval && return schedule.interval
    schedule isa IterationInterval && return schedule.interval * dt
    throw(ArgumentError("schedule must be a TimeInterval or IterationInterval"))
end

function _configure_output!(sim::Simulation; output=nothing, diagnostics=nothing,
    verbose=nothing)

    options = sim.run_options
    verbose !== nothing && (options.verbose = verbose)

    if output === false
        options.output = false
    elseif output !== nothing
        output isa NetCDFOutput || throw(ArgumentError("output must be a NetCDFOutput or false"))
        options.output = output
        options.output_dir = output.path
        interval = _schedule_in_seconds(output.schedule, sim.params.dt)
        options.save_interval = interval === nothing ? options.save_interval : interval
        options.save_psi = _saves_psi(output.fields)
        options.save_waves = _saves_waves(output.fields)
        options.save_velocities = output.velocities || :velocities in output.fields
    end

    if diagnostics isa IterationInterval
        options.diagnostics_interval = diagnostics.interval
    elseif diagnostics isa TimeInterval
        options.diagnostics_interval = max(1, round(Int, diagnostics.interval / sim.params.dt))
    elseif diagnostics isa Integer
        diagnostics > 0 || throw(ArgumentError("diagnostics interval must be positive"))
        options.diagnostics_interval = Int(diagnostics)
    elseif diagnostics !== nothing
        throw(ArgumentError("diagnostics must be an IterationInterval, TimeInterval, or integer"))
    end
    return sim
end

"""
    Simulation(model::QGYBJModel; Δt=nothing, stop_time=nothing,
               stop_iteration=nothing, output=nothing, diagnostics=nothing)

Configure the model clock, output, and diagnostics in an Oceananigans-style
workflow. The returned object shares the model state and uses ETD-RK2.
"""
function Simulation(model::Simulation; Δt=nothing, stop_time=nothing,
    stop_iteration=nothing, output=nothing, diagnostics=nothing, verbose=nothing)

    _configure_time_stepping!(model; Δt=Δt, stop_time=stop_time,
                              stop_iteration=stop_iteration)
    return _configure_output!(model; output=output, diagnostics=diagnostics,
                              verbose=verbose)
end

"""
    run!(sim::Simulation; kwargs...)

Run the simulation with specified options.

This wraps `run_simulation!` with a simpler interface.

# Keyword Arguments
- `output_dir`: Output directory (default: "output")
- `save_interval`: Save interval in simulation time units
- `diagnostics_interval`: Diagnostics interval in time steps (default: 10)
- `verbose`: Print progress (default: true on root)

# Example
```julia
run!(sim; output_dir="output")
```
"""
function run!(sim::Simulation;
    output_dir::Union{String, Nothing}=nothing,
    Δt=nothing,
    stop_time=nothing,
    stop_iteration=nothing,
    output=nothing,
    diagnostics=nothing,
    save_interval::Union{Real, Nothing}=nothing,
    diagnostics_interval::Union{Int, Nothing}=nothing,
    verbose::Union{Bool, Nothing}=nothing,
    progress::Union{Bool, Nothing}=nothing,
    save_psi::Union{Bool, Nothing}=nothing,
    save_waves::Union{Bool, Nothing}=nothing,
    save_velocities::Union{Bool, Nothing}=nothing)

    _configure_time_stepping!(sim; Δt=Δt, stop_time=stop_time,
                              stop_iteration=stop_iteration)
    _configure_output!(sim; output=output, diagnostics=diagnostics, verbose=verbose)
    options = sim.run_options
    progress !== nothing && (options.verbose = progress)
    output_dir !== nothing && (options.output_dir = output_dir)
    save_interval !== nothing && (options.save_interval = typeof(sim.params.dt)(save_interval))
    diagnostics_interval !== nothing && (options.diagnostics_interval = diagnostics_interval)
    save_psi !== nothing && (options.save_psi = save_psi)
    save_waves !== nothing && (options.save_waves = save_waves)
    save_velocities !== nothing && (options.save_velocities = save_velocities)

    options.diagnostics_interval > 0 ||
        throw(ArgumentError("diagnostics_interval must be positive"))

    G = sim.grid
    S = sim.state
    params = sim.params
    plans = sim.plans
    mpi_config = sim.mpi_config
    workspace = sim.workspace
    N2_profile = sim.N2_profile

    write_output = options.output !== false
    if write_output && mpi_config.is_root
        mkpath(options.output_dir)
    end
    MPI.Barrier(mpi_config.comm)

    # Compute default save interval (1 inertial period)
    T_inertial = 2π / params.f₀
    interval = options.save_interval === nothing ? T_inertial : options.save_interval
    interval > 0 || throw(ArgumentError("save_interval must be positive"))

    output_config = write_output ? OutputConfig(
        output_dir = options.output_dir,
        state_file_pattern = "state%04d.nc",
        psi_interval = interval,
        wave_interval = interval,
        diagnostics_interval = interval,
        save_psi = options.save_psi,
        save_waves = options.save_waves,
        save_velocities = options.save_velocities,
        save_vorticity = false,
        save_diagnostics = false
    ) : nothing

    # Run simulation
    run_simulation!(S, G, params, plans;
        output_config = output_config,
        mpi_config = mpi_config,
        workspace = workspace,
        N2_profile = N2_profile,
        print_progress = mpi_config.is_root && options.verbose,
        diagnostics_interval = options.diagnostics_interval
    )

    if mpi_config.is_root && options.verbose
        if write_output
            println("\nSimulation complete. Output saved to: $(options.output_dir)/")
        else
            println("\nSimulation complete.")
        end
    end

    return sim
end

#=
================================================================================
                        CLEANUP
================================================================================
=#

"""
    finalize_simulation!(sim::Simulation)

Clean up simulation resources and finalize MPI.

Call this at the end of your script to ensure proper cleanup.

# Example
```julia
finalize_simulation!(sim)
```
"""
function finalize_simulation!(sim::Simulation)
    MPI.Barrier(sim.mpi_config.comm)
    GC.gc(true)  # Force garbage collection before MPI finalization
    MPI.Finalize()
end

#=
================================================================================
                        UTILITY FUNCTIONS
================================================================================
=#

"""
    get_time(sim::Simulation, step::Int)

Get simulation time at a given step.
"""
get_time(sim::Simulation, step::Int) = step * sim.params.dt

"""
    get_inertial_period(sim::Simulation)

Get the inertial period T = 2π/f₀.
"""
get_inertial_period(sim::Simulation) = 2π / sim.params.f₀
inertial_period(sim::Simulation) = get_inertial_period(sim)

"""
    get_duration(sim::Simulation)

Get total simulation duration in seconds.
"""
get_duration(sim::Simulation) = sim.params.nt * sim.params.dt

"""
    get_duration_ip(sim::Simulation)

Get total simulation duration in inertial periods.
"""
get_duration_ip(sim::Simulation) = get_duration(sim) / get_inertial_period(sim)

"""
    summary(sim::Simulation)

Print a summary of the simulation configuration.
"""
function Base.summary(io::IO, sim::Simulation)
    if !sim.mpi_config.is_root
        return
    end

    G = sim.grid
    P = sim.params

    println(io, "QGYBJplus Simulation")
    println(io, "="^40)
    @printf(io, "Resolution: %d × %d × %d\n", G.nx, G.ny, G.nz)
    @printf(io, "Domain: %.1f km × %.1f km × %.1f m\n", G.Lx/1e3, G.Ly/1e3, G.Lz)
    @printf(io, "Coriolis: f₀ = %.2e s⁻¹\n", P.f₀)
    @printf(io, "Stratification: N² = %.2e s⁻²\n", P.N²)
    @printf(io, "Time step: dt = %.2f s, nt = %d\n", P.dt, P.nt)
    @printf(io, "Duration: %.1f inertial periods\n", get_duration_ip(sim))
    println(io, "MPI processes: $(sim.mpi_config.nprocs)")
    println(io, "="^40)
end

function Base.show(io::IO, sim::Simulation)
    print(io, "Simulation($(sim.grid.nx)×$(sim.grid.ny)×$(sim.grid.nz), ",
          "$(sim.mpi_config.nprocs) procs)")
end
