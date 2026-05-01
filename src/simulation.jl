"""
High-level Simulation API for QG-YBJ model.

Provides a simplified interface that hides MPI complexity from users:
- `Simulation` struct wraps all components (grid, state, plans, etc.)
- `QGYBJModel()` constructs a complete model in an Oceananigans-like style
- `set!()` initializes model fields
- `Simulation(model; Δt, stop_time, output, diagnostics)` configures the run
- `initialize_simulation()` handles all MPI setup automatically
- `set_mean_flow!()`, `set_surface_waves!()` for common initial conditions
- `set_exponential_surface_waves!()` for exponential vertical decay
- `run!()` for time integration

# Example
```julia
using QGYBJplus

grid = RectilinearGrid(size=(256, 256, 128),
                       x=(-35e3, 35e3),
                       y=(-35e3, 35e3),
                       z=(-3e3, 0))

model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1.24e-4),
                   stratification=ConstantStratification(N²=1e-5),
                   flow=:fixed,
                   feedback=:none)

κ = sqrt(2) * π / 70e3
U = 0.335
dipole = (x, y, z) -> begin
    x_rot = (x - y) / sqrt(2)
    y_rot = (x + y) / sqrt(2)
    (U / κ) * sin(κ * x_rot) * cos(κ * y_rot)
end

set!(model; ψ=dipole, pv_method=:barotropic,
     waves=SurfaceWave(amplitude=0.10, scale=30.0))

simulation = Simulation(model;
                        Δt=2.0,
                        stop_time=15 * inertial_period(model),
                        timestepper=:leapfrog,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(inertial_period(model)),
                                            fields=(:ψ, :waves)))

# Run simulation
run!(simulation)

# Cleanup
finalize_simulation!(simulation)
```
"""

using MPI
using Printf

#=
================================================================================
                        SIMULATION STRUCT
================================================================================
=#

"""
    Simulation{T, G, S, P, M, W}

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
mutable struct Simulation{T, G<:Grid, S<:State, P, M<:MPIConfig, W}
    grid::G
    state::S
    params::QGParams{T}
    plans::P
    mpi_config::M
    workspace::W
    N2_profile::Vector{T}
    run_options::Any
end

# Convenience accessors
is_root(sim::Simulation) = sim.mpi_config.is_root
nprocs(sim::Simulation) = sim.mpi_config.nprocs

#=
================================================================================
                        RUN CONFIGURATION
================================================================================
=#

"""
    TimeInterval(interval)

Output or callback schedule measured in model seconds.
"""
struct TimeInterval{T}
    interval::T
end

TimeInterval(interval::Real) = TimeInterval{typeof(float(interval))}(float(interval))

"""
    IterationInterval(interval)

Output or callback schedule measured in model iterations.
"""
struct IterationInterval
    interval::Int
end

IterationInterval(interval::Integer) = IterationInterval(Int(interval))

"""
    NetCDFOutput(; path="output", schedule=nothing, fields=(:ψ, :waves),
                   velocities=false)

Declarative NetCDF output configuration for `Simulation`.
"""
struct NetCDFOutput{S}
    path::String
    schedule::S
    fields::Tuple
    velocities::Bool
end

function NetCDFOutput(; path::AbstractString = "output",
    schedule = nothing,
    fields = (:ψ, :waves),
    velocities::Bool = false)

    return NetCDFOutput(String(path), schedule, Tuple(fields), velocities)
end

mutable struct SimulationRunOptions{T}
    timestepper::Symbol
    output_dir::String
    save_interval::Union{Nothing, T}
    diagnostics_interval::Int
    verbose::Bool
    save_psi::Bool
    save_waves::Bool
    save_velocities::Bool
end

function default_run_options(::Type{T}) where T
    return SimulationRunOptions{T}(
        :imex_cn,
        "output",
        nothing,
        10,
        true,
        true,
        true,
        false,
    )
end

#=
================================================================================
                        OCEANANIGANS-STYLE USER TYPES
================================================================================
=#

"""
    RectilinearGrid(; size, extent=nothing, x=nothing, y=nothing, z=nothing)

Describe a regular periodic-horizontal grid in the style of Oceananigans.

`size` is `(nx, ny, nz)`. `extent` is `(Lx, Ly, Lz)` in meters. Instead of
`extent`, users may pass coordinate ranges `x=(x₁, x₂)`, `y=(y₁, y₂)`, and
`z=(-Lz, 0)`. The vertical coordinate is always cell-centered with `z=0` at the
surface and `z=-Lz` at the bottom.
"""
struct RectilinearGridSpec{T}
    size::NTuple{3, Int}
    extent::NTuple{3, T}
    origin::NTuple{2, T}
end

function RectilinearGrid(; size::NTuple{3, Int},
    extent::Union{Nothing, NTuple{3, <:Real}} = nothing,
    x::Union{Nothing, NTuple{2, <:Real}} = nothing,
    y::Union{Nothing, NTuple{2, <:Real}} = nothing,
    z::Union{Nothing, NTuple{2, <:Real}} = nothing)

    if extent === nothing
        x === nothing && throw(ArgumentError("Provide either extent=(Lx, Ly, Lz) or x=(x₁, x₂)."))
        y === nothing && throw(ArgumentError("Provide either extent=(Lx, Ly, Lz) or y=(y₁, y₂)."))
        z === nothing && throw(ArgumentError("Provide either extent=(Lx, Ly, Lz) or z=(-Lz, 0)."))
        extent = (x[2] - x[1], y[2] - y[1], abs(z[2] - z[1]))
    end

    T = promote_type(map(typeof, extent)...)
    Lx, Ly, Lz = T.(extent)

    if x !== nothing && !isapprox(T(x[2] - x[1]), Lx; rtol=0, atol=10eps(T(max(abs(Lx), one(T)))))
        throw(ArgumentError("x range length must match Lx=$Lx."))
    end

    if y !== nothing && !isapprox(T(y[2] - y[1]), Ly; rtol=0, atol=10eps(T(max(abs(Ly), one(T)))))
        throw(ArgumentError("y range length must match Ly=$Ly."))
    end

    if z !== nothing && !isapprox(T(abs(z[2] - z[1])), Lz; rtol=0, atol=10eps(T(max(abs(Lz), one(T)))))
        throw(ArgumentError("z range length must match Lz=$Lz."))
    end

    x0 = x === nothing ? zero(T) : T(x[1])
    y0 = y === nothing ? zero(T) : T(y[1])

    return RectilinearGridSpec{T}(size, (Lx, Ly, Lz), (x0, y0))
end

"""
    FPlane(; f)

Constant Coriolis parameter container. This mirrors Oceananigans' `FPlane`
spelling while keeping the model equations dimensional.
"""
struct FPlane{T}
    f::T
end

FPlane(; f::Real) = FPlane(f)
FPlane(f::Real) = FPlane{typeof(float(f))}(float(f))

"""
    ConstantStratification(; N²)

Constant buoyancy frequency squared `N²` in dimensional units `[s⁻²]`.
"""
struct ConstantStratification{T}
    N²::T
end

ConstantStratification(; N²::Real) = ConstantStratification(N²)
ConstantStratification(N²::Real) = ConstantStratification{typeof(float(N²))}(float(N²))

"""
    HorizontalHyperdiffusivity(; flow=0.01, flow2=10.0, waves=0.0, waves2=10.0,
                               flow_laplacian_order=2, flow_laplacian_order2=6,
                               wave_laplacian_order=2, wave_laplacian_order2=6)

Horizontal flow and wave hyperdiffusion coefficients.

The implementation uses the same dimensional separable spectral operator as the
QG-YBJp Fortran code.
"""
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

function HorizontalHyperdiffusivity(;
    flow::Real = 0.01,
    flow2::Real = 10.0,
    flow_laplacian_order::Int = 2,
    flow_laplacian_order2::Int = 6,
    waves::Real = 0.0,
    waves2::Real = 10.0,
    wave_laplacian_order::Int = 2,
    wave_laplacian_order2::Int = 6)

    T = promote_type(typeof(float(flow)), typeof(float(flow2)),
                     typeof(float(waves)), typeof(float(waves2)))

    return HorizontalHyperdiffusivity{T}(
        T(flow), T(flow2), flow_laplacian_order, flow_laplacian_order2,
        T(waves), T(waves2), wave_laplacian_order, wave_laplacian_order2
    )
end

"""
    SurfaceWave(; amplitude, scale, profile=:gaussian)

Horizontally uniform surface-confined near-inertial wave initial condition.

`amplitude` is the dimensional velocity amplitude `[m s⁻¹]`; `scale` is the
vertical scale in meters.
"""
struct SurfaceWave{T}
    amplitude::T
    scale::T
    profile::Symbol
end

function SurfaceWave(; amplitude::Real, scale::Real, profile::Symbol = :gaussian)
    T = promote_type(typeof(float(amplitude)), typeof(float(scale)))
    return SurfaceWave{T}(T(amplitude), T(scale), profile)
end

const QGYBJModel = Simulation

_coriolis_frequency(f::Real) = f
_coriolis_frequency(coriolis::FPlane) = coriolis.f

_stratification_N²(N²::Real) = N²
_stratification_N²(stratification::ConstantStratification) = stratification.N²

function _feedback_flags(feedback)
    if feedback === false || feedback == :none || feedback == :off
        return true, true
    elseif feedback === true || feedback == :wave_mean || feedback == :on
        return false, false
    elseif feedback == :no_wave_feedback
        return false, true
    else
        throw(ArgumentError("Unknown feedback=$feedback. Use :none, :wave_mean, or :no_wave_feedback."))
    end
end

_fixed_flow(flow) = flow === true || flow == :fixed

"""
    QGYBJModel(; grid, coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
                 closure=HorizontalHyperdiffusivity(), flow=:evolving, feedback=:none,
                 ybj_plus=true, asselin_filter=1e-3, Δt=1.0, stop_iteration=1000)

Construct a complete QG-YBJ+ model while hiding MPI grids, FFT plans, workspaces,
and stratification bookkeeping.
"""
function QGYBJModel(; grid::RectilinearGridSpec,
    coriolis = FPlane(f = 1e-4),
    stratification = ConstantStratification(N² = 1e-5),
    closure::HorizontalHyperdiffusivity = HorizontalHyperdiffusivity(),
    flow = :evolving,
    feedback = :none,
    ybj_plus::Bool = true,
    asselin_filter::Real = 1e-3,
    Δt::Real = 1.0,
    stop_iteration::Int = 1000,
    topology = nothing,
    parallel_io::Bool = false,
    verbose::Bool = true)

    no_feedback, no_wave_feedback = _feedback_flags(feedback)
    nx, ny, nz = grid.size
    Lx, Ly, Lz = grid.extent
    x0, y0 = grid.origin

    return initialize_simulation(
        nx = nx, ny = ny, nz = nz,
        Lx = Lx, Ly = Ly, Lz = Lz,
        x0 = x0, y0 = y0,
        dt = Δt,
        nt = stop_iteration,
        f₀ = _coriolis_frequency(coriolis),
        N² = _stratification_N²(stratification),
        ybj_plus = ybj_plus,
        fixed_flow = _fixed_flow(flow),
        no_feedback = no_feedback,
        no_wave_feedback = no_wave_feedback,
        νₕ₁ = closure.flow,
        νₕ₂ = closure.flow2,
        ilap1 = closure.flow_laplacian_order,
        ilap2 = closure.flow_laplacian_order2,
        νₕ₁ʷ = closure.waves,
        νₕ₂ʷ = closure.waves2,
        ilap1w = closure.wave_laplacian_order,
        ilap2w = closure.wave_laplacian_order2,
        γ = asselin_filter,
        topology = topology,
        parallel_io = parallel_io,
        verbose = verbose
    )
end

function Base.show(io::IO, grid::RectilinearGridSpec)
    nx, ny, nz = grid.size
    Lx, Ly, Lz = grid.extent
    print(io, "RectilinearGrid(size=($nx, $ny, $nz), extent=($Lx, $Ly, $Lz))")
end

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

## Robert-Asselin filter
- `γ`: Filter coefficient (default: 1e-3)

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
    x0::Union{Real, Nothing} = nothing,
    y0::Union{Real, Nothing} = nothing,
    # Physical parameters
    f₀::Real = 1e-4,
    N²::Real = 1e-5,
    # Time stepping
    dt::Real = 1.0,
    nt::Int = 1000,
    # Model options
    ybj_plus::Bool = true,
    fixed_flow::Bool = false,
    no_feedback::Bool = true,
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
    # Robert-Asselin filter
    γ::Real = 1e-3,
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
    mpi_config = setup_mpi_environment(; topology=topology, parallel_io=parallel_io, verbose=verbose)

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
        x0 = x0,
        y0 = y0,
        dt = T(dt), nt = nt,
        f₀ = T(f₀), N² = T(N²),
        ybj_plus = ybj_plus,
        fixed_flow = fixed_flow,
        no_feedback = no_feedback,
        no_wave_feedback = no_wave_feedback,
        νₕ₁ = T(νₕ₁),
        νₕ₂ = T(νₕ₂),
        ilap1 = ilap1,
        ilap2 = ilap2,
        νₕ₁ʷ = T(νₕ₁ʷ),
        νₕ₂ʷ = T(νₕ₂ʷ),
        ilap1w = ilap1w,
        ilap2w = ilap2w,
        γ = T(γ)
    )

    # Initialize grid, plans, state, workspace
    grid = init_mpi_grid(params, mpi_config)
    plans = plan_mpi_transforms(grid, mpi_config)
    state = init_mpi_state(grid, plans, mpi_config)
    workspace = init_mpi_workspace(grid, mpi_config)

    # Compute stratification profile
    N2_profile = compute_stratification_profile(ConstantN{T}(sqrt(N²)), grid)

    if mpi_config.is_root && verbose
        println("Initialization complete.")
        println("="^70)
    end

    MPI.Barrier(mpi_config.comm)

    return Simulation{T, typeof(grid), typeof(state), typeof(plans),
                      typeof(mpi_config), typeof(workspace)}(
        grid, state, params, plans, mpi_config, workspace, N2_profile,
        default_run_options(T)
    )
end

#=
================================================================================
                        INITIAL CONDITIONS
================================================================================
=#

"""
    set_mean_flow!(sim::Simulation; psi_func, method=:function, pv_method=:qg,
                   amplitude=1.0, spectral_slope=-3.0, seed=0)

Set up the balanced mean flow from an analytical streamfunction or random noise.

For `method=:function`, `psi_func(x, y, z)` should return ψ in m²/s at the
cell centers. Coordinates respect the grid origin (`x0`, `y0`) and use `G.z`
for vertical levels. This works in MPI because each rank fills only its local
slab before the FFT.

For `method=:random`, a deterministic MPI-safe random spectrum is generated.

Use `pv_method=:barotropic` for the simple Asselin et al. (2020) initialization
`q̂ = -kₕ² ψ̂`. Use `pv_method=:qg` for full QG PV, including vertical stretching.

# Arguments
- `sim`: Simulation object
- `psi_func`: Function returning ψ(x, y, z) when `method=:function`
- `method`: `:function` or `:random` (alias `:analytical` for `:function`)
- `pv_method`: `:qg` for full QG PV or `:barotropic` for `q̂ = -kₕ² ψ̂`
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
set_mean_flow!(sim; psi_func=dipole, pv_method=:barotropic)
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

    if pv_method == :qg || pv_method == :balanced
        add_balanced_component!(S, G, sim.params, sim.plans; N2_profile=sim.N2_profile)
    elseif pv_method == :barotropic || pv_method == :asselin
        if sim.mpi_config.is_root && verbose
            println("Setting barotropic PV from streamfunction: q̂ = -kₕ² ψ̂")
        end
        compute_barotropic_q_from_psi!(S.q, S.psi, G)
    elseif pv_method == :none
        nothing
    else
        throw(ArgumentError("Unknown pv_method=$pv_method. Use :qg, :barotropic, or :none."))
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
    L⁺A_phys = allocate_fft_backward_dst(S.L⁺A, plans)
    L⁺A_arr = parent(L⁺A_phys)
    T = typeof(real(zero(eltype(L⁺A_arr))))

    dz = G.Lz / G.nz
    for k_local in axes(L⁺A_arr, 1)
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
            L⁺A_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        else
            # Could add horizontal structure here
            L⁺A_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        end
    end

    # Transform to spectral space
    fft_forward!(S.L⁺A, L⁺A_phys, plans)

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
    set!(model::Simulation; ψ=nothing, psi=nothing, mean_flow=nothing,
         pv_method=:qg, waves=nothing, B=nothing)

Set initial conditions with an Oceananigans-like interface.

Use `ψ`, `psi`, or `mean_flow` for a dimensional streamfunction function
`ψ(x, y, z)`. Use `waves=SurfaceWave(...)` or `B=SurfaceWave(...)` for a
surface-confined wave envelope.
"""
function set!(model::Simulation;
    ψ = nothing,
    psi = nothing,
    mean_flow = nothing,
    pv_method::Symbol = :qg,
    waves = nothing,
    B = nothing,
    verbose::Bool = false)

    flow = _first_notnothing(mean_flow, ψ, psi)
    wave = _first_notnothing(waves, B)

    if flow !== nothing
        flow isa Function || throw(ArgumentError("mean_flow/ψ must be a function of (x, y, z)."))
        set_mean_flow!(model; psi_func=flow, pv_method=pv_method, verbose=verbose)
    end

    if wave !== nothing
        if wave isa SurfaceWave
            set_surface_waves!(model;
                amplitude = wave.amplitude,
                surface_depth = wave.scale,
                profile = wave.profile,
                verbose = verbose)
        else
            throw(ArgumentError("waves/B must be a SurfaceWave."))
        end
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

    if sim.mpi_config.is_root
        println("Setting wave packet: kx=$kx, ky=$ky, σ_k=$sigma_k")
    end

    # Use the existing create_wave_packet function
    packet = create_wave_packet(G, kx, ky, sigma_k, amplitude)

    # Copy to state (handling MPI distribution)
    S.L⁺A .= scatter_from_root(packet, G, sim.mpi_config; plans=sim.plans)

    return sim
end

#=
================================================================================
                        RUNNING SIMULATIONS
================================================================================
=#

function _configure_time_stepping!(sim::Simulation; Δt=nothing, stop_time=nothing, stop_iteration=nothing)
    if Δt !== nothing
        Δt > 0 || throw(ArgumentError("Δt must be positive (got $Δt)."))
        sim.params.dt = typeof(sim.params.dt)(Δt)
    end

    if stop_iteration !== nothing
        stop_iteration > 0 || throw(ArgumentError("stop_iteration must be positive (got $stop_iteration)."))
        sim.params.nt = stop_iteration
    elseif stop_time !== nothing
        stop_time > 0 || throw(ArgumentError("stop_time must be positive (got $stop_time)."))
        sim.params.nt = max(1, round(Int, stop_time / sim.params.dt))
    end

    return sim
end

function _configure_output!(sim::Simulation; output=nothing, diagnostics=nothing,
    timestepper=nothing, verbose=nothing)

    options = sim.run_options

    if timestepper !== nothing
        options.timestepper = timestepper
    end

    if verbose !== nothing
        options.verbose = verbose
    end

    if output !== nothing
        if output isa NetCDFOutput
            options.output_dir = output.path
            options.save_interval = output.schedule isa TimeInterval ? output.schedule.interval : nothing
            options.save_psi = :ψ in output.fields || :psi in output.fields
            options.save_waves = :waves in output.fields || :B in output.fields || :L⁺A in output.fields
            options.save_velocities = output.velocities || :velocities in output.fields
        else
            throw(ArgumentError("output must be a NetCDFOutput."))
        end
    end

    if diagnostics !== nothing
        if diagnostics isa IterationInterval
            options.diagnostics_interval = diagnostics.interval
        elseif diagnostics isa Integer
            options.diagnostics_interval = Int(diagnostics)
        else
            throw(ArgumentError("diagnostics must be an IterationInterval or integer step interval."))
        end
    end

    return sim
end

"""
    Simulation(model::Simulation; Δt=nothing, stop_time=nothing, stop_iteration=nothing,
               timestepper=nothing, output=nothing, diagnostics=nothing, verbose=nothing)

Configure a model's run clock in the Oceananigans style and return the same
high-level simulation object.
"""
function Simulation(model::Simulation; Δt=nothing, stop_time=nothing, stop_iteration=nothing,
    timestepper=nothing, output=nothing, diagnostics=nothing, verbose=nothing)

    _configure_time_stepping!(model;
        Δt = Δt,
        stop_time = stop_time,
        stop_iteration = stop_iteration)

    return _configure_output!(model;
        timestepper = timestepper,
        output = output,
        diagnostics = diagnostics,
        verbose = verbose)
end

"""
    run!(sim::Simulation; kwargs...)

Run the simulation with specified options.

This wraps `run_simulation!` with a simpler interface.

# Keyword Arguments
- `output_dir`: Output directory (default: "output")
- `timestepper`: Time-stepping method, `:leapfrog` or `:imex_cn` (default: `:imex_cn`)
- `save_interval`: Save interval in simulation time units
- `diagnostics_interval`: Diagnostics interval in time steps (default: 10)
- `verbose`: Print progress (default: true on root)

# Example
```julia
run!(sim; output_dir="output", timestepper=:imex_cn)
```
"""
function run!(sim::Simulation;
    output_dir::Union{String, Nothing} = nothing,
    timestepper::Union{Symbol, Nothing} = nothing,
    Δt = nothing,
    stop_time = nothing,
    stop_iteration = nothing,
    output = nothing,
    diagnostics = nothing,
    save_interval::Union{Real, Nothing} = nothing,
    diagnostics_interval::Union{Int, Nothing} = nothing,
    verbose::Union{Bool, Nothing} = nothing,
    progress::Union{Nothing, Bool} = nothing,
    save_psi::Union{Bool, Nothing} = nothing,
    save_waves::Union{Bool, Nothing} = nothing,
    save_velocities::Union{Bool, Nothing} = nothing)

    _configure_time_stepping!(sim;
        Δt = Δt,
        stop_time = stop_time,
        stop_iteration = stop_iteration)

    _configure_output!(sim;
        timestepper = timestepper,
        output = output,
        diagnostics = diagnostics,
        verbose = verbose)

    if progress !== nothing
        sim.run_options.verbose = progress
    end

    options = sim.run_options

    if output_dir !== nothing
        options.output_dir = output_dir
    end

    if save_interval !== nothing
        options.save_interval = save_interval
    end

    if diagnostics_interval !== nothing
        options.diagnostics_interval = diagnostics_interval
    end

    if save_psi !== nothing
        options.save_psi = save_psi
    end

    if save_waves !== nothing
        options.save_waves = save_waves
    end

    if save_velocities !== nothing
        options.save_velocities = save_velocities
    end

    G = sim.grid
    S = sim.state
    params = sim.params
    plans = sim.plans
    mpi_config = sim.mpi_config
    workspace = sim.workspace
    N2_profile = sim.N2_profile

    # Create output directory
    if mpi_config.is_root
        mkpath(options.output_dir)
    end
    MPI.Barrier(mpi_config.comm)

    # Compute default save interval (1 inertial period)
    T_inertial = 2π / params.f₀
    interval = options.save_interval === nothing ? T_inertial : options.save_interval

    # Configure output
    output_config = OutputConfig(
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
    )

    # Run simulation
    run_simulation!(S, G, params, plans;
        output_config = output_config,
        mpi_config = mpi_config,
        workspace = workspace,
        N2_profile = N2_profile,
        print_progress = mpi_config.is_root && options.verbose,
        diagnostics_interval = options.diagnostics_interval,
        timestepper = options.timestepper
    )

    if mpi_config.is_root && options.verbose
        println("\nSimulation complete. Output saved to: $(options.output_dir)/")
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
