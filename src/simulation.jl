"""
High-level Simulation API for QG-YBJ model.

Provides a simplified interface that hides MPI complexity from users:
- `Simulation` struct wraps all components (grid, state, plans, etc.)
- `initialize_simulation()` handles all MPI setup automatically
- `set_dipole_flow!()`, `set_surface_waves!()` for common initial conditions
- `run!()` for time integration

# Example
```julia
using QGYBJplus

# Initialize simulation (handles all MPI setup internally)
sim = initialize_simulation(
    nx=256, ny=256, nz=128,
    Lx=70e3, Ly=70e3, Lz=2000.0,
    f₀=1.24e-4, N²=1e-5,
    dt=20.0, nt=10000
)

# Set initial conditions
set_dipole_flow!(sim; amplitude=0.335, k=sqrt(2)*π/70e3)
set_surface_waves!(sim; amplitude=0.10, surface_depth=30.0)

# Run simulation
run!(sim; output_dir="output", timestepper=:imex_cn)

# Cleanup
finalize_simulation!(sim)
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
struct Simulation{T, G<:Grid, S<:State, P, M<:MPIConfig, W}
    grid::G
    state::S
    params::QGParams{T}
    plans::P
    mpi_config::M
    workspace::W
    N2_profile::Vector{T}
end

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
    centered::Bool = false,  # Center domain at origin: x,y ∈ [-Lx/2, Lx/2)
    # Physical parameters
    f₀::Real = 1e-4,
    N²::Real = 1e-5,
    # Time stepping
    dt::Real = 1.0,
    nt::Int = 1000,
    # Model options
    ybj_plus::Bool = true,
    fixed_flow::Bool = false,
    no_wave_feedback::Bool = false,
    # Diffusion
    νₕ₁ʷ::Real = 0.0,
    ilap1w::Int = 2,
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
        dt = T(dt), nt = nt,
        f₀ = T(f₀), N² = T(N²),
        ybj_plus = ybj_plus,
        fixed_flow = fixed_flow,
        no_wave_feedback = no_wave_feedback,
        νₕ₁ʷ = T(νₕ₁ʷ),
        ilap1w = ilap1w,
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
        grid, state, params, plans, mpi_config, workspace, N2_profile
    )
end

#=
================================================================================
                        INITIAL CONDITIONS
================================================================================
=#

"""
    set_dipole_flow!(sim::Simulation; amplitude, k=nothing, rotated=true)

Set up a barotropic dipole flow (eddy).

The dipole streamfunction follows Asselin et al. (2020):
    ψ = (U/κ) sin(κx) cos(κy)

where U is the velocity amplitude and κ is the dipole wavenumber.

The domain origin is determined by the Grid's `x0`, `y0` fields, which are set via
`centered=true` in `initialize_simulation()` or `default_params()`.

# Arguments
- `sim`: Simulation object
- `amplitude`: Flow velocity scale U [m/s]
- `k`: Dipole wavenumber κ [rad/m]. Default: sqrt(2)π/Lx
- `rotated`: Use rotated coordinates x=(X-Y)/√2, y=(X+Y)/√2 (default: true)

# Example
```julia
sim = initialize_simulation(nx=256, ny=256, nz=128, Lx=70e3, Ly=70e3, Lz=2000.0,
                            centered=true, ...)  # Domain x,y ∈ [-35km, +35km)
set_dipole_flow!(sim; amplitude=0.335)  # U = 33.5 cm/s
```
"""
function set_dipole_flow!(sim::Simulation;
    amplitude::Real,
    k::Union{Real, Nothing} = nothing,
    rotated::Bool = true)

    G = sim.grid
    S = sim.state
    plans = sim.plans

    # Default wavenumber: sqrt(2)π/Lx
    κ = k === nothing ? sqrt(2) * π / G.Lx : k
    psi0 = amplitude / κ  # Streamfunction amplitude [m²/s]

    if sim.mpi_config.is_root
        println("Setting dipole flow: U = $(amplitude) m/s, κ = $(κ) rad/m")
    end

    # Get local ranges
    local_range = get_local_range_physical(plans)

    # Allocate physical-space array
    psi_phys = allocate_fft_backward_dst(S.psi, plans)
    psi_arr = parent(psi_phys)

    for k_local in axes(psi_arr, 1)
        for j_local in axes(psi_arr, 3)
            j_global = local_range[3][j_local]
            # Use Grid origin (x0, y0) for domain centering
            Y = G.y0 + (j_global - 1) * G.dy

            for i_local in axes(psi_arr, 2)
                i_global = local_range[2][i_local]
                # Use Grid origin (x0, y0) for domain centering
                X = G.x0 + (i_global - 1) * G.dx

                # Coordinates for streamfunction
                if rotated
                    # Rotated coordinates for dipole formula
                    x = (X - Y) / sqrt(2)
                    y = (X + Y) / sqrt(2)
                else
                    x = X
                    y = Y
                end

                psi_arr[k_local, i_local, j_local] = complex(psi0 * sin(κ * x) * cos(κ * y))
            end
        end
    end

    # Transform to spectral space
    fft_forward!(S.psi, psi_phys, plans)

    # Compute vorticity q = ∇²ψ (in spectral: q̂ = -kh² × ψ̂)
    local_range_spec = get_local_range_spectral(plans)
    q_arr = parent(S.q)
    psi_spec_arr = parent(S.psi)

    for k_local in axes(q_arr, 1)
        for j_local in axes(q_arr, 3)
            j_global = local_range_spec[3][j_local]
            for i_local in axes(q_arr, 2)
                i_global = local_range_spec[2][i_local]
                kh2 = G.kx[i_global]^2 + G.ky[j_global]^2
                q_arr[k_local, i_local, j_local] = -kh2 * psi_spec_arr[k_local, i_local, j_local]
            end
        end
    end

    return sim
end

"""
    set_surface_waves!(sim::Simulation; amplitude, surface_depth, uniform=true)

Set up surface-confined near-inertial waves.

The wave initial condition follows Asselin et al. (2020):
    u(t=0) = u₀ exp(-d²/s²), v(t=0) = 0

where d = -z is depth below the surface, u₀ is the wave velocity amplitude,
and s is the surface layer depth.

# Arguments
- `sim`: Simulation object
- `amplitude`: Wave velocity amplitude u₀ [m/s]
- `surface_depth`: Surface layer depth s [m]
- `uniform`: Horizontally uniform waves (default: true)

# Example
```julia
set_surface_waves!(sim; amplitude=0.10, surface_depth=30.0)  # u₀ = 10 cm/s
```
"""
function set_surface_waves!(sim::Simulation;
    amplitude::Real,
    surface_depth::Real,
    uniform::Bool = true)

    G = sim.grid
    S = sim.state
    plans = sim.plans

    if sim.mpi_config.is_root
        println("Setting surface waves: u₀ = $(amplitude) m/s, s = $(surface_depth) m")
    end

    # Get local ranges
    local_range = get_local_range_physical(plans)

    # Allocate physical-space array
    B_phys = allocate_fft_backward_dst(S.B, plans)
    B_arr = parent(B_phys)

    dz = G.Lz / G.nz
    for k_local in axes(B_arr, 1)
        k_global = local_range[1][k_local]
        # Depth from surface (z=0 is surface, z=-Lz is bottom).
        # Use a dz/2 shift so the top cell center corresponds to z=0.
        depth = max(zero(eltype(G.z)), -G.z[k_global] - dz / 2)
        wave_profile = exp(-(depth^2) / (surface_depth^2))

        if uniform
            # Horizontally uniform waves
            B_arr[k_local, :, :] .= complex(amplitude * wave_profile)
        else
            # Could add horizontal structure here
            B_arr[k_local, :, :] .= complex(amplitude * wave_profile)
        end
    end

    # Transform to spectral space
    fft_forward!(S.B, B_phys, plans)

    return sim
end

"""
    set_random_flow!(sim::Simulation; amplitude, spectral_slope=-3.0, seed=42)

Set up a random turbulent flow with specified spectral slope.

# Arguments
- `sim`: Simulation object
- `amplitude`: RMS velocity amplitude [m/s]
- `spectral_slope`: Energy spectrum slope (default: -3 for 2D turbulence)
- `seed`: Random seed for reproducibility

# Example
```julia
set_random_flow!(sim; amplitude=0.1, spectral_slope=-3.0)
```
"""
function set_random_flow!(sim::Simulation;
    amplitude::Real,
    spectral_slope::Real = -3.0,
    seed::Int = 42)

    if sim.mpi_config.is_root
        println("Setting random flow: amplitude = $(amplitude) m/s, slope = $(spectral_slope)")
    end

    init_random_psi!(sim.state.psi, sim.grid, amplitude; slope=spectral_slope)

    # Compute q from ψ for consistency
    add_balanced_component!(sim.state, sim.grid, sim.params, sim.plans;
                           N2_profile=sim.N2_profile)

    return sim
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
    S.B .= scatter_from_root(packet, G, sim.mpi_config; plans=sim.plans)

    return sim
end

#=
================================================================================
                        RUNNING SIMULATIONS
================================================================================
=#

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
    output_dir::String = "output",
    timestepper::Symbol = :imex_cn,
    save_interval::Union{Real, Nothing} = nothing,
    diagnostics_interval::Int = 10,
    verbose::Bool = true,
    save_psi::Bool = true,
    save_waves::Bool = true,
    save_velocities::Bool = false)

    G = sim.grid
    S = sim.state
    params = sim.params
    plans = sim.plans
    mpi_config = sim.mpi_config
    workspace = sim.workspace
    N2_profile = sim.N2_profile

    # Create output directory
    if mpi_config.is_root
        mkpath(output_dir)
    end
    MPI.Barrier(mpi_config.comm)

    # Compute default save interval (1 inertial period)
    T_inertial = 2π / params.f₀
    interval = save_interval === nothing ? T_inertial : save_interval

    # Configure output
    output_config = OutputConfig(
        output_dir = output_dir,
        state_file_pattern = "state%04d.nc",
        psi_interval = interval,
        wave_interval = interval,
        diagnostics_interval = interval,
        save_psi = save_psi,
        save_waves = save_waves,
        save_velocities = save_velocities,
        save_vorticity = false,
        save_diagnostics = false
    )

    # Run simulation
    run_simulation!(S, G, params, plans;
        output_config = output_config,
        mpi_config = mpi_config,
        workspace = workspace,
        N2_profile = N2_profile,
        print_progress = mpi_config.is_root && verbose,
        diagnostics_interval = diagnostics_interval,
        timestepper = timestepper
    )

    if mpi_config.is_root && verbose
        println("\nSimulation complete. Output saved to: $output_dir/")
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
