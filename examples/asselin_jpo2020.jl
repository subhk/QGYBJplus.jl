#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example - MPI Parallel Version
================================================================================

MPI-parallel version of the barotropic dipole simulation from:

    Asselin, O., L. N. Thomas, W. R. Young, and L. Rainville (2020)
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439-3454

USAGE:
------
    mpirun -n 4 julia --project examples/asselin_jpo2020.jl
    mpirun -n 16 julia --project examples/asselin_jpo2020.jl

TIME-STEPPING OPTIONS:
----------------------
This example supports two time-stepping methods:
- :leapfrog (default in Fortran): Explicit leapfrog with Robert-Asselin filter
                                  Requires dt ≤ 2f/N² ≈ 2s for dispersion CFL
- :imex_cn: IMEX Crank-Nicolson with implicit dispersion
            Allows dt ~ 20s (10x speedup), only limited by advection CFL

Set TIMESTEPPER in the parameters section below to choose.

================================================================================
=#

using MPI
using PencilArrays
using PencilFFTs
using QGYBJplus
using Printf

# ============================================================================
#                       SIMULATION PARAMETERS
# ============================================================================

const nx = 128
const ny = 128
const nz = 64

# Physical parameters from Asselin et al. (2020)
const f₀ = 1.24e-4           # Coriolis parameter [s⁻¹] (mid-latitude)
const N² = 1e-5              # Buoyancy frequency squared [s⁻²]

# Domain size [m] (Asselin et al. 2020)
# Grid is in cardinal (X,Y) coordinates with 70 km periodic domain
# Dipole formula uses rotated (x,y) coords: x=(X-Y)/√2, y=(X+Y)/√2
const Lx = 70e3              # 70 km horizontal domain in (X,Y)
const Ly = 70e3              # 70 km horizontal domain in (X,Y)
const Lz = 3000.0            # 3 km depth, surface at z = Lz

# Time stepping
const n_inertial_periods = 15
const T_inertial = 2π / f₀   # Inertial period = 2π/f [s] ≈ 14 hours

# Time-stepping method selection:
# - :leapfrog (default): Explicit leapfrog with Robert-Asselin filter
#                        CFL-limited by wave dispersion: dt ≤ 2f/N² ≈ 25s
# - :imex_cn: IMEX Crank-Nicolson with implicit dispersion
#             Allows larger dt (limited only by advection CFL)
const TIMESTEPPER = :imex_cn  # Options: :leapfrog or :imex_cn

# Timestep selection based on method
# IMEX allows ~10x larger timesteps since dispersion is treated implicitly
const dt = TIMESTEPPER == :imex_cn ? 20.0 : 2.0  # [s]
const nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave parameters
const u0_wave = 0.10         # Wave velocity amplitude [m/s] (u0 = 10 cm/s)
const surface_layer_depth = 30.0  # Surface layer depth [m] (s = 30 m)

# Flow parameters
const U0_flow = 0.335        # Flow velocity scale [m/s] (U = 33.5 cm/s)
const k_dipole = sqrt(2) * π / Lx  # κ = √2π/(70 km) per Asselin et al. (2020)
const psi0 = U0_flow / k_dipole  # Streamfunction amplitude [m²/s]

# Output settings
const output_dir = "output_asselin"
const save_interval_IP = 1.0  # Save every 1 inertial period
const diag_interval_IP = 0.5  # Print diagnostics every 0.5 inertial periods

# ============================================================================
#                       MAIN FUNCTION
# ============================================================================

function main()
    MPI.Init()
    mpi_config = QGYBJplus.setup_mpi_environment(parallel_io=false)
    is_root = mpi_config.is_root

    # Create output directory
    if is_root
        mkpath(output_dir)
        println("="^70)
        println("Asselin et al. (2020) Dipole - MPI Parallel")
        println("="^70)
        println("Processes: $(mpi_config.nprocs), Topology: $(mpi_config.topology)")
        @printf("Resolution: %d × %d × %d, Duration: %.1f IP\n", nx, ny, nz, n_inertial_periods)
        println("Output directory: $output_dir")
    end
    MPI.Barrier(mpi_config.comm)

    # Parameters matching Asselin et al. (2020)
    # Fully dimensional simulation with physical domain size
    # Biharmonic (4th order) hyperdiffusion for waves
    # For meaningful damping at grid scale: λʷ = dt × ν × k_max⁴ ≈ 0.1
    # k_max = π×nx/Lx ≈ 5.7e-3 m⁻¹, k_max⁴ ≈ 1.1e-9 m⁻⁴
    # νₕ₁ʷ ≈ 0.1 / (dt × k_max⁴) ≈ 1e7 m⁴/s
    νₕ₁ʷ_wave = 1.0e7  # [m⁴/s] - grid-scale damping ~10% per timestep

    par = QGYBJplus.default_params(
        nx = nx, ny = ny, nz = nz,
        Lx = Lx, Ly = Ly, Lz = Lz,  # Domain size [m]
        dt = dt, nt = nt,
        f₀ = f₀,               # Coriolis parameter [s⁻¹]
        N² = N²,               # Buoyancy frequency squared [s⁻²]
        ybj_plus = true,
        fixed_flow = true,
        no_wave_feedback = true,
        νₕ₁ʷ = νₕ₁ʷ_wave,      # ∇⁴ hyperdiffusion for waves
        ilap1w = 2,            # 4th order (biharmonic)
        γ = 0.01               # Stronger Robert-Asselin filter (default: 1e-3)
    )

    # Initialize distributed grid, plans, and state
    G = QGYBJplus.init_mpi_grid(par, mpi_config)
    plans = QGYBJplus.plan_mpi_transforms(G, mpi_config)
    S = QGYBJplus.init_mpi_state(G, plans, mpi_config)
    workspace = QGYBJplus.init_mpi_workspace(G, mpi_config)

    # Compute N2 profile for consistent physics across all operations
    # This is passed to run_simulation! for use in elliptic inversions and vertical velocity
    N2_profile = QGYBJplus.compute_stratification_profile(
        QGYBJplus.SkewedGaussian{T}(T(N02), T(N12), T(s_gauss), T(z0_gauss), T(α_sg)),
        G
    )

    # Local index ranges (physical vs spectral pencils)
    local_range_phys = QGYBJplus.get_local_range_physical(plans)
    local_range_spec = QGYBJplus.get_local_range_spectral(plans)

    # Set up dipole: ψ = U κ⁻¹ sin(κx) cos(κy) in rotated (x,y) coordinates
    # Grid is in cardinal (X,Y); transform via x=(X-Y)/√2, y=(X+Y)/√2 (Fig. 1)
    # This creates a barotropic dipole eddy with velocity scale U0_flow (Eq. 2 in paper)
    if is_root; println("\nSetting up dipole..."); end
    psi_phys = QGYBJplus.allocate_fft_backward_dst(S.psi, plans)
    psi_phys_arr = parent(psi_phys)
    for k_local in axes(psi_phys_arr, 1)
        for j_local in axes(psi_phys_arr, 3)
            j_global = local_range_phys[3][j_local]
            Y = (j_global - 1) * G.dy - G.Ly / 2  # Centered Y (cardinal coords)
            for i_local in axes(psi_phys_arr, 2)
                i_global = local_range_phys[2][i_local]
                X = (i_global - 1) * G.dx - G.Lx / 2  # Centered X (cardinal coords)
                # Transform to rotated (x,y) for dipole formula
                x = (X - Y) / sqrt(2)
                y = (X + Y) / sqrt(2)
                # Dimensional streamfunction [m²/s]
                psi_phys_arr[k_local, i_local, j_local] = complex(psi0 * sin(k_dipole * x) * cos(k_dipole * y))
            end
        end
    end
    QGYBJplus.fft_forward!(S.psi, psi_phys, plans)

    # Compute vorticity: ζ = ∇²ψ (on f-plane, q = ζ)
    # In spectral space: ζ̂ = -kh² × ψ̂
    # For dipole: ζ = -2κU sin(κx) cos(κy) per Eq. (3)
    q_local = parent(S.q)
    psi_local = parent(S.psi)
    for k_local in axes(q_local, 1)
        for j_local in axes(q_local, 3)
            j_global = local_range_spec[3][j_local]
            for i_local in axes(q_local, 2)
                i_global = local_range_spec[2][i_local]
                kh2 = G.kx[i_global]^2 + G.ky[j_global]^2
                q_local[k_local, i_local, j_local] = -kh2 * psi_local[k_local, i_local, j_local]
            end
        end
    end

    # Set up wave IC: surface-confined, horizontally uniform (k=0 mode only)
    # Initial condition: u(t=0) = u0 exp(-z^2/s^2), v(t=0) = 0 (Eq. 4 in paper)
    # For horizontally uniform waves, we initialize B directly with the wave profile.
    if is_root; println("Setting up waves..."); end
    B_phys = QGYBJplus.allocate_fft_backward_dst(S.B, plans)
    B_phys_arr = parent(B_phys)
    for k_local in axes(B_phys_arr, 1)
        k_global = local_range_phys[1][k_local]
        depth = G.Lz - G.z[k_global]  # Distance from surface [m]
        wave_profile = exp(-(depth^2) / (surface_layer_depth^2))
        wave_value = complex(u0_wave * wave_profile)
        B_phys_arr[k_local, :, :] .= wave_value
    end
    QGYBJplus.fft_forward!(S.B, B_phys, plans)

    # Configure output
    output_config = QGYBJplus.OutputConfig(
        output_dir = output_dir,
        state_file_pattern = "state%04d.nc",
        psi_interval = save_interval_IP * T_inertial,
        wave_interval = save_interval_IP * T_inertial,
        diagnostics_interval = diag_interval_IP * T_inertial,
        save_psi = true,
        save_waves = true,
        save_velocities = false,
        save_vorticity = false,
        save_diagnostics = false
    )

    # Compute diagnostics interval in steps
    diag_steps = max(1, round(Int, diag_interval_IP * T_inertial / dt))

    # Run simulation - all time-stepping handled automatically
    # This handles: state management, initial projection step,
    # output file saving, progress reporting, and energy diagnostics
    # The timestepper can be :leapfrog (explicit) or :imex_cn (implicit dispersion)
    QGYBJplus.run_simulation!(S, G, par, plans;
        output_config = output_config,
        mpi_config = mpi_config,
        workspace = workspace,
        N2_profile = N2_profile,  # Pass stratification profile for consistent physics
        print_progress = is_root,
        diagnostics_interval = diag_steps,
        timestepper = TIMESTEPPER  # :leapfrog or :imex_cn
    )

    if is_root
        println("\nOutput files saved to: $output_dir/")
        println("  - Variables: psi (flow), LAr, LAi (waves)")
    end

    # Ensure all MPI operations are complete before finalization
    MPI.Barrier(mpi_config.comm)

    # Force garbage collection BEFORE MPI.Finalize to prevent heap corruption
    # from Julia finalizers running after MPI is shut down
    GC.gc(true)  # Full garbage collection

    MPI.Finalize()
end

main()
