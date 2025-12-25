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
const Lx = 70e3              # 70 km horizontal domain
const Ly = 70e3              # 70 km horizontal domain
const Lz = 3000.0            # 3 km depth, surface at z = Lz

# Time stepping
const n_inertial_periods = 15
const T_inertial = 2π / f₀   # Inertial period = 2π/f [s] ≈ 14 hours
const dt = 100.0             # Time step [s]
const nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave parameters
const u0_wave = 0.10         # Wave velocity amplitude [m/s] (u0 = 10 cm/s)
const surface_layer_depth = 30.0  # Surface layer depth [m] (s = 30 m)

# Flow parameters
const U0_flow = 0.335        # Flow velocity scale [m/s] (U = 33.5 cm/s)
const k_dipole = 2π / Lx  # κ = 2π/(70 km) per Asselin et al. (2020)
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
    par = QGYBJplus.default_params(
        nx = nx, ny = ny, nz = nz,
        Lx = Lx, Ly = Ly, Lz = Lz,  # Domain size [m]
        dt = dt, nt = nt,
        f₀ = f₀,               # Coriolis parameter [s⁻¹]
        N² = N²,               # Buoyancy frequency squared [s⁻²]
        ybj_plus = true,
        fixed_flow = true,
        no_wave_feedback = true
    )

    # Initialize distributed grid, plans, and state
    G = QGYBJplus.init_mpi_grid(par, mpi_config)
    plans = QGYBJplus.plan_mpi_transforms(G, mpi_config)
    S = QGYBJplus.init_mpi_state(G, mpi_config)
    workspace = QGYBJplus.init_mpi_workspace(G, mpi_config)

    # All arrays use pencil_xy - get local index range
    local_range = QGYBJplus.get_local_range_xy(G)

    # Set up dipole: ψ = U κ⁻¹ sin(κx) cos(κy)
    # This creates a barotropic dipole eddy with velocity scale U0_flow (Eq. 2 in paper)
    if is_root; println("\nSetting up dipole..."); end
    psi_phys = similar(S.psi)
    psi_phys_arr = parent(psi_phys)
    for k_local in axes(psi_phys_arr, 3)
        for j_local in axes(psi_phys_arr, 2)
            j_global = local_range[2][j_local]
            y = (j_global - 1) * G.dy - G.Ly / 2  # Centered y (rotated coords)
            for i_local in axes(psi_phys_arr, 1)
                i_global = local_range[1][i_local]
                x = (i_global - 1) * G.dx - G.Lx / 2  # Centered x (rotated coords)
                # Dimensional streamfunction [m²/s]
                psi_phys_arr[i_local, j_local, k_local] = complex(psi0 * sin(k_dipole * x) * cos(k_dipole * y))
            end
        end
    end
    QGYBJplus.fft_forward!(S.psi, psi_phys, plans)

    # Compute q = -kh² × ψ (spectral space operation)
    q_local = parent(S.q)
    psi_local = parent(S.psi)
    for k_local in axes(q_local, 3)
        for j_local in axes(q_local, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(q_local, 1)
                i_global = local_range[1][i_local]
                kh2 = G.kx[i_global]^2 + G.ky[j_global]^2
                q_local[i_local, j_local, k_local] = -kh2 * psi_local[i_local, j_local, k_local]
            end
        end
    end

    # Set up wave IC: surface-confined, horizontally uniform (k=0 mode only)
    # Initial condition: u(t=0) = u0 exp(-z^2/s^2), v(t=0) = 0 (Eq. 4 in paper)
    if is_root; println("Setting up waves..."); end
    B_phys = similar(S.B)
    B_phys_arr = parent(B_phys)
    for k_local in axes(B_phys_arr, 3)
        k_global = local_range[3][k_local]
        depth = G.Lz - G.z[k_global]  # Distance from surface [m]
        wave_profile = exp(-(depth^2) / (surface_layer_depth^2))
        wave_value = complex(u0_wave * wave_profile)
        B_phys_arr[:, :, k_local] .= wave_value
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
    # This handles: leapfrog state management, initial projection step,
    # output file saving, progress reporting, and energy diagnostics
    QGYBJplus.run_simulation!(S, G, par, plans;
        output_config = output_config,
        mpi_config = mpi_config,
        workspace = workspace,
        print_progress = is_root,
        diagnostics_interval = 10 #diag_steps
    )

    if is_root
        println("\nOutput files saved to: $output_dir/")
        println("  - Variables: psi (flow), LAr, LAi (waves)")
    end

    MPI.Finalize()
end

main()
