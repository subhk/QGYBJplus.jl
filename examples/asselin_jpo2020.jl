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
    mpirun -n 4 julia --project examples/asselin_jpo2020_dipole_mpi.jl
    mpirun -n 16 julia --project examples/asselin_jpo2020_dipole_mpi.jl

================================================================================
=#

using MPI
using PencilArrays
using PencilFFTs
using QGYBJ
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

# Domain size [m] - typical mesoscale eddy domain
const Lx = 500e3             # 500 km horizontal domain
const Ly = 500e3             # 500 km horizontal domain
# Note: Vertical grid is nondimensional z ∈ [0, 2π], surface at z = 2π

# Time stepping
const n_inertial_periods = 15
const T_inertial = 2π / f₀   # Inertial period = 2π/f [s] ≈ 14 hours
const dt = 100.0             # Time step [s]
const nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave parameters
const u0_wave = 0.05         # Wave velocity amplitude [m/s]
const sigma_z = 0.01 * 2π    # Vertical decay scale (nondimensional, surface-confined)

# Flow parameters
const U0_flow = 0.5          # Flow velocity scale [m/s]
const psi0 = U0_flow * Lx / (2π)  # Streamfunction amplitude [m²/s]

# Output settings
const output_dir = "output_asselin"
const save_interval_IP = 1.0  # Save every 1 inertial period

# ============================================================================
#                       MAIN FUNCTION
# ============================================================================

function main()
    MPI.Init()
    mpi_config = QGYBJ.setup_mpi_environment()
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

    # Create parallel config for I/O
    parallel_config = QGYBJ.ParallelConfig(
        use_mpi = true,
        comm = mpi_config.comm,
        parallel_io = false  # Use gather-to-root approach
    )

    # Parameters matching Asselin et al. (2020)
    # Dimensional simulation with physical domain size
    par = QGYBJ.default_params(
        nx = nx, ny = ny, nz = nz,
        Lx = Lx, Ly = Ly,      # Domain size [m]
        dt = dt, nt = nt,
        f₀ = f₀,               # Coriolis parameter [s⁻¹]
        N² = N²,               # Buoyancy frequency squared [s⁻²]
        ybj_plus = true,
        fixed_flow = true,
        no_wave_feedback = true
    )

    # Initialize distributed grid and state
    G = QGYBJ.init_mpi_grid(par, mpi_config)
    S = QGYBJ.init_mpi_state(G, mpi_config)
    workspace = QGYBJ.init_mpi_workspace(G, mpi_config)
    plans = QGYBJ.plan_mpi_transforms(G, mpi_config)

    local_range = QGYBJ.get_local_range_xy(G)

    # Set up dipole: ψ = ψ₀ × sin(2πx/Lx - π/2) × cos(2πy/Ly)
    # This creates a barotropic dipole eddy with velocity scale U0_flow
    if is_root; println("\nSetting up dipole..."); end
    psi_local = parent(S.psi)
    for k_local in axes(psi_local, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(psi_local, 2)
            j_global = local_range[2][j_local]
            y = (j_global - 1) * G.dy  # Physical y coordinate [m]
            for i_local in axes(psi_local, 1)
                i_global = local_range[1][i_local]
                x = (i_global - 1) * G.dx  # Physical x coordinate [m]
                # Dimensional streamfunction [m²/s]
                psi_local[i_local, j_local, k_local] = complex(psi0 * sin(2π*x/Lx - π/2) * cos(2π*y/Ly))
            end
        end
    end
    QGYBJ.fft_forward!(S.psi, S.psi, plans)

    # Compute q = -kh² × ψ
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
    # Wave envelope B has amplitude u0_wave [m/s] at surface, decaying with depth
    if is_root; println("Setting up waves..."); end
    B_local = parent(S.B)
    for k_local in axes(B_local, 3)
        k_global = local_range[3][k_local]
        depth = 2π - G.z[k_global]  # Distance from surface (nondimensional)
        wave_profile = exp(-(depth^2) / (sigma_z^2))
        for j_local in axes(B_local, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(B_local, 1)
                i_global = local_range[1][i_local]
                if i_global == 1 && j_global == 1
                    # B in spectral space: k=0 mode scaled by (nx*ny) for FFT normalization
                    B_local[i_local, j_local, k_local] = u0_wave * wave_profile * (nx * ny)
                else
                    B_local[i_local, j_local, k_local] = 0.0
                end
            end
        end
    end

    # Time integration
    a_ell = QGYBJ.a_ell_ut(par, G)
    L_mask = QGYBJ.dealias_mask(G)
    QGYBJ.compute_velocities!(S, G; plans=plans, params=par, workspace=workspace)

    # Create OutputManager using codebase
    output_config = QGYBJ.OutputConfig(
        output_dir = output_dir,
        state_file_pattern = "state%04d.nc",
        psi_interval = save_interval_IP * T_inertial,
        wave_interval = save_interval_IP * T_inertial,
        diagnostics_interval = T_inertial,
        save_psi = true,
        save_waves = true,
        save_velocities = false,
        save_vorticity = false,
        save_diagnostics = false
    )
    output_manager = QGYBJ.OutputManager(output_config, par, parallel_config)

    if is_root
        println("\n" * "="^70)
        println("Starting time integration...")
        println("="^70)
    end

    output_interval = round(Int, T_inertial / dt)
    save_interval = round(Int, save_interval_IP * T_inertial / dt)

    # Initial diagnostics
    local_EB = sum(abs2.(parent(S.B)))
    global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)
    if is_root; @printf("\nt = 0.0 IP: E_B = %.4e\n", global_EB / (nx*ny*nz)); end

    # Save initial state using codebase OutputManager
    if is_root; println("\nSaving initial state..."); end
    QGYBJ.write_state_file(output_manager, S, G, plans, 0.0, parallel_config; params=par)

    QGYBJ.first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace)

    Sn, Snm1, Snp1 = deepcopy(S), deepcopy(S), deepcopy(S)

    for step in 1:nt
        QGYBJ.leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                             a=a_ell, dealias_mask=L_mask, workspace=workspace)
        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        current_time = step * dt

        # Print diagnostics every inertial period
        if step % output_interval == 0
            t_IP = current_time / T_inertial
            local_EB = sum(abs2.(parent(Sn.B)))
            global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)
            if is_root; @printf("t = %.1f IP: E_B = %.4e\n", t_IP, global_EB / (nx*ny*nz)); end
        end

        # Save state at specified interval using codebase OutputManager
        if step % save_interval == 0
            QGYBJ.write_state_file(output_manager, Sn, G, plans, current_time, parallel_config; params=par)
        end
    end

    if is_root
        println("\n" * "="^70)
        println("Simulation complete!")
        println("="^70)
        println("Output files saved to: $output_dir/")
        println("  - state0001.nc to state$(lpad(output_manager.psi_counter-1, 4, '0')).nc")
        println("  - Variables: psi (flow), LAr, LAi (waves)")
        println("="^70)
    end

    MPI.Finalize()
end

main()
