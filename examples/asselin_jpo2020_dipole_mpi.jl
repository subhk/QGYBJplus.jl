#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example - MPI Parallel Version
================================================================================

MPI-parallel version of the barotropic dipole simulation from:

    Asselin, O., L. N. Thomas, W. R. Young, and L. Rainville (2020)
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439-3454
    DOI: 10.1175/JPO-D-20-0109.1

USAGE:
------
    mpirun -n 4 julia --project examples/asselin_jpo2020_dipole_mpi.jl
    mpirun -n 16 julia --project examples/asselin_jpo2020_dipole_mpi.jl

For optimal performance, use power-of-2 process counts (4, 16, 64, etc.)

================================================================================
=#

using MPI
using PencilArrays
using PencilFFTs
using QGYBJ
using Printf

# ============================================================================
#                       PHYSICAL PARAMETERS (from paper)
# ============================================================================

# Domain dimensions (dimensional)
const L_domain = 70e3      # Horizontal domain size [m]
const H_depth = 3000.0     # Ocean depth [m]

# Coriolis parameter at 58.5N
const f_dim = 1.24e-4      # [s^-1]

# Stratification (uniform)
const N2_dim = 1e-5        # [s^-2]
const N_dim = sqrt(N2_dim) # [s^-1]

# Dipole flow parameters
const U_jet = 0.335        # Max jet velocity [m/s] (33.5 cm/s)
const gamma_max = 2.7e-9   # Max vorticity gradient [m^-1 s^-1]

# Compute dipole wavenumber
const kappa_dim = sqrt(gamma_max / (2 * U_jet))

# Wave initial condition
const u0_wave = 0.10       # Initial wave velocity [m/s] (10 cm/s)
const sigma_wave = 30.0    # Surface layer depth [m]

# ============================================================================
#                       NONDIMENSIONALIZATION
# ============================================================================

const L_scale = L_domain / (2π)
const H_scale = H_depth / (2π)
const U_scale = U_jet
const T_scale = L_scale / U_scale

# Nondimensional parameters
const Ro = U_scale / (f_dim * L_scale)
const Bu = (N_dim * H_scale / (f_dim * L_scale))^2
const W2F = (u0_wave / U_scale)^2

# Nondimensional wave parameters
const u0_nd = u0_wave / U_scale
const sigma_nd = sigma_wave / H_scale

# Inertial period
const T_inertial_dim = 2π / f_dim
const T_inertial_nd = T_inertial_dim / T_scale

# ============================================================================
#                       MAIN FUNCTION
# ============================================================================

function main()
    # Initialize MPI
    MPI.Init()
    mpi_config = QGYBJ.setup_mpi_environment()

    is_root = mpi_config.is_root

    if is_root
        println("="^70)
        println("Asselin et al. (2020) Dipole Simulation - MPI Parallel")
        println("="^70)
        println("\nRunning on $(mpi_config.nprocs) processes")
        println("Topology: $(mpi_config.topology)")

        println("\nDimensional Parameters:")
        @printf("  Domain:        %.0f km x %.0f km x %.1f km\n", L_domain/1e3, L_domain/1e3, H_depth/1e3)
        @printf("  Coriolis f:    %.2e s^-1\n", f_dim)
        @printf("  Stratification N^2: %.1e s^-2\n", N2_dim)
        @printf("  Jet velocity U:    %.1f cm/s\n", U_jet*100)
        @printf("  Wave velocity u0:  %.1f cm/s\n", u0_wave*100)

        println("\nNondimensional Numbers:")
        @printf("  Rossby number Ro:  %.3f\n", Ro)
        @printf("  Burger number Bu:  %.2f\n", Bu)
        @printf("  Nondim inertial period: %.2f\n", T_inertial_nd)
    end

    # ========================================================================
    #                       MODEL PARAMETERS
    # ========================================================================

    const nx = 128
    const ny = 128
    const nz = 64

    const n_inertial_periods = 15
    const dt = 0.001
    const nt = round(Int, n_inertial_periods * T_inertial_nd / dt)

    if is_root
        println("\nSimulation Setup:")
        @printf("  Resolution:    %d x %d x %d\n", nx, ny, nz)
        @printf("  Time step dt:  %.4f (nondim)\n", dt)
        @printf("  Total steps:   %d\n", nt)
    end

    # Create parameter struct
    par = QGYBJ.QGParams{Float64}(
        nx = nx, ny = ny, nz = nz,
        Lx = 2π, Ly = 2π,
        dt = dt, nt = nt,
        f0 = 1.0,
        Ro = Ro, Bu = Bu, W2F = W2F,
        gamma = 1e-3,
        nuh1 = 1e-4, ilap1 = 2,
        nuh2 = 1e-2, ilap2 = 6,
        nuh1w = 0.0, ilap1w = 2,
        nuh2w = 1e-2, ilap2w = 6,
        nuz = 0.0,
        nu_h = 0.0, nu_v = 0.0,
        linear_vert_structure = 0,
        stratification = :constant_N,
        inviscid = false,
        linear = false,
        no_dispersion = false,
        passive_scalar = false,
        ybj_plus = true,
        no_feedback = true,
        fixed_flow = true,
        no_wave_feedback = true,
        N02_sg = 1.0, N12_sg = 0.0, sigma_sg = 1.0, z0_sg = π, alpha_sg = 0.0
    )

    # ========================================================================
    #                       INITIALIZE DISTRIBUTED GRID AND STATE
    # ========================================================================

    if is_root
        println("\nInitializing distributed grid and state...")
    end

    G = QGYBJ.init_mpi_grid(par, mpi_config)
    S = QGYBJ.init_mpi_state(G, mpi_config)
    workspace = QGYBJ.init_mpi_workspace(G, mpi_config)
    plans = QGYBJ.plan_mpi_transforms(G, mpi_config)

    # Get local ranges for initialization
    local_range = QGYBJ.get_local_range_xy(G)

    z = G.z
    dx = G.dx
    dy = G.dy

    # ========================================================================
    #                       SET UP DIPOLE STREAMFUNCTION
    # ========================================================================

    if is_root
        println("Setting up dipole streamfunction...")
    end

    const k_dipole = 1.0
    const psi_amp = 1.0 / k_dipole

    # Initialize psi in real space using local indices
    # Access parent array for direct indexing
    psi_local = parent(S.psi)

    for k_local in axes(psi_local, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(psi_local, 2)
            j_global = local_range[2][j_local]
            y = (j_global - 1) * dy
            for i_local in axes(psi_local, 1)
                i_global = local_range[1][i_local]
                x = (i_global - 1) * dx

                # Dipole streamfunction (in real space for now)
                psi_val = psi_amp * sin(k_dipole * (x - π/2)) * cos(k_dipole * y)
                psi_local[i_local, j_local, k_local] = complex(psi_val)
            end
        end
    end

    # FFT the real-space psi to spectral space
    # Note: S.psi is already a PencilArray, the plans handle the distributed FFT
    QGYBJ.fft_forward!(S.psi, S.psi, plans)

    # Compute q = nabla^2 psi (in spectral space)
    q_local = parent(S.q)
    psi_local = parent(S.psi)

    for k_local in axes(q_local, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(q_local, 2)
            j_global = local_range[2][j_local]
            ky_val = G.ky[j_global]
            for i_local in axes(q_local, 1)
                i_global = local_range[1][i_local]
                kx_val = G.kx[i_global]
                kh2 = kx_val^2 + ky_val^2
                q_local[i_local, j_local, k_local] = -kh2 * psi_local[i_local, j_local, k_local]
            end
        end
    end

    # ========================================================================
    #                       SET UP WAVE INITIAL CONDITION
    # ========================================================================

    if is_root
        println("Setting up wave initial condition...")
    end

    # Surface-confined wave: only (kx,ky)=(0,0) mode is nonzero
    # This mode is on rank 0 (i_global=1, j_global=1)
    z_surface = 2π

    B_local = parent(S.B)

    for k_local in axes(B_local, 3)
        k_global = local_range[3][k_local]
        z_k = z[k_global]

        for j_local in axes(B_local, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(B_local, 1)
                i_global = local_range[1][i_local]

                # Only set (kx,ky)=(0,0) mode
                if i_global == 1 && j_global == 1
                    d_nd = z_surface - z_k
                    d_phys = d_nd * H_scale
                    wave_profile = exp(-(d_phys^2) / (sigma_wave^2))
                    B_local[i_local, j_local, k_local] = u0_nd * wave_profile * (nx * ny)
                else
                    B_local[i_local, j_local, k_local] = 0.0
                end
            end
        end
    end

    # ========================================================================
    #                       DIAGNOSTIC SETUP
    # ========================================================================

    a_ell = QGYBJ.a_ell_ut(par, G)
    L_mask = QGYBJ.dealias_mask(G)

    # Compute velocities from psi
    QGYBJ.compute_velocities!(S, G; plans=plans, params=par, workspace=workspace)

    # ========================================================================
    #                       TIME INTEGRATION
    # ========================================================================

    if is_root
        println("\n" * "="^70)
        println("Starting time integration...")
        println("="^70)
    end

    output_interval_IP = 1.0
    output_interval_steps = round(Int, output_interval_IP * T_inertial_nd / dt)

    # Initial diagnostics (local computation, then MPI reduce)
    local_EB = sum(abs2.(parent(S.B)))
    global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)

    if is_root
        @printf("\nt = 0.0 IP: E_B = %.4e\n", global_EB)
    end

    # First projection step
    if is_root
        println("\nRunning projection step...")
    end
    QGYBJ.first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace)

    # Create states for leapfrog
    Sn = deepcopy(S)
    Snm1 = deepcopy(S)
    Snp1 = deepcopy(S)

    # Main time loop
    for step in 1:nt
        QGYBJ.leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                             a=a_ell, dealias_mask=L_mask, workspace=workspace)

        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        # Output diagnostics
        if step % output_interval_steps == 0
            t_IP = step * dt / T_inertial_nd

            local_EB = sum(abs2.(parent(Sn.B)))
            global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)

            if is_root
                @printf("t = %.1f IP: E_B = %.4e\n", t_IP, global_EB)
            end
        end
    end

    if is_root
        println("\n" * "="^70)
        println("Simulation complete!")
        println("="^70)
    end

    MPI.Finalize()

    return nothing
end

# Run
main()
