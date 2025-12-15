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

const n_inertial_periods = 15
const T_inertial = 2π
const dt = 0.001
const nt = round(Int, n_inertial_periods * T_inertial / dt)

const u0_wave = 0.3
const sigma_z = 0.01 * 2π

# ============================================================================
#                       MAIN FUNCTION
# ============================================================================

function main()
    MPI.Init()
    mpi_config = QGYBJ.setup_mpi_environment()
    is_root = mpi_config.is_root

    if is_root
        println("="^70)
        println("Asselin et al. (2020) Dipole - MPI Parallel")
        println("="^70)
        println("Processes: $(mpi_config.nprocs), Topology: $(mpi_config.topology)")
        @printf("Resolution: %d × %d × %d, Duration: %.1f IP\n", nx, ny, nz, n_inertial_periods)
    end

    # Use default_params() - Ro=Bu=1 by default
    par = QGYBJ.default_params(
        nx = nx, ny = ny, nz = nz,
        dt = dt, nt = nt,
        W2F = u0_wave^2,
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

    # Set up dipole: ψ = sin(x - π/2) cos(y)
    if is_root; println("\nSetting up dipole..."); end
    psi_local = parent(S.psi)
    for k_local in axes(psi_local, 3)
        k_global = local_range[3][k_local]
        for j_local in axes(psi_local, 2)
            j_global = local_range[2][j_local]
            y = (j_global - 1) * G.dy
            for i_local in axes(psi_local, 1)
                i_global = local_range[1][i_local]
                x = (i_global - 1) * G.dx
                psi_local[i_local, j_local, k_local] = complex(sin(x - π/2) * cos(y))
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

    # Set up wave IC: surface-confined, (0,0) mode only
    if is_root; println("Setting up waves..."); end
    B_local = parent(S.B)
    for k_local in axes(B_local, 3)
        k_global = local_range[3][k_local]
        depth = 2π - G.z[k_global]
        wave_profile = exp(-(depth^2) / (sigma_z^2))
        for j_local in axes(B_local, 2)
            j_global = local_range[2][j_local]
            for i_local in axes(B_local, 1)
                i_global = local_range[1][i_local]
                if i_global == 1 && j_global == 1
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

    if is_root
        println("\n" * "="^70)
        println("Starting time integration...")
        println("="^70)
    end

    output_interval = round(Int, T_inertial / dt)

    local_EB = sum(abs2.(parent(S.B)))
    global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)
    if is_root; @printf("\nt = 0.0 IP: E_B = %.4e\n", global_EB / (nx*ny*nz)); end

    QGYBJ.first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask, workspace=workspace)

    Sn, Snm1, Snp1 = deepcopy(S), deepcopy(S), deepcopy(S)

    for step in 1:nt
        QGYBJ.leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                             a=a_ell, dealias_mask=L_mask, workspace=workspace)
        Snm1, Sn, Snp1 = Sn, Snp1, Snm1

        if step % output_interval == 0
            t_IP = step * dt / T_inertial
            local_EB = sum(abs2.(parent(Sn.B)))
            global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)
            if is_root; @printf("t = %.1f IP: E_B = %.4e\n", t_IP, global_EB / (nx*ny*nz)); end
        end
    end

    if is_root
        println("\n" * "="^70)
        println("Simulation complete!")
        println("="^70)
    end

    MPI.Finalize()
end

main()
