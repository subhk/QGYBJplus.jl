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
# NOTE: Wave dispersion CFL requires dt ≤ 2f/N² ≈ 25s for stability.
# Using smaller dt to avoid leapfrog computational mode instability.
const dt = 2.0               # Time step [s] (reduced for stability)
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

    # Set up wave IC in terms of wave amplitude A (velocity envelope)
    # u(t=0) = u0 exp(-z^2/s^2), v(t=0) = 0 (Eq. 4 in paper)
    if is_root; println("Setting up waves..."); end
    A_phys = similar(S.A)
    A_phys_arr = parent(A_phys)
    for k_local in axes(A_phys_arr, 3)
        k_global = local_range[3][k_local]
        depth = G.Lz - G.z[k_global]  # Distance from surface [m]
        wave_profile = exp(-(depth^2) / (surface_layer_depth^2))
        wave_value = complex(u0_wave * wave_profile)
        A_phys_arr[:, :, k_local] .= wave_value
    end
    QGYBJplus.fft_forward!(S.A, A_phys, plans)

    # Apply L⁺ operator to get B = L⁺A (consistent with YBJ+ formulation)
    a_ell = QGYBJplus.a_ell_ut(par, G)
    function apply_Lplus!(Bk, Ak, G, par, a; workspace=nothing)
        nz = G.nz
        need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

        ρ_ut = isdefined(QGYBJplus, :rho_ut) ? QGYBJplus.rho_ut(par, G) : ones(eltype(a), nz)
        ρ_st = isdefined(QGYBJplus, :rho_st) ? QGYBJplus.rho_st(par, G) : ones(eltype(a), nz)

        Δz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
        Δz² = Δz^2

        if need_transpose
            A_z = workspace !== nothing ? workspace.A_z : QGYBJplus.allocate_z_pencil(G, ComplexF64)
            B_z = workspace !== nothing ? workspace.B_z : QGYBJplus.allocate_z_pencil(G, ComplexF64)
            QGYBJplus.transpose_to_z_pencil!(A_z, Ak, G)

            A_z_arr = parent(A_z)
            B_z_arr = parent(B_z)
            nx_local, ny_local, nz_local = size(A_z_arr)

            for j_local in 1:ny_local, i_local in 1:nx_local
                i_global = QGYBJplus.local_to_global_z(i_local, 2, G)
                j_global = QGYBJplus.local_to_global_z(j_local, 3, G)
                kₓ = G.kx[i_global]
                kᵧ = G.ky[j_global]
                kₕ² = kₓ^2 + kᵧ^2

                if nz == 1
                    @inbounds B_z_arr[1, i_local, j_local] = -(kₕ²/4) * A_z_arr[1, i_local, j_local]
                    continue
                end

                @inbounds begin
                    # k = 1
                    d1 = -((ρ_ut[1]*a[1]) / ρ_st[1] + (kₕ²*Δz²)/4)
                    du1 = (ρ_ut[1]*a[1]) / ρ_st[1]
                    B_z_arr[1, i_local, j_local] = (d1*A_z_arr[1, i_local, j_local] +
                                                    du1*A_z_arr[2, i_local, j_local]) / Δz²

                    # interior
                    for k in 2:nz-1
                        dl = (ρ_ut[k-1]*a[k-1]) / ρ_st[k]
                        d  = -(((ρ_ut[k]*a[k] + ρ_ut[k-1]*a[k-1]) / ρ_st[k]) + (kₕ²*Δz²)/4)
                        du = (ρ_ut[k]*a[k]) / ρ_st[k]
                        B_z_arr[k, i_local, j_local] = (dl*A_z_arr[k-1, i_local, j_local] +
                                                        d*A_z_arr[k, i_local, j_local] +
                                                        du*A_z_arr[k+1, i_local, j_local]) / Δz²
                    end

                    # k = nz
                    dl = (ρ_ut[nz-1]*a[nz-1]) / ρ_st[nz]
                    d  = -((ρ_ut[nz-1]*a[nz-1]) / ρ_st[nz] + (kₕ²*Δz²)/4)
                    B_z_arr[nz, i_local, j_local] = (dl*A_z_arr[nz-1, i_local, j_local] +
                                                     d*A_z_arr[nz, i_local, j_local]) / Δz²
                end
            end

            QGYBJplus.transpose_to_xy_pencil!(Bk, B_z, G)
        else
            A_arr = parent(Ak)
            B_arr = parent(Bk)
            nz_local, nx_local, ny_local = size(A_arr)

            for j_local in 1:ny_local, i_local in 1:nx_local
                i_global = QGYBJplus.local_to_global(i_local, 2, G)
                j_global = QGYBJplus.local_to_global(j_local, 3, G)
                kₓ = G.kx[i_global]
                kᵧ = G.ky[j_global]
                kₕ² = kₓ^2 + kᵧ^2

                if nz == 1
                    @inbounds B_arr[1, i_local, j_local] = -(kₕ²/4) * A_arr[1, i_local, j_local]
                    continue
                end

                @inbounds begin
                    d1 = -((ρ_ut[1]*a[1]) / ρ_st[1] + (kₕ²*Δz²)/4)
                    du1 = (ρ_ut[1]*a[1]) / ρ_st[1]
                    B_arr[1, i_local, j_local] = (d1*A_arr[1, i_local, j_local] +
                                                  du1*A_arr[2, i_local, j_local]) / Δz²

                    for k in 2:nz-1
                        dl = (ρ_ut[k-1]*a[k-1]) / ρ_st[k]
                        d  = -(((ρ_ut[k]*a[k] + ρ_ut[k-1]*a[k-1]) / ρ_st[k]) + (kₕ²*Δz²)/4)
                        du = (ρ_ut[k]*a[k]) / ρ_st[k]
                        B_arr[k, i_local, j_local] = (dl*A_arr[k-1, i_local, j_local] +
                                                      d*A_arr[k, i_local, j_local] +
                                                      du*A_arr[k+1, i_local, j_local]) / Δz²
                    end

                    dl = (ρ_ut[nz-1]*a[nz-1]) / ρ_st[nz]
                    d  = -((ρ_ut[nz-1]*a[nz-1]) / ρ_st[nz] + (kₕ²*Δz²)/4)
                    B_arr[nz, i_local, j_local] = (dl*A_arr[nz-1, i_local, j_local] +
                                                   d*A_arr[nz, i_local, j_local]) / Δz²
                end
            end
        end
        return Bk
    end

    apply_Lplus!(S.B, S.A, G, par, a_ell; workspace=workspace)

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
