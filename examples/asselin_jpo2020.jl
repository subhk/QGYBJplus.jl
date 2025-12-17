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
using Dates
using NCDatasets

# ============================================================================
#                       SIMULATION PARAMETERS
# ============================================================================

const nx = 128
const ny = 128
const nz = 64

# Physical parameters from Asselin et al. (2020)
const f₀ = 1.24e-4           # Coriolis parameter [s⁻¹]
const N² = 1e-5              # Buoyancy frequency squared [s⁻²]

const n_inertial_periods = 15
const T_inertial = 2π / f₀   # Inertial period = 2π/f [s]
const dt = 100.0             # Time step [s]
const nt = round(Int, n_inertial_periods * T_inertial / dt)

const u0_wave = 0.3
const sigma_z = 0.01 * 2π

# Output settings
const output_dir = "output_asselin"
const save_interval_IP = 1.0  # Save every 1 inertial period

# ============================================================================
#                       MAIN FUNCTION
# ============================================================================

"""
    save_state_netcdf(S, G, plans, time, file_count, mpi_config)

Save flow (psi) and wave (LAr, LAi) variables to NetCDF file.
Uses gather-to-root approach for parallel I/O.
"""
function save_state_netcdf(S, G, plans, time, file_count, mpi_config)
    is_root = mpi_config.is_root

    # Gather fields to root process
    psi_global = QGYBJ.gather_to_root(S.psi, G, mpi_config)
    B_global = QGYBJ.gather_to_root(S.B, G, mpi_config)

    if is_root && psi_global !== nothing && B_global !== nothing
        # Create serial FFT plans for full domain
        temp_plans = QGYBJ.plan_transforms!(G)

        # Convert spectral fields to physical space
        psi_phys = zeros(ComplexF64, G.nx, G.ny, G.nz)
        B_phys = zeros(ComplexF64, G.nx, G.ny, G.nz)

        QGYBJ.fft_backward!(psi_phys, psi_global, temp_plans)
        QGYBJ.fft_backward!(B_phys, B_global, temp_plans)

        # Write to NetCDF
        filename = joinpath(output_dir, @sprintf("state%04d.nc", file_count))

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["x"] = G.nx
            ds.dim["y"] = G.ny
            ds.dim["z"] = G.nz

            # Coordinate variables
            x_var = defVar(ds, "x", Float64, ("x",))
            y_var = defVar(ds, "y", Float64, ("y",))
            z_var = defVar(ds, "z", Float64, ("z",))
            time_var = defVar(ds, "time", Float64, ())

            x_var[:] = collect(0:G.dx:(2π - G.dx))
            y_var[:] = collect(0:G.dy:(2π - G.dy))
            z_var[:] = G.z
            time_var[] = time

            # Flow variable: streamfunction psi
            psi_var = defVar(ds, "psi", Float64, ("x", "y", "z"))
            psi_var[:,:,:] = real.(psi_phys)
            psi_var.attrib["units"] = "m²/s"
            psi_var.attrib["long_name"] = "streamfunction"

            # Wave variables: L+A real and imaginary parts
            LAr_var = defVar(ds, "LAr", Float64, ("x", "y", "z"))
            LAi_var = defVar(ds, "LAi", Float64, ("x", "y", "z"))
            LAr_var[:,:,:] = real.(B_phys)
            LAi_var[:,:,:] = imag.(B_phys)
            LAr_var.attrib["units"] = "m/s"
            LAr_var.attrib["long_name"] = "wave envelope real part (L+A)"
            LAi_var.attrib["units"] = "m/s"
            LAi_var.attrib["long_name"] = "wave envelope imaginary part (L+A)"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Simulation: Asselin et al. (2020) Dipole"
            ds.attrib["created_at"] = string(now())
            ds.attrib["time_IP"] = time / T_inertial
            ds.attrib["nx"] = G.nx
            ds.attrib["ny"] = G.ny
            ds.attrib["nz"] = G.nz
            ds.attrib["f0"] = f₀
            ds.attrib["N2"] = N²
        end

        @printf("  Saved: %s (t = %.2f IP)\n", filename, time / T_inertial)
    end

    MPI.Barrier(mpi_config.comm)
end

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

    # Parameters matching Asselin et al. (2020)
    par = QGYBJ.default_params(
        nx = nx, ny = ny, nz = nz,
        dt = dt, nt = nt,
        f₀ = f₀,               # Coriolis parameter [s⁻¹]
        N² = N²,               # Buoyancy frequency squared [s⁻²]
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
    save_interval = round(Int, save_interval_IP * T_inertial / dt)
    file_count = 0

    # Initial diagnostics
    local_EB = sum(abs2.(parent(S.B)))
    global_EB = QGYBJ.mpi_reduce_sum(local_EB, mpi_config)
    if is_root; @printf("\nt = 0.0 IP: E_B = %.4e\n", global_EB / (nx*ny*nz)); end

    # Save initial state
    if is_root; println("\nSaving initial state..."); end
    save_state_netcdf(S, G, plans, 0.0, file_count, mpi_config)
    file_count += 1

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

        # Save state at specified interval
        if step % save_interval == 0
            save_state_netcdf(Sn, G, plans, current_time, file_count, mpi_config)
            file_count += 1
        end
    end

    if is_root
        println("\n" * "="^70)
        println("Simulation complete!")
        println("="^70)
        println("Output files saved to: $output_dir/")
        println("  - state0000.nc to state$(lpad(file_count-1, 4, '0')).nc")
        println("  - Variables: psi (flow), LAr, LAi (waves)")
        println("="^70)
    end

    MPI.Finalize()
end

main()
