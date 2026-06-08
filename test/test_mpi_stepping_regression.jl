#=
================================================================================
        MPI stepping regression: serial vs parallel must agree
================================================================================

Runs a few `exp_rk2_step!` steps on (a) a serial grid and (b) a 4-rank 2D-pencil
grid started from the SAME deterministic initial condition (set by global index),
then gathers the parallel result to root and compares. The 2D-decomposition code
paths (vertical velocity, elliptic inversion, dissipation, convolutions) are only
exercised under MPI, so this is the regression net for the 2D-pencil workspace
refactor: an aliasing/buffer bug produces O(field) errors, while serial-FFTW vs
PencilFFTs roundoff stays ~1e-10.

RUN:
    mpiexec -n 4 julia --project test/test_mpi_stepping_regression.jl
================================================================================
=#

using Test
using MPI
using PencilArrays
using QGYBJplus
using QGYBJplus: setup_model, exp_rk2_step!, copy_state, dealias_mask, ExpRK2Workspace,
                 setup_mpi_environment, init_mpi_grid, plan_mpi_transforms, init_mpi_state,
                 a_ell_ut, gather_to_root, get_local_range_xy

const TEST_Lx = 500e3
const TEST_Ly = 500e3
const TEST_Lz = 4000.0
const NSTEPS  = 3

# Deterministic global initial condition (low wavenumbers, small amplitude so the
# solution stays bounded over a few steps). Pure function of (k, i_global, j_global)
# → identical content on serial and parallel regardless of decomposition.
ic_q(k, ig, jg) = (ig <= 3 && jg <= 3) ? 0.01 * cis(0.31ig + 0.71jg + 0.13k) : 0.0im
ic_B(k, ig, jg) = (ig <= 3 && jg <= 3) ? 0.02 * cis(0.17ig + 0.53jg + 0.23k) : 0.0im

# Set IC using explicit global-index ranges (zrange, xrange, yrange) so the index
# convention matches gather_to_root / scatter_from_root exactly.
function fill_ic!(field, ic, ranges)
    arr = parent(field)
    zr, xr, yr = ranges
    @inbounds for (kl, kg) in enumerate(zr), (jl, jg) in enumerate(yr), (il, ig) in enumerate(xr)
        arr[kl, il, jl] = ic(kg, ig, jg)
    end
    return field
end

function step_n!(S, G, par, plans, a, L, K)
    ws = ExpRK2Workspace(S, plans; G=G)
    Sn = S
    Snp1 = copy_state(S)
    for _ in 1:K
        exp_rk2_step!(Snp1, Sn, G, par, plans; a=a, dealias_mask=L, timestep_workspace=ws)
        Sn, Snp1 = Snp1, Sn
    end
    return Sn
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

par = default_params(nx=32, ny=32, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                     ybj_plus=true, no_feedback=false, no_wave_feedback=false,
                     inviscid=false, dt=1e-3, nt=NSTEPS)

# ---- parallel run (all ranks) ----
mpi_config = setup_mpi_environment()
gridp  = init_mpi_grid(par, mpi_config)
plansp = plan_mpi_transforms(gridp, mpi_config)
statep = init_mpi_state(gridp, plansp, mpi_config)
ap = a_ell_ut(par, gridp)
Lp = dealias_mask(gridp)
lr_p = get_local_range_xy(gridp)
fill_ic!(statep.q, ic_q, lr_p)
fill_ic!(statep.L⁺A, ic_B, lr_p)

# Stage 0: gather the IC and compare to the analytic global field. This isolates
# gather/index correctness (test plumbing) from solver correctness. Should be ~0.
q0 = gather_to_root(statep.q, gridp, mpi_config)
B0 = gather_to_root(statep.L⁺A, gridp, mpi_config)
if rank == 0
    nz_, nx_, ny_ = size(q0)
    refq = [ic_q(k, i, j) for k in 1:nz_, i in 1:nx_, j in 1:ny_]
    refB = [ic_B(k, i, j) for k in 1:nz_, i in 1:nx_, j in 1:ny_]
    println("STAGE0 gather(IC): errq=", maximum(abs.(q0 .- refq)),
            " errB=", maximum(abs.(B0 .- refB)))
end

Sp = step_n!(statep, gridp, par, plansp, ap, Lp, NSTEPS)

qp   = gather_to_root(Sp.q,   gridp, mpi_config)
Bp   = gather_to_root(Sp.L⁺A, gridp, mpi_config)
psip = gather_to_root(Sp.psi, gridp, mpi_config)

# ---- serial reference (root only, same global IC) ----
if rank == 0
    Gs, Ss, planss, _ = setup_model(par)
    as = a_ell_ut(par, Gs)
    Ls = dealias_mask(Gs)
    serial_ranges = (1:par.nz, 1:par.nx, 1:par.ny)
    fill_ic!(Ss.q, ic_q, serial_ranges)
    fill_ic!(Ss.L⁺A, ic_B, serial_ranges)
    Ss_final = step_n!(Ss, Gs, par, planss, as, Ls, NSTEPS)

    errq   = maximum(abs.(parent(Ss_final.q)   .- qp))
    errB   = maximum(abs.(parent(Ss_final.L⁺A) .- Bp))
    errpsi = maximum(abs.(parent(Ss_final.psi) .- psip))
    println("REGRESSION (", nprocs, " ranks, ", NSTEPS, " steps): errq=", errq,
            " errB=", errB, " errpsi=", errpsi)

    @testset "Serial vs parallel stepping agree" begin
        @test errq   < 1e-7
        @test errB   < 1e-7
        @test errpsi < 1e-7
    end
end

MPI.Barrier(comm)
MPI.Finalize()
