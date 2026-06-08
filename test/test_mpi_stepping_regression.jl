#=
================================================================================
        MPI stepping regression for the 2D-pencil workspace refactor
================================================================================

The 2D-decomposition code paths (vertical velocity, elliptic inversion,
dissipation, convolutions) only run under genuine multi-rank MPI, so the serial
test suite cannot cover them. This file is their regression net.

Two layout-free checks (no gather, so independent of index conventions):

  1. REFACTOR EQUIVALENCE — step the SAME parallel IC twice: once with fresh
     per-call allocations (no timestep_workspace) and once with the reusable
     workspace (z-pencil scratch). The refactor is a pure performance change, so
     the two must agree bit-for-bit. Compared on local arrays + Allreduce(MAX).
     An aliasing bug in the workspace reuse would make these differ.

  2. SERIAL vs PARALLEL global norms — Parseval norms ∑|·|² are decomposition
     invariant, so serial and 4-rank results must agree to FFT roundoff. Catches
     gross distributed-solver bugs without depending on gather index order.

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
                 a_ell_ut, scatter_from_root

const TEST_Lx = 500e3
const TEST_Ly = 500e3
const TEST_Lz = 4000.0
const NSTEPS  = 3

# Deterministic global IC (low wavenumbers, small amplitude). Pure function of
# global (k, i, j) → identical content regardless of decomposition.
ic_q(k, ig, jg) = (ig <= 3 && jg <= 3) ? 0.01 * cis(0.31ig + 0.71jg + 0.13k) : 0.0im
ic_B(k, ig, jg) = (ig <= 3 && jg <= 3) ? 0.02 * cis(0.17ig + 0.53jg + 0.23k) : 0.0im

global_field(ic, par) = ComplexF64[ic(k, i, j) for k in 1:par.nz, i in 1:par.nx, j in 1:par.ny]

gnorm2(field, comm) = MPI.Allreduce(sum(abs2, parent(field)), MPI.SUM, comm)

function step_n!(Sn, G, par, plans, a, L, K; ws=nothing)
    Snp1 = copy_state(Sn)
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

mpi_config = setup_mpi_environment()
gridp  = init_mpi_grid(par, mpi_config)
plansp = plan_mpi_transforms(gridp, mpi_config)
ap = a_ell_ut(par, gridp)
Lp = dealias_mask(gridp)

# Build the global IC on root once; scatter to the parallel state's pencil so the
# parallel and serial runs start from an identical global field.
glob_q = rank == 0 ? global_field(ic_q, par) : nothing
glob_B = rank == 0 ? global_field(ic_B, par) : nothing

function mk_state()
    st = init_mpi_state(gridp, plansp, mpi_config)
    parent(st.q)   .= parent(scatter_from_root(glob_q, gridp, mpi_config; plans=plansp))
    parent(st.L⁺A) .= parent(scatter_from_root(glob_B, gridp, mpi_config; plans=plansp))
    st
end

# IC-norm diagnostic: confirm the parallel IC matches the analytic global IC norm
# before any stepping (isolates IC/index setup from the solver).
nq_ic_p = gnorm2(mk_state().q, comm)
nB_ic_p = gnorm2(mk_state().L⁺A, comm)
if rank == 0
    println("IC |q|²: parallel=", nq_ic_p, " analytic=", sum(abs2, glob_q),
            " | IC |B|²: parallel=", nB_ic_p, " analytic=", sum(abs2, glob_B))
end

# ---- Check 1: refactor equivalence (fresh allocations vs reusable workspace) ----
SA = step_n!(mk_state(), gridp, par, plansp, ap, Lp, NSTEPS; ws=nothing)
stB = mk_state()
wsB = ExpRK2Workspace(stB, plansp; G=gridp)
SB = step_n!(stB, gridp, par, plansp, ap, Lp, NSTEPS; ws=wsB)

dq   = MPI.Allreduce(maximum(abs.(parent(SA.q)   .- parent(SB.q))),   MPI.MAX, comm)
dB   = MPI.Allreduce(maximum(abs.(parent(SA.L⁺A) .- parent(SB.L⁺A))), MPI.MAX, comm)
dpsi = MPI.Allreduce(maximum(abs.(parent(SA.psi) .- parent(SB.psi))), MPI.MAX, comm)

# ---- Check 3: the workspace reduces per-step allocations in the 2D paths ----
function one_step_alloc(use_ws)
    st = mk_state()
    Snp1 = copy_state(st)
    ws = use_ws ? ExpRK2Workspace(st, plansp; G=gridp) : nothing
    exp_rk2_step!(Snp1, st, gridp, par, plansp; a=ap, dealias_mask=Lp, timestep_workspace=ws)  # warmup
    return @allocated exp_rk2_step!(Snp1, st, gridp, par, plansp; a=ap, dealias_mask=Lp, timestep_workspace=ws)
end
allocWS   = MPI.Allreduce(one_step_alloc(true),  MPI.SUM, comm)
allocFresh = MPI.Allreduce(one_step_alloc(false), MPI.SUM, comm)

# ---- Check 2: serial vs parallel global norms (use the workspace result SB) ----
nqp = gnorm2(SB.q, comm)
nBp = gnorm2(SB.L⁺A, comm)
npp = gnorm2(SB.psi, comm)

if rank == 0
    Gs, Ss, planss, _ = setup_model(par)
    as = a_ell_ut(par, Gs)
    Ls = dealias_mask(Gs)
    parent(Ss.q)   .= glob_q
    parent(Ss.L⁺A) .= glob_B
    wss = ExpRK2Workspace(Ss, planss; G=Gs)
    Ssf = step_n!(Ss, Gs, par, planss, as, Ls, NSTEPS; ws=wss)
    nqs = sum(abs2, parent(Ssf.q))
    nBs = sum(abs2, parent(Ssf.L⁺A))
    nps = sum(abs2, parent(Ssf.psi))

    relnorm(a, b) = abs(a - b) / max(abs(b), eps())
    rq, rB, rp = relnorm(nqp, nqs), relnorm(nBp, nBs), relnorm(npp, nps)
    println("EQUIV (ws vs fresh, local): dq=", dq, " dB=", dB, " dpsi=", dpsi)
    println("NORMS serial/parallel rel diff: q=", rq, " B=", rB, " psi=", rp,
            "   (|q|²ser=", nqs, " par=", nqp, ")")
    println("ALLOC/step (4 ranks, bytes): fresh=", allocFresh, " workspace=", allocWS,
            "  (", round(100*(1 - allocWS/allocFresh), digits=1), "% less)")

    @testset "2D-pencil workspace refactor" begin
        @testset "workspace reuse == fresh allocation (bit-identical)" begin
            @test dq   < 1e-12
            @test dB   < 1e-12
            @test dpsi < 1e-12
        end
        @testset "serial and parallel agree (global norms)" begin
            @test rq < 1e-6
            @test rB < 1e-6
            @test rp < 1e-6
        end
        @testset "workspace cuts per-step allocations" begin
            @test allocWS < allocFresh
        end
    end
end

MPI.Barrier(comm)
MPI.Finalize()
