#=
================================================================================
                    MPI stepping regression tests
================================================================================

The distributed vertical operators and FFT paths require a genuine multi-rank
run. This test checks:

  1. Reusable MPI workspace execution agrees with allocation-on-demand.
  2. Distributed and serial stepping produce the same global spectral norms.

RUN:
    mpiexec -n 4 julia --project=. test/test_mpi_stepping_regression.jl
================================================================================
=#

using Test
using FFTW
using MPI
using QGYBJplus
using QGYBJplus: setup_model, copy_fields, dealias_mask,
                 exp_rk2_step!, ExpRK2Workspace,
                 setup_mpi_environment, init_mpi_grid, plan_mpi_transforms,
                 init_mpi_state, init_mpi_workspace, a_ell_ut, scatter_from_root

const TEST_Lx = 2pi
const TEST_Ly = 2pi
const TEST_Lz = 1.0
const NSTEPS = 3

gnorm2(field, comm) = MPI.Allreduce(sum(abs2, parent(field)), MPI.SUM, comm)

function step_n!(initial, G, par, plans, a, L, nsteps;
                 workspace=nothing, reuse_timestep_workspace=false)
    nsteps > 0 || return copy_fields(initial)

    Sn = copy_fields(initial)
    Snp1 = copy_fields(Sn)
    timestep_workspace = reuse_timestep_workspace ? ExpRK2Workspace(Sn, plans; G=G) : nothing
    for _ in 1:nsteps
        exp_rk2_step!(Snp1, Sn, G, par, plans;
                      a=a, dealias_mask=L, workspace=workspace,
                      timestep_workspace=timestep_workspace)
        Sn, Snp1 = Snp1, Sn
    end
    return Sn
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

try
    par = default_params(nx=8, ny=8, nz=8,
                         Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true,
                         no_feedback=false, no_wave_feedback=false,
                         inviscid=false, dt=1e-3, nt=NSTEPS)

    mpi_config = setup_mpi_environment()
    gridp = init_mpi_grid(par, mpi_config)
    plansp = plan_mpi_transforms(gridp, mpi_config)
    ap = a_ell_ut(par, gridp)
    Lp = dealias_mask(gridp)

    # Generate deterministic global initial conditions in physical space and
    # transform them on root. q is real-valued; the wave envelope is complex.
    glob_q = nothing
    glob_B = nothing
    if rank == 0
        q_phys = zeros(Float64, par.nz, par.nx, par.ny)
        B_phys = zeros(ComplexF64, par.nz, par.nx, par.ny)
        for k in 1:par.nz, j in 1:par.ny, i in 1:par.nx
            x = 2pi * (i - 1) / par.nx
            y = 2pi * (j - 1) / par.ny
            vertical = sin(pi * (k - 0.5) / par.nz)
            q_phys[k, i, j] = 1e-2 * vertical * sin(x) * cos(y)
            B_phys[k, i, j] = 2e-2 * vertical * (cos(x) + im * sin(y))
        end
        glob_q = FFTW.fft(q_phys, (2, 3))
        glob_B = FFTW.fft(B_phys, (2, 3))
    end

    function make_parallel_state()
        state = init_mpi_state(gridp, plansp, mpi_config)
        state.q .= scatter_from_root(glob_q, gridp, mpi_config; plans=plansp)
        state.B .= scatter_from_root(glob_B, gridp, mpi_config; plans=plansp)
        return state
    end

    fresh = step_n!(make_parallel_state(), gridp, par, plansp, ap, Lp, NSTEPS)
    workspace = init_mpi_workspace(gridp, mpi_config)
    reused = step_n!(make_parallel_state(), gridp, par, plansp, ap, Lp, NSTEPS;
                     workspace=workspace, reuse_timestep_workspace=true)

    dq = MPI.Allreduce(maximum(abs.(parent(fresh.q) .- parent(reused.q))), MPI.MAX, comm)
    dB = MPI.Allreduce(maximum(abs.(parent(fresh.B) .- parent(reused.B))), MPI.MAX, comm)
    dpsi = MPI.Allreduce(maximum(abs.(parent(fresh.psi) .- parent(reused.psi))), MPI.MAX, comm)

    nqp = gnorm2(reused.q, comm)
    nBp = gnorm2(reused.B, comm)
    npsip = gnorm2(reused.psi, comm)

    nqs = 0.0
    nBs = 0.0
    npsis = 0.0
    if rank == 0
        Gs, Ss, planss, as = setup_model(par)
        Ss.q .= glob_q
        Ss.B .= glob_B
        serial = step_n!(Ss, Gs, par, planss, as, dealias_mask(Gs), NSTEPS)
        nqs = sum(abs2, serial.q)
        nBs = sum(abs2, serial.B)
        npsis = sum(abs2, serial.psi)
    end
    nqs = MPI.bcast(nqs, 0, comm)
    nBs = MPI.bcast(nBs, 0, comm)
    npsis = MPI.bcast(npsis, 0, comm)

    relative_error(a, b) = abs(a - b) / max(abs(b), eps())

    @testset "MPI stepping" begin
        @test nprocs > 1
        @test dq < 1e-12
        @test dB < 1e-12
        @test dpsi < 1e-12
        @test relative_error(nqp, nqs) < 1e-6
        @test relative_error(nBp, nBs) < 1e-6
        @test relative_error(npsip, npsis) < 1e-6
    end
finally
    if !MPI.Finalized()
        MPI.Finalize()
    end
end
