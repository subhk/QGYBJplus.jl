#=
================================================================================
        MPI particle advection regression: periodic seams + migration
================================================================================

The parallel particle paths (halo-based interpolation, cross-rank migration,
periodic wrap-around between edge ranks) only run under multi-rank MPI, so the
serial suite cannot cover them. This file is their regression net.

Design: a steady barotropic streamfunction
    ψ(x,y) = -U0·Ly/(2π)·sin(2πy/Ly) + V0·Lx/(2π)·sin(2πx/Lx)
gives u = U0·cos(2πy/Ly), v = V0·cos(2πx/Lx) — rows of particles translate in
x at row-dependent speed, columns translate in y. Particles starting on the
x=0 / y=0 lines in counter-flowing rows/columns cross the periodic seam
backwards within a few steps; interior particles cross internal rank
boundaries. The flow is frozen (no model stepping), so a serial reference
advection on the gathered global velocity fields must match the distributed
result particle-by-particle.

Checks:
  1. CONSERVATION — every particle id exists on exactly one rank after each
     step (migration loses nothing, duplicates nothing).
  2. SERIAL ↔ PARALLEL — final (x, y) per particle id agree with a serial
     reference (same trilinear scheme on the global fields) in the periodic
     metric. Catches halo-interpolation and migration-targeting bugs.
  3. SEAM CROSSING — the reference actually wrapped at least one particle in
     x and in y, so check 2 genuinely exercises the periodic seam.

RUN:
    mpiexec -n 4 julia --project test/test_mpi_particles_periodic.jl
================================================================================
=#

using Test
using MPI
using PencilArrays
using QGYBJplus
using QGYBJplus: setup_model, setup_mpi_environment, init_mpi_grid,
                 plan_mpi_transforms, init_mpi_state, scatter_from_root,
                 fft_forward!, compute_velocities!

const UPA = QGYBJplus.UnifiedParticleAdvection

const Lx = 500e3
const Ly = 500e3
const Lz = 4000.0
const NX, NY, NZ = 32, 32, 8
const NP = 8                  # particles per direction (NP² total)
const NSTEPS = 25
const DT = 2500.0             # seconds; max drift = U0·DT·NSTEPS = 62.5 km
const U0 = 1.0                # m/s
const V0 = 0.8                # m/s

ψ_phys(x, y) = -U0 * Ly / (2π) * sin(2π * y / Ly) + V0 * Lx / (2π) * sin(2π * x / Lx)

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

par = default_params(nx=NX, ny=NY, nz=NZ, Lx=Lx, Ly=Ly, Lz=Lz)
z_mid = -Lz / 2

mpi_config = setup_mpi_environment()
gridp = init_mpi_grid(par, mpi_config)
plansp = plan_mpi_transforms(gridp, mpi_config)
Sp = init_mpi_state(gridp, plansp, mpi_config)

# ---- Root: global spectral ψ plus the serial reference velocity fields ----
glob_psik = nothing
u_ref = v_ref = w_ref = nothing
dz1 = 0.0
if rank == 0
    Gs, Ss, planss, _ = setup_model(par)
    phys = zeros(ComplexF64, NZ, NX, NY)
    for j in 1:NY, i in 1:NX, k in 1:NZ
        phys[k, i, j] = ψ_phys((i - 1) * Lx / NX, (j - 1) * Ly / NY)
    end
    glob_psik = similar(phys)
    fft_forward!(glob_psik, phys, planss)
    parent(Ss.psi) .= glob_psik
    compute_velocities!(Ss, Gs; plans=planss, params=par, compute_w=false)
    u_ref = copy(parent(Ss.u))
    v_ref = copy(parent(Ss.v))
    w_ref = zeros(size(u_ref))
    dz1 = Gs.dz[1]
end

parent(Sp.psi) .= parent(scatter_from_root(glob_psik, gridp, mpi_config; plans=plansp))

# ---- Parallel tracker: particles distributed by position, global ids ----
cfg = ParticleConfig{Float64}(x_max=Lx, y_max=Ly, z_level=z_mid,
                              nx_particles=NP, ny_particles=NP,
                              use_3d_advection=false,
                              interpolation_method=QGYBJplus.TRILINEAR)
tracker = ParticleTracker(cfg, gridp, mpi_config; plans=plansp)
initialize_particles!(tracker, cfg)

NPtot = NP * NP
np_global_init = MPI.Allreduce(tracker.particles.np, MPI.SUM, comm)

count_ok = true
for _ in 1:NSTEPS
    advect_particles!(tracker, Sp, gridp, DT; params=par)
    np_global = MPI.Allreduce(tracker.particles.np, MPI.SUM, comm)
    global count_ok = count_ok && (np_global == NPtot)
end

# Collect final (x, y) keyed by global particle id. After migration each id
# lives on exactly one rank, so a SUM reduction reassembles the global set and
# the per-id counter doubles as a loss/duplication check.
xs_par = zeros(NPtot); ys_par = zeros(NPtot); cnt = zeros(Int, NPtot)
let p = tracker.particles
    for n in 1:p.np
        xs_par[p.id[n]] += p.x[n]
        ys_par[p.id[n]] += p.y[n]
        cnt[p.id[n]] += 1
    end
end
MPI.Allreduce!(xs_par, +, comm)
MPI.Allreduce!(ys_par, +, comm)
MPI.Allreduce!(cnt, +, comm)

if rank == 0
    # ---- Serial reference: same trilinear scheme on the global fields ----
    grid_info = (dx=Lx / NX, dy=Ly / NY, dz=dz1, Lx=Lx, Ly=Ly, Lz=Lz,
                 z_min=-Lz, z_max=0.0, z0=-Lz + dz1 / 2)
    bcs = (periodic_x=true, periodic_y=true, periodic_z=false)

    dxp = Lx / NP; dyp = Ly / NP
    # id = (j-1)*NP + i (i fastest) — must match initialize_particles_parallel!
    xs = Float64[(i - 1) * dxp for j in 1:NP for i in 1:NP]
    ys = Float64[(j - 1) * dyp for j in 1:NP for i in 1:NP]
    xs_un = copy(xs); ys_un = copy(ys)   # unwrapped, to detect seam crossings

    for _ in 1:NSTEPS
        for n in 1:NPtot
            u, v, _ = interpolate_velocity_advanced(xs[n], ys[n], z_mid,
                                                    u_ref, v_ref, w_ref,
                                                    grid_info, bcs, QGYBJplus.TRILINEAR)
            xs[n] = mod(xs[n] + DT * u, Lx)
            ys[n] = mod(ys[n] + DT * v, Ly)
            xs_un[n] += DT * u
            ys_un[n] += DT * v
        end
    end

    crossed_x = count(x -> abs(fld(x, Lx)) >= 1, xs_un)
    crossed_y = count(y -> abs(fld(y, Ly)) >= 1, ys_un)

    perid_dist(a, b, L) = abs(mod(a - b + L / 2, L) - L / 2)
    max_dx = maximum(perid_dist(xs_par[n], xs[n], Lx) for n in 1:NPtot)
    max_dy = maximum(perid_dist(ys_par[n], ys[n], Ly) for n in 1:NPtot)

    println("PARTICLES init np=", np_global_init, "/", NPtot,
            " | seam crossings: x=", crossed_x, " y=", crossed_y)
    println("MAX serial↔parallel mismatch (periodic metric, m): dx=", max_dx, " dy=", max_dy)

    @testset "MPI particle advection across periodic seams" begin
        @testset "particle conservation under migration" begin
            @test np_global_init == NPtot
            @test count_ok
            @test all(cnt .== 1)
        end
        @testset "positions stay in the periodic domain" begin
            @test all(x -> 0.0 <= x < Lx, xs_par)
            @test all(y -> 0.0 <= y < Ly, ys_par)
        end
        @testset "seam crossings actually happened" begin
            @test crossed_x >= 1
            @test crossed_y >= 1
        end
        @testset "parallel matches serial reference per particle" begin
            @test max_dx < 1e-6
            @test max_dy < 1e-6
        end
    end
end

MPI.Barrier(comm)
MPI.Finalize()
