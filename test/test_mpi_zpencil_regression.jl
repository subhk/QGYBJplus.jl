#=
================================================================================
    Distributed-z (z-pencil) regression for the secondary operator paths
================================================================================

Spectral model fields live on the PencilFFTs *output* pencil. With a 2D process
topology (px > 1, e.g. `-n 4` -> (2, 2)) that pencil decomposes dimension 1,
so **z is distributed** and `size(parent(S.q), 1) < nz` on every rank.

`step!` and `compute_velocities!` already handle this. The operators exercised
here did not: they either asserted `nz_local == nz`, or ran vertical finite
differences straight over `1:nz_local` (treating each rank slab edge as a
physical boundary), or indexed a global `nz`-length N² / a_ell profile with a
local `k`. Those failures are invisible in serial and invisible at one rank.

Reference model: every operator is also evaluated on the root rank through the
fully serial code path (`decomposition === nothing`, plain `Array` fields,
FFTW plans) starting from the *gathered global* input fields. The distributed
answer must equal the serial answer to roundoff, independent of rank count.

A non-constant N² profile is deliberate: with constant N² the "local k into a
global profile" bug is silent because every profile entry is identical.

RUN:
    mpiexec -n 1 julia --project=. test/test_mpi_zpencil_regression.jl
    mpiexec -n 4 julia --project=. test/test_mpi_zpencil_regression.jl
================================================================================
=#

using Test
using MPI
using QGYBJplus

const Q = QGYBJplus

const NX = 16
const NY = 16
const NZ = 8
const LX = 2π
const LY = 2π
const LZ = 1.0
const FCOR = 1.0

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

# Strong vertical variation so a local/global profile-index slip cannot cancel.
stratification() = ExponentialProfile(2.0, 0.25, 0.4)

build_grid() = RectilinearGrid(size=(NX, NY, NZ), extent=(LX, LY, LZ))

model_physics() = Q.ModelPhysics(FPlane(f=FCOR), stratification(),
                                 EvolvingFlow(), WaveMeanFeedback(), YBJPlus())

model_numerics() = Q.ModelNumerics(
    HorizontalHyperdiffusivity(flow=FlowHyperdiffusivity(coefficient=0),
                               wave=WaveHyperdiffusivity(coefficient=0)),
    VerticalDiffusivity(), Dissipative(), NonlinearDynamics(), Dispersive())

"""Collective: every rank must call this together."""
function build_model(grid)
    physics = model_physics()
    numerics = model_numerics()
    return QGYBJModel(
        grid=grid,
        coriolis=physics.coriolis,
        stratification=physics.stratification,
        closure=numerics.closure,
        flow=physics.flow,
        feedback=physics.feedback,
        formulation=physics.formulation,
        verbose=false,
    )
end

"""Deterministic global spectral field with genuine vertical structure."""
function global_spectral_field(seed::Int)
    field = zeros(ComplexF64, NZ, NX, NY)
    # A handful of modes inside the radial 2/3 cutoff, each with its own
    # vertical profile so vertical derivatives are non-trivial at every level.
    modes = ((2, 1), (1, 2), (2, 2), (3, 2), (2, 3))
    for (index, (i, j)) in enumerate(modes)
        for k in 1:NZ
            z = (k - 0.5) / NZ
            amplitude = cospi(index * z) + 0.5 * sinpi((index + seed) * z)
            phase = 0.37 * index + 0.11 * seed
            value = amplitude * cis(phase)
            field[k, i, j] += value
            # Keep the field Hermitian so its physical-space image is real.
            ci = i == 1 ? 1 : NX - i + 2
            cj = j == 1 ? 1 : NY - j + 2
            field[k, ci, cj] += conj(value)
        end
    end
    return field
end

"""Serial (non-distributed) twin of the runtime geometry."""
function serial_context(grid)
    geometry = Q.RuntimeGeometry(grid, grid.kh2, nothing)
    fields = Q.ModelFields(Float64, (NZ, NX, NY))
    plans = Q.plan_transforms!(geometry)
    N² = Float64.(compute_stratification_profile(stratification(), grid))
    return (geometry=geometry, fields=fields, plans=plans,
            N²=N², a=a_ell_from_N2(N², FPlane(f=FCOR)), mask=dealias_mask(grid))
end

scatter!(destination, global_field, model) =
    (destination .= scatter_from_root(global_field, model.runtime.geometry,
                                      model.runtime.mpi;
                                      plans=model.runtime.plans);
     destination)

gather(array, model) =
    gather_to_root(array, model.runtime.geometry, model.runtime.mpi)

"""Relative sup-norm difference; `nothing` off root."""
function relative_error(distributed, reference)
    distributed === nothing && return nothing
    scale = max(maximum(abs, reference), eps())
    return maximum(abs, Array(distributed) .- reference) / scale
end

const TOLERANCE = 1e-11

"""`true` when `thunk` runs to completion; rethrows nothing, reports the error."""
function runs_without_error(thunk)
    try
        thunk()
        return true
    catch exception
        RANK == 0 && @error "operator threw" exception
        return false
    end
end

@testset "Distributed-z operator paths ($NPROCS ranks)" begin
    grid = build_grid()
    q_global = global_spectral_field(1)
    B_global = global_spectral_field(2)
    ψ_global = global_spectral_field(3)

    @testset "set! derives q from ψ on a distributed-z pencil" begin
        model = build_model(grid)
        try
            # The README quickstart shape must not throw on a distributed-z pencil.
            @test runs_without_error() do
                set!(model; ψ=(x, y, z) -> sinpi(2x / LX) * cospi(2y / LY) * (1 + z / LZ),
                     verbose=false)
            end

            reference = serial_context(grid)
            ψ_distributed = gather(model.fields.psi, model)
            q_distributed = gather(model.fields.q, model)
            if RANK == 0
                copyto!(reference.fields.psi, ψ_distributed)
                Q.compute_q_from_psi!(reference.fields.q, reference.fields.psi,
                                      reference.geometry, reference.a, grid.dz)
                @test relative_error(q_distributed, reference.fields.q) < TOLERANCE
            end
        finally
            finalize_model!(model)
        end
    end

    @testset "compute_ybj_vertical_velocity! matches the serial path" begin
        model = build_model(grid)
        try
            scatter!(model.fields.B, B_global, model)
            @test runs_without_error(() -> Q.compute_ybj_vertical_velocity!(model))

            reference = serial_context(grid)
            w_distributed = gather(model.fields.w, model)
            if RANK == 0
                copyto!(reference.fields.B, B_global)
                Q.compute_ybj_vertical_velocity!(
                    reference.fields, reference.geometry, reference.plans;
                    f=FCOR, N2=first(reference.N²), N2_profile=reference.N²)
                @test relative_error(w_distributed, reference.fields.w) < TOLERANCE
            end
        finally
            finalize_model!(model)
        end
    end

    @testset "compute_total_velocities! matches the serial path" begin
        model = build_model(grid)
        try
            scatter!(model.fields.q, q_global, model)
            scatter!(model.fields.B, B_global, model)
            Q.invert_q_to_psi!(model)
            Q.invert_B_to_A!(model)
            Q.compute_total_velocities!(model)

            reference = serial_context(grid)
            u_distributed = gather(model.fields.u, model)
            v_distributed = gather(model.fields.v, model)
            w_distributed = gather(model.fields.w, model)
            if RANK == 0
                copyto!(reference.fields.q, q_global)
                copyto!(reference.fields.B, B_global)
                Q.invert_q_to_psi!(reference.fields, reference.geometry;
                                   a=reference.a)
                Q.invert_B_to_A!(reference.fields, reference.geometry, reference.a)
                Q.compute_total_velocities!(
                    reference.fields, reference.geometry;
                    plans=reference.plans, f=FCOR, N2=first(reference.N²),
                    N2_profile=reference.N², dealias_mask=reference.mask)
                @test relative_error(u_distributed, reference.fields.u) < TOLERANCE
                @test relative_error(v_distributed, reference.fields.v) < TOLERANCE
                @test relative_error(w_distributed, reference.fields.w) < TOLERANCE
            end
        finally
            finalize_model!(model)
        end
    end

    @testset "step! and compute_velocities! stay rank-independent" begin
        model = build_model(grid)
        try
            scatter!(model.fields.q, q_global, model)
            scatter!(model.fields.B, B_global, model)
            scatter!(model.fields.psi, ψ_global, model)
            step!(model, ExponentialRungeKutta2(Δt=1e-3))

            reference = serial_context(grid)
            q_distributed = gather(model.fields.q, model)
            B_distributed = gather(model.fields.B, model)
            u_distributed = gather(model.fields.u, model)
            w_distributed = gather(model.fields.w, model)
            if RANK == 0
                copyto!(reference.fields.q, q_global)
                copyto!(reference.fields.B, B_global)
                copyto!(reference.fields.psi, ψ_global)
                # Build the stepper options directly: constructing a model here
                # would run collectives on the root rank alone and deadlock.
                options = Q.ETDModelOptions(model_physics(), model_numerics())
                next = Q.copy_fields(reference.fields)
                Q._advance_etdrk2!(next, reference.fields, reference.geometry,
                                   options, reference.plans;
                                   Δt=1e-3, a=reference.a,
                                   dealias_mask=reference.mask,
                                   N2_profile=reference.N²)
                @test relative_error(q_distributed, next.q) < TOLERANCE
                @test relative_error(B_distributed, next.B) < TOLERANCE
                @test relative_error(u_distributed, next.u) < TOLERANCE
                @test relative_error(w_distributed, next.w) < TOLERANCE
            end
        finally
            finalize_model!(model)
        end
    end
end

MPI.Barrier(COMM)
