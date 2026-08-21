"""
Composition-first MPI stepping regression.

Run with one or more ranks, for example:

    mpiexec -n 2 julia --project=. test/test_mpi_stepping_regression.jl

The fixed-flow, passive-wave configuration has a closed-form solution: each
wave Fourier coefficient is damped by the ETD integrating factor. This gives
the same rank-independent reference without constructing a second legacy
serial data model.
"""

using Test
using MPI
using QGYBJplus

const NX = 8
const NY = 8
const NZ = 8
const NSTEPS = 3
const ΔT = 1e-3
const WAVE_DIFFUSIVITY = 0.3

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

function stepping_model(grid)
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1),
        stratification=ConstantStratification(N²=1),
        closure=HorizontalHyperdiffusivity(
            flow=0,
            flow2=0,
            waves=WAVE_DIFFUSIVITY,
            waves2=0,
            wave_laplacian_order=1,
        ),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        verbose=false,
    )
end

function scatter_initial_fields!(model, q, B, ψ)
    runtime = model.runtime
    for (destination, global_field) in (
        (model.fields.q, q),
        (model.fields.B, B),
        (model.fields.psi, ψ),
    )
        destination .= scatter_from_root(
            global_field,
            runtime.geometry,
            runtime.mpi;
            plans=runtime.plans,
        )
    end
    return model
end

function global_maximum(value)
    return MPI.Allreduce(value, MPI.MAX, comm)
end

grid = RectilinearGrid(size=(NX, NY, NZ), extent=(2π, 2π, 1.0))
q_initial = B_initial = ψ_initial = nothing
B_expected = nothing

if rank == 0
    q_initial = zeros(ComplexF64, NZ, NX, NY)
    B_initial = zeros(ComplexF64, NZ, NX, NY)
    ψ_initial = zeros(ComplexF64, NZ, NX, NY)

    # All selected modes lie within the radial two-thirds cutoff.
    q_initial[2, 2, 1] = 0.5 - 0.1im
    q_initial[2, NX, 1] = conj(q_initial[2, 2, 1])
    ψ_initial[2, 2, 1] = -0.2 + 0.05im
    ψ_initial[2, NX, 1] = conj(ψ_initial[2, 2, 1])
    B_initial[3, 2, 2] = 1.2 - 0.7im
    B_initial[6, 3, 1] = -0.4 + 0.9im

    B_expected = similar(B_initial)
    for k in 1:NZ, i in 1:NX, j in 1:NY
        damping = exp(-NSTEPS * ΔT * WAVE_DIFFUSIVITY * grid.kh2[i, j])
        B_expected[k, i, j] = damping * B_initial[k, i, j]
    end
end

let lazy_model=nothing, preallocated_model=nothing
try
    lazy_model = stepping_model(grid)
    preallocated_model = stepping_model(grid)
    scatter_initial_fields!(lazy_model, q_initial, B_initial, ψ_initial)
    scatter_initial_fields!(preallocated_model, q_initial, B_initial, ψ_initial)

    lazy = Simulation(
        lazy_model;
        Δt=ΔT,
        stop_iteration=NSTEPS,
        output=false,
        verbose=false,
    )
    preallocated = Simulation(
        preallocated_model;
        Δt=ΔT,
        stop_iteration=NSTEPS,
        output=false,
        verbose=false,
    )
    preallocated.timestepper.workspace = ExponentialRungeKutta2Workspace(
        preallocated_model.fields,
        preallocated_model.runtime.plans;
        G=preallocated_model.runtime.geometry,
    )
    explicit_workspace = preallocated.timestepper.workspace

    run!(lazy)
    run!(preallocated)

    @testset "MPI ETD-RK2 stepping" begin
        @test lazy.clock.iteration == NSTEPS
        @test preallocated.clock.iteration == NSTEPS
        @test lazy.timestepper.workspace isa ExponentialRungeKutta2Workspace
        @test preallocated.timestepper.workspace === explicit_workspace

        q_difference = global_maximum(maximum(abs,
            parent(lazy_model.fields.q) .- parent(preallocated_model.fields.q)))
        B_difference = global_maximum(maximum(abs,
            parent(lazy_model.fields.B) .- parent(preallocated_model.fields.B)))
        ψ_difference = global_maximum(maximum(abs,
            parent(lazy_model.fields.psi) .- parent(preallocated_model.fields.psi)))
        @test q_difference < 1e-13
        @test B_difference < 1e-13
        @test ψ_difference < 1e-13

        q_result = gather_to_root(
            lazy_model.fields.q,
            lazy_model.runtime.geometry,
            lazy_model.runtime.mpi,
        )
        B_result = gather_to_root(
            lazy_model.fields.B,
            lazy_model.runtime.geometry,
            lazy_model.runtime.mpi,
        )

        q_error = rank == 0 ? maximum(abs, q_result .- q_initial) : 0.0
        B_error = rank == 0 ? maximum(abs, B_result .- B_expected) : 0.0
        q_error = MPI.bcast(q_error, 0, comm)
        B_error = MPI.bcast(B_error, 0, comm)
        @test q_error < 1e-13
        @test B_error < 1e-13
    end
finally
    lazy_model === nothing || finalize_model!(lazy_model)
    preallocated_model === nothing || finalize_model!(preallocated_model)
    MPI.Barrier(comm)
    MPI.Finalize()
end
end
