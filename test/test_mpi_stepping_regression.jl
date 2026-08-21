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
using NCDatasets
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

function normal_ybj_model(grid, dispersion)
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=2),
        stratification=ConstantStratification(N²=3),
        closure=HorizontalHyperdiffusivity(
            flow=0, flow2=0, waves=0, waves2=0),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=YBJ(),
        linear=LinearDynamics(),
        no_dispersion=dispersion,
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
    B_initial[3, 3, 1] = 0.6 + 0.2im
    B_initial[6, 3, 1] = -0.4 + 0.9im

    B_expected = similar(B_initial)
    for k in 1:NZ, i in 1:NX, j in 1:NY
        damping = exp(-NSTEPS * ΔT * WAVE_DIFFUSIVITY * grid.kh2[i, j])
        B_expected[k, i, j] = damping * B_initial[k, i, j]
    end
end

let lazy_model=nothing, preallocated_model=nothing,
    normal_model=nothing, nonfinite_model=nothing
try
    lazy_model = stepping_model(grid)
    preallocated_model = stepping_model(grid)
    scatter_initial_fields!(lazy_model, q_initial, B_initial, ψ_initial)
    scatter_initial_fields!(preallocated_model, q_initial, B_initial, ψ_initial)

    @testset "MPI dimensional wave feedback" begin
        context = QGYBJplus._operator_context(lazy_model)
        BRk = similar(lazy_model.fields.B)
        BIk = similar(lazy_model.fields.B)
        qw_f1 = similar(lazy_model.fields.q)
        qw_f2 = similar(lazy_model.fields.q)
        qw_split = similar(lazy_model.fields.q)

        QGYBJplus.split_B_to_real_imag!(
            BRk, BIk, lazy_model.fields.B, context.plans)
        QGYBJplus.compute_qw_complex!(
            qw_f1, lazy_model.fields.B, context.grid, context.plans;
            f=1.0, Lmask=context.mask)
        QGYBJplus.compute_qw_complex!(
            qw_f2, lazy_model.fields.B, context.grid, context.plans;
            f=2.0, Lmask=context.mask)
        QGYBJplus.compute_qw!(
            qw_split, BRk, BIk, context.grid, context.plans;
            f=2.0, Lmask=context.mask)

        feedback_energy = MPI.Allreduce(
            sum(abs2, parent(qw_f1)), +, comm)
        scaling_error = global_maximum(maximum(abs,
            parent(qw_f2) .- 0.5 .* parent(qw_f1)))
        split_error = global_maximum(maximum(abs,
            parent(qw_split) .- parent(qw_f2)))
        @test feedback_energy > 0
        @test scaling_error < 1e-13
        @test split_error < 1e-13
    end

    diagnostic_dir = mktempdir()
    lazy = Simulation(
        lazy_model;
        Δt=ΔT,
        stop_iteration=NSTEPS,
        output=false,
        diagnostics=EnergyDiagnosticsOutput(
            path=diagnostic_dir,
            schedule=IterationInterval(1),
        ),
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

    @testset "Runtime-local transpose cache identity" begin
        lazy_key = QGYBJplus._transpose_buffer_key(
            lazy_model.fields.q,
            lazy_model.runtime.workspace.q_z,
            eltype(lazy_model.fields.q),
        )
        preallocated_key = QGYBJplus._transpose_buffer_key(
            preallocated_model.fields.q,
            preallocated_model.runtime.workspace.q_z,
            eltype(preallocated_model.fields.q),
        )
        lazy_plan_key = QGYBJplus._plan_transpose_buffer_key(
            lazy_model.runtime.plans, eltype(lazy_model.fields.q))
        preallocated_plan_key = QGYBJplus._plan_transpose_buffer_key(
            preallocated_model.runtime.plans,
            eltype(preallocated_model.fields.q))
        lazy_buffer = QGYBJplus._get_transpose_buffer(
            lazy_model.fields.q,
            lazy_model.runtime.workspace.q_z,
            eltype(lazy_model.fields.q),
        )
        preallocated_buffer = QGYBJplus._get_transpose_buffer(
            preallocated_model.fields.q,
            preallocated_model.runtime.workspace.q_z,
            eltype(preallocated_model.fields.q),
        )
        lazy_plan_buffer = QGYBJplus._get_plan_transpose_buffer(
            lazy_model.runtime.plans, eltype(lazy_model.fields.q))
        preallocated_plan_buffer = QGYBJplus._get_plan_transpose_buffer(
            preallocated_model.runtime.plans,
            eltype(preallocated_model.fields.q))
        @test !isequal(lazy_key, preallocated_key)
        @test !isequal(lazy_plan_key, preallocated_plan_key)
        @test lazy_buffer !== preallocated_buffer
        @test lazy_plan_buffer !== preallocated_plan_buffer
    end

    run!(lazy)
    run!(preallocated)

    @testset "MPI ETD-RK2 stepping" begin
        @test lazy.clock.iteration == NSTEPS
        @test preallocated.clock.iteration == NSTEPS
        @test lazy.timestepper.workspace isa ExponentialRungeKutta2Workspace
        @test preallocated.timestepper.workspace === explicit_workspace
        @test lazy.diagnostics_manager.closed
        @test lazy.diagnostics_manager.time ≈
              collect(0:NSTEPS) .* ΔT

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
        if rank == 0
            diagnostic_file = joinpath(diagnostic_dir, "total_energy.nc")
            @test isfile(diagnostic_file)
            NCDataset(diagnostic_file, "r") do dataset
                @test dataset["time"][:] ≈ collect(0:NSTEPS) .* ΔT
                @test all(isfinite, dataset["total_energy"][:])
            end
        end
    end

    for dispersion in (Dispersive(), NoDispersion())
        normal_model = normal_ybj_model(grid, dispersion)
        try
            B_normal = zeros(ComplexF64, NZ, NX, NY)
            B_normal[:, 2, 1] .= (
                1 + 0.5im, -1 - 0.2im, 1 + 0.4im, -1 - 0.1im,
                1 + 0.3im, -1 - 0.6im, 1 + 0.2im, -1 - 0.5im)
            set!(normal_model;
                B=FieldArray(B_normal; space=:spectral), verbose=false)
            A_energy = MPI.Allreduce(
                sum(abs2, parent(normal_model.fields.A)), +, comm)
            C_energy = MPI.Allreduce(
                sum(abs2, parent(normal_model.fields.C)), +, comm)
            A_imaginary_energy = MPI.Allreduce(
                sum(abs2, imag.(parent(normal_model.fields.A))), +, comm)
            C_imaginary_energy = MPI.Allreduce(
                sum(abs2, imag.(parent(normal_model.fields.C))), +, comm)
            @test A_energy > 0
            @test C_energy > 0
            @test A_imaginary_energy > 0
            @test C_imaginary_energy > 0

            A_result = gather_to_root(
                normal_model.fields.A,
                normal_model.runtime.geometry,
                normal_model.runtime.mpi,
            )
            C_result = gather_to_root(
                normal_model.fields.C,
                normal_model.runtime.geometry,
                normal_model.runtime.mpi,
            )
            reconstruction_error = 0.0
            derivative_error = 0.0
            if rank == 0
                expected_A = zeros(ComplexF64, NZ)
                expected_C = zeros(ComplexF64, NZ)
                cumulative_B = 0.0im
                Δz = grid.z[2] - grid.z[1]
                for k in 2:NZ
                    cumulative_B += B_normal[k-1, 2, 1]
                    expected_A[k] = expected_A[k-1] +
                        cumulative_B * (3 / 2^2) * Δz^2
                end
                expected_A .-= sum(expected_A) / NZ
                for k in 1:(NZ-1)
                    expected_C[k] =
                        (expected_A[k+1] - expected_A[k]) / Δz
                end
                reconstruction_error = maximum(abs,
                    A_result[:, 2, 1] .- expected_A)
                derivative_error = maximum(abs,
                    C_result[:, 2, 1] .- expected_C)
            end
            reconstruction_error = MPI.bcast(reconstruction_error, 0, comm)
            derivative_error = MPI.bcast(derivative_error, 0, comm)
            @test reconstruction_error < 1e-13
            @test derivative_error < 1e-13

            normal_diagnostic_dir = MPI.bcast(
                rank == 0 ? mktempdir() : "", 0, comm)
            normal_simulation = Simulation(
                normal_model;
                Δt=ΔT,
                stop_iteration=1,
                output=false,
                diagnostics=EnergyDiagnosticsOutput(
                    path=normal_diagnostic_dir,
                    schedule=IterationInterval(1),
                ),
                verbose=false,
            )
            run!(normal_simulation)
            if rank == 0
                NCDataset(joinpath(
                    normal_diagnostic_dir, "total_energy.nc"), "r") do dataset
                    @test all(>(0), dataset["wave_KE"][:])
                    @test all(isfinite, dataset["total_energy"][:])
                end
            end
        finally
            finalize_model!(normal_model)
            normal_model = nothing
        end
    end

    nonfinite_model = stepping_model(grid)
    fill!(parent(nonfinite_model.fields.B), 0)
    initialize_particles!(nonfinite_model, particles_in_box(
        Float64,
        -0.5;
        x_max=grid.extent[1],
        y_max=grid.extent[2],
        nx=2,
        ny=2,
        use_3d_advection=false,
    ))
    particles = nonfinite_model.particles.particles
    initial_x = copy(particles.x)
    initial_y = copy(particles.y)
    initial_z = copy(particles.z)
    initial_particle_time = particles.time
    offender_rank = min(1, MPI.Comm_size(comm) - 1)
    if rank == offender_rank
        B_local = parent(nonfinite_model.fields.B)
        injected = false
        for j in axes(B_local, 3)
            for i in axes(B_local, 2)
                i_global = local_to_global(i, 2, nonfinite_model.fields.B)
                j_global = local_to_global(j, 3, nonfinite_model.fields.B)
                if nonfinite_model.runtime.dealias_mask[i_global, j_global] &&
                   grid.kh2[i_global, j_global] > 0
                    B_local[1, i, j] = complex(NaN)
                    injected = true
                    break
                end
            end
            injected && break
        end
        injected || error("no retained nonzero spectral mode found on rank $rank")
    end
    nonfinite_simulation = Simulation(
        nonfinite_model;
        Δt=ΔT,
        stop_iteration=1,
        output=false,
        diagnostics=false,
        verbose=false,
    )
    @testset "Collective non-finite termination" begin
        @test_throws ErrorException run!(nonfinite_simulation)
        @test nonfinite_simulation.state == Failed
        @test particles.x == initial_x
        @test particles.y == initial_y
        @test particles.z == initial_z
        @test particles.time == initial_particle_time
        @test nonfinite_simulation.clock.iteration == 0
        @test nonfinite_simulation.clock.time == 0
    end
finally
    lazy_model === nothing || finalize_model!(lazy_model)
    preallocated_model === nothing || finalize_model!(preallocated_model)
    normal_model === nothing || finalize_model!(normal_model)
    nonfinite_model === nothing || finalize_model!(nonfinite_model)
    MPI.Barrier(comm)
    MPI.Finalize()
end
end
