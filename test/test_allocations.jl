#=
Per-call allocation budgets for the hot path.

The model reuses runtime-owned workspaces, so steady-state stepping should not
allocate grid-sized temporaries. These bounds are expressed as multiples of one
grid-sized ComplexF64 array so they stay meaningful if the test grid changes.
They are budgets, not exact figures: they catch a kernel that starts allocating
a fresh field per call, which is the regression that actually hurts.
=#

using Test
using QGYBJplus

const Q = QGYBJplus

@testset "Hot-path allocation budgets" begin
    nx, ny, nz = 32, 32, 16
    grid_array_bytes = nx * ny * nz * sizeof(ComplexF64)

    grid = RectilinearGrid(size=(nx, ny, nz), extent=(1.0, 1.0, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1e-4),
        stratification=ConstantStratification(N²=1e-5),
        flow=EvolvingFlow(),
        feedback=WaveMeanFeedback(),
        formulation=YBJPlus(),
        verbose=false,
    )

    try
        set!(model;
             ψ=(x, y, z) -> 1e-3 * sinpi(2x) * cospi(2y),
             waves=SurfaceWave(amplitude=0.1, scale=0.1),
             verbose=false)

        context = Q._operator_context(model)
        fields = model.fields

        """Mean bytes allocated per call, after a warm-up call."""
        function bytes_per_call(thunk; repetitions::Int=5)
            thunk()
            return (@allocated begin
                for _ in 1:repetitions
                    thunk()
                end
            end) / repetitions
        end

        @testset "elliptic inversions reuse their solver scratch" begin
            # The Thomas solver used to copy two nz-vectors per horizontal mode,
            # i.e. 2 * nx * ny * 2 copies per inversion.
            psi_bytes = bytes_per_call() do
                Q.invert_q_to_psi!(fields, context.grid;
                                   a=context.a, workspace=context.workspace)
            end
            @test psi_bytes < 0.25 * grid_array_bytes

            wave_bytes = bytes_per_call() do
                Q.invert_B_to_A!(fields, context.grid, context.a;
                                 workspace=context.workspace)
            end
            @test wave_bytes < 0.25 * grid_array_bytes
        end

        @testset "velocity diagnosis reuses workspace buffers" begin
            velocity_bytes = bytes_per_call() do
                Q.compute_velocities!(fields, context.grid;
                                      plans=context.plans, f=context.f,
                                      N2=first(context.N2), compute_w=false,
                                      N2_profile=context.N2,
                                      workspace=context.workspace,
                                      dealias_mask=context.mask)
            end
            @test velocity_bytes < 0.25 * grid_array_bytes
        end

        @testset "ETD workspace layout validation is allocation-free" begin
            timestepper = ExponentialRungeKutta2(Δt=1.0)
            step!(model, timestepper)
            timestep_workspace =
                timestepper.workspace::ExponentialRungeKutta2Workspace
            validation_bytes = bytes_per_call() do
                Q._etdrk2_workspace_matches(timestep_workspace, model.fields)
            end
            @test validation_bytes == 0
        end

        @testset "one ETD-RK2 step stays within budget" begin
            timestepper = ExponentialRungeKutta2(Δt=1.0)
            step_bytes = bytes_per_call(() -> step!(model, timestepper))
            @test step_bytes < 8 * grid_array_bytes
        end
    finally
        finalize_model!(model)
    end
end

@testset "Persistent model memory stays within budget" begin
    # Counted in grid-sized complex arrays: the fields hold 6.5, the ETD-RK2
    # workspace 4 (the stage q and B and their first-stage tendencies), and
    # the scratch pool at most 5 after stepping, velocity, Stokes drift,
    # NetCDF output, and energy diagnostics.
    nx, ny, nz = 16, 16, 8
    grid_array_bytes = nx * ny * nz * sizeof(ComplexF64)
    grid = RectilinearGrid(size=(nx, ny, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> 1 + 0.5cos(3z); returns=:N²),
        flow=EvolvingFlow(), feedback=WaveMeanFeedback(), formulation=YBJPlus(),
        verbose=false)
    mktempdir() do directory
        set!(model; ψ=(x, y, z) -> 0.3sin(x) * cos(y) * (1 + z),
             waves=SurfaceWave(amplitude=0.1, scale=0.3), verbose=false)
        simulation = Simulation(model; Δt=0.01, stop_iteration=2,
            output=NetCDFOutput(path=joinpath(directory, "out"),
                schedule=IterationInterval(1), fields=(:ψ, :waves), velocities=true),
            diagnostics=EnergyDiagnosticsOutput(path=joinpath(directory, "diag"),
                schedule=IterationInterval(1)),
            verbose=false)
        initialize_particles!(simulation, particles_in_box(Float64, -0.5;
            x_max=2π, y_max=2π, nx=2, ny=2))
        q = model.fields.q
        run!(simulation)
        compute_total_velocities!(model; t=0.3)
        compute_ybj_vertical_velocity!(model; t=0.3)

        # step! updates the fields in place.
        @test model.fields.q === q
        @test length(model.runtime.workspace.pool.buffers) <= 5
        workspace = simulation.timestepper.workspace
        workspace_bytes = sum(sizeof, (workspace.stage_q, workspace.stage_B,
            workspace.rhsq0, workspace.rhsB0))
        @test workspace_bytes <= 4 * grid_array_bytes
        finalize_simulation!(simulation)
    end
end
