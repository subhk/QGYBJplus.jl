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
        topology=(1, 1),
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
