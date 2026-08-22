using Test
using QGYBJplus

@testset "Model-owned exponential Runge-Kutta 2" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0.3, order=2),
        ),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )

    try
        model.fields.q[2, 2, 1] = 0.5 - 0.1im
        model.fields.B[2, 2, 1] = 1.2 - 0.7im
        q_initial = model.fields.q[2, 2, 1]
        B_initial = model.fields.B[2, 2, 1]

        timestepper = ExponentialRungeKutta2(Δt=0.1)
        @test timestepper.Δt == 0.1
        @test timestepper.workspace === nothing

        step!(model, timestepper)
        damping = exp(-0.1 * 0.3 * grid.kh2[2, 1])
        @test model.fields.q[2, 2, 1] == q_initial
        @test model.fields.B[2, 2, 1] ≈ damping * B_initial rtol=1e-14
        @test timestepper.workspace isa ExponentialRungeKutta2Workspace

        workspace = timestepper.workspace
        step!(model, timestepper)
        @test timestepper.workspace === workspace
        @test model.fields.B[2, 2, 1] ≈ damping^2 * B_initial rtol=2e-14

        @test !isdefined(QGYBJplus, :ExpRK2Workspace)
        @test !isdefined(QGYBJplus, :leapfrog_step!)
        @test !isdefined(QGYBJplus, :imex_cn_step!)
    finally
        finalize_model!(model)
    end
end
