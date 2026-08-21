using Test
using QGYBJplus

@testset "Simulation clock and lifecycle" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=0, flow2=0, waves=0, waves2=0),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )
    model.fields.B[2, 2, 1] = 1 + 0im

    simulation = Simulation(model;
        Δt=0.1, stop_iteration=2, output=false, verbose=false)
    @test simulation.model === model
    @test simulation.clock.time == 0
    @test simulation.clock.iteration == 0
    @test simulation.state == Ready

    simulation.state = Running
    @test_throws InvalidStateException run!(simulation)
    simulation.state = Ready

    run!(simulation)
    @test simulation.clock.iteration == 2
    @test simulation.clock.time ≈ 0.2
    @test simulation.state == Stopped
    @test simulation.timestepper isa ExponentialRungeKutta2

    finalize_simulation!(simulation)
    finalize_simulation!(simulation)
    @test simulation.state == Finalized
    @test_throws InvalidStateException run!(simulation)
end
