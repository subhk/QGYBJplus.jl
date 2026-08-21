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
        Δt=0.1, stop_iteration=2, output=false,
        diagnostics=false, verbose=false)
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
    @test simulation.diagnostics_manager === nothing
    @test_throws InvalidStateException run!(simulation; stop_iteration=3)

    time_limited = Simulation(model;
        Δt=0.5, stop_time=1.0, output=false, verbose=false)
    run!(time_limited; Δt=0.25)
    @test time_limited.clock.iteration == 4
    @test time_limited.clock.time ≈ 1.0
    @test time_limited.stop_iteration === nothing

    decimal_limited = Simulation(model;
        Δt=0.1, stop_time=1.0, output=false, verbose=false)
    run!(decimal_limited)
    @test decimal_limited.clock.iteration == 10
    @test decimal_limited.clock.time ≈ 1.0

    for state in (Running, Failed)
        time_limited.state = state
        @test_throws InvalidStateException set!(
            time_limited; waves=SurfaceWave(amplitude=0.1, scale=0.2))
        @test_throws InvalidStateException set_mean_flow!(
            time_limited; psi_func=(x, y, z) -> x, verbose=false)
        @test_throws InvalidStateException set_surface_waves!(
            time_limited; amplitude=0.1, surface_depth=0.2, verbose=false)
        @test_throws InvalidStateException set_wave_packet!(
            time_limited; amplitude=0.1, kx=1, ky=1, sigma_k=0.5)
    end
    time_limited.state = Stopped

    finalize_simulation!(simulation)
    finalize_simulation!(simulation)
    @test simulation.state == Finalized
    @test_throws InvalidStateException run!(simulation)
    @test_throws InvalidStateException set!(
        simulation; waves=SurfaceWave(amplitude=0.1, scale=0.2))
end
