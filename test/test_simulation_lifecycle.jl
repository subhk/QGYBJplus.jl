using Test
using QGYBJplus

@testset "Simulation clock and lifecycle" begin
    function lifecycle_model(; flow=FixedFlow(), formulation=PassiveWave(), f=1.0)
        grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
        return QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=f),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=flow,
            feedback=NoFeedback(),
            formulation=formulation,
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            verbose=false,
        )
    end

    negative_f_model = lifecycle_model(f=-2.0)
    try
        @test inertial_period(negative_f_model) ≈ π
    finally
        finalize_model!(negative_f_model)
    end

    model = lifecycle_model()
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

    # Progress reporting (used by the examples) prints one line per report.
    progress_simulation = Simulation(model; Δt=0.25, stop_iteration=2,
        output=false, diagnostics=false, verbose=false)
    progress_text = mktemp() do _, io
        redirect_stdout(io) do
            run!(progress_simulation; progress=true, diagnostics_interval=1)
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    @test count(line -> startswith(line, "Iteration: "), split(progress_text, '\n')) == 2

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

    @test_throws ArgumentError Simulation(
        model; Δt=0.1, stop_time=Inf, output=false, verbose=false)
    invalid_override = Simulation(
        model; Δt=0.1, stop_iteration=1, output=false, verbose=false)
    @test_throws ArgumentError run!(invalid_override; Δt=Inf)
    @test invalid_override.state == Ready

    nonfinite_model = lifecycle_model()
    try
        initialize_particles!(nonfinite_model, ParticleConfig{Float64}(
            x_max=2π,
            y_max=2π,
            z_level=-0.5,
            nx_particles=2,
            ny_particles=2,
            use_3d_advection=false,
        ))
        particles = nonfinite_model.particles.particles
        initial_x = copy(particles.x)
        initial_y = copy(particles.y)
        initial_z = copy(particles.z)
        initial_particle_time = particles.time
        nonfinite_model.fields.B[2, 2, 1] = complex(NaN)
        nonfinite_simulation = Simulation(
            nonfinite_model;
            Δt=0.1,
            stop_iteration=1,
            output=false,
            diagnostics=false,
            verbose=false,
        )
        @test_throws ErrorException run!(nonfinite_simulation)
        @test nonfinite_simulation.state == Failed
        @test particles.x == initial_x
        @test particles.y == initial_y
        @test particles.z == initial_z
        @test particles.time == initial_particle_time
        @test nonfinite_simulation.clock.iteration == 0
        @test nonfinite_simulation.clock.time == 0
    finally
        finalize_model!(nonfinite_model)
    end

    blowup_model = lifecycle_model(flow=EvolvingFlow())
    try
        blowup_model.fields.q[2, 2, 1] = 1e14 + 0im
        blowup_simulation = Simulation(
            blowup_model;
            Δt=0.1,
            stop_iteration=1,
            output=false,
            diagnostics=false,
            verbose=false,
        )
        @test_throws ErrorException run!(blowup_simulation)
        @test blowup_simulation.state == Failed
    finally
        finalize_model!(blowup_model)
    end

    finalize_simulation!(simulation)
    finalize_simulation!(simulation)
    @test simulation.state == Finalized
    @test_throws InvalidStateException run!(simulation)
    @test_throws InvalidStateException set!(
        simulation; waves=SurfaceWave(amplitude=0.1, scale=0.2))
end

@testset "A run whose output setup fails can be retried" begin
    mktempdir() do directory
        grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
        build() = QGYBJModel(; grid, coriolis=FPlane(f=1.0),
                             stratification=ConstantStratification(N²=1.0),
                             verbose=false)
        output(path) = NetCDFOutput(path=joinpath(directory, path),
                                    schedule=IterationInterval(1))
        first_run = Simulation(build(); Δt=0.1, stop_iteration=1,
                               output=output("taken"), diagnostics=false, verbose=false)
        run!(first_run)
        finalize_simulation!(first_run)

        # The second run finds the first one's files: nothing is written, so
        # it stays ready and runs once given another path.
        simulation = Simulation(build(); Δt=0.1, stop_iteration=1,
                                output=output("taken"), diagnostics=false, verbose=false)
        @test_throws ArgumentError run!(simulation)
        @test simulation.state == Ready
        @test simulation.output_manager === nothing
        run!(simulation; output=output("fresh"))
        @test simulation.state == Stopped
        @test isfile(joinpath(directory, "fresh", "state0002.nc"))
        finalize_simulation!(simulation)
    end
end
