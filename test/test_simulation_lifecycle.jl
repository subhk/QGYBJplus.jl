using Test
using QGYBJplus

struct SlowProgressModel
    runtime
end

function QGYBJplus._progress_maxima(::SlowProgressModel)
    sleep(0.2)
    return (wave_speed=1.0, flow_speed=2.0)
end

@testset "Simulation clock and lifecycle" begin
    @test isdefined(QGYBJplus, :_prettytime)
    if isdefined(QGYBJplus, :_prettytime)
        @test QGYBJplus._prettytime(0.0) == "0 seconds"
        @test QGYBJplus._prettytime(0.25) == "250 ms"
        @test QGYBJplus._prettytime(1.25) == "1.250 seconds"
        @test QGYBJplus._prettytime(60.0) == "1 minute"
        @test QGYBJplus._prettytime(5400.0) == "1.500 hours"
        @test QGYBJplus._prettytime(129600.0) == "1.500 days"
    end

    slow_model = SlowProgressModel((mpi=(is_root=true,),))
    slow_simulation = Simulation(
        slow_model,
        Clock(Float64),
        (Δt=0.25,),
        nothing,
        1,
        QGYBJplus.default_run_options(Float64),
        nothing,
        nothing,
        nothing,
        Ready,
    )
    slow_simulation.clock.iteration = 1
    slow_simulation.clock.time = 0.25
    slow_progress_text = mktemp() do _, io
        run_wall_start = time_ns() - UInt64(2_000_000_000)
        redirect_stdout(io) do
            QGYBJplus._print_detailed_progress(
                slow_simulation, run_wall_start)
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    slow_wall_match = match(
        r"wall time: ([0-9.]+) seconds", slow_progress_text)
    @test slow_wall_match !== nothing
    if slow_wall_match !== nothing
        @test parse(Float64, slow_wall_match.captures[1]) >= 2.15
    end

    function lifecycle_model(; flow=FixedFlow(), formulation=PassiveWave())
        grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
        return QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=flow,
            feedback=NoFeedback(),
            formulation=formulation,
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
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

    progress_model = lifecycle_model()
    try
        set!(progress_model;
            ψ=(x, y, _) -> 3sin(x) + 4sin(y),
            pv_method=:none,
        )
        progress_model.fields.B[1, 1, 1] = 1 + 0im
        progress_simulation = Simulation(progress_model;
            Δt=0.25,
            stop_time=0.5,
            output=false,
            diagnostics=false,
            verbose=false,
        )
        progress_text = mktemp() do _, io
            redirect_stdout(io) do
                run!(progress_simulation;
                    progress=true,
                    diagnostics_interval=1,
                )
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end

        progress_lines = filter(!isempty, split(progress_text, '\n'))
        @test length(progress_lines) == 2
        @test startswith(progress_lines[1],
            "Iteration: 0001, time: 250 ms, Δt: 250 ms, ")
        @test startswith(progress_lines[2],
            "Iteration: 0002, time: 500 ms, Δt: 250 ms, ")
        @test all(occursin("max(|LA|) = ", line) for line in progress_lines)
        @test all(occursin("max(|uₕ|) = ", line) for line in progress_lines)
        @test all(occursin(
            r", wall time: (?:[0-9.eE+-]+ (?:ns|μs|ms|second|seconds|minute|minutes|hour|hours|day|days))$",
            line) for line in progress_lines)
        wave_matches = collect(eachmatch(
            r"max\(\|LA\|\) = ([0-9.eE+-]+)", progress_text))
        flow_matches = collect(eachmatch(
            r"max\(\|uₕ\|\) = ([0-9.eE+-]+)", progress_text))
        @test length(wave_matches) == 2
        @test length(flow_matches) == 2
        @test all(parse(Float64, match.captures[1]) > 0 for match in wave_matches)
        @test all(isapprox(parse(Float64, match.captures[1]), 5.0;
                           rtol=1e-3) for match in flow_matches)
        @test progress_simulation.clock.iteration == 2
        @test progress_simulation.clock.time ≈ 0.5
        @test progress_simulation.stop_iteration === nothing
    finally
        finalize_model!(progress_model)
    end

    ybj_plus_model = lifecycle_model(formulation=YBJPlus())
    try
        ybj_plus_model.fields.B[1, 2, 1] = 1 + 0im
        ybj_plus_model.fields.A[1, 2, 1] = 4 + 0im
        maxima = QGYBJplus._progress_maxima(ybj_plus_model)
        @test maxima.wave_speed ≈ 2 / 64
    finally
        finalize_model!(ybj_plus_model)
    end

    ybj_model = lifecycle_model(formulation=YBJ())
    try
        ybj_model.fields.B[1, 2, 1] = 2 + 0im
        ybj_model.fields.A[1, 2, 1] = 100 + 0im
        maxima = QGYBJplus._progress_maxima(ybj_model)
        @test maxima.wave_speed ≈ 2 / 64
    finally
        finalize_model!(ybj_model)
    end

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
