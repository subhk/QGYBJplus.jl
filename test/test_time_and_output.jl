using Test
using NCDatasets
using QGYBJplus

time_test_model(; f=1e-4, kwargs...) = QGYBJModel(;
    grid=RectilinearGrid(size=(8, 8, 4), extent=(1e5, 1e5, 1000.0)),
    coriolis=FPlane(f=f), stratification=ConstantStratification(N²=1e-5),
    verbose=false, kwargs...)

"""Iterations recorded by the state files in `path`, in file order."""
state_iterations(path) = map(sort(filter(endswith(".nc"), readdir(path)))) do name
    NCDataset(dataset -> dataset.attrib["iteration"], joinpath(path, name))
end

@testset "The clock is t₀ + nΔt in Float64" begin
    # Accumulating Δt drifted past stop_time: Tᵢ/200 to Tᵢ took 201 steps.
    model = time_test_model()
    period = inertial_period(model)
    simulation = Simulation(model; Δt=period / 200, stop_time=period,
                            output=false, diagnostics=false, verbose=false)
    run!(simulation)
    @test simulation.clock.iteration == 200
    @test simulation.clock.time ≈ period

    # Every multiple-of-the-step stop time, as the review sampled them.
    overshoots = 0
    for n in (7, 30, 200, 400), m in (1, 3)
        model = time_test_model()
        simulation = Simulation(model; Δt=period / n, stop_time=m * period,
                                output=false, diagnostics=false, verbose=false)
        run!(simulation)
        overshoots += simulation.clock.iteration != n * m
    end
    @test overshoots == 0

    # A single-precision Δt still gives a Float64 clock.
    simulation = Simulation(time_test_model(); Δt=62.83f0, stop_iteration=3,
                            output=false, diagnostics=false, verbose=false)
    @test simulation.clock.time isa Float64
    @test simulation.timestepper.Δt isa Float64
end

@testset "Time schedules fire at multiples of their interval" begin
    mktempdir() do directory
        first_run = joinpath(directory, "first")
        simulation = Simulation(time_test_model(); Δt=0.3, stop_iteration=12,
            output=NetCDFOutput(path=first_run, schedule=TimeInterval(1.0)),
            diagnostics=false, verbose=false)
        run!(simulation)
        # t = 1.2, 2.1, 3.0 are the first steps at or after 1, 2, 3.
        @test state_iterations(first_run) == [0, 4, 7, 10, 12]

        # A run restored from t = 1.2 keeps those times.
        restored = joinpath(directory, "restored")
        simulation = Simulation(time_test_model(); Δt=0.3, stop_iteration=12,
            output=NetCDFOutput(path=restored, schedule=TimeInterval(1.0)),
            diagnostics=false, verbose=false)
        restore!(simulation, joinpath(first_run, "state0002.nc"))
        @test simulation.clock.iteration == 4
        run!(simulation)
        @test state_iterations(restored) == [4, 7, 10, 12]
    end

    @test TimeInterval(0.5).interval === 0.5
    @test TimeInterval(1).interval === 1.0
    @test_throws ArgumentError TimeInterval(0)
    @test_throws ArgumentError TimeInterval(-1.0)
    @test_throws ArgumentError TimeInterval(Inf)
    # The default constructor used to bypass the check for Int.
    @test_throws ArgumentError IterationInterval(0)
    @test_throws ArgumentError IterationInterval(-3)
    @test IterationInterval(Int32(5)).interval === 5
end

@testset "run! applies its keywords together or not at all" begin
    simulation = Simulation(time_test_model(); Δt=1.0, stop_iteration=2,
                            output=false, diagnostics=false, verbose=false)
    @test_throws ArgumentError run!(simulation; Δt=0.01, diagnostics_interval=0)
    @test simulation.timestepper.Δt == 1.0
    @test simulation.run_options.diagnostics_interval == 10
    @test simulation.state === Ready
    run!(simulation)
    @test simulation.clock.iteration == 2
end

@testset "Output setup checks everything before removing a file" begin
    mktempdir() do directory
        state_path = joinpath(directory, "state")
        run!(Simulation(time_test_model(); Δt=1.0, stop_iteration=1,
            output=NetCDFOutput(path=state_path, schedule=IterationInterval(1)),
            diagnostics=false, verbose=false))
        @test length(state_iterations(state_path)) == 2

        # The particle output check fails (no particles), so the state files
        # that overwrite=true would replace must still be there.
        simulation = Simulation(time_test_model(); Δt=1.0, stop_iteration=1,
            output=NetCDFOutput(path=state_path, schedule=IterationInterval(1),
                               overwrite=true),
            diagnostics=false, verbose=false,
            particle_output=ParticleOutputManager(directory; save_interval_iter=1))
        @test_throws ArgumentError run!(simulation)
        @test simulation.state === Ready
        @test length(state_iterations(state_path)) == 2
    end
end

@testset "Particle output lifecycle" begin
    @test_throws ArgumentError Simulation(time_test_model(); particle_output=3)

    mktempdir() do directory
        # Without particles, run! fails cleanly and so does the cleanup that
        # the docs recommend, which used to throw a MethodError.
        manager = ParticleOutputManager(directory; save_interval_iter=1)
        simulation = Simulation(time_test_model(); Δt=10.0, stop_iteration=2,
            output=false, diagnostics=false, particle_output=manager, verbose=false)
        @test_throws ArgumentError run!(simulation)
        @test simulation.state === Ready
        @test finalize_simulation!(simulation) === simulation
        @test simulation.state === Finalized
        @test !manager.closed   # never opened, so it may serve another run
    end

    mktempdir() do directory
        model = time_test_model()
        initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5,
                                                      nx=2, ny=2))
        manager = ParticleOutputManager(directory; save_interval_iter=1)
        run!(Simulation(model; Δt=10.0, stop_iteration=2, output=false,
                        diagnostics=false, particle_output=manager, verbose=false))
        @test manager.closed
        @test manager.save_count == 3

        # A closed manager used to record nothing, silently.
        continuation = Simulation(model; Δt=10.0, stop_iteration=2, output=false,
            diagnostics=false, particle_output=manager, verbose=false)
        @test_throws ArgumentError run!(continuation)
        @test continuation.state === Ready

        # Existing files block a new manager without touching them.
        blocked = Simulation(model; Δt=10.0, stop_iteration=2, output=false,
            diagnostics=false, verbose=false,
            particle_output=ParticleOutputManager(directory; save_interval_iter=1))
        @test_throws ArgumentError run!(blocked)
        @test isfile(joinpath(directory, "particles", "particles_trajectory.nc"))
    end
end

@testset "finalize_simulation! leaves the model usable" begin
    model = time_test_model()
    simulation = Simulation(model; Δt=60.0, stop_iteration=2, output=false,
                            diagnostics=false, verbose=false)
    try
        run!(simulation)
    finally
        finalize_simulation!(simulation)
    end
    @test simulation.state === Finalized
    @test !model.runtime.finalized
    continuation = Simulation(model; Δt=60.0, stop_iteration=2, output=false,
                              diagnostics=false, verbose=false)
    run!(continuation)
    @test continuation.clock.iteration == 2
    @test finalize_simulation!(model) === model
    @test model.runtime.finalized
end

@testset "NetCDFOutput rejects unknown fields" begin
    @test_throws ArgumentError NetCDFOutput(fields=(:ψ, :velocity))
    @test_throws ArgumentError NetCDFOutput(fields=:vorticity)
    @test NetCDFOutput(fields=(:psi, :B, :velocities)).fields == (:psi, :B, :velocities)
end

@testset "A FixedFlow restart restores the prescribed streamfunction" begin
    prescribed(x, y, z) = 1e3 * sin(2π * x / 1e5) * cos(2π * y / 1e5) * (1 + z / 1000)
    for pv_method in (:none, :barotropic)
        mktempdir() do directory
            model = time_test_model(flow=FixedFlow())
            set!(model; ψ=prescribed, pv_method)
            run!(Simulation(model; Δt=60.0, stop_iteration=2,
                output=NetCDFOutput(path=directory, schedule=IterationInterval(1)),
                diagnostics=false, verbose=false))
            path = joinpath(directory, "state0003.nc")
            NCDataset(path, "r") do dataset
                @test dataset.attrib["flow_evolution"] == "FixedFlow"
                @test haskey(dataset, "psi_hat_real")
            end

            restored = time_test_model(flow=FixedFlow())
            restore!(restored, path)
            # Re-inverting q gave ψ = 0 for :none and a 30% error for :barotropic.
            @test restored.fields.psi ≈ model.fields.psi
            @test restored.fields.u ≈ model.fields.u
            @test_throws ArgumentError invert_q_to_psi!(restored)
        end
    end
end
