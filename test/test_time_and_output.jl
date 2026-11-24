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

@testset "A single-precision Δt keeps its decimal value" begin
    # Read as 0.0099999998, Δt = 0.01f0 took 101 steps to t = 1, and every
    # schedule fired a step late; 0.7f0 lands just short of 3.5 and 7.
    for (Δt, decimal, stop_time, interval, iterations) in (
            (0.01f0, 0.01, 1.0, 0.25, [0, 25, 50, 75, 100]),
            (0.7f0, 0.7, 7.0, 3.5, [0, 5, 10]))
        mktempdir() do directory
            simulation = Simulation(time_test_model(); Δt, stop_time,
                output=NetCDFOutput(path=directory, schedule=TimeInterval(interval)),
                diagnostics=TimeInterval(interval), verbose=false)
            run!(simulation)
            @test simulation.clock.iteration == last(iterations)
            @test state_iterations(directory) == iterations
            energy_times = NCDataset(dataset -> dataset["time"][:],
                joinpath(directory, "diagnostic", "total_energy.nc"))
            @test energy_times ≈ decimal .* iterations
        end
    end
end

@testset "run! applies its keywords together or not at all" begin
    simulation = Simulation(time_test_model(); Δt=1.0, stop_iteration=2,
                            output=false, diagnostics=false, verbose=false)
    @test_throws ArgumentError run!(simulation; Δt=0.01, diagnostics_interval=0)
    @test simulation.timestepper.Δt == 1.0
    @test simulation.run_options.diagnostics_interval == 10
    @test simulation.state === Ready

    # So does output that cannot be set up, here because another run's files
    # are in its way: the retry ran 7 steps of 0.5.
    mktempdir() do directory
        run!(Simulation(time_test_model(); Δt=1.0, stop_iteration=1,
            output=NetCDFOutput(path=directory), diagnostics=false, verbose=false))
        @test_throws ArgumentError run!(simulation; Δt=0.5, stop_iteration=7,
                                        output=NetCDFOutput(path=directory))
        @test simulation.state === Ready
        @test simulation.timestepper.Δt == 1.0
        @test simulation.stop_iteration == 2
        @test simulation.run_options.output === false
    end
    run!(simulation)
    @test simulation.clock.iteration == 2
    @test simulation.clock.time == 2.0
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

        # Nor may a diagnostics directory that cannot be created, because a
        # file is in its way, fail after the state files are removed.
        blocked = joinpath(directory, "blocked")
        touch(blocked)
        replacing(diagnostics) = Simulation(time_test_model(); Δt=1.0,
            stop_iteration=1, diagnostics, verbose=false,
            output=NetCDFOutput(path=state_path, schedule=IterationInterval(1),
                                overwrite=true))
        simulation = replacing(EnergyDiagnosticsOutput(path=blocked))
        @test_throws Base.IOError run!(simulation)
        @test simulation.state === Ready
        @test length(state_iterations(state_path)) == 2

        # The same for a directory that takes no new files, where the file
        # system enforces that (not for root).
        locked = mkpath(joinpath(directory, "locked"))
        chmod(locked, 0o555)
        try
            enforced = try
                touch(joinpath(locked, "probe"))
                false
            catch
                true
            end
            if enforced
                simulation = replacing(EnergyDiagnosticsOutput(path=locked))
                @test_throws SystemError run!(simulation)
                @test simulation.state === Ready
                @test length(state_iterations(state_path)) == 2
            end
        finally
            chmod(locked, 0o755)
        end
    end
end

@testset "An unreadable state file does not shift the numbering" begin
    mktempdir() do directory
        output(overwrite) = NetCDFOutput(path=directory,
            schedule=IterationInterval(2), overwrite=overwrite)
        run!(Simulation(time_test_model(); Δt=60.0, stop_iteration=4,
            output=output(false), diagnostics=false, verbose=false))
        # A run killed while writing leaves a file that cannot be read. It
        # used to stay the latest file after its removal, so the restored
        # run wrote its restart state again, as state0004.
        write(joinpath(directory, "state0003.nc"), "truncated")
        restored = Simulation(time_test_model(); Δt=60.0, stop_iteration=6,
            output=output(true), diagnostics=false, verbose=false)
        restore!(restored, joinpath(directory, "state0002.nc"))
        run!(restored)
        @test state_iterations(directory) == [0, 2, 4, 6]
    end
end

@testset "Output files cannot replace directories" begin
    mktempdir() do path
        collisions = (joinpath(path, "wave_KE.nc"), joinpath(path, "total_energy.nc"))
        for collision in collisions, overwrite in (false, true)
            mkpath(collision)
            marker = joinpath(collision, "keep.txt")
            write(marker, "keep this file")
            simulation = Simulation(time_test_model(); stop_iteration=1, verbose=false,
                diagnostics=EnergyDiagnosticsOutput(; path, overwrite))
            @test_throws ArgumentError run!(simulation)
            @test simulation.state === Ready
            @test read(marker, String) == "keep this file"
            mv(collision, tempname(path))
        end
        # This name was previously used as the diagnostics writer's temp file.
        existing = joinpath(path, "wave_KE.nc.tmp")
        write(existing, "keep this too")
        run!(Simulation(time_test_model(); stop_iteration=1, verbose=false,
            diagnostics=EnergyDiagnosticsOutput(; path)))
        @test read(existing, String) == "keep this too"
        @test isfile(joinpath(path, "wave_KE.nc"))
    end
end

@testset "Failed streaming setup preserves checkpoints" begin
    mktempdir() do path
        output = NetCDFOutput(; path, schedule=IterationInterval(1), overwrite=true)
        run!(Simulation(time_test_model(); stop_iteration=2, output, verbose=false))
        checkpoint = joinpath(path, "state0002.nc")
        later = joinpath(path, "state0003.nc")
        original = read(later)
        model = time_test_model()
        initialize_particles!(model,
            particles_in_box(-500.0; x_max=1e5, y_max=1e5, nx=2, ny=2))
        make_simulation(particle_output) = restore!(
            Simulation(model; stop_iteration=3, output, particle_output, verbose=false),
            checkpoint)
        particle_output = ParticleOutputManager(
            path; save_interval_iter=1, output_mode=:streaming, overwrite=true)
        collision = mkpath(joinpath(path, "particles", "particles_stream.nc"))
        write(joinpath(collision, "keep.txt"), "keep this directory")
        blocked = make_simulation(particle_output)
        @test_throws ArgumentError run!(blocked)
        @test blocked.state === Ready
        @test !particle_output.initialized
        @test read(later) == original
        @test isfile(joinpath(collision, "keep.txt"))
        mv(collision, joinpath(path, "kept"))

        # NetCDF does not support this precision. Failure while defining the
        # streaming variables must also leave the existing output in place.
        unsupported = ParticleOutputManager{BigFloat}(
            path; save_interval_iter=1, output_mode=:streaming)
        broken = make_simulation(unsupported)
        @test_throws Exception run!(broken)
        @test broken.state === Ready
        @test !unsupported.initialized
        @test read(later) == original
        @test isempty(readdir(joinpath(path, "particles")))

        run!(blocked)
        @test blocked.state === Stopped
        @test state_iterations(path) == [0, 1, 2, 3]
    end
end

@testset "State output continues only from the restored checkpoint" begin
    mktempdir() do path
        destination = joinpath(path, "destination")
        source = joinpath(path, "source")
        for (directory, amplitude) in ((destination, 1.0), (source, 2.0))
            model = time_test_model(flow=FixedFlow())
            set!(model; ψ=(x, y, z) -> amplitude * cospi(2x / 1e5), verbose=false)
            run!(Simulation(model; stop_iteration=1, verbose=false,
                diagnostics=IterationInterval(1),
                output=NetCDFOutput(path=directory)))
        end
        checkpoint = joinpath(source, "state0002.nc")
        simulation = Simulation(time_test_model(flow=FixedFlow());
            stop_iteration=2, verbose=false, diagnostics=IterationInterval(1),
            output=NetCDFOutput(path=destination))
        restore!(simulation, checkpoint)
        existing = joinpath(destination, "state0002.nc")
        original = read(existing)
        @test_throws ArgumentError run!(simulation)
        @test simulation.state === Ready
        @test read(existing) == original
        @test state_iterations(destination) == [0, 1]

        run!(simulation; output=NetCDFOutput(path=destination, overwrite=true))
        @test state_iterations(destination) == [1, 2]
        @test NCDataset(ds -> maximum(abs, ds["psi"][:]),
            joinpath(destination, "state0001.nc")) ≈ 2.0
        @test NCDataset(ds -> ds["time"][:],
            joinpath(destination, "diagnostic", "total_energy.nc")) == [1.0, 2.0]
    end
end

@testset "Restore validates every field before modifying the model" begin
    mktempdir() do path
        file = joinpath(path, "restart.nc")
        grid = RectilinearGrid(Float32; size=(4, 4, 4), extent=(2π, 2π, 1.0))
        model = QGYBJModel(; grid, verbose=false)
        set!(model; ψ=(x, y, z) -> cos(x), verbose=false)
        original = copy_fields(model.fields)
        simulation = Simulation(model; stop_iteration=2, verbose=false)
        for bad in (-999.0, 1e50)
            NCDataset(file, "c") do dataset
                dims = ("x", "y", "z")
                for dimension in dims
                    dataset.dim[dimension] = 4
                end
                for name in ("q_hat_real", "q_hat_imag", "B_hat_imag")
                    defVar(dataset, name, zeros(4, 4, 4), dims)
                end
                variable = defVar(dataset, "B_hat_real", Float64, dims; fillvalue=-999.0)
                variable[:, :, :] = fill(bad, (4, 4, 4))
                dataset.attrib["iteration"] = 1
            end
            @test_throws ArgumentError restore!(simulation, file)
            @test simulation.clock.iteration == 0
            @test all(name -> getproperty(model.fields, name) == getproperty(original, name),
                      propertynames(original))
        end
    end
end

@testset "Diagnostics without state output go beneath output/" begin
    # Run with output=false, a schedule wrote beneath the path of the output
    # the simulation was built with.
    mktempdir() do directory
        cd(directory) do
            simulation = Simulation(time_test_model(); Δt=1.0, stop_iteration=1,
                output=NetCDFOutput(path="first"), verbose=false)
            run!(simulation; output=false, diagnostics=IterationInterval(1))
            @test isfile(joinpath("output", "diagnostic", "total_energy.nc"))
            @test !ispath("first")
        end
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

@testset "save_velocities overrides a :velocities field" begin
    for save_velocities in (false, true)
        mktempdir() do directory
            fields = save_velocities ? (:ψ,) : (:ψ, :velocities)
            simulation = Simulation(time_test_model(); Δt=1.0, stop_iteration=1,
                output=NetCDFOutput(path=directory, fields=fields),
                diagnostics=false, verbose=false)
            run!(simulation; save_velocities)
            NCDataset(joinpath(directory, "state0001.nc"), "r") do dataset
                @test haskey(dataset, "psi")
                @test haskey(dataset, "u") == save_velocities
            end
        end
    end
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

@testset "Output does not change a run" begin
    # Each step of run! starts from the diagnosis the previous one ended
    # with; a state write recomputes the diagnostics, and the step after it
    # diagnoses them again from q and B. A run gives the same fields whether
    # or not it writes output at every step.
    fields(model) = map(name -> Array(getproperty(model.fields, name)),
                        (:q, :B, :psi, :u, :v, :A, :C))
    for (formulation, feedback, flow) in (
            (YBJPlus(), WaveMeanFeedback(), EvolvingFlow()),
            (YBJ(), WaveMeanFeedback(), EvolvingFlow()),
            (YBJ(), :none, FixedFlow()))
        runs = map((true, false)) do with_output
            model = QGYBJModel(;
                grid=RectilinearGrid(size=(16, 16, 8), extent=(2π, 2π, 1.0)),
                coriolis=FPlane(f=1.0), stratification=ConstantStratification(N²=1.0),
                formulation, feedback, flow, verbose=false)
            set!(model; ψ=(x, y, z) -> 0.2sin(x) * cos(2y) * (1 + z), verbose=false)
            set_wave_packet!(model; amplitude=0.1, kx=3, ky=1, sigma_k=1.5)
            output = with_output ?
                NetCDFOutput(path=mktempdir(), schedule=IterationInterval(1)) : false
            simulation = Simulation(model; Δt=0.01, stop_iteration=5, output,
                                    diagnostics=false, verbose=false)
            try
                run!(simulation)
                fields(model)
            finally
                finalize_simulation!(simulation)
                finalize_model!(model)
            end
        end
        @test runs[1] == runs[2]
    end
end
