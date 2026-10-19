using Test
using NCDatasets
using QGYBJplus

@testset "Model-owned particle state and advection" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        verbose=false,
    )

    try
        @test hasproperty(model, :particles)
        @test model.particles === nothing

        box = particles_in_box(
            Float64,
            -0.5;
            x_max=grid.extent[1],
            y_max=grid.extent[2],
            nx=2,
            ny=3,
        )
        @test initialize_particles!(model, box) === model
        @test model.particles isa ParticleTracker
        @test model.particles.model === model
        @test model.particles.particles.np == 6

        distributions = (
            (particles_in_grid_3d(
                x_max=grid.extent[1], y_max=grid.extent[2],
                z_max=grid.extent[3], nx=2, ny=2, nz=2,
                precision=Float64), 8),
            (particles_in_layers(
                [-0.75, -0.25];
                x_max=grid.extent[1], y_max=grid.extent[2],
                nx=2, ny=2, precision=Float64), 8),
            (particles_random_3d(
                5;
                x_max=grid.extent[1], y_max=grid.extent[2],
                z_max=grid.extent[3], seed=42, precision=Float64), 5),
            (particles_custom(
                [(0.2, 0.3, -0.8), (1.2, 1.3, -0.2)];
                precision=Float64), 2),
        )
        for (configuration, expected_count) in distributions
            initialize_particles!(model, configuration)
            @test model.particles.particles.np == expected_count
        end

        for interpolation in (TRILINEAR, TRICUBIC, QUINTIC)
            configuration = particles_in_box(
                Float64,
                -0.5;
                x_max=grid.extent[1],
                y_max=grid.extent[2],
                nx=2,
                ny=2,
                interpolation_method=interpolation,
            )
            initialize_particles!(model, configuration)
            tracker = model.particles
            # Uniform horizontal velocity is reproduced everywhere.
            fill!(tracker.current.u, 1.0)
            fill!(tracker.current.v, -0.5)
            @test all((-0.5, -0.05, 0.0, -1.0)) do z
                velocity = interpolate_velocity_at_position(
                    Float64(π), Float64(π), z, tracker)
                all(isapprox.(velocity, (1.0, -0.5)))
            end
        end

        for method in (:euler, :rk2, :rk4)
            configuration = particles_in_box(
                Float64,
                -0.5;
                x_max=grid.extent[1],
                y_max=grid.extent[2],
                nx=2,
                ny=2,
                integration_method=method,
            )
            initialize_particles!(model, configuration)
            initial_positions = copy(model.particles.particles.x)
            @test advect_particles!(model, 0.1; current_time=0.0) === model
            @test model.particles.particles.x ≈ initial_positions
            @test model.particles.particles.time ≈ 0.1
        end

        @testset "particles keep their depth and wrap around the domain" begin
            initialize_particles!(model, box)
            tracker = model.particles
            particles = tracker.particles
            fill!(tracker.current.u, 1.0)
            fill!(tracker.current.v, -0.5)
            x₀, y₀, z₀ = copy(particles.x), copy(particles.y), copy(particles.z)

            # Far enough to cross the periodic boundaries several times.
            QGYBJplus.UnifiedParticleAdvection._advance_particles!(tracker, 0.0, 20.0)

            @test particles.x ≈ mod.(x₀ .+ 20, 2π)
            @test particles.y ≈ mod.(y₀ .- 10, 2π)
            @test particles.z == z₀
            @test particles.u ≈ fill(1.0, 6)
            @test particles.v ≈ fill(-0.5, 6)
        end

        @testset "default particle and output precisions interoperate" begin
            default_box = particles_in_box(
                -0.5;
                x_max=grid.extent[1],
                y_max=grid.extent[2],
                nx=2,
                ny=2,
            )
            @test default_box isa ParticleConfig{Float32}
            initialize_particles!(model, default_box)
            tracker = model.particles

            mktempdir() do output_dir
                manager = ParticleOutputManager(
                    output_dir;
                    save_interval_iter=1,
                    save_interval_time=0.0,
                    output_mode=:trajectory,
                )
                operations_succeeded = try
                    QGYBJplus.setup_particle_output!(manager, tracker)
                    QGYBJplus.save_particle_positions!(manager, tracker, 0, 0.0)
                    QGYBJplus.finalize_particle_output!(manager, tracker)
                    true
                catch
                    false
                end

                @test operations_succeeded
                @test manager.time_series == [0.0]
                @test length(manager.x_series) == 1
                @test eltype(only(manager.x_series)) === Float64
                @test only(manager.x_series) ≈ Float64.(tracker.particles.x)
                @test isfile(joinpath(
                    output_dir, "particles", "particles_trajectory.nc"))
            end
        end

        @testset "automatic trajectory splitting is transactional" begin
            initialize_particles!(model, box)
            tracker = model.particles
            particles = tracker.particles
            empty!(particles.x_history)
            empty!(particles.y_history)
            empty!(particles.z_history)
            empty!(particles.id_history)
            empty!(particles.time_history)

            mktempdir() do output_dir
                base = joinpath(output_dir, "trajectory")
                @test_throws ArgumentError QGYBJplus.UnifiedParticleAdvection.enable_auto_file_splitting!(
                    tracker, base; max_points_per_file=0)
                QGYBJplus.UnifiedParticleAdvection.enable_auto_file_splitting!(
                    tracker, base; max_points_per_file=1)
                particles.time = 0.0
                QGYBJplus.UnifiedParticleAdvection.save_particle_state!(tracker)
                particles.time = 1.0
                QGYBJplus.UnifiedParticleAdvection.save_particle_state!(tracker)

                first_segment = "$(base).nc"
                @test isfile(first_segment)
                if isfile(first_segment)
                    NCDataset(first_segment, "r") do dataset
                        @test dataset["time"][:] ≈ [0.0]
                    end
                end
                @test particles.time_history ≈ [1.0]
                @test tracker.output_file_sequence == 1
            end

            initialize_particles!(model, box)
            tracker = model.particles
            particles = tracker.particles
            empty!(particles.x_history)
            empty!(particles.y_history)
            empty!(particles.z_history)
            empty!(particles.id_history)
            empty!(particles.time_history)
            mktempdir() do output_dir
                base = joinpath(output_dir, "missing", "trajectory")
                QGYBJplus.UnifiedParticleAdvection.enable_auto_file_splitting!(
                    tracker, base; max_points_per_file=1)
                particles.time = 0.0
                QGYBJplus.UnifiedParticleAdvection.save_particle_state!(tracker)
                particles.time = 1.0

                split_failed = try
                    QGYBJplus.UnifiedParticleAdvection.save_particle_state!(tracker)
                    false
                catch
                    true
                end

                @test split_failed
                @test particles.time_history ≈ [0.0]
                @test tracker.output_file_sequence == 0
            end
        end

        # A tracker that also keeps its in-memory history.
        initialize_particles!(model, particles_in_box(
            Float64, -0.5; x_max=grid.extent[1], y_max=grid.extent[2],
            nx=2, ny=3, save_interval=0.1))
        mktempdir() do output_dir
            particle_output = ParticleOutputManager(
                output_dir;
                save_interval_iter=1,
                save_interval_time=0.0,
                output_mode=:trajectory,
            )
            simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=2,
                output=false,
                particle_output=particle_output,
                verbose=false,
            )
            @test simulation.particle_output_manager === particle_output
            run!(simulation)
            @test model.particles.particles.time ≈ simulation.clock.time
            @test model.particles.particles.time_history ≈ [0.0, 0.1, 0.2]
            @test particle_output.save_count == 3
            @test particle_output.time_series ≈ [0.0, 0.1, 0.2]
            @test particle_output.closed
            trajectory_path = joinpath(
                output_dir, "particles", "particles_trajectory.nc")
            @test isfile(trajectory_path)
            NCDataset(trajectory_path, "r") do dataset
                @test dataset["time"][:] ≈ [0.0, 0.1, 0.2]
                @test size(dataset["x"]) == (6, 3)
            end
        end

        mktempdir() do output_dir
            # Make the first snapshot target a directory so NetCDF creation
            # fails after manager setup has succeeded.
            blocked_snapshot = joinpath(
                output_dir, "particles", "particles_000000.nc")
            mkpath(blocked_snapshot)
            particle_output = ParticleOutputManager(
                output_dir;
                save_interval_iter=1,
                save_interval_time=0.0,
                output_mode=:snapshots,
            )
            failing_simulation = Simulation(
                model;
                Δt=0.1,
                stop_iteration=1,
                output=false,
                particle_output=particle_output,
                verbose=false,
            )
            @test_throws Exception run!(failing_simulation)
            @test failing_simulation.state == Failed
            @test particle_output.save_count == 0
            @test particle_output.closed
        end
    finally
        finalize_model!(model)
    end
end

@testset "Single-precision particles advect with every integrator" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), verbose=false)
    try
        set!(model; ψ=(x, y, z) -> 0.1sin(x) * cos(y), pv_method=:barotropic,
             verbose=false)
        for method in (:euler, :rk2, :rk4)
            # particles_in_box defaults to Float32 storage.
            box = particles_in_box(-0.5; x_max=grid.extent[1], y_max=grid.extent[2],
                                   nx=2, ny=2, integration_method=method)
            initialize_particles!(model, box)
            advect_particles!(model, 0.1; current_time=0.0)
            positions = model.particles.particles.x
            @test eltype(positions) === Float32
            @test all(isfinite, positions)
        end
    finally
        finalize_model!(model)
    end
end

"""Model with a prescribed wave field B(x, y, z), no mean flow and no feedback."""
function wave_only_model(wave_at; f=1.0)
    grid = RectilinearGrid(size=(32, 32, 16), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=f),
        stratification=ConstantStratification(N²=1.0), flow=FixedFlow(),
        feedback=NoFeedback(), formulation=YBJPlus(), verbose=false)
    nx, ny, nz = grid.size
    wave = ComplexF64[wave_at(grid.x[i], grid.y[j], grid.z[k])
                      for k in 1:nz, i in 1:nx, j in 1:ny]
    set!(model; B=FieldArray(wave), verbose=false)
    return model
end

"""Advect particles in `model` for `periods` inertial periods; displacements."""
function particle_displacements(model, motion; periods=1, steps=128, f=1.0)
    initialize_particles!(model, particles_in_box(Float64, -0.3;
        x_min=0.5, x_max=5.5, y_min=0.5, y_max=5.5, nx=3, ny=3,
        wave_motion=motion, integration_method=:rk4,
        interpolation_method=TRICUBIC))
    particles = model.particles.particles
    x0, y0 = copy(particles.x), copy(particles.y)
    Δt = 2π / f / steps
    half = nothing
    for n in 0:(periods * steps - 1)
        advect_particles!(model, Δt; current_time=n * Δt)
        n + 1 == steps ÷ 2 && (half = hypot.(particles.x .- x0, particles.y .- y0))
    end
    return (horizontal=hypot.(particles.x .- x0, particles.y .- y0), half=half)
end

@testset "Particles follow the Lagrangian mean or the inertial oscillation" begin
    profile(z) = cos(π * (z + 1))   # zero depth mean
    # A horizontally uniform wave: no Stokes drift and no mean flow.
    LA = 0.05 + 0.02im
    model = wave_only_model((x, y, z) -> LA * profile(z))
    try
        averaged = particle_displacements(model, :averaged)
        @test maximum(averaged.horizontal) == 0
        # Instantaneous particles trace inertial circles of diameter 2|LA|/f.
        instantaneous = particle_displacements(model, :instantaneous)
        @test maximum(instantaneous.horizontal) < 1e-12
        @test instantaneous.half ≈ fill(2abs(LA * profile(-0.3)), 9) rtol=1e-3
    finally
        finalize_model!(model)
    end

    # Waves with horizontal structure and no mean flow: the Lagrangian mean is
    # zero, so over a period the orbits' own Stokes drift cancels the
    # Eulerian mean, -Stokes drift. Adding the drift instead would leave twice
    # the Stokes displacement.
    a = 0.01
    model = wave_only_model((x, y, z) -> a * profile(z) * (1 + 0.5cos(x) + 0.3im * sin(y)))
    try
        compute_velocities!(model)
        compute_wave_velocities!(model; include_wave_velocity=false)
        stokes = maximum(hypot.(model.fields.u, model.fields.v)) * 2π
        drift = particle_displacements(model, :instantaneous; steps=256)
        @test maximum(drift.horizontal) < 0.02 * stokes
    finally
        finalize_model!(model)
    end

    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), verbose=false)
    try
        @test_throws ArgumentError initialize_particles!(model,
            particles_in_box(Float64, -0.5; x_max=2π, y_max=2π, use_ybj_w=true))
        @test_throws ArgumentError initialize_particles!(model,
            particles_in_box(Float64, -0.5; x_max=2π, y_max=2π, wave_motion=:eulerian))
        # Particles move horizontally.
        @test_throws ArgumentError initialize_particles!(model,
            particles_in_box(Float64, -0.5; x_max=2π, y_max=2π, use_3d_advection=true))
        @test_throws ArgumentError particles_in_grid_3d(x_max=2π, y_max=2π, z_max=1.0,
            nx=2, ny=2, nz=2, use_3d_advection=true)
    finally
        finalize_model!(model)
    end
end

@testset "Particle defaults and method validation" begin
    configuration = ParticleConfig{Float64}(x_max=1.0, y_max=1.0, z_level=-0.5)
    @test configuration.integration_method === :rk2
    @test configuration.interpolation_method == TRICUBIC
    volume = particles_in_grid_3d(x_max=1.0, y_max=1.0, z_max=1.0,
                                  nx=2, ny=2, nz=2, precision=Float64)
    @test volume.integration_method === :rk2
    @test volume.interpolation_method == TRICUBIC

    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), verbose=false)
    try
        @test_throws ArgumentError initialize_particles!(model, particles_in_box(
            Float64, -0.5; x_max=2π, y_max=2π, integration_method=:rk3))
        @test_logs (:warn, r"ADAPTIVE") initialize_particles!(model,
            particles_in_box(Float64, -0.5; x_max=2π, y_max=2π,
                             interpolation_method=ADAPTIVE))
    finally
        finalize_model!(model)
    end

    # ADAPTIVE interpolates exactly as TRICUBIC.
    u = [sin(1.0i + 2.0j + 3.0k) for k in 1:4, i in 1:8, j in 1:8]
    v, w = cos.(u), u .^ 2
    info = (dx=2π / 8, dy=2π / 8, dz=0.25, Lx=2π, Ly=2π, Lz=1.0,
            z_min=-1.0, z_max=0.0, z0=-0.875)
    bc = (periodic_x=true, periodic_y=true)
    for (x, y, z) in ((0.3, 5.9, -0.97), (2.2, 1.1, -0.4), (6.1, 0.2, -0.02))
        @test interpolate_velocity_advanced(x, y, z, u, v, w, info, bc, ADAPTIVE) ==
              interpolate_velocity_advanced(x, y, z, u, v, w, info, bc, TRICUBIC)
    end
end

@testset "Particle interpolation reproduces grid values and converges" begin
    interpolate_fields = QGYBJplus.UnifiedParticleAdvection.interpolate_fields
    L = 2π
    # cos(πz) is even and sin(πz) odd about both z = 0 and z = -1, like the
    # horizontal and the vertical velocity, so their mirrored columns are
    # smooth and the order holds in the boundary cells too.
    even(x, y, z) = sin(x + 0.3) * cos(2y) * cos(π * z)
    odd(x, y, z) = sin(x + 0.3) * cos(2y) * sin(π * z)
    function sampled(n)
        dx, dz = L / n, 1 / n
        z0 = -1 + dz / 2
        values(f) = [f((i - 1) * dx, (j - 1) * dx, z0 + (k - 1) * dz)
                     for k in 1:n, i in 1:n, j in 1:n]
        info = (dx=dx, dy=dx, dz=dz, Lx=L, Ly=L, Lz=1.0,
                z_min=-1.0, z_max=0.0, z0=z0)
        return values(even), values(odd), info
    end
    bc = (periodic_x=true, periodic_y=true)
    # Quasi-random points over the whole domain, plus the boundaries.
    points = [(L * mod(0.618034m, 1), L * mod(0.414214m, 1), -mod(0.732051m, 1))
              for m in 1:200]
    append!(points, [(1.3, 2.1, 0.0), (4.4, 0.7, -1.0), (2.9, 5.5, -0.003)])
    expected_order = Dict(TRILINEAR => 1.8, TRICUBIC => 2.8, QUINTIC => 5.5)
    for method in (TRILINEAR, TRICUBIC, QUINTIC)
        errors = map((16, 32)) do n
            e, o, info = sampled(n)
            maximum(points) do (x, y, z)
                ie, io = interpolate_fields((e, o), (1, -1), x, y, z, info, bc, method)
                max(abs(ie - even(x, y, z)), abs(io - odd(x, y, z)))
            end
        end
        @test log2(errors[1] / errors[2]) > expected_order[method]

        e, o, info = sampled(8)
        for (k, i, j) in ((1, 1, 1), (2, 5, 8), (8, 3, 4))
            x, y = (i - 1) * info.dx, (j - 1) * info.dy
            z = info.z0 + (k - 1) * info.dz
            ie, io = interpolate_fields((e, o), (1, -1), x, y, z, info, bc, method)
            @test ie ≈ e[k, i, j] atol=1e-14
            @test io ≈ o[k, i, j] atol=1e-14
        end
        # Vertical velocity vanishes at the lid and the bottom.
        for z in (0.0, -1.0)
            @test abs(interpolate_fields((o,), (-1,), 1.3, 2.1, z, info, bc, method)[1]) < 1e-14
        end
    end
end

@testset "Particles stay in place until their release time" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), flow=FixedFlow(),
        feedback=NoFeedback(), verbose=false)
    try
        set!(model; ψ=(x, y, z) -> 0.1sin(x) * cos(y), verbose=false)
        initialize_particles!(model, particles_in_box(Float64, -0.5;
            x_min=0.3, x_max=2.3, y_min=0.4, y_max=2.4, nx=2, ny=2,
            particle_advec_time=0.25))
        particles = model.particles.particles
        x₀ = copy(particles.x)
        for n in 0:2
            advect_particles!(model, 0.1; current_time=0.1n)
        end
        @test particles.x == x₀
        @test particles.time ≈ 0.3
        advect_particles!(model, 0.1; current_time=0.3)
        @test particles.x != x₀
    finally
        finalize_model!(model)
    end
end

@testset "Particle trajectories are second-order accurate in time" begin
    # One Fourier mode ψ = -c(t) sin y decays under Laplacian damping,
    # c = exp(-t), which ETD-RK2 integrates exactly. Particles move with
    # u = c(t) cos y, so x(1) - x₀ = cos(y₀)(1 - e⁻¹). Starting on grid nodes
    # in y makes the interpolation exact, leaving only the time error.
    grid = RectilinearGrid(size=(16, 16, 4), extent=(2π, 2π, 1.0))
    y₀ = [grid.y[3], grid.y[6]]
    function displacement_error(method, steps; manual=false)
        model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0), flow=EvolvingFlow(),
            closure=FlowHyperdiffusivity(coefficient=1.0, order=2),
            feedback=NoFeedback(), verbose=false)
        try
            set!(model; ψ=(x, y, z) -> -sin(y), verbose=false)
            initialize_particles!(model, particles_custom(
                [(1.0, y₀[1], -0.5), (2.0, y₀[2], -0.5)]; precision=Float64,
                integration_method=method))
            particles = model.particles.particles
            x₀ = copy(particles.x)
            Δt = 1 / steps
            if manual
                timestepper = ExponentialRungeKutta2(Δt=Δt)
                for n in 0:(steps - 1)
                    step!(model, timestepper)
                    advect_particles!(model, Δt; current_time=n * Δt)
                end
            else
                run!(Simulation(model; Δt, stop_iteration=steps, output=false,
                                verbose=false))
            end
            return maximum(abs, (particles.x .- x₀) .- cos.(y₀) .* (1 - exp(-1)))
        finally
            finalize_model!(model)
        end
    end

    order(e) = log2(e[1] / e[2])
    rk2 = [displacement_error(:rk2, steps) for steps in (10, 20)]
    @test order(rk2) > 1.95
    # Linear interpolation in time: the trapezoidal rule's error.
    @test rk2[1] ≈ 0.1^2 / 12 * (1 - exp(-1)) * maximum(abs, cos.(y₀)) rtol=0.02
    @test order([displacement_error(:rk4, steps) for steps in (10, 20)]) > 1.95
    @test 0.9 < order([displacement_error(:euler, steps) for steps in (10, 20)]) < 1.1
    @test order([displacement_error(:rk2, steps; manual=true) for steps in (10, 20)]) > 1.9
end

"""Model with a steady depth-dependent flow, no damping, for output tests."""
function particle_output_test_model()
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        verbose=false)
    set!(model; ψ=(x, y, z) -> 0.3sin(x) * cos(y) * cos(π * z), verbose=false)
    return model
end

"""Times and (particle, time) matrices of positions and velocities in a run's output."""
function read_particle_output(path, mode)
    directory = joinpath(path, "particles")
    if mode === :snapshots
        files = sort(filter(name -> endswith(name, ".nc"), readdir(directory)))
        snapshots = [NCDataset(joinpath(directory, file), "r") do dataset
                         (dataset.attrib["time"],
                          [dataset[name][:] for name in ("x", "y", "z", "u", "v")])
                     end for file in files]
        columns(k) = reduce(hcat, [last(snapshot)[k] for snapshot in snapshots])
        return (time=first.(snapshots), x=columns(1), y=columns(2), z=columns(3),
                u=columns(4), v=columns(5))
    end
    file = mode === :trajectory ? "particles_trajectory.nc" : "particles_stream.nc"
    NCDataset(joinpath(directory, file), "r") do dataset
        (time=dataset["time"][:], x=dataset["x"][:, :], y=dataset["y"][:, :],
         z=dataset["z"][:, :], u=dataset["u"][:, :], v=dataset["v"][:, :])
    end
end

@testset "Particle output modes agree and store velocities at the recorded times" begin
    records = Dict{Symbol, Any}()
    run_with(model, path, mode; overwrite=false) = redirect_stdout(devnull) do
        simulation = Simulation(model; Δt=0.05, stop_iteration=3, output=false,
            particle_output=ParticleOutputManager(path; save_interval_iter=1,
                save_interval_time=0.0, output_mode=mode, overwrite),
            verbose=false)
        run!(simulation)
    end
    for mode in (:trajectory, :streaming, :snapshots)
        model = particle_output_test_model()
        try
            initialize_particles!(model, particles_in_box(Float64, -0.4;
                x_min=0.2, x_max=6.0, y_min=0.3, y_max=6.1, nx=3, ny=2))
            mktempdir() do path
                run_with(model, path, mode)
                records[mode] = read_particle_output(path, mode)
                @test records[mode].time ≈ [0.0, 0.05, 0.1, 0.15]
                tracker = model.particles
                particles = tracker.particles
                @test records[mode].x[:, end] == particles.x
                velocity = [interpolate_velocity_at_position(particles.x[i],
                                particles.y[i], particles.z[i], tracker, particles.time)
                            for i in 1:particles.np]
                @test records[mode].u[:, end] == first.(velocity)
                @test records[mode].v[:, end] == last.(velocity)
                # Particles keep their depth.
                @test all(records[mode].z .== records[mode].z[:, 1])
                # The initial record holds the initial velocities too.
                @test all(>(0), records[mode].u[:, 1] .^ 2 .+ records[mode].v[:, 1] .^ 2)

                # Another run may not replace these files, unless asked to.
                @test_throws ArgumentError run_with(model, path, mode)
                run_with(model, path, mode; overwrite=true)
                @test length(read_particle_output(path, mode).time) == 4
            end
        finally
            finalize_model!(model)
        end
    end
    @test records[:streaming] == records[:trajectory]
    @test records[:snapshots] == records[:trajectory]
end

@testset "Tracker history is opt-in and keeps its schedule" begin
    model = particle_output_test_model()
    box(; kwargs...) = particles_in_box(Float64, -0.5; x_max=2π, y_max=2π,
                                        nx=2, ny=1, kwargs...)
    advance!(steps) = foreach(n -> advect_particles!(model, 0.05;
                                                      current_time=0.05n), 0:(steps - 1))
    try
        mktempdir() do path
            initialize_particles!(model, box())
            advance!(4)
            @test isempty(model.particles.particles.time_history)
            @test_logs (:warn, r"save_interval") write_particle_trajectories(
                joinpath(path, "none.nc"), model.particles)

            # Records fall on the nominal times despite rounding (0.25 + 0.05
            # is below 0.3 in floating point).
            initialize_particles!(model, box(save_interval=0.1))
            advance!(10)
            history = model.particles.particles
            @test history.time_history ≈ 0.0:0.1:0.5

            file = joinpath(path, "trajectories.nc")
            write_particle_trajectories(file, model.particles)
            trajectories = read_particle_trajectories(file)
            @test trajectories.x == reduce(hcat, history.x_history)
            @test trajectories.z == reduce(hcat, history.z_history)
            @test trajectories.time == history.time_history

            # A full history warns once instead of dropping records silently.
            initialize_particles!(model, box(save_interval=0.0, max_save_points=3))
            @test_logs (:warn, r"max_save_points") advance!(6)
            @test length(model.particles.particles.time_history) == 3
        end
    finally
        finalize_model!(model)
    end
end

@testset "Standalone particle files grow and report errors" begin
    model = particle_output_test_model()
    try
        initialize_particles!(model, particles_in_box(Float64, -0.5;
            x_max=2π, y_max=2π, nx=2, ny=2))
        tracker = model.particles
        mktempdir() do path
            file = joinpath(path, "series.nc")
            create_particle_output_file(file, tracker)
            for n in 0:2
                advect_particles!(model, 0.05; current_time=0.05n)
                QGYBJplus.ParticleIO.append_particle_data!(file, tracker, 0)
            end
            NCDataset(file, "r") do dataset
                @test dataset["time"][:] ≈ [0.05, 0.1, 0.15]
                @test dataset["x"][:, end] == tracker.particles.x
                @test dataset["u"][:, end] == tracker.particles.u
            end
            missing_directory = joinpath(path, "missing")
            @test_throws Exception write_particle_snapshot(
                joinpath(missing_directory, "snapshot.nc"), tracker, 0.0)
            @test_throws Exception create_particle_output_file(
                joinpath(missing_directory, "series.nc"), tracker)
        end
    finally
        finalize_model!(model)
    end
end
