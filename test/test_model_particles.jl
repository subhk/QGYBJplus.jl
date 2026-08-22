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
            flow=0, flow2=0, waves=0, waves2=0),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
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

        for interpolation in (TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC)
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
            fill!(tracker.u_field, 1.0)
            fill!(tracker.v_field, -0.5)
            fill!(tracker.w_field, 0.25)
            velocity = interpolate_velocity_at_position(
                Float64(π), Float64(π), -0.5, tracker)
            @test all(isapprox.(velocity, (1.0, -0.5, 0.25)))
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

        initialize_particles!(model, box)
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
            @test last(model.particles.particles.time_history) ≈
                  simulation.clock.time
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
