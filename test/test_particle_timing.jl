using Test
using NCDatasets
using QGYBJplus

const UPA = QGYBJplus.UnifiedParticleAdvection

particle_test_model(; f=1.0, n=8, kwargs...) = QGYBJModel(;
    grid=RectilinearGrid(size=(n, n, 4), extent=(1e5, 1e5, 1000.0)),
    coriolis=FPlane(f=f), stratification=ConstantStratification(N²=1e-5),
    verbose=false, kwargs...)

cellular_flow(x, y, z) = 1e3 * sin(2π * x / 1e5) * cos(2π * y / 1e5)

@testset "Particle settings are times in Float64" begin
    config = particles_in_box(Float32, -500.0; x_max=1e5, y_max=1e5, nx=2, ny=2,
                              save_interval=0.1, particle_advec_time=0.3)
    @test config.save_interval === 0.1
    @test config.particle_advec_time === 0.3

    # Stored as Float32, 0.1 rounded up and the record at t = 0.1 was skipped,
    # and the release came a step late.
    model = particle_test_model()
    initialize_particles!(model, config)
    simulation = Simulation(model; Δt=0.1, stop_iteration=6, output=false,
                            diagnostics=false, verbose=false)
    set!(simulation; ψ=cellular_flow)
    run!(simulation)
    particles = model.particles.particles
    @test particles.time_history ≈ (0:6) .* 0.1
    @test particles.y_history[4] == particles.y_history[1]   # t = 0.3: not yet moved
    @test particles.y_history[5] != particles.y_history[4]   # moved over 0.3 to 0.4
end

@testset "Particle helpers default to Float64" begin
    # Float32 positions near x = 1000 km round steps of a few cm away.
    @test particles_in_box(-500.0; x_max=1e6, y_max=1e6) isa ParticleConfig{Float64}
    @test particles_in_grid_3d(x_max=1e6, y_max=1e6, z_max=1000.0) isa ParticleConfig3D{Float64}
    @test particles_in_layers([-100.0]; x_max=1e6, y_max=1e6) isa ParticleConfig3D{Float64}
    @test particles_random_3d(4; x_max=1e6, y_max=1e6, z_max=1000.0) isa ParticleConfig3D{Float64}
    @test particles_custom([(1.0, 2.0, -3.0)]) isa ParticleConfig3D{Float64}
    @test particles_in_circle(-500.0; center=(5e5, 5e5), radius=1e4) isa ParticleConfig3D{Float64}
    @test particles_in_circle(-500.0; center=(5e5, 5e5), radius=1e4,
                              precision=Float32) isa ParticleConfig3D{Float32}

    # A slow, uniform flow on a large domain moves every particle the same way.
    model = QGYBJModel(grid=RectilinearGrid(size=(8, 8, 2), extent=(1e6, 1e6, 1000.0)),
        coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
        flow=FixedFlow(), verbose=false)
    speed = 4e-4
    set!(model; ψ=(x, y, z) -> -speed * 1e6 / (2π) * sin(2π * y / 1e6), pv_method=:none)
    initialize_particles!(model, particles_in_box(-500.0; x_max=1e6, y_max=1e6, nx=4, ny=1))
    simulation = Simulation(model; Δt=60.0, stop_iteration=1440, output=false,
                            diagnostics=false, verbose=false)
    start = copy(model.particles.particles.x)
    run!(simulation)
    moved = model.particles.particles.x .- start
    @test all(isapprox.(moved, speed * 86400; rtol=1e-6))
end

@testset "Particle helpers place what they promise" begin
    # The top level used to be dropped.
    config = particles_in_grid_3d(x_max=1.0, y_max=1.0, z_min=-500.0, z_max=-100.0,
                                  nx=1, ny=1, nz=5)
    model = particle_test_model()
    initialize_particles!(model, config)
    @test sort(Array(model.particles.particles.z)) ≈ [-500.0, -400.0, -300.0, -200.0, -100.0]

    # The random disk honours its seed.
    disk(seed) = particles_in_circle(-500.0; center=(5e4, 5e4), radius=1e4, n=20,
                                     pattern=:random, seed)
    @test disk(1).custom_x == disk(1).custom_x
    @test disk(1).custom_x != disk(2).custom_x
    @test all(hypot.(disk(1).custom_x .- 5e4, disk(1).custom_y .- 5e4) .<= 1e4 * (1 + 1e-12))
    @test_throws ArgumentError particles_in_circle(-500.0; center=(0.0, 0.0), radius=-1.0)
    @test_throws ArgumentError particles_in_circle(-500.0; center=(0.0, 0.0), radius=1.0,
                                                   pattern=:spiral)
end

@testset "Particle configurations are validated" begin
    @test_throws ArgumentError particles_in_box(-500.0; x_max=1.0, y_max=1.0, save_interval=-1.0)
    @test_throws ArgumentError particles_in_box(-500.0; x_max=1.0, y_max=1.0, max_save_points=0)
    @test_throws ArgumentError particles_in_box(-500.0; x_max=1.0, y_max=1.0, nx=0)
    @test_throws ArgumentError particles_in_box(-500.0; x_max=1.0, y_max=1.0,
                                                particle_advec_time=NaN)
    @test_throws ArgumentError particles_in_box(NaN; x_max=1.0, y_max=1.0)
    # ParticleConfig3D raised AssertionErrors, which can be compiled out.
    @test_throws ArgumentError ParticleConfig3D{Float64}(x_max=1.0, y_max=1.0, z_min=0.0,
                                                         z_max=-1.0)
    @test_throws ArgumentError particles_in_layers([-1.0, -1.0]; x_max=1.0, y_max=1.0)
    @test_throws ArgumentError particles_custom(Tuple{Float64, Float64, Float64}[])

    # A depth given as a positive number lies above the lid: an error, rather
    # than particles clamped to the surface.
    model = particle_test_model()
    @test_throws ArgumentError initialize_particles!(
        model, particles_in_box(500.0; x_max=1e5, y_max=1e5))
    @test model.particles === nothing
end

@testset "Re-initializing a tracker replaces its particles" begin
    model = particle_test_model()
    initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5,
                                                  nx=3, ny=3, save_interval=0.0))
    tracker = model.particles
    smaller = particles_in_box(-200.0; x_max=1e5, y_max=1e5, nx=2, ny=1,
                               integration_method=:euler, wave_motion=:instantaneous)
    initialize_particles!(tracker, smaller)
    particles = tracker.particles
    @test particles.np == 2
    @test length(particles.x) == length(particles.z) == length(particles.id) == 2
    @test all(==(-200.0), Array(particles.z))
    @test tracker.config === smaller
    @test isempty(particles.time_history)   # the new config keeps no history
    @test !tracker.current.valid
end

@testset "Particles join the time axis of each run" begin
    # Restored: the initial record takes the restart time.
    mktempdir() do directory
        model = particle_test_model(f=1e-4)
        simulation = Simulation(model; Δt=10.0, stop_iteration=3,
            output=NetCDFOutput(path=directory, schedule=IterationInterval(1)),
            diagnostics=false, verbose=false)
        set!(simulation; ψ=cellular_flow)
        run!(simulation)

        restored_model = particle_test_model(f=1e-4)
        restored = Simulation(restored_model; Δt=10.0, stop_iteration=6,
                              output=false, diagnostics=false, verbose=false)
        initialize_particles!(restored, particles_in_box(-500.0; x_max=1e5, y_max=1e5,
                                                         nx=2, ny=2, save_interval=20.0))
        restore!(restored, joinpath(directory, "state0004.nc"))
        run!(restored)
        # Recorded at the restart time, then at multiples of 20 s.
        @test restored_model.particles.particles.time_history ≈ [30.0, 40.0, 60.0]
    end

    # Continued with a new simulation: released particles keep moving, and
    # the history continues, with a warning that the time axis restarts.
    model = particle_test_model()
    initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5,
                                                  nx=2, ny=2, save_interval=0.1,
                                                  particle_advec_time=0.15))
    first_run = Simulation(model; Δt=0.1, stop_iteration=3, output=false,
                           diagnostics=false, verbose=false)
    set!(first_run; ψ=cellular_flow)
    run!(first_run)
    @test model.particles.released
    before = copy(Array(model.particles.particles.y))
    records = length(model.particles.particles.time_history)
    second_run = Simulation(model; Δt=0.1, stop_iteration=1, output=false,
                            diagnostics=false, verbose=false)
    @test_logs (:warn, r"continue from t = 0.0") match_mode=:any run!(second_run)
    # The clock restarted below particle_advec_time, which no longer holds them.
    @test Array(model.particles.particles.y) != before
    @test model.particles.particles.time_history[(records + 1):end] ≈ [0.0, 0.1]
end

@testset "Tracker-level advection uses its model's coefficients" begin
    model = particle_test_model(f=1e-4)
    set!(model; ψ=cellular_flow, waves=SurfaceWave(amplitude=0.1, scale=200.0))
    config = particles_in_box(-100.0; x_max=1e5, y_max=1e5, nx=2, ny=2,
                              wave_motion=:instantaneous)
    initialize_particles!(model, config)
    advect_particles!(model, 60.0; current_time=0.0)
    expected = (copy(Array(model.particles.particles.x)),
                copy(Array(model.particles.particles.y)))

    initialize_particles!(model, config)
    tracker = model.particles
    # f = 1 and N² = 1 used to be the defaults here.
    advect_particles!(tracker, model.fields, model.runtime.geometry, 60.0, 0.0)
    @test Array(tracker.particles.x) == expected[1]
    @test Array(tracker.particles.y) == expected[2]
    @test tracker.inertial_frequency == 1e-4
end

@testset "Interpolation needs a recorded velocity" begin
    model = particle_test_model()
    initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5))
    # It returned (0, 0) before the first snapshot.
    @test_throws ArgumentError interpolate_velocity_at_position(
        5e4, 5e4, -500.0, model.particles)
    advect_particles!(model, 1.0)
    @test all(isfinite, interpolate_velocity_at_position(5e4, 5e4, -500.0, model.particles))
end

@testset "Particle files sort and are named uniquely" begin
    mktempdir() do directory
        model = particle_test_model()
        initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5,
                                                      nx=2, ny=2))
        manager = ParticleOutputManager(directory; save_interval_iter=1,
                                        output_mode=:snapshots)
        run!(Simulation(model; Δt=1.0, stop_iteration=2, output=false,
                        diagnostics=false, particle_output=manager, verbose=false))
        names = filter(endswith(".nc"), readdir(joinpath(directory, "particles")))
        @test names == ["particles_00000000.nc", "particles_00000001.nc",
                        "particles_00000002.nc"]
    end

    # Levels closer than the old 3-decimal rounding no longer collide.
    mktempdir() do directory
        model = particle_test_model()
        initialize_particles!(model, particles_in_layers(
            [-123.4, -123.1]; x_max=1e5, y_max=1e5, nx=2, ny=2, save_interval=0.0))
        files = write_particle_trajectories_by_zlevel(
            joinpath(directory, "levels"), model.particles)
        @test length(files) == 2
        @test allunique(values(files))
        @test all(isfile, values(files))
        @test sort(basename.(collect(values(files)))) ==
              ["levels_z-123p1.nc", "levels_z-123p4.nc"]
    end

    # Splitting needs a history.
    model = particle_test_model()
    initialize_particles!(model, particles_in_box(-500.0; x_max=1e5, y_max=1e5))
    @test_throws ArgumentError UPA.enable_auto_file_splitting!(model.particles, "unused")
end

# The integrators by themselves, with an analytic cellular velocity.
struct CellularVelocity end
UPA._particle_velocity(x, y, z, τ, ::CellularVelocity) =
    (-sin(x) * cos(y), cos(x) * sin(y))

function integrate_cellular(method, Δt, duration)
    x, y = 0.3, 1.1
    for _ in 1:round(Int, duration / Δt)
        x, y = UPA._integrate(Val(method), x, y, 0.0, 0.0, Δt, CellularVelocity())
    end
    return [x, y]
end

@testset "Particle integrators converge at their order" begin
    # A velocity that varies along the trajectory, so every stage position
    # matters (with RK2's midpoint at the start, all 169 earlier particle
    # tests still passed).
    for (method, low, high) in ((:euler, 1.8, 2.2), (:rk2, 3.6, 4.4), (:rk4, 12.0, 20.0))
        reference = integrate_cellular(method, 1e-4, 2.0)
        errors = [maximum(abs, integrate_cellular(method, Δt, 2.0) .- reference)
                  for Δt in (0.1, 0.05, 0.025)]
        ratios = errors[1:2] ./ errors[2:3]
        @test all(ratio -> low < ratio < high, ratios)
    end

    # The same through a model with the steady flow ψ = sin x sin y: the
    # particles follow its streamlines, and the errors fall at the method's order.
    grid = RectilinearGrid(size=(32, 32, 2), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), flow=FixedFlow(),
        closure=HorizontalHyperdiffusivity(flow=FlowHyperdiffusivity(coefficient=0),
                                           wave=WaveHyperdiffusivity(coefficient=0)),
        verbose=false)
    set!(model; ψ=(x, y, z) -> sin(x) * sin(y), pv_method=:none)
    starts = [(0.3, 1.1, -0.5), (2.0, 0.7, -0.5), (4.0, 5.0, -0.5)]
    function final_positions(method, interpolation, Δt)
        initialize_particles!(model, particles_custom(starts;
            integration_method=method, interpolation_method=interpolation))
        run!(Simulation(model; Δt, stop_time=2.0, output=false, diagnostics=false,
                        verbose=false))
        return copy(model.particles.particles.x), copy(model.particles.particles.y)
    end
    for (method, interpolation, low, high, drift) in
            ((:rk2, TRICUBIC, 3.6, 4.4, 1e-4), (:rk4, QUINTIC, 12.0, 20.0, 1e-8))
        x, y = final_positions(method, interpolation, 1 / 512)
        errors = map((1 / 8, 1 / 16, 1 / 32)) do Δt
            xΔ, yΔ = final_positions(method, interpolation, Δt)
            maximum(hypot.(xΔ .- x, yΔ .- y))
        end
        @test all(ratio -> low < ratio < high, errors[1:2] ./ errors[2:3])
        ψ₀ = [sin(x₀) * sin(y₀) for (x₀, y₀, _) in starts]
        @test maximum(abs, sin.(x) .* sin.(y) .- ψ₀) < drift
    end
end
