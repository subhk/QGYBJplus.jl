using Test
using FFTW
using SpecialFunctions: erf
using QGYBJplus

# Exported names that the other test files do not exercise directly.

api_test_model(; kwargs...) = QGYBJModel(;
    grid=RectilinearGrid(size=(8, 8, 8), extent=(1e5, 1e5, 1000.0)),
    coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
    verbose=false, kwargs...)

@testset "Stratification profiles evaluate their formulas" begin
    @test evaluate_N2(ConstantN{Float64}(0.01), -123.0) ≈ 1e-4

    skewed = SkewedGaussian{Float64}(1e-5, 4e-5, 200.0, 300.0, 1.5)
    depth = 250.0
    @test evaluate_N2(skewed, -depth) ≈
          4e-5 * exp(-(depth - 300)^2 / 200^2) * (1 + erf(1.5 * (depth - 300) / (200 * sqrt(2)))) +
          1e-5

    tanh_profile = TanhProfile{Float64}(0.01, 0.002, 300.0, 50.0)
    @test evaluate_N2(tanh_profile, -300.0) ≈ ((0.01 + 0.002) / 2)^2
    @test evaluate_N2(tanh_profile, 0.0) ≈ 0.01^2 rtol=1e-4
    @test evaluate_N2(tanh_profile, -1e4) ≈ 0.002^2 rtol=1e-4

    exponential = ExponentialProfile{Float64}(0.01, 200.0, 0.001)
    @test evaluate_N2(exponential, 0.0) ≈ 0.01^2
    @test evaluate_N2(exponential, -200.0) ≈ (0.001 + 0.009 / ℯ)^2

    grid = RectilinearGrid(size=(4, 4, 4), extent=(1.0, 1.0, 1000.0))
    @test compute_stratification_profile(exponential, grid) ≈
          [evaluate_N2(exponential, z) for z in grid.z]
end

@testset "Time and period helpers" begin
    model = api_test_model()
    simulation = Simulation(model; Δt=30.0, stop_iteration=10, output=false,
                            diagnostics=false, verbose=false)
    @test get_time(simulation, 4) == 120.0
    @test inertial_period(model) ≈ 2π / 1e-4
    @test get_inertial_period(model) == inertial_period(model)
    @test inertial_period(simulation) == inertial_period(model)
    @test get_inertial_period(simulation) == inertial_period(model)
    @test Clock() isa Clock{Float64}
    @test Clock().time == 0 && Clock().iteration == 0
    @test simulation.state === Ready
    @test all(state -> state isa SimulationState, (Ready, Running, Stopped, Failed, Finalized))
end

@testset "Surface-wave and mean-flow setters" begin
    model = api_test_model()
    grid = model.grid
    top = last(grid.z_faces)
    # The profile is taken at each cell's upper face, and its depth mean is
    # removed (the wave velocity has no barotropic part).
    depths = [top - z - grid.dz / 2 for z in grid.z]
    function check(raw)
        B = real(ifft(Array(model.fields.B), (2, 3)))
        @test B[:, 1, 1] ≈ raw .- sum(raw) / length(raw)
        @test B[:, 3, 5] ≈ B[:, 1, 1]   # horizontally uniform
    end
    set_surface_waves!(model; amplitude=0.1, surface_depth=200.0, verbose=false)
    check(0.1 .* exp.(-(depths .^ 2) ./ 200^2))
    set_surface_waves!(model; amplitude=0.1, surface_depth=200.0, profile=:exponential,
                       verbose=false)
    check(0.1 .* exp.(-depths ./ 200))
    set_exponential_surface_waves!(model; amplitude=0.1, efold_depth=200.0, verbose=false)
    check(0.1 .* exp.(-depths ./ 200))
    @test_throws ArgumentError set_surface_waves!(model; amplitude=0.1, surface_depth=-1.0)
    @test_throws MethodError set_surface_waves!(model; amplitude=0.1, surface_depth=1.0,
                                                uniform=false)   # removed: never used

    flow(x, y, z) = 1e3 * sin(2π * x / 1e5) * (1 + z / 1000)
    set_mean_flow!(model; psi_func=flow, verbose=false)
    ψ = real(ifft(Array(model.fields.psi), (2, 3)))
    @test ψ ≈ [flow(x, y, z) for z in grid.z, x in grid.x, y in grid.y]
end

@testset "Model component types" begin
    model = api_test_model()
    @test model.fields isa ModelFields
    @test model.physics isa ModelPhysics
    @test model.numerics isa ModelNumerics
    @test model.runtime isa ModelRuntime
    @test model.runtime.coefficients isa OperatorCoefficients
    @test model.physics.coriolis isa AbstractCoriolis
    @test model.physics.stratification isa AbstractStratification
    @test model.physics.flow isa FlowEvolution
    @test model.physics.feedback isa FeedbackMode
    @test model.physics.formulation isa WaveFormulation
    @test model.numerics.closure isa AbstractClosure
    @test model.numerics.dissipation isa DissipationMode
    @test model.numerics.dynamics isa DynamicsMode
    @test model.numerics.dispersion isa DispersionMode
    @test TimeInterval(1.0) isa AbstractSchedule
    @test architecture(model) isa AbstractArchitecture
end

@testset "Particle types and distributions" begin
    @test all(kind -> kind isa ParticleDistribution, (UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM))
    @test particles_in_layers([-100.0]; x_max=1.0, y_max=1.0).distribution_type == LAYERED
    @test particles_random_3d(3; x_max=1.0, y_max=1.0, z_max=1.0).distribution_type == RANDOM_3D
    @test particles_in_grid_3d(x_max=1.0, y_max=1.0, z_max=1.0).distribution_type == UNIFORM_GRID
    @test particles_custom([(0.5, 0.5, -0.5)]).distribution_type == CUSTOM
    state = ParticleState{Float64}(3)
    @test state.np == 3 && state.time == 0.0 && state.id == [1, 2, 3]
    @test isempty(state.time_history)

    # A layered configuration on a model, written level by level.
    model = api_test_model()
    initialize_particles!(model, particles_in_layers([-250.0, -750.0]; x_max=1e5, y_max=1e5,
                                                     nx=2, ny=2, save_interval=60.0))
    run!(Simulation(model; Δt=60.0, stop_iteration=2, output=false, diagnostics=false,
                    verbose=false))
    @test model.particles.particles.time_history ≈ [0.0, 60.0, 120.0]
    mktempdir() do directory
        files = write_particle_trajectories_by_zlevel(joinpath(directory, "p"), model.particles)
        @test sort(collect(keys(files))) ≈ [-750.0, -250.0]
        for (level, file) in files
            trajectories = read_particle_trajectories(file)
            @test all(==(level), trajectories.z)
            @test trajectories.time ≈ [0.0, 60.0, 120.0]
            @test size(trajectories.x) == (4, 3)
        end
    end
end
