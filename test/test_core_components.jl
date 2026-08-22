using Test
using QGYBJplus

@testset "RectilinearGrid geometry" begin
    grid = RectilinearGrid(size = (8, 6, 4),
                           x = (-4.0, 4.0),
                           y = (-3.0, 3.0),
                           z = (-2.0, 0.0))

    @test nameof(typeof(grid)) == :RectilinearGrid
    @test grid.size == (8, 6, 4)
    @test grid.extent == (8.0, 6.0, 2.0)
    @test grid.origin == (-4.0, -3.0)

    geometry_properties = (:x, :y, :z, :x_faces, :y_faces, :z_faces,
                           :dx, :dy, :dz, :kx, :ky, :kh2)
    for property in geometry_properties
        @test hasproperty(grid, property)
    end

    if all(property -> hasproperty(grid, property), geometry_properties)
        # Horizontal Fourier collocation nodes start at the declared origin.
        @test grid.x == collect(-4.0:1.0:3.0)
        @test grid.y == collect(-3.0:1.0:2.0)
        @test grid.z == [-1.75, -1.25, -0.75, -0.25]
        @test grid.x_faces == collect(-4.0:1.0:4.0)
        @test grid.y_faces == collect(-3.0:1.0:3.0)
        @test grid.z_faces == collect(-2.0:0.5:0.0)
        @test (grid.dx, grid.dy, grid.dz) == (1.0, 1.0, 0.5)
        @test grid.kx[1] == 0
        @test grid.ky[1] == 0
        @test size(grid.kh2) == (8, 6)
        @test grid.kh2[2, 2] == grid.kx[2]^2 + grid.ky[2]^2
    end

    centered = RectilinearGrid(size = (4, 4, 2),
                               extent = (8, 4, 2),
                               centered = true)
    @test centered.origin == (-4.0, -2.0)

    @test_throws ArgumentError RectilinearGrid(size = (0, 8, 4),
                                                extent = (1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, -1, 1))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, 1, 1),
                                                x = (0, 2))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, 1, 1),
                                                centered = true,
                                                x = (-0.5, 0.5))
end

@testset "Typed physics components" begin
    @test FPlane(f = 1) isa FPlane{Float64}
    @test ConstantStratification(N² = 1) isa ConstantStratification{Float64}
    @test_throws ArgumentError FPlane(f = 0)
    @test_throws ArgumentError ConstantStratification(N² = 0)

    N²_function = z -> 1e-5 + z^2 * 1e-10
    N²_profile = AnalyticalProfile(N²_function; returns=:N²)
    @test evaluate_N2(N²_profile, -100.0) == N²_function(-100.0)
    N_profile = AnalyticalProfile(z -> 1e-2; returns=:N,
                                  precision=Float32)
    @test N_profile isa AnalyticalProfile{Float32}
    @test evaluate_N2(N_profile, -100.0) ≈ 1e-4
    @test_throws ArgumentError AnalyticalProfile(N²_function; returns=:invalid)

    flow_closure = FlowHyperdiffusivity(coefficient=0)
    wave_closure = WaveHyperdiffusivity(coefficient=1e5)
    closure = HorizontalHyperdiffusivity(
        flow=flow_closure,
        wave=wave_closure,
    )

    @test flow_closure isa FlowHyperdiffusivity
    @test flow_closure.coefficients == (0.0,)
    @test flow_closure.orders == (4,)
    @test wave_closure isa WaveHyperdiffusivity
    @test wave_closure.coefficients == (1e5,)
    @test wave_closure.orders == (4,)
    @test closure.flow === flow_closure
    @test closure.wave === wave_closure

    defaults = HorizontalHyperdiffusivity()
    @test defaults.flow.coefficients == (0.01, 10.0)
    @test defaults.flow.orders == (4, 12)
    @test defaults.wave.coefficients == (0.0, 10.0)
    @test defaults.wave.orders == (4, 12)

    @test FlowHyperdiffusivity(2; order=6).orders == (6,)
    @test WaveHyperdiffusivity(2; order=6).orders == (6,)
    @test_throws ArgumentError FlowHyperdiffusivity(coefficient=-1)
    @test_throws ArgumentError FlowHyperdiffusivity(
        coefficient=1e5, order=3)
    @test_throws ArgumentError WaveHyperdiffusivity(coefficient=-1)
    @test_throws ArgumentError WaveHyperdiffusivity(
        coefficient=1e5, order=3)

    kx, ky, dt = 3.0, 4.0, 2.0
    @test QGYBJplus.int_factor(kx, ky, dt, closure) == 0
    @test QGYBJplus.int_factor(kx, ky, dt, closure; waves=true) ==
          dt * 1e5 * (kx^2 + ky^2)^2
    @test QGYBJplus.int_factor(
        kx,
        ky,
        dt,
        FlowHyperdiffusivity(coefficient=0.25),
    ) == dt * 0.25 * (kx^2 + ky^2)^2
    @test QGYBJplus.int_factor(kx, ky, dt, wave_closure) == 0

    component_types = (:AbstractCoriolis, :AbstractStratification,
                       :FlowEvolution, :FixedFlow, :EvolvingFlow,
                       :FeedbackMode, :NoFeedback, :WaveMeanFeedback,
                       :NoWaveFeedback, :WaveFormulation, :YBJPlus, :YBJ)
    for type_name in component_types
        @test isdefined(QGYBJplus, type_name)
    end

    if all(type_name -> isdefined(QGYBJplus, type_name), component_types)
        @test FixedFlow() isa FlowEvolution
        @test EvolvingFlow() isa FlowEvolution
        @test NoFeedback() isa FeedbackMode
        @test WaveMeanFeedback() isa FeedbackMode
        @test NoWaveFeedback() isa FeedbackMode
        @test YBJPlus() isa WaveFormulation
        @test YBJ() isa WaveFormulation
    end
end
