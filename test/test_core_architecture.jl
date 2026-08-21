using Test
using QGYBJplus

@testset "Composition-first type identities" begin
    grid = RectilinearGrid(size = (8, 8, 4),
                           x = (-π, π),
                           y = (-π, π),
                           z = (-1.0, 0.0))

    @test nameof(typeof(grid)) == :RectilinearGrid
    @test !isdefined(QGYBJplus, :RectilinearGridSpec)

    model = QGYBJModel(grid = grid,
                       coriolis = FPlane(f = 1.0),
                       stratification = ConstantStratification(N² = 1.0),
                       closure = HorizontalHyperdiffusivity(flow = 0,
                                                             flow2 = 0,
                                                             waves = 0,
                                                             waves2 = 0),
                       verbose = false)

    @test nameof(typeof(model)) == :QGYBJModel
    @test model.grid === grid
    @test hasproperty(model, :fields)
    @test hasproperty(model, :physics)
    @test hasproperty(model, :runtime)

    simulation = Simulation(model;
                            Δt = 0.01,
                            stop_iteration = 1,
                            output = false)

    @test nameof(typeof(simulation)) == :Simulation
    has_model = hasproperty(simulation, :model)
    @test has_model
    has_model && @test simulation.model === model
    @test typeof(model) !== typeof(simulation)
end
