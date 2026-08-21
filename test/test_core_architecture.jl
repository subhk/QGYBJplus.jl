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

    @testset "legacy API is absent" begin
        forbidden_symbols = (
            :Grid,
            :State,
            :QGParams,
            :RectilinearGridSpec,
            :QGYBJSimulation,
            :default_params,
            :setup_model,
            :initialize_simulation,
            :setup_simulation,
            :run_simulation!,
            :create_simple_config,
            :run_simple_simulation,
            :DomainConfig,
            :ModelConfig,
        )
        for symbol in forbidden_symbols
            @test !isdefined(QGYBJplus, symbol)
        end

        @test !hasproperty(model.runtime, :parameters)
        @test !hasproperty(model.runtime, :computational_grid)
    end

    finalize_simulation!(simulation)
end

@testset "source contains one data model" begin
    source_directory = normpath(joinpath(@__DIR__, "..", "src"))
    source = join(
        (read(joinpath(root, file), String)
         for (root, _, files) in walkdir(source_directory)
         for file in files if endswith(file, ".jl")),
        '\n',
    )
    for legacy_name in ("Grid", "State", "QGParams",
                        "RectilinearGridSpec", "QGYBJSimulation")
        @test isnothing(match(Regex("\\b$(legacy_name)\\b"), source))
    end
end
