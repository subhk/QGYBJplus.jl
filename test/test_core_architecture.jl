using Test
using QGYBJplus

@testset "Composition-first type identities" begin
    grid = RectilinearGrid(size = (8, 8, 4),
                           x = (-π, π),
                           y = (-π, π),
                           z = (-1.0, 0.0))

    @test nameof(typeof(grid)) == :RectilinearGrid
    @test !isdefined(QGYBJplus, :RectilinearGridSpec)

    initialization_logger = Test.TestLogger(
        min_level = Base.CoreLogging.Info)
    model = Base.CoreLogging.with_logger(initialization_logger) do
        QGYBJModel(grid = grid,
                   coriolis = FPlane(f = 1.0),
                   stratification = ConstantStratification(N² = 1.0),
                   closure = HorizontalHyperdiffusivity(
                       flow = FlowHyperdiffusivity(coefficient = 0),
                       wave = WaveHyperdiffusivity(coefficient = 0)),
                   verbose = true)
    end

    initialization_logs = Dict(
        record.message => record for record in initialization_logger.logs)
    expected_messages = (
        "MPI initialized with 2D decomposition",
        "Topology validation passed",
        "Pencil decompositions created",
        "QGYBJModel runtime initialized",
    )
    @test all(haskey(initialization_logs, message)
              for message in expected_messages)
    @test initialization_logs[expected_messages[1]].kwargs[:nprocs] ==
          model.runtime.mpi.nprocs
    @test initialization_logs[expected_messages[1]].kwargs[:topology] ==
          model.runtime.mpi.topology
    @test initialization_logs[expected_messages[2]].kwargs[:nx] == grid.size[1]
    @test initialization_logs[expected_messages[2]].kwargs[:ny] == grid.size[2]
    @test initialization_logs[expected_messages[2]].kwargs[:nz] == grid.size[3]
    @test initialization_logs[expected_messages[2]].kwargs[:decomp_dims] ==
          (2, 3)
    @test initialization_logs[expected_messages[3]].kwargs[:xy_decomp] ==
          (2, 3)
    @test initialization_logs[expected_messages[3]].kwargs[:xz_decomp] ==
          (1, 3)
    @test initialization_logs[expected_messages[3]].kwargs[:z_decomp] ==
          (2, 3)
    @test initialization_logs[expected_messages[4]].kwargs[:size] == grid.size
    @test initialization_logs[expected_messages[4]].kwargs[:ranks] ==
          model.runtime.mpi.nprocs

    @test nameof(typeof(model)) == :QGYBJModel
    @test model.grid === grid
    @test hasproperty(model, :fields)
    @test hasproperty(model, :physics)
    @test hasproperty(model, :runtime)
    @test !hasproperty(model.runtime.coefficients, :stratification)

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
        removed_particle_aliases = (
            :create_particle_config,
            :create_particle_config_3d,
            :create_uniform_3d_grid,
            :create_layered_distribution,
            :create_random_3d_distribution,
            :create_custom_distribution,
        )
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
            removed_particle_aliases...,
        )
        for symbol in forbidden_symbols
            @test !isdefined(QGYBJplus, symbol)
        end

        for symbol in removed_particle_aliases
            @test !isdefined(QGYBJplus.UnifiedParticleAdvection, symbol)
        end
        for symbol in (:copy_local_to_extended!, :pack_halo_data!, :unpack_halo_data!)
            @test !isdefined(QGYBJplus.UnifiedParticleAdvection.HaloExchange, symbol)
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
    for obsolete_density_name in ("rho_u", "rho_s", "rho_ut", "rho_st")
        @test isnothing(match(Regex("\\b$(obsolete_density_name)\\b"), source))
    end
end
