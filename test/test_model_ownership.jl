using Test
using MPI
using QGYBJplus

@testset "QGYBJModel ownership" begin
    required_names = (:ModelPhysics, :ModelNumerics, :OperatorCoefficients,
                      :ModelRuntime, :QGYBJModel, :finalize_model!)
    for name in required_names
        @test isdefined(QGYBJplus, name)
    end

    mpi_was_initialized = MPI.Initialized()
    grid = RectilinearGrid(size = (8, 8, 4), extent = (2π, 2π, 1.0))
    model = QGYBJModel(grid = grid,
                       coriolis = FPlane(f = 1),
                       stratification = ConstantStratification(N² = 1),
                       closure = HorizontalHyperdiffusivity(
                           flow = FlowHyperdiffusivity(coefficient = 0),
                           wave = WaveHyperdiffusivity(coefficient = 0)),
                       flow = :fixed,
                       feedback = :none,
                       ybj_plus = true,
                       topology = (1, 1),
                       verbose = false)

    @test nameof(typeof(model)) == :QGYBJModel
    @test model.grid === grid
    @test !hasproperty(model, :params)

    ownership_properties = (:fields, :physics, :numerics, :runtime)
    for property in ownership_properties
        @test hasproperty(model, property)
    end

    if all(property -> hasproperty(model, property), ownership_properties)
        @test model.fields isa ModelFields
        @test model.physics isa ModelPhysics
        @test model.numerics isa ModelNumerics
        @test model.runtime isa ModelRuntime
        @test model.physics.coriolis.f == 1
        @test model.physics.flow isa FixedFlow
        @test model.physics.feedback isa NoFeedback
        @test model.physics.formulation isa YBJPlus
        @test model.runtime.plans !== nothing
        @test model.runtime.decomposition !== nothing
        @test model.runtime.dealias_mask !== nothing
        @test model.runtime.coefficients isa OperatorCoefficients
        @test model.runtime.owns_mpi == !mpi_was_initialized

        finalize_model!(model)
        @test model.runtime.finalized
        finalize_model!(model)
        @test model.runtime.finalized

        if mpi_was_initialized
            @test !MPI.Finalized()
        else
            @test MPI.Finalized()
        end
    end
end
