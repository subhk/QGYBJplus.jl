using Test
using NCDatasets
using QGYBJplus

include(joinpath(@__DIR__, "..", "examples", "asselin_jpo2020.jl"))

@testset "Reduced Asselin ETD-RK2 example" begin
    mktempdir() do output_dir
        simulation = run_asselin_example(
            size=(4, 4, 2),
            Δt=2.0,
            stop_iteration=1,
            output_dir=output_dir,
            output_schedule=IterationInterval(1),
            diagnostics=IterationInterval(1),
            verbose=false,
        )

        @test simulation.state == Finalized
        @test simulation.clock.iteration == 1
        @test simulation.timestepper isa ExponentialRungeKutta2
        @test simulation.model.grid isa RectilinearGrid
        @test simulation.model.runtime.finalized

        files = sort(filter(name -> endswith(name, ".nc"), readdir(output_dir)))
        @test files == ["state0001.nc", "state0002.nc"]
        NCDataset(joinpath(output_dir, last(files)), "r") do dataset
            @test dataset.attrib["iteration"] == 1
            @test dataset["time"][1] ≈ 2.0
            @test size(dataset["psi"]) == (4, 4, 2)
            @test size(dataset["B_real"]) == (4, 4, 2)
        end
    end
end
