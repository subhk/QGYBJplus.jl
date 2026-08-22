using Test
using NCDatasets
using QGYBJplus

const ASSELIN_EXAMPLE = joinpath(
    @__DIR__, "..", "examples", "asselin_jpo2020.jl")

@testset "Direct Asselin example" begin
    source = read(ASSELIN_EXAMPLE, String)
    @test Meta.parse("begin\n" * source * "\nend") isa Expr
    @test !occursin(r"(?m)^\s*function\b", source)
    @test !occursin("QGYBJ_ASSELIN", source)
    for api in ("RectilinearGrid", "QGYBJModel", "WaveHyperdiffusivity", "YBJPlus",
                "Simulation", "run!", "finalize_simulation!")
        @test occursin(api, source)
    end

    mktempdir() do output_dir
        L = 70.0e3
        H = 3.0e3
        f = 1.24e-4
        κ = sqrt(2) * π / L
        ψ_scale = 0.335 / κ
        ψ₀ = (X, Y, _) -> ψ_scale *
            sin(κ * (X - Y) / sqrt(2)) * cos(κ * (X + Y) / sqrt(2))

        grid = RectilinearGrid(
            size=(4, 4, 2),
            extent=(L, L, H),
            centered=true,
        )
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=f),
            stratification=ConstantStratification(N²=1.0e-5),
            closure=WaveHyperdiffusivity(coefficient=1.0e5),
            flow=FixedFlow(),
            feedback=NoFeedback(),
            formulation=YBJPlus(),
            verbose=false,
        )
        set!(
            model;
            ψ=ψ₀,
            pv_method=:barotropic,
            waves=SurfaceWave(amplitude=0.10, scale=30.0),
            verbose=false,
        )

        simulation = Simulation(
            model;
            Δt=2.0,
            stop_iteration=1,
            output=NetCDFOutput(
                path=output_dir,
                schedule=IterationInterval(1),
            ),
            verbose=false,
        )
        try
            run!(simulation)
        finally
            finalize_simulation!(simulation)
        end

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
            @test size(dataset["B_hat_real"]) == (4, 4, 2)
            @test size(dataset["LA_real"]) == (4, 4, 2)
        end
    end
end
