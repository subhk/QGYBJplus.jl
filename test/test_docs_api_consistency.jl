using Test
using QGYBJplus

@testset "Published documentation uses the composition-first API" begin
    documentation_root = joinpath(@__DIR__, "..", "docs", "src")
    pages = String[]
    for (directory, _, files) in walkdir(documentation_root)
        for file in files
            endswith(file, ".md") || continue
            push!(pages, read(joinpath(directory, file), String))
        end
    end
    published_documentation = join(pages, '\n')

    removed_names = (
        "`Grid`",
        "`State`",
        "`QGParams`",
        "default_params(",
        "setup_model(",
        "initialize_simulation(",
        "QGYBJSimulation",
        "run_simulation!",
        "exp_rk2_step!",
        "compute_detailed_wave_energy",
        "EnergyDiagnosticsManager",
        "ModelOutputManager",
        "Leapfrog",
        "IMEX-CN",
    )

    for name in removed_names
        @test !occursin(name, published_documentation)
    end
    for name in ("RectilinearGrid", "QGYBJModel", "Simulation", "run!")
        @test occursin(name, published_documentation)
    end

    int_factor_doc = sprint(
        show,
        MIME("text/plain"),
        Base.Docs.doc(QGYBJplus.int_factor),
    )
    @test occursin("Compute hyperdiffusion integrating factor", int_factor_doc)
end
