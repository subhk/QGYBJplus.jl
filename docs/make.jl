using Pkg
# Ensure the package in this repo is available in the docs environment
Pkg.develop(PackageSpec(path=pwd()*"/.."))
Pkg.instantiate()

using Documenter
using QGYBJ

makedocs(
    sitename = "QGYBJ.jl",
    modules = [QGYBJ],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "") == "true"),
    pages = [
        "Home" => "src/index.md",
        "Getting Started" => "src/getting_started.md",
        "Configuration" => "src/configuration.md",
        "Stratification" => "src/stratification.md",
        "Simulation" => "src/simulation.md",
        "I/O" => "src/io.md",
        "Parallel & Particles" => "src/parallel_particles.md",
        "Diagnostics" => "src/diagnostics.md",
        "Worked Example" => "src/worked_example.md",
        "Troubleshooting" => "src/troubleshooting.md",
        "API Reference" => "src/api.md",
    ],
)

deploydocs(
    repo = "github.com/subhk/QGYBJ.jl.git",
    devbranch = "main",
    push_preview = true,
)
