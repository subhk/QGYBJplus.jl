using Pkg
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()

using Documenter
using QGYBJplus

makedocs(
    sitename = "QGYBJ+.jl",
    modules = [QGYBJplus],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "") == "true",
        canonical = "https://subhk.github.io/QGYBJplus.jl/stable/",
        sidebar_sitename = true,
        collapselevel = 2,
        size_threshold = 500_000,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started.md",
            "Quick Start" => "quickstart.md",
            "Key Concepts" => "concepts.md",
            "Worked Example" => "worked_example.md",
        ],
        "Physics & Theory" => [
            "Model Overview" => "physics/overview.md",
            "QG Equations" => "physics/qg_equations.md",
            "YBJ+ Wave Model" => "physics/ybj_plus.md",
            "Wave-Mean Interaction" => "physics/wave_mean.md",
            "Numerical Methods" => "physics/numerical_methods.md",
        ],
        "User Guide" => [
            "Configuration" => "guide/configuration.md",
            "Stratification" => "guide/stratification.md",
            "Initial Conditions" => "guide/initial_conditions.md",
            "Running Simulations" => "guide/simulation.md",
            "I/O and Output" => "guide/io.md",
            "Diagnostics" => "guide/diagnostics.md",
        ],
        "Advanced Topics" => [
            "MPI Parallelization" => "advanced/parallel.md",
            "Particle Advection" => "advanced/particles.md",
            "Parallel Particle Algorithm" => "advanced/parallel_particles.md",
            "Interpolation" => "advanced/interpolation.md",
            "Performance Tips" => "advanced/performance.md",
        ],
        "API Reference" => [
            "Core Types" => "api/types.md",
            "Geometry & Fields" => "api/grid_state.md",
            "Physics Functions" => "api/physics.md",
            "Time Stepping" => "api/timestepping.md",
            "Particles" => "api/particles.md",
            "API Map" => "api/index.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
    ],
    doctest = false,
    # Internal kernels are intentionally omitted from the user-facing API.
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/subhk/QGYBJplus.jl.git",
    devbranch = "main",
    push_preview = true,
)
