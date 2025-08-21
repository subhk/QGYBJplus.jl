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
    ],
)

deploydocs(
    repo = "github.com/subhk/QGYBJ.jl",
    devbranch = "main",
    push_preview = true,
)

