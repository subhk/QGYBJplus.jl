using QGYBJplus
using NCDatasets
using FFTW

"""Reconstruct the snapshot geometry and its FFTW-ordered wavenumbers."""
function grid_from_snapshot(path)
    NCDataset(path, "r") do dataset
        size = (length(dataset["x"]),
                length(dataset["y"]),
                length(dataset["z"]))
        extent = (dataset.attrib["Lx"],
                  dataset.attrib["Ly"],
                  dataset.attrib["Lz"])
        return RectilinearGrid(size=size, extent=extent, centered=true)
    end
end

function horizontal_plans(grid)
    template = zeros(ComplexF64, grid.size)
    return (
        forward=FFTW.plan_fft(template, (1, 2)),
        backward=FFTW.plan_ifft(template, (1, 2)),
    )
end

function horizontal_velocity(ψ, grid, plans)
    ψ_hat = plans.forward * complex.(ψ)
    kx = reshape(grid.kx, :, 1, 1)
    ky = reshape(grid.ky, 1, :, 1)
    u = real.(plans.backward * ((-im .* ky) .* ψ_hat))
    v = real.(plans.backward * ((im .* kx) .* ψ_hat))
    return u, v
end

function write_energy_snapshot(input_path, output_path, grid,
                               plans=horizontal_plans(grid))
    data = NCDataset(input_path, "r") do dataset
        ψ = Array(dataset["psi"][:, :, :])
        LA = Array(dataset["LA_real"][:, :, :]) .+
             im .* Array(dataset["LA_imag"][:, :, :])
        coordinates = (Array(dataset["x"][:]),
                       Array(dataset["y"][:]),
                       Array(dataset["z"][:]))
        return (; ψ, LA, coordinates, time=dataset["time"][1])
    end

    size(data.ψ) == grid.size ||
        throw(DimensionMismatch("snapshot dimensions do not match the grid"))

    u, v = horizontal_velocity(data.ψ, grid, plans)
    flow_KE = 0.5 .* (u .^ 2 .+ v .^ 2)
    wave_KE = 0.5 .* abs2.(data.LA)

    NCDataset(output_path, "c") do dataset
        nx, ny, nz = grid.size
        dataset.dim["x"] = nx
        dataset.dim["y"] = ny
        dataset.dim["z"] = nz
        dataset.dim["time"] = 1

        for (name, values) in zip(("x", "y", "z"), data.coordinates)
            coordinate = defVar(dataset, name, Float64, (name,))
            coordinate[:] = values
        end
        time = defVar(dataset, "time", Float64, ("time",))
        time[1] = data.time

        for (name, values) in (
            "flow_KE" => flow_KE,
            "wave_KE" => wave_KE,
            "total_KE" => flow_KE .+ wave_KE,
            "u" => u,
            "v" => v,
        )
            field = defVar(dataset, name, Float64, ("x", "y", "z"))
            field[:, :, :] = values
        end

        dataset.attrib["source_file"] = basename(input_path)
        dataset.attrib["flow_KE_formula"] = "(u² + v²) / 2"
        dataset.attrib["wave_KE_formula"] = "|LA|² / 2"
    end
end

function main(output_dir)
    files = sort(filter(name -> startswith(name, "state") &&
                                endswith(name, ".nc"), readdir(output_dir)))
    isempty(files) && error("no state*.nc files found in $output_dir")

    grid = grid_from_snapshot(joinpath(output_dir, first(files)))
    plans = horizontal_plans(grid)
    println("grid = $grid")
    println("kx = grid.kx; ky = grid.ky")

    for filename in files
        output_name = replace(filename, "state" => "energy"; count=1)
        write_energy_snapshot(joinpath(output_dir, filename),
                              joinpath(output_dir, output_name), grid, plans)
        println("$filename -> $output_name")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        error("usage: julia --project=. examples/compute_energy.jl OUTPUT_DIR")
    main(only(ARGS))
end
