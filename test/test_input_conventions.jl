using Test
using FFTW
using JLD2
using NCDatasets
using QGYBJplus

physical_values(field) = real(ifft(Array(field), (2, 3)))

function write_field(path, dimensions, values; fillvalue=nothing, coordinates=())
    NCDataset(path, "c") do dataset
        for (name, length) in dimensions
            dataset.dim[name] = length
        end
        for (name, vector) in coordinates
            defVar(dataset, name, vector, (name,))
        end
        variable = fillvalue === nothing ?
            defVar(dataset, "field", Float64, first.(dimensions)) :
            defVar(dataset, "field", Float64, first.(dimensions); fillvalue)
        variable[ntuple(_ -> Colon(), length(dimensions))...] = values
    end
    return path
end

@testset "FieldFile reads NetCDF dimensions by name" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(1.0, 1.0, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0), flow=FixedFlow(),
                       stratification=ConstantStratification(N²=1.0), verbose=false)
    nx, ny, nz = grid.size
    xyz = [sin(2π * x) * (1 + 0.5cos(2π * y)) * (1 + 0.1k)
           for x in grid.x, y in grid.y, k in 1:nz]
    expected = permutedims(xyz, (3, 1, 2))
    mktempdir() do directory
        # A (y, x, z) variable on a square grid passed every size check and
        # was silently transposed.
        path = write_field(joinpath(directory, "yxz.nc"),
                           [("y", ny), ("x", nx), ("z", nz)], permutedims(xyz, (2, 1, 3)))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi) ≈ expected

        # A separate x dimension of the same length can be present in staggered output.
        x_staggered = grid.x .+ grid.dx / 2
        staggered = [sin(2π * x) * (1 + 0.5cos(2π * y)) * (1 + 0.1k)
                     for x in x_staggered, y in grid.y, k in 1:nz]
        path = joinpath(directory, "staggered.nc")
        NCDataset(path, "c") do dataset
            defVar(dataset, "x", grid.x, ("x",))
            defVar(dataset, "xc", x_staggered, ("xc",))
            defVar(dataset, "y", grid.y, ("y",))
            defVar(dataset, "z", grid.z, ("z",))
            defVar(dataset, "field", staggered, ("xc", "y", "z"))
        end
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        centered = (staggered .+ circshift(staggered, (1, 0, 0))) ./ 2
        @test physical_values(model.fields.psi) ≈ permutedims(centered, (3, 1, 2))

        # Common alternative names, and a third dimension implied by the other two.
        path = write_field(joinpath(directory, "lonlat.nc"),
                           [("lat", ny), ("lon", nx), ("s_rho", nz)],
                           permutedims(xyz, (2, 1, 3)))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi) ≈ expected

        # Unnamed dimensions follow `layout`.
        path = write_field(joinpath(directory, "unnamed.nc"),
                           [("d1", nz), ("d2", nx), ("d3", ny)], expected)
        set!(model; ψ=FieldFile(path, "field"; layout=:zxy), pv_method=:none)
        @test physical_values(model.fields.psi) ≈ expected
        @test_throws ArgumentError set!(model; ψ=FieldFile(
            write_field(joinpath(directory, "partial.nc"),
                        [("x", nx), ("d2", ny), ("d3", nz)], xyz), "field"),
            pv_method=:none)

        # A single time is dropped; a time series is an error.
        path = joinpath(directory, "time.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["x"] = nx; dataset.dim["y"] = ny; dataset.dim["z"] = nz
            dataset.dim["time"] = 1
            defVar(dataset, "field", Float64, ("x", "y", "z", "time"))[:, :, :, 1] = xyz
        end
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi) ≈ expected
        path = joinpath(directory, "series.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["x"] = nx; dataset.dim["y"] = ny; dataset.dim["z"] = nz
            dataset.dim["time"] = 2
            variable = defVar(dataset, "field", Float64, ("x", "y", "z", "time"))
            variable[:, :, :, 1] = xyz
            variable[:, :, :, 2] = xyz
        end
        failure = try
            set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
            nothing
        catch exception
            exception
        end
        @test failure isa ArgumentError
        # Displaying the error used to read the closed file and kill Julia.
        @test occursin("time", sprint(showerror, failure))

        # Fill values and NaN are errors while the file is open.
        values = Array{Union{Missing, Float64}}(xyz)
        values[1, 1, 1] = missing
        path = write_field(joinpath(directory, "fill.nc"),
                           [("x", nx), ("y", ny), ("z", nz)], values; fillvalue=-999.0)
        @test_throws ArgumentError set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        bad = copy(xyz)
        bad[2] = NaN
        path = write_field(joinpath(directory, "nan.nc"), [("x", nx), ("y", ny), ("z", nz)], bad)
        @test_throws ArgumentError set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
    end

    # FieldArray values are checked as well.
    bad = copy(expected)
    bad[1] = Inf
    @test_throws ArgumentError set!(model; ψ=bad, pv_method=:none)
    @test_throws ArgumentError set!(model; ψ=FieldArray(bad), pv_method=:none)
end

@testset "Depths are measured below the lid of a shifted grid" begin
    shifted = RectilinearGrid(size=(4, 4, 8), x=(0.0, 1e5), y=(0.0, 1e5), z=(0.0, 1000.0))
    standard = RectilinearGrid(size=(4, 4, 8), extent=(1e5, 1e5, 1000.0))
    # Profiles took depth as -z: on z = (0, 1000), a tanh profile was uniform
    # and an exponential one gave N² up to 0.96 s⁻².
    for profile in (TanhProfile{Float64}(0.01, 0.002, 300.0, 50.0),
                    ExponentialProfile{Float64}(0.01, 200.0, 0.001),
                    PiecewiseProfile([0.0, 300.0, 1000.0], [0.01, 0.002]),
                    SkewedGaussian{Float64}(1e-5, 4e-5, 200.0, 300.0, 1.0),
                    FileProfile{Float64}("cast", [0.0, 500.0, 1000.0], [1e-4, 1e-5, 1e-6]))
        @test compute_stratification_profile(profile, shifted) ≈
              compute_stratification_profile(profile, standard)
    end
    # An analytical profile is a function of the grid's own z.
    analytical = AnalyticalProfile(z -> 1e-5 * (1 + z / 1000))
    @test compute_stratification_profile(analytical, shifted) ≈
          [1e-5 * (1 + z / 1000) for z in shifted.z]

    model = QGYBJModel(grid=shifted, coriolis=FPlane(f=1e-4), flow=FixedFlow(),
                       stratification=ConstantStratification(N²=1e-5), verbose=false)
    levels = [Float64(k) for i in 1:4, j in 1:4, k in 1:8]
    mktempdir() do directory
        # z at the model's (positive) levels was taken for a depth and collapsed.
        path = write_field(joinpath(directory, "z.nc"), [("x", 4), ("y", 4), ("z", 8)], levels;
                           coordinates=(("x", collect(shifted.x)), ("y", collect(shifted.y)),
                                        ("z", collect(shifted.z))))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi)[:, 1, 1] ≈ 1:8

        path = write_field(joinpath(directory, "depth.nc"),
                           [("x", 4), ("y", 4), ("depth", 8)], levels;
                           coordinates=(("x", collect(shifted.x)), ("y", collect(shifted.y)),
                                        ("depth", collect(1000.0 .- shifted.z))))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi)[:, 1, 1] ≈ 1:8
    end
end

@testset "FieldFile finds coordinates under the names of the axes" begin
    # A (lon, lat, depth) field, surface first as in ocean output, was taken to
    # be on the model's grid, bottom level first.
    grid = RectilinearGrid(size=(4, 4, 8), extent=(1.0, 1.0, 800.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0), flow=FixedFlow(),
                       stratification=ConstantStratification(N²=1.0), verbose=false)
    depth = collect(50.0:100.0:750.0)
    values = [d for x in grid.x, y in grid.y, d in depth]   # the depth of each point
    mktempdir() do directory
        # Longitude and latitude are in degrees, not the model's units: they
        # were read as x and y in metres, and the field interpolated across
        # the domain. Only the depths place such a field.
        degrees = collect(10.0:13.0)
        for (x, y, z, horizontal, vertical) in (
                ("lon", "lat", "depth", (degrees, degrees), depth),
                ("xC", "yC", "zC", (collect(grid.x), collect(grid.y)), -depth))
            path = write_field(joinpath(directory, "$x.nc"), [(x, 4), (y, 4), (z, 8)], values;
                               coordinates=((x, horizontal[1]), (y, horizontal[2]),
                                            (z, vertical)))
            set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
            @test physical_values(model.fields.psi)[:, 1, 1] ≈ -grid.z
        end
        columns = [100.0i + j for i in 1:4, j in 1:4, d in depth]
        path = write_field(joinpath(directory, "columns.nc"),
                           [("lon", 4), ("lat", 4), ("depth", 8)], columns;
                           coordinates=(("lon", degrees), ("lat", degrees), ("depth", depth)))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi)[1, :, :] ≈ columns[:, :, 1]

        # A level index is not a depth: the field is on the grid as stored.
        levels = [Float64(k) for i in 1:4, j in 1:4, k in 1:8]
        path = write_field(joinpath(directory, "level.nc"),
                           [("x", 4), ("y", 4), ("level", 8)], levels;
                           coordinates=(("x", collect(grid.x)), ("y", collect(grid.y)),
                                        ("level", collect(1.0:8.0))))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi)[:, 1, 1] ≈ 1:8
    end
end

@testset "Field and stratification files share a vertical convention" begin
    # On a grid whose lid is at z = 1000, the stratification reader took heights
    # stated to be positive up for depths, and the field reader depths stated
    # to be positive down for heights.
    grid = RectilinearGrid(size=(4, 4, 8), x=(0.0, 1e5), y=(0.0, 1e5), z=(0.0, 1000.0))
    levels = [Float64(k) for i in 1:4, j in 1:4, k in 1:8]
    N² = 1e-6 .* (1 .+ grid.z ./ 1000)
    mktempdir() do directory
        for (positive, coordinate) in (("up", grid.z), ("down", 1000 .- grid.z))
            path = joinpath(directory, "$positive.nc")
            NCDataset(path, "c") do dataset
                dataset.dim["x"] = 4; dataset.dim["y"] = 4; dataset.dim["z"] = 8
                defVar(dataset, "x", collect(grid.x), ("x",))
                defVar(dataset, "y", collect(grid.y), ("y",))
                defVar(dataset, "z", collect(coordinate), ("z",); attrib=Dict("positive" => positive))
                defVar(dataset, "N2", N², ("z",))
                defVar(dataset, "field", levels, ("x", "y", "z"))
            end
            model = QGYBJModel(grid=grid, coriolis=FPlane(f=1e-4), flow=FixedFlow(),
                               stratification=load_stratification_from_file(path),
                               verbose=false)
            @test model.runtime.coefficients.N² ≈ N²
            set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
            @test physical_values(model.fields.psi)[:, 1, 1] ≈ 1:8
        end
    end
end

@testset "JLD2 fields drop dimensions of length one, as NetCDF fields do" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(1.0, 1.0, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0), flow=FixedFlow(),
                       stratification=ConstantStratification(N²=1.0), verbose=false)
    xyz = [sin(2π * x) * (1 + 0.5cos(2π * y)) * (1 + 0.1k)
           for x in grid.x, y in grid.y, k in 1:4]
    mktempdir() do directory
        path = joinpath(directory, "time.jld2")
        jldsave(path; field=reshape(xyz, size(xyz)..., 1), series=cat(xyz, xyz; dims=4))
        set!(model; ψ=FieldFile(path, "field"), pv_method=:none)
        @test physical_values(model.fields.psi) ≈ permutedims(xyz, (3, 1, 2))
        @test_throws ArgumentError set!(model; ψ=FieldFile(path, "series"), pv_method=:none)
    end

    # With one level, the leading z of a (z, x, y, time) array was dropped in
    # place of the trailing time.
    flat = RectilinearGrid(size=(8, 8, 1), extent=(1.0, 1.0, 1.0))
    flat_model = QGYBJModel(grid=flat, coriolis=FPlane(f=1.0), flow=FixedFlow(),
                            stratification=ConstantStratification(N²=1.0), verbose=false)
    xy = [sin(2π * x) * (1 + 0.5cos(2π * y)) for x in flat.x, y in flat.y]
    mktempdir() do directory
        path = joinpath(directory, "flat.jld2")
        jldsave(path; zxy=reshape(xy, 1, 8, 8, 1), xyz=reshape(xy, 8, 8, 1, 1))
        for (name, layout) in (("zxy", :zxy), ("xyz", :xyz))
            set!(flat_model; ψ=FieldFile(path, name; layout), pv_method=:none)
            @test physical_values(flat_model.fields.psi)[1, :, :] ≈ xy
        end
    end
end

@testset "Stratification inputs are validated" begin
    # The documented constructor accepted descending depths and mismatched lengths.
    @test_throws ArgumentError FileProfile{Float64}("cast", [100.0, 50.0], [1e-5, 2e-5])
    @test_throws ArgumentError FileProfile{Float64}("cast", [50.0, 100.0], [1e-5])
    @test_throws ArgumentError FileProfile{Float64}("cast", Float64[], Float64[])
    @test_throws ArgumentError FileProfile{Float64}("cast", [0.0, NaN], [1e-5, 1e-5])

    errors, warnings = validate_stratification([1e-5, NaN, 2e-5])
    @test any(contains("Non-finite"), errors)
    errors, warnings = validate_stratification([1e-5, 2e-5, 3e-5])
    @test isempty(errors)
    errors, _ = validate_stratification([1e-5, -1e-6])
    @test !isempty(errors)
    @test !isdefined(QGYBJplus, :create_stratification_profile)
end
