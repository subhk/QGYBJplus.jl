using Test
using FFTW
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
