using Test
using JLD2
using NCDatasets
using QGYBJplus

@testset "File stratification coordinates" begin
    mktempdir() do directory
        path = joinpath(directory, "signed_z.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["level"] = 3
            z = defVar(dataset, "z", Float64, ("level",))
            N² = defVar(dataset, "N2", Float64, ("level",))
            z[:] = [-100.0, -50.0, 0.0]
            N²[:] = [1.0, 2.0, 3.0]
        end

        profile = load_stratification_from_file(path)
        @test profile.z_data == [0.0, 50.0, 100.0]
        @test profile.N2_data == [3.0, 2.0, 1.0]
        @test evaluate_N2(profile, -50.0) == 2.0
        @test evaluate_N2(profile, -25.0) == 2.5
    end

    mktempdir() do directory
        path = joinpath(directory, "unsorted_depth.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["level"] = 3
            depth = defVar(dataset, "depth", Float64, ("level",))
            N² = defVar(dataset, "N2", Float64, ("level",))
            depth[:] = [100.0, 0.0, 50.0]
            N²[:] = [1.0, 3.0, 2.0]
        end

        profile = load_stratification_from_file(path)
        @test profile.z_data == [0.0, 50.0, 100.0]
        @test profile.N2_data == [3.0, 2.0, 1.0]
    end

    mktempdir() do directory
        path = joinpath(directory, "legacy_positive_z.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["level"] = 3
            z = defVar(dataset, "z", Float64, ("level",))
            N² = defVar(dataset, "N2", Float64, ("level",))
            z[:] = [0.0, 50.0, 100.0]
            N²[:] = [3.0, 2.0, 1.0]
        end

        profile = load_stratification_from_file(path)
        @test profile.z_data == [0.0, 50.0, 100.0]
        @test evaluate_N2(profile, -50.0) == 2.0
    end

    mktempdir() do directory
        path = joinpath(directory, "duplicate_depth.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["level"] = 2
            depth = defVar(dataset, "depth", Float64, ("level",))
            N² = defVar(dataset, "N2", Float64, ("level",))
            depth[:] = [20.0, 20.0]
            N²[:] = [1.0, 2.0]
        end

        @test_throws ArgumentError load_stratification_from_file(path)
    end

    mktempdir() do directory
        path = joinpath(directory, "grouped.nc")
        NCDataset(path, "c") do dataset
            ocean = defGroup(dataset, "ocean")
            ocean.dim["level"] = 3
            defVar(ocean, "depth", Float64, ("level",))[:] = [100.0, 0.0, 50.0]
            defVar(ocean, "N2", Float64, ("level",))[:] = [1.0, 3.0, 2.0]
        end

        profile = load_stratification_from_file(path; group="ocean")
        @test profile.z_data == [0.0, 50.0, 100.0]
        @test profile.N2_data == [3.0, 2.0, 1.0]
        @test_throws "has no coordinate named z or depth" load_stratification_from_file(path)
        @test_throws "has no group" load_stratification_from_file(path; group="land")
    end
end

@testset "JLD2 stratification files" begin
    mktempdir() do directory
        path(name) = joinpath(directory, name)
        depth, N² = [100.0, 0.0, 50.0], [1.0, 3.0, 2.0]
        sorted(profile) = profile.z_data == [0.0, 50.0, 100.0] &&
                          profile.N2_data == [3.0, 2.0, 1.0]

        jldsave(path("plain.jld2"); depth, N2=N²)
        @test sorted(load_stratification_from_file(path("plain.jld2")))
        # The NetCDF sign convention applies, and ranges are accepted.
        jldsave(path("range.jld2"); z=range(-100, 0; length=3), N2=[1.0, 2.0, 3.0])
        @test sorted(load_stratification_from_file(path("range.jld2")))

        jldopen(path("nested.jld2"), "w") do file
            file["ocean/cast/depth"] = depth
            file["ocean/cast/N²"] = N²
            file["profile"] = (z=-depth, n2=Float32.(N²))
            file["table"] = Dict("depth" => depth, "N2" => N²)
            file["metadata"] = (source="CTD",)
        end
        for group in ("ocean/cast", "profile", "table")
            @test sorted(load_stratification_from_file(path("nested.jld2"); group))
        end
        @test_throws "it contains metadata, ocean, profile, table" load_stratification_from_file(path("nested.jld2"))
        @test_throws "has no group" load_stratification_from_file(
            path("nested.jld2"); group="ocean/ship")
        @test_throws "not a group" load_stratification_from_file(
            path("nested.jld2"); group="ocean/cast/depth")

        jldsave(path("gaps.jld2"); depth, N2=[1.0, missing, 2.0])
        @test_throws "without missing values" load_stratification_from_file(path("gaps.jld2"))
    end
end

@testset "File profiles are sampled on the model's vertical grid" begin
    mktempdir() do directory
        path = joinpath(directory, "profile.jld2")
        depth = [0.0, 50.0, 150.0, 400.0, 900.0, 1600.0, 2500.0]
        N² = [1e-5, 8e-5, 2e-5, 6e-6, 3e-6, 2e-6, 1.5e-6]
        jldsave(path; depth, N2=N²)
        profile = load_stratification_from_file(path)
        # Independent reference: linear in depth, end values held beyond the data.
        function interpolated(d)
            i = clamp(searchsortedlast(depth, d), 1, length(depth) - 1)
            α = clamp((d - depth[i]) / (depth[i + 1] - depth[i]), 0, 1)
            return (1 - α) * N²[i] + α * N²[i + 1]
        end
        build(grid) = QGYBJModel(; grid, coriolis=FPlane(f=1e-4),
                                 stratification=profile, verbose=false)

        grid = RectilinearGrid(size=(4, 4, 24), extent=(1e4, 1e4, 2500.0))
        model = @test_logs build(grid)
        try
            coefficients = model.runtime.coefficients
            @test coefficients.N² ≈ interpolated.(-grid.z)
            @test coefficients.N²_face ≈ interpolated.(-grid.z_faces[2:end])
        finally
            finalize_model!(model)
        end

        deeper = RectilinearGrid(size=(4, 4, 24), extent=(1e4, 1e4, 3000.0))
        model = @test_logs (:warn, r"cover depths 0\.0 to 2500\.0, but the domain spans depths 0\.0 to 3000\.0") build(deeper)
        try
            @test model.runtime.coefficients.N²[1] == N²[end]
        finally
            finalize_model!(model)
        end
    end
end

@testset "Single-level stratification validation" begin
    errors, warnings = validate_stratification([1.0])
    @test isempty(errors)
    @test isempty(warnings)
end
