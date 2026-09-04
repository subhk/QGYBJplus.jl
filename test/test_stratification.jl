using Test
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
end

@testset "Single-level stratification validation" begin
    errors, warnings = validate_stratification([1.0])
    @test isempty(errors)
    @test isempty(warnings)
end
