using Test
using LinearAlgebra
using NCDatasets
using QGYBJplus

@testset "Model-owned field initialization" begin
    grid = RectilinearGrid(
        size=(8, 8, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=0, flow2=0, waves=0, waves2=0),
        topology=(1, 1),
        verbose=false,
    )

    try
        set!(model; ψ=(x, y, z) -> sin(x) * cos(y),
            pv_method=:barotropic)

        ψ = parent(model.fields.psi)
        q = parent(model.fields.q)
        for k in axes(ψ, 1), j in axes(ψ, 3), i in axes(ψ, 2)
            i_global = QGYBJplus.local_to_global(i, 2, model.fields.psi)
            j_global = QGYBJplus.local_to_global(j, 3, model.fields.psi)
            @test q[k, i, j] ≈ -grid.kh2[i_global, j_global] * ψ[k, i, j]
        end
        @test all(isfinite, parent(model.fields.u))
        @test all(isfinite, parent(model.fields.v))

        set!(model; waves=SurfaceWave(
            amplitude=0.2, scale=0.1, profile=:gaussian))
        B_physical = QGYBJplus.allocate_fft_backward_dst(
            model.fields.B, model.runtime)
        QGYBJplus.fft_backward!(
            B_physical, model.fields.B, model.runtime.plans)
        B = parent(B_physical)
        @test maximum(abs, B[end, :, :] .- 0.2) < 1e-12
        @test all(B[:, :, 1] .≈ B[:, :, end])
        initialized_A = copy(parent(model.fields.A))
        @test !iszero(norm(initialized_A))
        fill!(parent(model.fields.A), 0)
        fill!(parent(model.fields.C), 0)
        invert_B_to_A!(model)
        @test parent(model.fields.A) ≈ initialized_A

        set!(model; waves=SurfaceWave(
            amplitude=0.15, scale=0.2, profile=:exponential))
        @test !iszero(norm(parent(model.fields.A)))

        set_wave_packet!(model;
            amplitude=0.1, kx=1, ky=1, sigma_k=0.5)
        @test !iszero(norm(parent(model.fields.B)))
        @test !iszero(norm(parent(model.fields.A)))

        ψ_direct = Array{Float64}(undef, grid.size[3], grid.size[1], grid.size[2])
        for k in axes(ψ_direct, 1), i in axes(ψ_direct, 2), j in axes(ψ_direct, 3)
            ψ_direct[k, i, j] = sin(grid.x[i]) * cos(grid.y[j])
        end
        set!(model; ψ=ψ_direct, pv_method=:barotropic)
        ψ_physical = QGYBJplus.allocate_fft_backward_dst(
            model.fields.psi, model.runtime)
        QGYBJplus.fft_backward!(
            ψ_physical, model.fields.psi, model.runtime.plans)
        @test parent(ψ_physical) ≈ ψ_direct

        mktempdir() do directory
            path = joinpath(directory, "waves.nc")
            wave_values = fill(0.05, grid.size...)
            NCDataset(path, "c") do dataset
                dataset.dim["x"] = grid.size[1]
                dataset.dim["y"] = grid.size[2]
                dataset.dim["z"] = grid.size[3]
                variable = defVar(dataset, "B", Float64, ("x", "y", "z"))
                variable[:, :, :] = wave_values
            end
            set!(model; B=FieldFile(path, "B"))
            B_from_file = QGYBJplus.allocate_fft_backward_dst(
                model.fields.B, model.runtime)
            QGYBJplus.fft_backward!(
                B_from_file, model.fields.B, model.runtime.plans)
            @test parent(B_from_file) ≈ permutedims(wave_values, (3, 1, 2))
            @test !iszero(norm(parent(model.fields.A)))
        end

        random = RandomStreamfunction(
            amplitude=0.4, spectral_slope=-3, seed=42)
        set!(model; mean_flow=random, pv_method=:qg)
        first_random = copy(parent(model.fields.psi))
        set!(model; mean_flow=random, pv_method=:qg)
        @test parent(model.fields.psi) == first_random
        @test all(isfinite, parent(model.fields.q))

        @test_throws ArgumentError set!(model; mean_flow=:invalid)
        @test_throws ArgumentError FieldArray(zeros(2, 2, 2); space=:invalid)
        @test_throws ArgumentError FieldFile("missing.nc", "B"; layout=:invalid)
    finally
        finalize_model!(model)
    end
end
