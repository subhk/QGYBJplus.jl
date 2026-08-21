using Test
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

        random = RandomStreamfunction(
            amplitude=0.4, spectral_slope=-3, seed=42)
        set!(model; mean_flow=random, pv_method=:qg)
        first_random = copy(parent(model.fields.psi))
        set!(model; mean_flow=random, pv_method=:qg)
        @test parent(model.fields.psi) == first_random
        @test all(isfinite, parent(model.fields.q))

        @test_throws ArgumentError set!(model; mean_flow=:invalid)
    finally
        finalize_model!(model)
    end
end
