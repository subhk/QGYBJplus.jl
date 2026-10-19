using Test
using JLD2
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
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        verbose=false,
    )

    try
        set!(model; ψ=(x, y, z) -> sin(x) * cos(y),
            pv_method=:barotropic)

        ψ = parent(model.fields.psi)
        q = parent(model.fields.q)
        # Barotropic PV: q = ∇²ψ = -kₕ²ψ in every column.
        @test q ≈ -reshape(grid.kh2, 1, size(grid.kh2)...) .* ψ
        @test all(isfinite, parent(model.fields.u))
        @test all(isfinite, parent(model.fields.v))

        set!(model; waves=SurfaceWave(
            amplitude=0.2, scale=0.1, profile=:gaussian))
        B_physical = QGYBJplus.allocate_fft_backward_dst(
            model.fields.B, model.runtime)
        QGYBJplus.fft_backward!(
            B_physical, model.fields.B, model.runtime.plans)
        B = parent(B_physical)
        # The Gaussian profile, less its depth mean: a horizontally uniform
        # wave velocity has no barotropic part.
        nz = grid.size[3]
        depth = [max(0.0, -grid.z[k] - 0.5 / nz) for k in 1:nz]
        profile = 0.2 .* exp.(-(depth .^ 2) ./ 0.1^2)
        profile .-= sum(profile) / nz
        @test maximum(abs, B .- reshape(profile, :, 1, 1)) < 1e-12
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
            # Vertical structure with zero depth mean in the horizontally
            # uniform part, which every wave envelope has.
            wave_values = [0.05 * cospi((k - 0.5) / grid.size[3]) * (1 + 0.3cos(grid.x[i]))
                           for i in 1:grid.size[1], j in 1:grid.size[2], k in 1:grid.size[3]]
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
        # A GPU's sqrt returns NaN rather than throwing, so this is checked up front.
        @test_throws ArgumentError set_mean_flow!(model; method=:random,
                                                  amplitude=-1.0, verbose=false)
        # Kernels read a dealiasing mask unchecked, so its size is checked first.
        @test_throws DimensionMismatch compute_velocities!(model.fields,
            model.runtime.geometry; plans=model.runtime.plans, dealias_mask=trues(4, 4))
        @test_throws ArgumentError FieldArray(zeros(2, 2, 2); space=:invalid)
        @test_throws ArgumentError FieldFile("missing.nc", "B"; layout=:invalid)
    finally
        finalize_model!(model)
    end
end

@testset "Vorticity input and fields on other grids" begin
    grid = RectilinearGrid(size=(16, 16, 4), x=(-π, π), y=(-π, π), z=(-1.0, 0.0))
    build() = QGYBJModel(; grid, coriolis=FPlane(f=1.0),
                         stratification=ConstantStratification(N²=1.0), verbose=false)
    ψ_exact(x, y, z) = (sin(x) * cos(2y) + 0.3cos(3x + y)) * (1 + 0.5z)
    ζ_exact(x, y, z) = (-5sin(x) * cos(2y) - 3cos(3x + y)) * (1 + 0.5z)
    exact_psi = [ψ_exact(grid.x[i], grid.y[j], grid.z[k]) for k in 1:4, i in 1:16, j in 1:16]
    sampled(f, xs, ys, zs) = [f(x, y, z) for x in xs, y in ys, z in zs]
    # ψ on the grid after `set!(model; initial...)`, with q built barotropically.
    function psi_after(; initial...)
        model = build()
        try
            set!(model; pv_method=:barotropic, initial...)
            psi = QGYBJplus.allocate_fft_backward_dst(model.fields.psi, model.runtime)
            QGYBJplus.fft_backward!(psi, model.fields.psi, model.runtime.plans)
            return real.(parent(psi))
        finally
            finalize_model!(model)
        end
    end

    # ψ follows from ζ = ∇ₕ²ψ exactly, and q is the same as when ψ is given.
    @test maximum(abs, psi_after(ζ=ζ_exact) .- exact_psi) < 1e-13
    from_psi, from_zeta = build(), build()
    try
        set!(from_psi; ψ=ψ_exact)
        set!(from_zeta; ζ=ζ_exact)
        @test parent(from_zeta.fields.q) ≈ parent(from_psi.fields.q) rtol=1e-12
    finally
        finalize_model!(from_psi)
        finalize_model!(from_zeta)
    end

    mktempdir() do directory
        # Data on other grids, in the periodic frame [0, 2π) and half a data
        # spacing off the model's points: linear interpolation is second order.
        zs = collect(range(-1, 0; length=6))
        function interpolation_error(n)
            xs = [(i - 0.5) * 2π / n for i in 1:n]
            path = joinpath(directory, "zeta$n.jld2")
            jldsave(path; x=xs, y=xs, depth=-zs, zeta=sampled(ζ_exact, xs, xs, zs))
            return maximum(abs, psi_after(ζ=FieldFile(path, "zeta")) .- exact_psi)
        end
        coarse, fine = interpolation_error(16), interpolation_error(32)
        @test 3.7 < coarse / fine < 4.3

        # NetCDF with a signed z reads the same data as JLD2 with depths.
        xs = [(i - 0.5) * 2π / 16 for i in 1:16]
        path = joinpath(directory, "zeta.nc")
        NCDataset(path, "c") do dataset
            dataset.dim["x"], dataset.dim["y"], dataset.dim["z"] = 16, 16, 6
            defVar(dataset, "x", xs, ("x",))
            defVar(dataset, "y", xs, ("y",))
            defVar(dataset, "z", zs, ("z",))
            defVar(dataset, "zeta", sampled(ζ_exact, xs, xs, zs), ("x", "y", "z"))
        end
        @test psi_after(ζ=FieldFile(path, "zeta")) ==
              psi_after(ζ=FieldFile(joinpath(directory, "zeta16.jld2"), "zeta"))

        # ψ and B from files on other grids are interpolated the same way; the
        # error is about h²/8 max|∇²ψ| ≈ 0.0096 for this spacing h.
        dense = [(i - 0.5) * 2π / 64 for i in 1:64]
        jldopen(joinpath(directory, "flow.jld2"), "w") do file
            file["flow/x"], file["flow/y"], file["flow/z"] = dense, dense, zs
            file["flow/psi"] = sampled(ψ_exact, dense, dense, zs)
        end
        @test maximum(abs, psi_after(ψ=FieldFile(joinpath(directory, "flow.jld2"),
                                                  "psi"; group="flow")) .- exact_psi) < 0.01
    end

    # A last point one period after the first repeats it, and the model's own
    # points are copied rather than interpolated.
    xs = collect(range(0, 2π; length=17))
    zs = collect(range(-1, 0; length=6))
    periodic = sampled(ζ_exact, xs, xs, zs)
    @test psi_after(ζ=FieldArray(periodic; layout=:xyz, x=xs, y=xs, z=zs)) ==
          psi_after(ζ=FieldArray(periodic[1:16, 1:16, :]; layout=:xyz,
                                 x=xs[1:16], y=xs[1:16], z=zs))
    own = sampled(ζ_exact, grid.x, grid.y, grid.z)
    @test psi_after(ζ=FieldArray(own; layout=:xyz, x=grid.x, y=grid.y, z=grid.z)) ==
          psi_after(ζ=FieldArray(own; layout=:xyz))

    @test_logs (:warn, r"horizontal mean of up to 0.5") psi_after(
        ζ=(x, y, z) -> ζ_exact(x, y, z) + 0.5)
    shallow = [-0.4, -0.2, 0.0]
    @test_logs (:warn, r"covers depths 0.0 to 0.4, but the domain spans depths 0.0 to 1.0") psi_after(
        ζ=FieldArray(sampled(ζ_exact, grid.x, grid.y, shallow); layout=:xyz,
                     x=grid.x, y=grid.y, z=shallow))
    regional = collect(range(-π, π / 2; length=12))
    @test_logs (:warn, r"leaving a gap of 1.57") psi_after(
        ζ=FieldArray(sampled(ζ_exact, regional, grid.y, grid.z); layout=:xyz,
                     x=regional, y=grid.y, z=grid.z))

    @test_throws "more than the domain's period" psi_after(
        ζ=FieldArray(zeros(8, 16, 4); layout=:xyz, x=range(0, 4π; length=8),
                     y=grid.y, z=grid.z))
    @test_throws ArgumentError FieldArray(zeros(4, 16, 16); space=:spectral,
                                          x=grid.x, y=grid.y, z=grid.z)
    @test_throws ArgumentError FieldArray(zeros(4, 16, 16); x=grid.x)
    @test_throws DimensionMismatch psi_after(ζ=zeros(3, 16, 16))
    @test_throws ArgumentError psi_after(ψ=ψ_exact, ζ=ζ_exact)
end

@testset "Waves are measured from the top of a shifted domain" begin
    # The same domain with its lid at z = 0 or at z = -100 m must get the
    # same wave profiles below the lid.
    profiles = map(((-1000.0, 0.0), (-1100.0, -100.0))) do z_bounds
        grid = RectilinearGrid(size=(4, 4, 16), x=(0.0, 1e4), y=(0.0, 1e4), z=z_bounds)
        model = QGYBJModel(; grid, verbose=false)
        try
            set!(model; waves=SurfaceWave(amplitude=0.1, scale=200.0))
            surface = Array(model.fields.B[:, 1, 1])
            set_wave_packet!(model; amplitude=0.1, kx=1, ky=0, sigma_k=0.5,
                             z_center=300.0, z_width=100.0)
            # The packet's phases are random; its envelope is not.
            (surface, abs.(Array(model.fields.B[:, 2, 1])))
        finally
            finalize_model!(model)
        end
    end
    @test profiles[1][1] ≈ profiles[2][1]
    @test profiles[1][2] ≈ profiles[2][2]
    # Levels lie (16 - k + 1/2) 62.5 m below the lid: 281 m is nearest 300 m.
    @test argmax(profiles[2][2]) == 12
end
