using Test
using QGYBJplus

@testset "RectilinearGrid geometry" begin
    grid = RectilinearGrid(size = (8, 6, 4),
                           x = (-4.0, 4.0),
                           y = (-3.0, 3.0),
                           z = (-2.0, 0.0))

    @test grid.size == (8, 6, 4)
    @test grid.extent == (8.0, 6.0, 2.0)
    @test grid.origin == (-4.0, -3.0)

    # Horizontal Fourier collocation nodes start at the declared origin.
    @test grid.x == collect(-4.0:1.0:3.0)
    @test grid.y == collect(-3.0:1.0:2.0)
    @test grid.z == [-1.75, -1.25, -0.75, -0.25]
    @test grid.x_faces == collect(-4.0:1.0:4.0)
    @test grid.y_faces == collect(-3.0:1.0:3.0)
    @test grid.z_faces == collect(-2.0:0.5:0.0)
    @test (grid.dx, grid.dy, grid.dz) == (1.0, 1.0, 0.5)
    @test grid.kx ≈ [0, 1, 2, 3, -4, -3, -2, -1] .* (2π / 8)
    @test grid.ky ≈ [0, 1, 2, -3, -2, -1] .* (2π / 6)
    @test grid.kh2[2, 2] == grid.kx[2]^2 + grid.ky[2]^2

    centered = RectilinearGrid(size = (4, 4, 2),
                               extent = (8, 4, 2),
                               centered = true)
    @test centered.origin == (-4.0, -2.0)

    @test_throws ArgumentError RectilinearGrid(size = (0, 8, 4),
                                                extent = (1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, -1, 1))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, 1, 1),
                                                x = (0, 2))
    @test_throws ArgumentError RectilinearGrid(size = (8, 8, 4),
                                                extent = (1, 1, 1),
                                                centered = true,
                                                x = (-0.5, 0.5))
end

@testset "Typed physics components" begin
    @test_throws ArgumentError FPlane(f = 0)
    @test_throws ArgumentError ConstantStratification(N² = 0)

    N²_function = z -> 1e-5 + z^2 * 1e-10
    N²_profile = AnalyticalProfile(N²_function; returns=:N²)
    @test evaluate_N2(N²_profile, -100.0) == N²_function(-100.0)
    N_profile = AnalyticalProfile(z -> 1e-2; returns=:N,
                                  precision=Float32)
    @test N_profile isa AnalyticalProfile{Float32}
    @test evaluate_N2(N_profile, -100.0) ≈ 1e-4
    @test_throws ArgumentError AnalyticalProfile(N²_function; returns=:invalid)

    flow_closure = FlowHyperdiffusivity(coefficient=0)
    wave_closure = WaveHyperdiffusivity(coefficient=1e5)
    closure = HorizontalHyperdiffusivity(
        flow=flow_closure,
        wave=wave_closure,
    )

    @test FlowHyperdiffusivity(2; order=6).orders == (6,)
    @test WaveHyperdiffusivity(2; order=6).orders == (6,)
    @test_throws ArgumentError FlowHyperdiffusivity(coefficient=-1)
    @test_throws ArgumentError FlowHyperdiffusivity(
        coefficient=1e5, order=3)
    @test_throws ArgumentError WaveHyperdiffusivity(coefficient=-1)
    @test_throws ArgumentError WaveHyperdiffusivity(
        coefficient=1e5, order=3)

    kx, ky, dt = 3.0, 4.0, 2.0
    @test QGYBJplus.int_factor(kx, ky, dt, closure) == 0
    @test QGYBJplus.int_factor(kx, ky, dt, closure; waves=true) ==
          dt * 1e5 * (kx^2 + ky^2)^2
    @test QGYBJplus.int_factor(
        kx,
        ky,
        dt,
        FlowHyperdiffusivity(coefficient=0.25),
    ) == dt * 0.25 * (kx^2 + ky^2)^2
    @test QGYBJplus.int_factor(kx, ky, dt, wave_closure) == 0

    # Zero-valued terms must be ignored before exponentiation. Otherwise an
    # intentionally disabled high-order term can evaluate as 0*Inf = NaN.
    sparse_wave_closure = WaveHyperdiffusivity(
        (0.0, 0.25), (1024, 2))
    @test QGYBJplus.int_factor(
        2.0, 0.0, 1.0, sparse_wave_closure; waves=true) == 1.0
end

@testset "Model construction and finalization" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    # Symbols and booleans select the typed components.
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1),
        stratification=ConstantStratification(N²=1),
        flow=:fixed, feedback=:none, ybj_plus=false, verbose=false)
    @test model.physics.flow isa FixedFlow
    @test model.physics.feedback isa NoFeedback
    @test model.physics.formulation isa YBJ
    @test model.runtime.dealias_mask == dealias_mask(grid)

    copied = copy_fields(model.fields)
    copied.q[1] = 1 + 2im
    @test model.fields.q[1] == 0

    finalize_model!(model)
    finalize_model!(model)
    @test model.runtime.finalized
    @test_throws ErrorException invert_q_to_psi!(model)
end

@testset "Hyperdiffusion coefficients target the dealiasing cutoff" begin
    dx, dt, order = 1e3, 10.0, 8
    ν = compute_hyperdiff_coeff(; dx, dy=dx, dt, order)
    # One e-fold every efold_steps = 10 steps at the largest wavenumber the
    # two-thirds rule keeps; the Nyquist target is (3/2)^order weaker there.
    @test 10 * dt * ν * (2π / (3dx))^order ≈ 1
    @test compute_hyperdiff_coeff(; dx, dy=dx, dt, order, kmax_fraction=1) ≈
          ν * (2 / 3)^order
    @test compute_hyperdiff_params(nx=64, ny=64, Lx=64dx, Ly=64dx, dt=dt,
                                   order=order).ν ≈ ν
    parameters = dimensional_hyperdiff_params(nx=64, ny=64, Lx=64dx, Ly=64dx,
                                              dt=dt, order=order)
    @test parameters.νₕ₁ ≈ ν
    @test parameters.νₕ₁ʷ ≈ ν
    @test parameters.ilap1 == order ÷ 2
end

@testset "The default closure damps the grid scale in any units" begin
    # The former default had fixed coefficients: it damped nothing on a
    # dimensional grid and even kₕ = 1 within 0.1 time units on a 2π one. The
    # default damps the largest wavenumber the grid keeps by one e-fold every
    # 10 steps, and half of it 2⁸ times more weakly, in any units and for
    # any time step.
    for (extent, f, N², Δt) in (((500e3, 500e3, 4000.0), 1e-4, 1e-5, 600.0),
                                ((2π, 2π, 1.0), 1.0, 1.0, 0.01))
        grid = RectilinearGrid(size=(16, 16, 4), extent=extent)
        model = @test_logs min_level=Base.CoreLogging.Warn QGYBJModel(; grid,
            coriolis=FPlane(f=f), stratification=ConstantStratification(N²=N²),
            linear=true, verbose=false)
        try
            closure = model.numerics.closure
            @test closure isa GridScaleHyperdiffusivity
            @test closure.kmax² == maximum(grid.kh2[dealias_mask(grid)])
            k = sqrt(closure.kmax²)
            # A closure that no model has put on a grid has no rate to give;
            # it used to report no damping at all.
            @test_throws ArgumentError QGYBJplus.int_factor(
                k, 0.0, Δt, GridScaleHyperdiffusivity())
            for step in (Δt, 10Δt)
                @test QGYBJplus.int_factor(k, 0.0, step, closure) ≈ 0.1
                @test QGYBJplus.int_factor(k / 2, 0.0, step, closure;
                                           waves=true) ≈ 0.1 / 2^8
            end
            # A step of a linear flow decays that mode by exactly e^(-1/10).
            set!(model; ψ=(x, y, z) -> cos(k * x), verbose=false)
            before = maximum(abs, model.fields.q)
            step!(model, ExponentialRungeKutta2(; Δt))
            @test maximum(abs, model.fields.q) / before ≈ exp(-0.1) rtol=1e-10
        finally
            finalize_model!(model)
        end
    end
    @test_throws ArgumentError GridScaleHyperdiffusivity(order=3)
    @test_throws ArgumentError GridScaleHyperdiffusivity(efold_steps=0)
end
