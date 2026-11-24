using Test
using FFTW
using QGYBJplus

initial_test_model(; kwargs...) = QGYBJModel(;
    grid=RectilinearGrid(size=(16, 16, 6), extent=(5e5, 5e5, 1000.0)),
    coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
    verbose=false, kwargs...)

physical_field(field) = ifft(Array(field), (2, 3))
rms_speed(model) = sqrt(sum(model.fields.u .^ 2 .+ model.fields.v .^ 2) / length(model.fields.u))

@testset "RandomStreamfunction sets an rms velocity" begin
    # amplitude=0.4 gave max |u| = 2e-7 m/s on a 500 km grid, with
    # uncorrelated levels.
    for size in (16, 32)
        model = QGYBJModel(grid=RectilinearGrid(size=(size, size, 4),
                                                extent=(5e5, 5e5, 1000.0)),
            coriolis=FPlane(f=1e-4), stratification=ConstantStratification(N²=1e-5),
            verbose=false)
        set!(model; mean_flow=RandomStreamfunction(amplitude=0.4, seed=42))
        @test rms_speed(model) ≈ 0.4 rtol=1e-8
    end

    model = initial_test_model()
    set!(model; mean_flow=RandomStreamfunction(amplitude=0.4, seed=42))
    psi = real(physical_field(model.fields.psi))
    @test psi[1, :, :] ≈ psi[end, :, :]   # the same at every level

    profiled = initial_test_model()
    set!(profiled; mean_flow=RandomStreamfunction(amplitude=0.4, seed=42,
                                                  profile=z -> exp(z / 300)))
    @test rms_speed(profiled) ≈ 0.4 rtol=1e-8
    psi = real(physical_field(profiled.fields.psi))
    ratio = psi[end, :, :] ./ psi[1, :, :]
    @test all(isapprox.(ratio, exp((profiled.grid.z[end] - profiled.grid.z[1]) / 300);
                        rtol=1e-8))

    # Reproducible, and different for another seed.
    again = initial_test_model()
    set!(again; mean_flow=RandomStreamfunction(amplitude=0.4, seed=42))
    @test again.fields.psi == model.fields.psi
    other = initial_test_model()
    set!(other; mean_flow=RandomStreamfunction(amplitude=0.4, seed=7))
    @test other.fields.psi != model.fields.psi
    # Any Int is a seed: a negative one threw an InexactError.
    set!(other; mean_flow=RandomStreamfunction(amplitude=0.4, seed=-42))
    @test rms_speed(other) ≈ 0.4 rtol=1e-8
    set!(again; mean_flow=RandomStreamfunction(amplitude=0.4, seed=-42))
    @test again.fields.psi == other.fields.psi
    @test other.fields.psi != model.fields.psi

    @test_throws ArgumentError RandomStreamfunction(amplitude=-1.0)
    @test_throws ArgumentError RandomStreamfunction(amplitude=1.0, profile=3)
    @test_throws UndefKeywordError RandomStreamfunction()
end

@testset "The energy spectrum bins every mode, for odd sizes too" begin
    # For odd n, the mode kx = n ÷ 2 was taken for -(n ÷ 2 + 1) and dropped.
    for n in (8, 9)
        grid = RectilinearGrid(size=(n, n, 1), extent=(2π, 2π, 1.0))
        geometry = QGYBJplus.RuntimeGeometry(grid)
        binned = map(Iterators.product(1:n, 1:n)) do (i, j)
            mode = zeros(ComplexF64, 1, n, n)
            mode[1, i, j] = 1
            spectrum = QGYBJplus.compute_energy_spectrum(mode, geometry)
            shell = round(Int, hypot(grid.kx[i], grid.ky[j]))   # integer wavenumbers
            1 <= shell <= n ÷ 2 ? spectrum[shell] > 0 : iszero(spectrum)
        end
        @test all(binned)
    end
end

@testset "set_wave_packet! is localized, deterministic, and scaled" begin
    model = initial_test_model()
    set_wave_packet!(model; amplitude=0.1, kx=3, ky=0, sigma_k=1.5)
    first_packet = copy(model.fields.B)
    set_wave_packet!(model; amplitude=0.1, kx=3, ky=0, sigma_k=1.5)
    @test model.fields.B == first_packet   # no random phases

    B = physical_field(model.fields.B)
    level = argmax([maximum(abs, B[k, :, :]) for k in axes(B, 1)])
    plane = abs.(B[level, :, :])
    peak = argmax(plane)
    grid = model.grid
    @test abs(grid.x[peak[1]] - 2.5e5) <= grid.dx
    @test abs(grid.y[peak[2]] - 2.5e5) <= grid.dy
    @test plane[1, 1] < 0.1 * maximum(plane)       # localized
    @test 0.05 < maximum(abs, B) < 0.15             # amplitude 0.1, up to the projection

    set_wave_packet!(model; amplitude=0.1, kx=3, ky=0, sigma_k=1.5,
                     center=(1e5, 4e5))
    plane = abs.(physical_field(model.fields.B)[level, :, :])
    peak = argmax(plane)
    @test abs(grid.x[peak[1]] - 1e5) <= grid.dx
    @test abs(grid.y[peak[2]] - 4e5) <= grid.dy

    @test_throws ArgumentError set_wave_packet!(model; amplitude=0.1, kx=1, ky=0,
                                                sigma_k=-1.0)

    # A negative amplitude keeps its sign: the rescaling applied it twice.
    set_wave_packet!(model; amplitude=0.1, kx=3, ky=0, sigma_k=1.5)
    positive = copy(model.fields.B)
    set_wave_packet!(model; amplitude=-0.1, kx=3, ky=0, sigma_k=1.5)
    @test model.fields.B ≈ -positive
end

struct CallableStreamfunction
    amplitude::Float64
end
(flow::CallableStreamfunction)(x, y, z) = flow.amplitude * sin(2π * x / 5e5)

@testset "set_mean_flow! takes any callable" begin
    # A callable that is not a Function was a MethodError.
    model = initial_test_model()
    set_mean_flow!(model; psi_func=(x, y, z) -> 1e3 * sin(2π * x / 5e5), verbose=false)
    expected = copy(model.fields.psi)
    set_mean_flow!(model; psi_func=CallableStreamfunction(1e3), verbose=false)
    @test model.fields.psi == expected
    @test_throws ArgumentError set_mean_flow!(model; psi_func=1.0, verbose=false)
    @test model.fields.psi == expected
end

@testset "set! checks every argument before it changes the model" begin
    model = initial_test_model()
    untouched = copy(model.fields.psi)
    flow(x, y, z) = 1e3 * sin(2π * x / 5e5)
    @test_throws ArgumentError set!(model; ψ=flow, pv_method=:typo)
    @test model.fields.psi == untouched
    # A wave array of the wrong size is read before ψ is assigned.
    @test_throws DimensionMismatch set!(model; ψ=flow, B=zeros(3, 3, 3))
    @test model.fields.psi == untouched
    @test_throws ArgumentError set!(model; ψ=flow, particles=:not_a_config)
    @test model.fields.psi == untouched
    @test_throws ArgumentError set!(model; ψ=flow, particles=particles_in_box(
        500.0; x_max=5e5, y_max=5e5))   # a depth, above the lid
    @test model.fields.psi == untouched
    @test model.particles === nothing

    # pv_method=:none keeps q, which an evolving flow inverts at once.
    @test_throws ArgumentError set!(model; ψ=flow, pv_method=:none)
    @test_throws ArgumentError set_mean_flow!(model; psi_func=flow, pv_method=:none,
                                              verbose=false)
    @test model.fields.psi == untouched
    fixed = initial_test_model(flow=FixedFlow())
    set!(fixed; ψ=flow, pv_method=:none)
    @test maximum(abs, fixed.fields.psi) > 0
    # Waves may keep q (the PV-conserving choice).
    set!(model; waves=SurfaceWave(amplitude=0.1, scale=100.0), pv_method=:none)
    @test maximum(abs, model.fields.B) > 0
end

@testset "A wave packet nearly uniform in depth is not scaled up" begin
    # The envelope is restricted to what a wave amplitude can produce and
    # then scaled back to `amplitude`. What YBJ() keeps of a packet much
    # deeper than the domain is its small vertical residual, which was scaled
    # up without bound: 87 times for z_width = 2 on a unit depth, and 1e13
    # times, from round-off, for a uniform one.
    peak(model) = maximum(abs, physical_field(model.fields.B))
    packet!(model; kwargs...) = set_wave_packet!(model; amplitude=0.1, kx=3, ky=1,
                                                 sigma_k=1.5, kwargs...)
    unit_model(formulation) = QGYBJModel(;
        grid=RectilinearGrid(size=(16, 16, 8), extent=(2π, 2π, 1.0)),
        coriolis=FPlane(f=1.0), stratification=ConstantStratification(N²=1.0),
        formulation, verbose=false)

    model = unit_model(YBJ())
    # A packet with vertical structure keeps its amplitude.
    @test_logs min_level=Base.CoreLogging.Warn packet!(model)
    @test peak(model) ≈ 0.1
    @test_logs (:warn, r"nearly uniform in depth") packet!(model; z_width=2.0)
    @test 0 < peak(model) < 0.01
    @test_logs (:warn, r"nearly uniform in depth") packet!(model; z_width=20.0)
    @test 0 < peak(model) < 1e-3
    # A broad packet can still have a representable vertical residual.
    @test_logs (:warn, r"nearly uniform in depth") packet!(model; z_width=2e6)
    @test 0 < peak(model) < 1e-12
    # Once all levels round to the same value, the packet is removed.
    @test_logs (:warn, r"uniform in depth and was removed") match_mode=:any packet!(
        model; z_width=2e8)
    @test peak(model) == 0
    @test all(iszero, model.fields.A)

    # YBJ⁺ removes the depth mean of the horizontally uniform column only.
    plus = unit_model(YBJPlus())
    @test_logs min_level=Base.CoreLogging.Warn packet!(plus; z_width=20.0)
    @test peak(plus) ≈ 0.1
end

@testset "A large depth mean cannot erase a resolved Float32 wave" begin
    grid = RectilinearGrid(Float32; size=(8, 8, 4), extent=(2π, 2π, 1))
    B = repeat(reshape(Float32[1 - 1e-4, 1 - 1e-4, 1 + 1e-4, 1 + 1e-4],
                       4, 1, 1), 1, 8, 8)
    for formulation in (YBJPlus(), YBJ())
        model = QGYBJModel(; grid, formulation, verbose=false)
        set!(model; B)
        @test physical_field(model.fields.B) ≈ B .- 1 rtol=1e-5
    end
end
