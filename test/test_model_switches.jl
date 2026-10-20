#=
Behavioural checks for the typed physics switches.

Every one of these is a component that gates a branch in the ETD-RK2 kernels.
Constructing them and checking `isa` (test_core_components.jl) proves the types
exist; these testsets prove that flipping each switch actually changes the
answer, and changes it in the documented direction.

A single Fourier mode is an exact steady state of the Jacobian, so the flow
tests use a multi-mode streamfunction — with one mode an evolving flow looks
frozen and the test would pass for the wrong reason.
=#

using Test
using QGYBJplus

const NX, NY, NZ = 16, 16, 8
const STEPS = 25
const ΔT = 2e-2

switch_grid() = RectilinearGrid(size=(NX, NY, NZ), extent=(2π, 2π, 1.0))

function switch_model(grid; flow, feedback, formulation)
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=flow, feedback=feedback, formulation=formulation,
        verbose=false,
    )
end

"""Multi-mode streamfunction: a single mode would not advect."""
multimode_psi(x, y, z) =
    0.8 * (sinpi(x / π) * cospi(y / π) +
           0.6 * sinpi(2x / π) * cospi(3y / π) +
           0.4 * cospi(3x / π) * sinpi(y / π)) * (1 + 0.5z)

"""
Wave envelope with genuine horizontal structure, so that the formulations'
different dispersion shows up in the comparison.
"""
function structured_wave()
    return ComplexF64[
        (0.5 * sinpi(2(i - 1) / NX) * cospi(2(j - 1) / NY) +
         0.3 * cospi(4(i - 1) / NX)) * exp(-((k - 0.5) / NZ) / 0.3)
        for k in 1:NZ, i in 1:NX, j in 1:NY]
end

"""Advance a configured model and return copies of the fields that matter."""
function evolve_switch(grid; flow, feedback, formulation, waves=structured_wave())
    model = switch_model(grid; flow, feedback, formulation)
    try
        set!(model; ψ=multimode_psi, B=waves, verbose=false)
        initial_q = copy(parent(model.fields.q))
        initial_B = copy(parent(model.fields.B))
        timestepper = ExponentialRungeKutta2(Δt=ΔT)
        for _ in 1:STEPS
            step!(model, timestepper)
        end
        return (q=copy(parent(model.fields.q)),
                B=copy(parent(model.fields.B)),
                psi=copy(parent(model.fields.psi)),
                u=copy(parent(model.fields.u)),
                A=copy(parent(model.fields.A)),
                C=copy(parent(model.fields.C)),
                initial_q=initial_q, initial_B=initial_B)
    finally
        finalize_model!(model)
    end
end

magnitude(array) = maximum(abs, array)
relative(a, b) = maximum(abs, a .- b) /
                 max(magnitude(a), magnitude(b), eps())

@testset "Typed physics switches change the solution" begin
    grid = switch_grid()

    @testset "flow: FixedFlow holds q, EvolvingFlow advances it" begin
        fixed = evolve_switch(grid; flow=FixedFlow(),
                              feedback=NoFeedback(), formulation=YBJPlus())
        evolving = evolve_switch(grid; flow=EvolvingFlow(),
                                 feedback=NoFeedback(), formulation=YBJPlus())

        @test magnitude(fixed.initial_q) > 1
        @test relative(fixed.q, fixed.initial_q) < 1e-12
        @test relative(evolving.q, evolving.initial_q) > 1e-2

        # A frozen flow also steers the waves differently.
        @test relative(fixed.psi, evolving.psi) > 1e-3
        @test relative(fixed.B, evolving.B) > 1e-3
    end

    @testset "feedback: WaveMeanFeedback alters the balanced inversion" begin
        without = evolve_switch(grid; flow=EvolvingFlow(),
                                feedback=NoFeedback(), formulation=YBJPlus())
        with = evolve_switch(grid; flow=EvolvingFlow(),
                             feedback=WaveMeanFeedback(), formulation=YBJPlus())

        @test relative(without.psi, with.psi) > 1e-3
        @test relative(without.u, with.u) > 1e-3
        @test relative(without.q, with.q) > 1e-3

        # NoWaveFeedback is the evolving-flow companion of NoFeedback: the docs
        # define both as omitting wave PV from the inversion, so they must stay
        # numerically identical. If that ever changes it should be deliberate.
        neither = evolve_switch(grid; flow=EvolvingFlow(),
                                feedback=NoWaveFeedback(), formulation=YBJPlus())
        @test neither.psi == without.psi
        @test neither.B == without.B
    end

    @testset "formulation: YBJPlus, YBJ and PassiveWave differ" begin
        plus = evolve_switch(grid; flow=EvolvingFlow(),
                             feedback=WaveMeanFeedback(), formulation=YBJPlus())
        normal = evolve_switch(grid; flow=EvolvingFlow(),
                               feedback=WaveMeanFeedback(), formulation=YBJ())
        passive = evolve_switch(grid; flow=EvolvingFlow(),
                                feedback=WaveMeanFeedback(), formulation=PassiveWave())

        # Each formulation must actually carry a wave field.
        @test magnitude(plus.B) > 1e-3
        @test magnitude(normal.B) > 1e-3
        @test magnitude(passive.B) > 1e-3

        @test relative(plus.B, normal.B) > 1e-3
        @test relative(plus.B, passive.B) > 1e-3

        # Every formulation recovers an amplitude; the passive envelope keeps
        # the YBJ⁺ relation B = L⁺A after a step, as set! and output define it.
        @test magnitude(plus.A) > 1e-3
        @test magnitude(normal.A) > 1e-3
        @test magnitude(passive.A) > 1e-3
    end

    @testset "FixedFlow with WaveMeanFeedback warns that feedback is inactive" begin
        @test_logs (:warn, r"no effect with flow=FixedFlow") finalize_model!(
            switch_model(grid; flow=FixedFlow(), feedback=WaveMeanFeedback(),
                         formulation=YBJPlus()))
    end
end

"""
Undamped model on a square grid for the kₕ = 0 tests: the jet ψ = U sin x
leaves a wave envelope that depends on x and z only unadvected, J(ψ, B) = 0,
so with no dispersion B(x, z, t) = B₀(z) exp(i U t sin(x) / 2) exactly.
"""
function jet_model(formulation; dispersion=false, n=32, nz=16)
    grid = RectilinearGrid(size=(n, n, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=FixedFlow(), feedback=NoFeedback(), formulation=formulation,
        no_dispersion=dispersion ? Dispersive() : NoDispersion(), verbose=false)
    set!(model; ψ=(x, y, z) -> sin(x), verbose=false)
    return model
end

"""Depth mean of the horizontally uniform column of `B`, relative to `B`."""
uniform_depth_mean(B) = abs(sum(B[:, 1, 1])) / size(B, 1) / maximum(abs, B)

@testset "Horizontally uniform waves in YBJ and YBJ⁺" begin
    @testset "a surface wave keeps its depth-varying part" begin
        for formulation in (YBJ(), YBJPlus())
            model = jet_model(formulation)
            try
                set!(model; waves=SurfaceWave(amplitude=0.1, scale=0.3), verbose=false)
                B = Array(model.fields.B)
                @test maximum(abs, B[:, 1, 1]) > 1     # still there, also for YBJ
                @test uniform_depth_mean(B) < 1e-14    # ... without its depth mean
            finally
                finalize_model!(model)
            end
        end
    end

    @testset "refraction of a uniform wave by a jet is exact" begin
        nz, T = 16, 1.0
        z = collect(range(-1 + 0.5 / nz, step=1 / nz, length=nz))
        profile = exp.(-(z .^ 2) ./ 0.3^2)
        profile .-= sum(profile) / nz
        results = Dict()
        for formulation in (YBJ(), YBJPlus()), dispersion in (false, true)
            model = jet_model(formulation; dispersion)
            try
                grid = model.grid
                set!(model; B=FieldArray([0.1 * profile[k] + 0im
                                          for k in 1:nz, i in 1:32, j in 1:32]),
                     verbose=false)
                step = ExponentialRungeKutta2(Δt=0.005)
                for _ in 1:200
                    step!(model, step)
                end
                B = QGYBJplus.allocate_fft_backward_dst(model.fields.B, model.runtime)
                QGYBJplus.fft_backward!(B, model.fields.B, model.runtime.plans)
                results[(nameof(typeof(formulation)), dispersion)] = Array(B)
                @test uniform_depth_mean(Array(model.fields.B)) < 1e-12
                if !dispersion
                    exact = [0.1 * profile[k] * cis(T * sin(grid.x[i]) / 2)
                             for k in 1:nz, i in 1:32, j in 1:32]
                    @test maximum(abs, Array(B) .- exact) < 1e-5 * maximum(abs, exact)
                end
            finally
                finalize_model!(model)
            end
        end
        # At these large scales YBJ and YBJ⁺ dispersion nearly coincide.
        ybj, plus = results[(:YBJ, true)], results[(:YBJPlus, true)]
        @test maximum(abs, ybj .- plus) < 1e-3 * maximum(abs, plus)
    end

    @testset "the amplitude reproduces the kₕ = 0 envelope and its energy" begin
        for formulation in (YBJ(), YBJPlus())
            model = jet_model(formulation; dispersion=true, n=16)
            try
                set!(model; waves=SurfaceWave(amplitude=0.1, scale=0.3), verbose=false)
                nz = model.grid.size[3]
                a = Array(model.runtime.coefficients.a_ell)
                A, B = Array(model.fields.A), Array(model.fields.B)
                LA = [((k < nz ? a[k] * (A[k + 1, 1, 1] - A[k, 1, 1]) : 0) -
                       (k > 1 ? a[k - 1] * (A[k, 1, 1] - A[k - 1, 1, 1]) : 0)) * nz^2
                      for k in 1:nz]
                @test maximum(abs, LA .- B[:, 1, 1]) < 1e-12 * maximum(abs, B[:, 1, 1])
                # The wave kinetic energy diagnostic is ½⟨|LA|²⟩ of the
                # (horizontally uniform) wave velocity.
                manager = QGYBJplus.EnergyDiagnosticsManager(
                    EnergyDiagnosticsOutput(path=mktempdir()), Float64)
                wave_KE = first(QGYBJplus._current_energy_components(model, manager))
                n = model.grid.size[1]
                @test wave_KE ≈ 0.5 * sum(abs2, B[:, 1, 1]) / (n^4 * nz) rtol=1e-12
            finally
                finalize_model!(model)
            end
        end
    end

    @testset "a depth-uniform envelope warns and is removed" begin
        for formulation in (YBJ(), YBJPlus())
            model = jet_model(formulation)
            try
                uniform = fill(0.1 + 0im, 16, 32, 32)
                @test_logs (:warn, r"uniform in depth") match_mode = :any set!(
                    model; B=FieldArray(uniform), verbose=false)
                # Only round-off of the spectral amplitude 0.1 * 32² remains.
                @test maximum(abs, Array(model.fields.B)) < 1e-12
            finally
                finalize_model!(model)
            end
        end
    end
end

@testset "Two-thirds dealiasing ellipse" begin
    """The previous rule: a disk of radius ⌊min(nx, ny)/3⌋."""
    function disk(i, j, nx, ny)
        nmin = min(nx, ny)
        kmax = fld(nmin, 3)
        ix = i - 1 <= nx ÷ 2 ? i - 1 : i - 1 - nx
        jy = j - 1 <= ny ÷ 2 ? j - 1 : j - 1 - ny
        return nmin % 3 == 0 ? ix^2 + jy^2 < kmax^2 : ix^2 + jy^2 <= kmax^2
    end
    signed(index, n) = index - 1 <= n ÷ 2 ? index - 1 : index - 1 - n

    # Square grids keep exactly the previous disk.
    for n in (8, 9, 12, 15, 16, 24, 30, 32)
        @test all(is_dealiased(i, j, n, n) == disk(i, j, n, n) for i in 1:n, j in 1:n)
    end

    # A long grid keeps its long direction resolved.
    @test is_dealiased(11, 1, 32, 8)       # kx = 10 = ⌊32/3⌋
    @test !is_dealiased(12, 1, 32, 8)
    @test is_dealiased(1, 3, 32, 8)        # ky = 2 = ⌊8/3⌋
    @test !is_dealiased(1, 4, 32, 8)
    @test count(i -> is_dealiased(i, 1, 16, 1), 1:16) == 11   # |kx| ≤ 5 on a slice

    # No product of two retained modes aliases onto a retained mode.
    for (nx, ny) in ((32, 8), (12, 8), (9, 24), (16, 6), (10, 30))
        kept = [(signed(i, nx), signed(j, ny))
                for i in 1:nx, j in 1:ny if is_dealiased(i, j, nx, ny)]
        retained = Set(kept)
        aliased = false
        for (a, b) in kept, (c, d) in kept
            sx, sy = a + c, b + d
            wx = mod(sx + nx ÷ 2, nx) - nx ÷ 2      # the index the product lands on
            wy = mod(sy + ny ÷ 2, ny) - ny ÷ 2
            (wx, wy) != (sx, sy) && (wx, wy) in retained && (aliased = true)
        end
        @test !aliased
    end
end
