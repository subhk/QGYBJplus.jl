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
        topology=(1, 1), verbose=false,
    )
end

"""Multi-mode streamfunction: a single mode would not advect."""
multimode_psi(x, y, z) =
    0.8 * (sinpi(x / π) * cospi(y / π) +
           0.6 * sinpi(2x / π) * cospi(3y / π) +
           0.4 * cospi(3x / π) * sinpi(y / π)) * (1 + 0.5z)

"""
Wave envelope with genuine horizontal structure. `SurfaceWave` is horizontally
uniform (pure kₕ = 0), which the normal-YBJ constraint legitimately discards,
so it cannot be used to compare formulations.
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

        # YBJ⁺ recovers an amplitude; the passive envelope has none.
        @test magnitude(plus.A) > 1e-3
        @test magnitude(normal.A) > 1e-3
        @test all(iszero, passive.A)
        @test all(iszero, passive.C)
    end

    @testset "YBJ warns when the wave field is entirely kₕ = 0" begin
        # SurfaceWave is horizontally uniform, so the normal-YBJ solvability
        # constraint (which divides by kₕ²) discards all of it. That is correct
        # but must not happen silently: the run would otherwise proceed with no
        # waves at all.
        model = switch_model(grid; flow=EvolvingFlow(),
                             feedback=WaveMeanFeedback(), formulation=YBJ())
        try
            @test_logs (:warn, r"horizontally uniform") match_mode = :any set!(
                model; waves=SurfaceWave(amplitude=0.5, scale=0.3), verbose=false)
            @test all(iszero, parent(model.fields.B))
        finally
            finalize_model!(model)
        end

        # YBJ⁺ keeps the same initial condition, so it must not warn.
        plus_model = switch_model(grid; flow=EvolvingFlow(),
                                  feedback=WaveMeanFeedback(), formulation=YBJPlus())
        try
            set!(plus_model; waves=SurfaceWave(amplitude=0.5, scale=0.3), verbose=false)
            @test magnitude(parent(plus_model.fields.B)) > 1e-3
        finally
            finalize_model!(plus_model)
        end
    end
end
