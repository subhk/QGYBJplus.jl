#=
Physics checks of the YBJ⁺ implementation (Asselin & Young 2019) against
properties of the discrete equations:

- a vertical eigenmode of the discrete L⁺ oscillates at the YBJ⁺ dispersion
  frequency (f/2)kₕ²/(|λ| + kₕ²/4), converging at second order in Δt;
- advection, refraction, and dispersion conserve the wave action ∫|B|², so
  inviscid stepping drifts only by time-truncation error;
- two-way coupling conserves the energy of A&Y equation (3.7); without wave
  feedback the same diagnostic drifts by an amount independent of Δt, so the
  check is sensitive to the coupling signs.

Refraction, -(i/2)ζB, is checked against an exact solution for both
formulations in test_model_switches.jl.
=#

using Test
using QGYBJplus

const YBJP = QGYBJplus

ybj_plus_inviscid_closure() = HorizontalHyperdiffusivity(
    flow=FlowHyperdiffusivity(coefficient=0),
    wave=WaveHyperdiffusivity(coefficient=0))

@testset "YBJ⁺ eigenmode follows the discrete dispersion relation" begin
    nx, ny, nz, f, N², m, i0 = 16, 16, 8, 1.0, 1.0, 2, 2
    grid = RectilinearGrid(size=(nx, ny, nz), extent=(2π, 2π, 1.0))
    kh² = grid.kx[i0]^2
    λ = -4 * (f^2 / N²) * sin(m * π / (2nz))^2 / grid.dz^2
    vertical = cos.(m * π .* ((1:nz) .- 0.5) ./ nz)
    B0 = zeros(ComplexF64, nz, nx, ny)
    B0[:, i0, 1] .= (1 + 0.5im) .* vertical

    for (formulation, μ) in ((YBJPlus(), λ - kh² / 4), (YBJ(), λ))
        ω = (f / 2) * kh² / (-μ)
        T = 0.5 / ω
        errors = map((300, 600)) do steps
            model = QGYBJModel(; grid, coriolis=FPlane(f=f),
                stratification=ConstantStratification(N²=N²),
                closure=ybj_plus_inviscid_closure(),
                flow=FixedFlow(), feedback=NoFeedback(), formulation,
                linear=LinearDynamics(), inviscid=Inviscid(), verbose=false)
            try
                set!(model; B=FieldArray(B0; space=:spectral), verbose=false)
                timestepper = ExponentialRungeKutta2(Δt=T / steps)
                for _ in 1:steps
                    step!(model, timestepper)
                end
                exact = B0[:, i0, 1] .* exp(-im * ω * T)
                maximum(abs, model.fields.B[:, i0, 1] .- exact) / maximum(abs, exact)
            finally
                finalize_model!(model)
            end
        end
        @test errors[1] < 1e-5
        @test 3.8 < errors[1] / errors[2] < 4.2
    end
end

function ybj_plus_inviscid_drift(grid; flow, feedback, steps, T=1.0)
    model = QGYBJModel(; grid, coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> 1.0 + 0.5cos(3z); returns=:N²),
        closure=ybj_plus_inviscid_closure(), flow, feedback,
        formulation=YBJPlus(), inviscid=Inviscid(), verbose=false)
    try
        wave = ComplexF64[
            0.5 * (1 + 0.3cos(π * (grid.z[k] + 1))) *
            (cos(grid.x[i]) + 0.6im * sin(grid.y[j]) + 0.4cis(grid.x[i] + grid.y[j]))
            for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]]
        set!(model;
             ψ=(x, y, z) -> 0.3 * (sin(x) * cos(y) + 0.5sin(2x + y)) * (1 + 0.3z),
             B=FieldArray(wave), verbose=false)
        # A&Y (3.7): mean-flow KE and PE plus wave PE and correction energy.
        energy() = sum(YBJP._energy_components(model, model.fields)[2:5])
        action₀, energy₀ = sum(abs2, model.fields.B), energy()
        timestepper = ExponentialRungeKutta2(Δt=T / steps)
        for _ in 1:steps
            step!(model, timestepper)
        end
        return (action=abs(sum(abs2, model.fields.B) / action₀ - 1),
                energy=abs(energy() / energy₀ - 1))
    finally
        finalize_model!(model)
    end
end

@testset "Inviscid YBJ⁺ conserves wave action and coupled energy" begin
    grid = RectilinearGrid(size=(16, 16, 6), extent=(2π, 2π, 1.0))

    action = [ybj_plus_inviscid_drift(grid; flow=FixedFlow(), feedback=NoFeedback(),
                                      steps).action for steps in (50, 100)]
    @test action[1] < 1e-3
    @test action[1] / action[2] > 6

    coupled = [ybj_plus_inviscid_drift(grid; flow=EvolvingFlow(),
                                       feedback=WaveMeanFeedback(), steps).energy
               for steps in (50, 100)]
    @test coupled[1] < 1e-3
    @test coupled[1] / coupled[2] > 6

    # Without wave feedback the waves exchange energy with a flow that does not
    # feel them, so the same diagnostic is not conserved at any Δt.
    uncoupled = [ybj_plus_inviscid_drift(grid; flow=EvolvingFlow(),
                                         feedback=NoWaveFeedback(), steps).energy
                 for steps in (50, 100)]
    @test uncoupled[2] > 10 * coupled[2]
    @test uncoupled[1] / uncoupled[2] < 1.5
end
