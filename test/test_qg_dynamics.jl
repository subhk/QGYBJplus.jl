#=
Does `flow=EvolvingFlow()` actually integrate the QG potential-vorticity
equation?

The switch tests (test_model_switches.jl) show that q *changes* when the flow
evolves. That alone would also be satisfied by a stepper advancing the wrong
equation. These testsets pin the dynamics against properties the QG equation
must have, independently of this implementation:

    ∂q/∂t + J(ψ, q) = (vertical diffusion) - (horizontal hyperdiffusion)

with the hyperdiffusion integrated exactly by the ETD integrating factor and
the rest evaluated at two Runge-Kutta stages.

These are verification tests for behaviour that already exists, so they pass on
first run; their value is that a regression in the Jacobian, the integrating
factor, or the Runge-Kutta weights breaks at least one of them.
=#

using Test
using QGYBJplus

const Q = QGYBJplus

const QG_NX, QG_NY, QG_NZ = 16, 16, 8

qg_grid() = RectilinearGrid(size=(QG_NX, QG_NY, QG_NZ), extent=(2π, 2π, 1.0))

function qg_model(grid; coefficient=0.0, order=4,
                  dynamics=NonlinearDynamics(), dissipation=Dissipative())
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(; coefficient, order),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=EvolvingFlow(), feedback=NoFeedback(), formulation=YBJPlus(),
        linear=dynamics, inviscid=dissipation,
        topology=(1, 1), verbose=false,
    )
end

"""A single horizontal mode: `J(ψ, q)` vanishes identically."""
single_mode(x, y, z) = sinpi(x / π) * cospi(y / π) * (1 + 0.5z)

"""Two horizontal modes, so the Jacobian is genuinely active."""
two_modes(x, y, z) =
    (sinpi(x / π) * cospi(y / π) + 0.7 * sinpi(2x / π) * cospi(3y / π)) *
    (1 + 0.5z)

function advance!(model, steps, Δt)
    timestepper = ExponentialRungeKutta2(; Δt)
    for _ in 1:steps
        step!(model, timestepper)
    end
    return model
end

"""Seed ψ (which derives q), advance, and return the final q."""
function evolved_q(grid, psi_function; steps, Δt, kwargs...)
    model = qg_model(grid; kwargs...)
    try
        set!(model; ψ=psi_function, verbose=false)
        advance!(model, steps, Δt)
        return copy(parent(model.fields.q))
    finally
        finalize_model!(model)
    end
end

peak(array) = maximum(abs, array)
discrepancy(a, b) = peak(a .- b) / peak(b)

@testset "EvolvingFlow integrates the QG potential-vorticity equation" begin
    grid = qg_grid()

    @testset "a single mode is an exact steady state of J(ψ, q)" begin
        # For ψ = Ψ(z)F(x,y) with F one horizontal mode, q = G(z)F, so
        # J(ψ, q) = ΨG J(F, F) = 0. An inviscid nonlinear run must therefore
        # hold q fixed: any drift is spurious tendency from the Jacobian.
        model = qg_model(grid; dissipation=Inviscid())
        try
            set!(model; ψ=single_mode, verbose=false)
            initial = copy(parent(model.fields.q))
            @test peak(initial) > 1
            advance!(model, 200, 1e-2)
            @test discrepancy(parent(model.fields.q), initial) < 1e-10
        finally
            finalize_model!(model)
        end
    end

    @testset "linear dynamics reproduce the analytic hyperdiffusive decay" begin
        # With advection switched off, every mode obeys
        # q̂(t) = exp(-ν kₕ^order t) q̂(0). ETD integrates that linear operator
        # exactly, so this must hold to roundoff, not to truncation order.
        ν, order, Δt, steps = 3e-3, 4, 5e-3, 60
        model = qg_model(grid; coefficient=ν, order, dynamics=LinearDynamics())
        try
            set!(model; ψ=two_modes, verbose=false)
            initial = copy(parent(model.fields.q))
            advance!(model, steps, Δt)
            final = parent(model.fields.q)

            time = steps * Δt
            worst = 0.0
            for j in 1:QG_NY, i in 1:QG_NX
                kh² = grid.kx[i]^2 + grid.ky[j]^2
                damping = exp(-ν * kh²^(order ÷ 2) * time)
                for k in 1:QG_NZ
                    abs(initial[k, i, j]) < 1e-10 && continue
                    worst = max(worst,
                        abs(final[k, i, j] - damping * initial[k, i, j]) /
                        abs(initial[k, i, j]))
                end
            end
            @test worst < 1e-10
        finally
            finalize_model!(model)
        end
    end

    @testset "the stepped tendency is exactly -J(ψ, q)" begin
        model = qg_model(grid; dissipation=Inviscid())
        try
            set!(model; ψ=two_modes, verbose=false)
            context = Q._operator_context(model)
            options = Q.ETDModelOptions(model.physics, model.numerics)

            tendency_q = similar(model.fields.q)
            tendency_B = similar(model.fields.B)
            Q._compute_etdrk2_rhs!(tendency_q, tendency_B, model.fields,
                context.grid, options, context.plans;
                a=context.a, dealias_mask=context.mask,
                workspace=context.workspace, N2_profile=context.N2,
                N2_face_profile=context.N2_face,
                timestep_workspace=nothing)

            # Independent evaluation of the advection term on the same state.
            jacobian = similar(model.fields.q)
            Q.compute_velocities!(model; compute_w=false)
            Q.convol_waqg_q!(jacobian, model.fields.u, model.fields.v,
                model.fields.q, context.grid, context.plans;
                Lmask=context.mask, workspace=context.workspace)

            @test peak(parent(jacobian)) > 0
            # No vertical diffusivity is configured, so the tendency is purely
            # minus the Jacobian.
            @test peak(parent(tendency_q) .+ parent(jacobian)) /
                  peak(parent(jacobian)) < 1e-12
        finally
            finalize_model!(model)
        end
    end

    @testset "the Jacobian sign matches an independent manufactured tendency" begin
        barotropic_grid = RectilinearGrid(
            size=(16, 16, 1), extent=(2π, 2π, 1.0))
        model = qg_model(barotropic_grid; dissipation=Inviscid())
        try
            amplitude = 0.3
            # For ψ = sin(x)sin(y) + a cos(2x), q = ∇²ψ and direct
            # differentiation gives -J(ψ,q) = 4a sin(x)sin(2x)cos(y).
            set!(model;
                ψ=(x, y, z) -> sin(x) * sin(y) + amplitude * cos(2x),
                verbose=false)
            context = Q._operator_context(model)
            options = Q.ETDModelOptions(model.physics, model.numerics)
            tendency_q = similar(model.fields.q)
            tendency_B = similar(model.fields.B)
            Q._compute_etdrk2_rhs!(
                tendency_q, tendency_B, model.fields,
                context.grid, options, context.plans;
                a=context.a,
                dealias_mask=context.mask,
                workspace=context.workspace,
                N2_profile=context.N2,
                N2_face_profile=context.N2_face,
            )

            physical = Q.allocate_fft_backward_dst(
                tendency_q, model.runtime)
            Q.fft_backward!(physical, tendency_q, context.plans)
            expected = [
                4amplitude * sin(barotropic_grid.x[i]) *
                sin(2barotropic_grid.x[i]) * cos(barotropic_grid.y[j])
                for k in 1:1, i in 1:16, j in 1:16
            ]
            @test real.(parent(physical)) ≈ expected atol=2e-13
            @test maximum(abs, imag.(parent(physical))) < 2e-13
        finally
            finalize_model!(model)
        end
    end

    @testset "a finite-difference dq/dt converges to that tendency" begin
        model = qg_model(grid; dissipation=Inviscid())
        reference_tendency = nothing
        initial = nothing
        try
            set!(model; ψ=two_modes, verbose=false)
            initial = copy(parent(model.fields.q))
            context = Q._operator_context(model)
            options = Q.ETDModelOptions(model.physics, model.numerics)
            reference_tendency = similar(model.fields.q)
            scratch_B = similar(model.fields.B)
            Q._compute_etdrk2_rhs!(reference_tendency, scratch_B, model.fields,
                context.grid, options, context.plans;
                a=context.a, dealias_mask=context.mask,
                workspace=context.workspace, N2_profile=context.N2,
                N2_face_profile=context.N2_face,
                timestep_workspace=nothing)
            reference_tendency = copy(parent(reference_tendency))
        finally
            finalize_model!(model)
        end

        # (q(Δt) - q(0))/Δt differs from the tendency at t=0 by O(Δt), so the
        # error must halve when Δt halves.
        errors = map((1e-4, 5e-5, 2.5e-5)) do Δt
            final = evolved_q(grid, two_modes; steps=1, Δt, dissipation=Inviscid())
            difference = (final .- initial) ./ Δt
            peak(difference .- reference_tendency) / peak(reference_tendency)
        end
        @test errors[1] < 1e-3
        @test 1.8 < errors[1] / errors[2] < 2.2
        @test 1.8 < errors[2] / errors[3] < 2.2
    end

    @testset "ETD-RK2 converges at second order in Δt" begin
        # The decisive check: a stepper solving a different equation, or using
        # wrong Runge-Kutta weights, would not converge at the design order.
        horizon = 0.4
        reference = evolved_q(grid, two_modes;
                              steps=640, Δt=horizon / 640, dissipation=Inviscid())
        errors = map((40, 80, 160)) do steps
            final = evolved_q(grid, two_modes;
                              steps, Δt=horizon / steps, dissipation=Inviscid())
            discrepancy(final, reference)
        end
        @test all(>(0), errors)
        @test 3.4 < errors[1] / errors[2] < 4.6
        @test 3.4 < errors[2] / errors[3] < 4.6
    end

    @testset "nonlinear QG plus horizontal hyperdiffusion remains second order" begin
        # This exercises the actual semilinear split: the Jacobian is explicit
        # while horizontal diffusion is carried by the exponential factor.
        horizon = 0.1
        coarse = evolved_q(grid, two_modes;
            steps=10, Δt=horizon / 10, coefficient=0.02, order=2)
        medium = evolved_q(grid, two_modes;
            steps=20, Δt=horizon / 20, coefficient=0.02, order=2)
        fine = evolved_q(grid, two_modes;
            steps=40, Δt=horizon / 40, coefficient=0.02, order=2)
        ratio = peak(coarse .- medium) / peak(medium .- fine)
        @test 3.5 < ratio < 4.5
    end

    @testset "inviscid nonlinear stepping nearly conserves enstrophy" begin
        # J(ψ, q) conserves ∫q²; only the time truncation should erode it.
        model = qg_model(grid; dissipation=Inviscid())
        try
            set!(model; ψ=two_modes, verbose=false)
            initial = sum(abs2, parent(model.fields.q))
            advance!(model, 200, 2e-3)
            final = sum(abs2, parent(model.fields.q))
            @test abs(final - initial) / initial < 1e-6
        finally
            finalize_model!(model)
        end
    end

    @testset "ψ is re-inverted from the evolving q, not left stale" begin
        # Each Runge-Kutta stage runs invert_q_to_psi! before the advection
        # kernels read u and v, so after a run the streamfunction must be the
        # inversion of the *current* q — a stale ψ would still satisfy the
        # steady-state and linear-decay checks above.
        model = qg_model(grid; dissipation=Inviscid())
        try
            set!(model; ψ=two_modes, verbose=false)
            initial_psi = copy(parent(model.fields.psi))
            advance!(model, 50, 2e-2)
            evolved_psi = copy(parent(model.fields.psi))

            @test discrepancy(evolved_psi, initial_psi) > 1e-3

            # Invert the final q independently and compare.
            context = Q._operator_context(model)
            recovered = Q.copy_fields(model.fields)
            Q.invert_q_to_psi!(recovered, context.grid;
                               a=context.a, workspace=context.workspace)
            @test discrepancy(parent(recovered.psi), evolved_psi) < 1e-12
        finally
            finalize_model!(model)
        end
    end

    @testset "FixedFlow leaves the same initial condition untouched" begin
        # The direct contrast: identical non-steady ψ, only the flow switch
        # differs.
        evolving = evolved_q(grid, two_modes;
                             steps=50, Δt=2e-2, dissipation=Inviscid())
        model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=FixedFlow(), feedback=NoFeedback(), formulation=YBJPlus(),
            inviscid=Inviscid(), topology=(1, 1), verbose=false)
        try
            set!(model; ψ=two_modes, verbose=false)
            initial = copy(parent(model.fields.q))
            advance!(model, 50, 2e-2)
            @test discrepancy(parent(model.fields.q), initial) < 1e-12
            @test discrepancy(evolving, initial) > 1e-3
        finally
            finalize_model!(model)
        end
    end
end
