using Test
using QGYBJplus

@testset "ETD coefficient accuracy and stiff limit" begin
    h = 0.37
    for x in (0.0, eps(Float64), 1e-12, 1e-6, 1e-5, 1e-3,
              1.0, 50.0, 1e160)
        E, hφ1, hφ2 = QGYBJplus._etd_coefficients(x, h)
        reference = setprecision(256) do
            xb = BigFloat(x)
            hb = BigFloat(h)
            if iszero(xb)
                return (one(xb), hb, hb / 2)
            end
            (
                exp(-xb),
                hb * (-expm1(-xb)) / xb,
                hb * (exp(-xb) - 1 + xb) / xb^2,
            )
        end
        @test E ≈ Float64(reference[1]) rtol=5e-13 atol=0
        @test hφ1 ≈ Float64(reference[2]) rtol=5e-13 atol=0
        @test hφ2 ≈ Float64(reference[3]) rtol=5e-13 atol=0
    end

    # An overflowed non-negative damping exponent is the infinitely stiff
    # limit, not an indeterminate operation.
    @test QGYBJplus._etd_coefficients(Inf, h) == (0.0, 0.0, 0.0)
end

@testset "Model-owned exponential Runge-Kutta 2" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0.3, order=2),
        ),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )

    try
        model.fields.q[2, 2, 1] = 0.5 - 0.1im
        model.fields.B[2, 2, 1] = 1.2 - 0.7im
        q_initial = model.fields.q[2, 2, 1]
        B_initial = model.fields.B[2, 2, 1]

        timestepper = ExponentialRungeKutta2(Δt=0.1)
        fields_reference = model.fields
        @test timestepper.Δt == 0.1
        @test timestepper.workspace === nothing

        step!(model, timestepper)
        @test model.fields === fields_reference
        damping = exp(-0.1 * 0.3 * grid.kh2[2, 1])
        @test model.fields.q[2, 2, 1] == q_initial
        @test model.fields.B[2, 2, 1] ≈ damping * B_initial rtol=1e-14
        @test timestepper.workspace isa ExponentialRungeKutta2Workspace

        workspace = timestepper.workspace
        step!(model, timestepper)
        @test model.fields === fields_reference
        @test timestepper.workspace === workspace
        @test model.fields.B[2, 2, 1] ≈ damping^2 * B_initial rtol=2e-14

        # The coefficient table is rebuilt, so changing the step size between
        # calls must use the new value without reallocating the workspace.
        timestepper.Δt = 0.2
        step!(model, timestepper)
        @test timestepper.workspace === workspace
        @test model.fields.B[2, 2, 1] ≈ damping^4 * B_initial rtol=3e-14

        # `Δt` is mutable to support this workflow, so direct low-level calls
        # must revalidate it before touching the model state.
        timestepper.Δt = NaN
        @test_throws ArgumentError step!(model, timestepper)

        @test !isdefined(QGYBJplus, :ExpRK2Workspace)
        @test !isdefined(QGYBJplus, :leapfrog_step!)
        @test !isdefined(QGYBJplus, :imex_cn_step!)
    finally
        finalize_model!(model)
    end
end

@testset "ETD-RK2 mixed linear/exponential convergence" begin
    horizon = 0.4
    f = 0.7
    damping_rate = 0.3
    initial_B = 1.2 - 0.7im

    function dispersive_mode_error(Δt)
        grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=f),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(
                    coefficient=damping_rate, order=2)),
            flow=FixedFlow(),
            feedback=NoFeedback(),
            formulation=YBJPlus(),
            linear=LinearDynamics(),
            topology=(1, 1),
            verbose=false,
        )
        try
            # With nz=1 and kₕ²=1, L⁺A=B gives A=-4B. The remaining
            # explicit dispersive tendency is therefore Bₜ=-2ifB, while the
            # integrating factor supplies the exact -damping_rate*B part.
            model.fields.B[1, 2, 1] = initial_B
            timestepper = ExponentialRungeKutta2(; Δt)
            for _ in 1:round(Int, horizon / Δt)
                step!(model, timestepper)
            end
            exact = exp((-damping_rate - 2im * f) * horizon) * initial_B
            return abs(model.fields.B[1, 2, 1] - exact)
        finally
            finalize_model!(model)
        end
    end

    errors = dispersive_mode_error.((0.1, 0.05, 0.025))
    @test all(>(0), errors)
    @test 3.8 < errors[1] / errors[2] < 4.2
    @test 3.8 < errors[2] / errors[3] < 4.2
end

@testset "ETD-RK2 advances a discrete Neumann diffusion eigenmode by Heun" begin
    nz = 6
    vertical_diffusivity = 0.2
    Δt = 0.05
    vertical_mode = 2
    grid = RectilinearGrid(size=(8, 8, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        vertical_diffusion=VerticalDiffusivity(
            coefficient=vertical_diffusivity),
        flow=EvolvingFlow(),
        feedback=NoFeedback(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )

    try
        # The boundary stencil is [-1, 1] rather than the doubled-ghost-cell
        # stencil. Its exact eigenvectors are cos(mπ(k-1/2)/nz), with
        # eigenvalues -4sin²(mπ/(2nz))/Δz².
        eigenvector = cos.(vertical_mode * π .* ((1:nz) .- 0.5) ./ nz)
        amplitude = 0.7 - 0.4im
        parent(model.fields.q)[:, 2, 1] .= amplitude .* eigenvector
        parent(model.fields.q)[:, end, 1] .= conj(amplitude) .* eigenvector
        initial_q = copy(parent(model.fields.q))

        eigenvalue = -4vertical_diffusivity / grid.dz^2 *
                     sin(vertical_mode * π / (2nz))^2
        z = Δt * eigenvalue
        heun_amplification = 1 + z + z^2 / 2
        euler_amplification = 1 + z

        timestepper = ExponentialRungeKutta2(; Δt)
        step!(model, timestepper)

        # Zero horizontal diffusivity makes the ETD linear operator L exactly
        # zero, so hφ1=h and hφ2=h/2 and the explicit vertical operator must
        # reduce to Heun's stability polynomial on this eigenmode.
        @test timestepper.workspace.etd.Eq[2, 1] == 1
        @test timestepper.workspace.etd.hphi1q[2, 1] == Δt
        @test timestepper.workspace.etd.hphi2q[2, 1] == Δt / 2
        @test parent(model.fields.q) ≈
              heun_amplification .* initial_q rtol=2e-13 atol=2e-13
        @test maximum(abs,
            parent(model.fields.q) .- euler_amplification .* initial_q) > 1e-2
    finally
        finalize_model!(model)
    end
end

@testset "Overflowed hyperdiffusion damps instead of producing NaN" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(
                coefficient=floatmax(Float64), order=2)),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )
    try
        model.fields.B[1, 3, 1] = 1 + im
        step!(model, ExponentialRungeKutta2(Δt=1.0))
        @test all(isfinite, parent(model.fields.B))
        @test iszero(model.fields.B[1, 3, 1])
    finally
        finalize_model!(model)
    end
end

@testset "A timestepper cache follows the model that owns it" begin
    function cache_test_model(n)
        grid = RectilinearGrid(size=(n, n, 1), extent=(2π, 2π, 1.0))
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0.3, order=2)),
            flow=FixedFlow(),
            formulation=PassiveWave(),
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
        return model
    end

    first_model = cache_test_model(8)
    second_model = cache_test_model(4)
    timestepper = ExponentialRungeKutta2(Δt=0.1)
    try
        first_model.fields.B[1, 2, 1] = 1 - 0.5im
        step!(first_model, timestepper)

        initial = 0.7 + 0.2im
        second_model.fields.B[1, 2, 1] = initial
        step!(second_model, timestepper)
        @test second_model.fields.B[1, 2, 1] ≈
              exp(-0.03) * initial rtol=1e-14
        @test size(parent(timestepper.workspace.next.q)) ==
              size(parent(second_model.fields.q))
    finally
        finalize_model!(first_model)
        finalize_model!(second_model)
    end
end

@testset "A manually preallocated cache follows its field layout" begin
    function same_size_cache_model()
        grid = RectilinearGrid(size=(4, 4, 1), extent=(2π, 2π, 1.0))
        return QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0.3, order=2)),
            flow=FixedFlow(),
            formulation=PassiveWave(),
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
    end

    first_model = same_size_cache_model()
    second_model = same_size_cache_model()
    timestepper = ExponentialRungeKutta2(Δt=0.1)
    try
        foreign_workspace = ExponentialRungeKutta2Workspace(
            first_model.fields,
            first_model.runtime.plans;
            G=first_model.runtime.geometry,
        )
        timestepper.workspace = foreign_workspace

        initial = 0.7 + 0.2im
        second_model.fields.B[1, 2, 1] = initial
        step!(second_model, timestepper)

        @test timestepper.workspace !== foreign_workspace
        @test second_model.fields.B[1, 2, 1] ≈
              exp(-0.03) * initial rtol=1e-14
    finally
        finalize_model!(first_model)
        finalize_model!(second_model)
    end
end

@testset "Fixed-flow stages refresh a changed prescribed flow" begin
    function fixed_advection_model(grid)
        return QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=FixedFlow(),
            formulation=PassiveWave(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
    end

    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    reused = fixed_advection_model(grid)
    fresh = fixed_advection_model(grid)
    timestepper = ExponentialRungeKutta2(Δt=0.05)
    try
        # Populate the reusable stage/next buffers with an older fixed flow.
        set!(reused; ψ=(x, y, z) -> 0.0, verbose=false)
        step!(reused, timestepper)

        wave = ComplexF64[
            sin(grid.x[i]) + 0.2im * cos(grid.y[j])
            for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]
        ]
        prescribed_flow = (x, y, z) -> sin(y) + 0.3cos(x)
        set!(reused; ψ=prescribed_flow, B=FieldArray(wave), verbose=false)
        set!(fresh; ψ=prescribed_flow, B=FieldArray(wave), verbose=false)
        expected_psi = copy(parent(reused.fields.psi))

        step!(reused, timestepper)
        step!(fresh, ExponentialRungeKutta2(Δt=0.05))
        @test parent(reused.fields.psi) == expected_psi
        @test parent(reused.fields.psi) ≈ parent(fresh.fields.psi)
        @test parent(reused.fields.B) ≈ parent(fresh.fields.B) atol=2e-13
    finally
        finalize_model!(reused)
        finalize_model!(fresh)
    end
end

@testset "Fixed flow preserves PV outside the nonlinear dealiasing mask" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=FixedFlow(),
        formulation=PassiveWave(),
        linear=LinearDynamics(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )
    try
        # kₓ=3 is resolved on this grid but deliberately excluded by the
        # two-thirds mask. The mask applies to nonlinear tendencies; it must not
        # alter q when FixedFlow disables the q equation altogether.
        set!(model; ψ=(x, y, z) -> sin(3x), verbose=false)
        @test !is_dealiased(4, 1, grid)
        initial_q = copy(parent(model.fields.q))
        initial_psi = copy(parent(model.fields.psi))
        @test maximum(abs, initial_q) > 0

        step!(model, ExponentialRungeKutta2(Δt=0.1))

        @test parent(model.fields.q) == initial_q
        @test parent(model.fields.psi) == initial_psi
    finally
        finalize_model!(model)
    end
end

@testset "Linear QG modes outside the nonlinear cutoff follow the ETD operator" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    Δt = 0.1

    for diffusivity in (0.0, 0.2)
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(
                    coefficient=diffusivity, order=2),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=EvolvingFlow(),
            feedback=NoFeedback(),
            formulation=PassiveWave(),
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
        try
            # kx=3 is FFT-resolved on this grid but excluded from the radial
            # two-thirds mask. With nonlinear advection disabled, that mask has
            # no role in the q equation: the resolved mode must undergo only
            # its configured linear ETD damping.
            set!(model; ψ=(x, y, z) -> sin(3x), verbose=false)
            @test !is_dealiased(4, 1, grid)
            initial_q = copy(parent(model.fields.q))
            @test maximum(abs, initial_q) > 0

            step!(model, ExponentialRungeKutta2(; Δt))

            damping = exp(-diffusivity * 3^2 * Δt)
            @test parent(model.fields.q) ≈ damping .* initial_q atol=2e-13
        finally
            finalize_model!(model)
        end
    end
end

@testset "The first nonlinear QG RHS projects both Jacobian operands" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=EvolvingFlow(),
        feedback=NoFeedback(),
        formulation=PassiveWave(),
        inviscid=Inviscid(),
        topology=(1, 1),
        verbose=false,
    )
    try
        # (kx, ky)=(2,1) is outside this grid's radial cutoff, while (1,0)
        # is retained. Projecting both ψ and q therefore leaves one plane wave,
        # whose self-Jacobian is exactly zero. If velocity is instead diagnosed
        # from the unprojected ψ, their cross-product aliases into retained modes.
        set!(model; ψ=(x, y, z) -> cos(2x + y) + cos(x), verbose=false)
        @test !is_dealiased(3, 2, grid)
        context = QGYBJplus._operator_context(model)
        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        rhsq = similar(model.fields.q)
        rhsB = similar(model.fields.B)

        QGYBJplus._compute_etdrk2_rhs!(
            rhsq, rhsB, model.fields, context.grid, options, context.plans;
            a=context.a,
            dealias_mask=context.mask,
            workspace=context.workspace,
            N2_profile=context.N2,
            N2_face_profile=context.N2_face,
        )

        @test maximum(abs, parent(rhsq)) < 2e-12
    finally
        finalize_model!(model)
    end
end

@testset "Linear wave modes outside the nonlinear cutoff follow ETD-RK2" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    ν = 0.02
    Δt = 0.1
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=ν, order=2)),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=YBJPlus(),
        linear=LinearDynamics(),
        inviscid=Dissipative(),
        no_dispersion=Dispersive(),
        topology=(1, 1),
        verbose=false,
    )

    try
        B = zeros(ComplexF64, 1, 8, 8)
        B[1, 4, 1] = 1.0 + 0.25im # kx=3, outside the radial cutoff.
        set!(model; B=FieldArray(B; space=:spectral), verbose=false)
        @test !is_dealiased(4, 1, grid)

        context = QGYBJplus._operator_context(model)
        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        rhsq = similar(model.fields.q)
        rhsB = similar(model.fields.B)
        QGYBJplus._compute_etdrk2_rhs!(
            rhsq, rhsB, model.fields, context.grid, options, context.plans;
            a=context.a,
            dealias_mask=context.mask,
            workspace=context.workspace,
            N2_profile=context.N2,
            N2_face_profile=context.N2_face,
        )

        initial = B[1, 4, 1]
        dispersion_rate = -2im # A=-4B/k² gives (i f k²/2)A=-2ifB.
        @test parent(rhsB)[1, 4, 1] ≈ dispersion_rate * initial atol=2e-13

        x = ν * 3^2 * Δt
        E, hφ1, hφ2 = QGYBJplus._etd_coefficients(x, Δt)
        stage = E * initial + hφ1 * dispersion_rate * initial
        expected = E * initial + hφ1 * dispersion_rate * initial +
                   hφ2 * (dispersion_rate * stage -
                          dispersion_rate * initial)
        step!(model, ExponentialRungeKutta2(Δt=Δt))
        @test parent(model.fields.B)[1, 4, 1] ≈ expected atol=2e-13
    finally
        finalize_model!(model)
    end
end

@testset "Linear normal-YBJ recovery retains modes outside the cutoff" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    ν = 0.02
    Δt = 0.1
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=ν, order=2)),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=YBJ(),
        linear=LinearDynamics(),
        inviscid=Dissipative(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )

    try
        vertical_mode = ComplexF64[1 + 0.2im, -1 - 0.2im,
                                   1 + 0.2im, -1 - 0.2im]
        B = zeros(ComplexF64, 4, 8, 8)
        B[:, 4, 1] .= vertical_mode
        set!(model; B=FieldArray(B; space=:spectral), verbose=false)
        @test !is_dealiased(4, 1, grid)
        @test parent(model.fields.B)[:, 4, 1] ≈ vertical_mode atol=2e-13
        @test maximum(abs, parent(model.fields.A)[:, 4, 1]) > 0

        step!(model, ExponentialRungeKutta2(Δt=Δt))
        expected = exp(-ν * 3^2 * Δt) .* vertical_mode
        @test parent(model.fields.B)[:, 4, 1] ≈ expected atol=2e-13
        @test maximum(abs, parent(model.fields.A)[:, 4, 1]) > 0
    finally
        finalize_model!(model)
    end
end

@testset "Fixed-flow nonlinear advection projects the velocity operand" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=PassiveWave(),
        linear=NonlinearDynamics(),
        inviscid=Inviscid(),
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )

    try
        wave = [complex(cos(grid.x[i]))
                for k in 1:1, i in 1:8, j in 1:8]
        set!(model;
            ψ=(x, y, z) -> cos(2x + y) + cos(x),
            B=FieldArray(wave),
            verbose=false)
        prescribed_psi = copy(parent(model.fields.psi))
        @test !is_dealiased(3, 2, grid) # (kx,ky)=(2,1).

        context = QGYBJplus._operator_context(model)
        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        rhsq = similar(model.fields.q)
        rhsB = similar(model.fields.B)
        QGYBJplus._compute_etdrk2_rhs!(
            rhsq, rhsB, model.fields, context.grid, options, context.plans;
            a=context.a,
            dealias_mask=context.mask,
            workspace=context.workspace,
            N2_profile=context.N2,
            N2_face_profile=context.N2_face,
        )

        # Projecting ψ leaves only cos(x); it advects B=cos(x) by zero.
        @test maximum(abs, parent(rhsB)) < 2e-12
        @test parent(model.fields.psi) == prescribed_psi

        step!(model, ExponentialRungeKutta2(Δt=0.05))
        reference = copy_fields(model.fields)
        QGYBJplus.compute_velocities!(
            reference, context.grid;
            plans=context.plans,
            f=context.f,
            N2=first(context.N2),
            N2_profile=context.N2,
            compute_w=false,
            workspace=context.workspace,
            dealias_mask=context.mask,
        )
        @test parent(model.fields.u) ≈ parent(reference.u) atol=2e-13
        @test parent(model.fields.v) ≈ parent(reference.v) atol=2e-13
    finally
        finalize_model!(model)
    end
end

@testset "NoDispersion retains diagnosed wave amplitude" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    for formulation in (YBJPlus(), YBJ())
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=FixedFlow(),
            feedback=NoFeedback(),
            formulation=formulation,
            linear=LinearDynamics(),
            no_dispersion=NoDispersion(),
            topology=(1, 1),
            verbose=false,
        )
        try
            B = zeros(ComplexF64, 4, 8, 8)
            B[:, 2, 1] .= (1 + 0.2im, -1 - 0.2im,
                           1 + 0.2im, -1 - 0.2im)
            set!(model; B=FieldArray(B; space=:spectral), verbose=false)
            initial_B = copy(parent(model.fields.B))
            @test sum(abs2, parent(model.fields.A)) > 0
            @test sum(abs2, parent(model.fields.C)) > 0

            step!(model, ExponentialRungeKutta2(Δt=0.1))
            @test parent(model.fields.B) ≈ initial_B atol=2e-13
            @test sum(abs2, parent(model.fields.A)) > 0
            @test sum(abs2, parent(model.fields.C)) > 0
        finally
            finalize_model!(model)
        end
    end
end
