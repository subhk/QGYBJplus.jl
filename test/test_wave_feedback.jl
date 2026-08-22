using Test
using QGYBJplus

function wave_feedback_test_model(grid, formulation; f=2e-4, linear=NonlinearDynamics())
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=f),
        stratification=ConstantStratification(N²=1e-4),
        closure=HorizontalHyperdiffusivity(
            flow=0, flow2=0, waves=0, waves2=0),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=formulation,
        linear=linear,
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )
end

function set_analytic_wave!(model)
    grid = model.grid
    physical = QGYBJplus.allocate_fft_backward_dst(
        model.fields.B, model.runtime)
    values = parent(physical)
    @inbounds for k in axes(values, 1), i in axes(values, 2), j in axes(values, 3)
        values[k, i, j] = sin(grid.x[i]) + im * sin(grid.y[j])
    end
    QGYBJplus.fft_forward!(model.fields.B, physical, model.runtime.plans)
    return physical
end

@testset "Dimensional wave feedback" begin
    f = 2e-4
    grid = RectilinearGrid(
        size=(16, 16, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = wave_feedback_test_model(grid, YBJPlus(); f)

    try
        wave_physical = set_analytic_wave!(model)
        context = QGYBJplus._operator_context(model)

        qwave_complex = similar(model.fields.q)
        QGYBJplus.compute_qw_complex!(
            qwave_complex, model.fields.B, context.grid, context.plans;
            f, Lmask=context.mask)
        qwave_physical = QGYBJplus.allocate_fft_backward_dst(
            qwave_complex, model.runtime)
        QGYBJplus.fft_backward!(
            qwave_physical, qwave_complex, context.plans)

        expected = similar(real.(parent(qwave_physical)))
        @inbounds for k in axes(expected, 1), i in axes(expected, 2), j in axes(expected, 3)
            x = grid.x[i]
            y = grid.y[j]
            expected[k, i, j] = (
                -cos(x) * cos(y) + 0.5 * (cos(2x) + cos(2y))) / f
        end
        @test real.(parent(qwave_physical)) ≈ expected rtol=5e-13 atol=5e-11
        @test maximum(abs, imag.(parent(qwave_physical))) < 5e-11

        BRk = similar(model.fields.B)
        BIk = similar(model.fields.B)
        QGYBJplus.split_B_to_real_imag!(
            BRk, BIk, model.fields.B, context.plans)

        BR_physical = QGYBJplus.allocate_fft_backward_dst(BRk, model.runtime)
        BI_physical = QGYBJplus.allocate_fft_backward_dst(BIk, model.runtime)
        QGYBJplus.fft_backward!(BR_physical, BRk, context.plans)
        QGYBJplus.fft_backward!(BI_physical, BIk, context.plans)
        @test real.(parent(BR_physical)) ≈
              real.(parent(wave_physical)) atol=2e-13
        @test real.(parent(BI_physical)) ≈
              imag.(parent(wave_physical)) atol=2e-13

        qwave_split = similar(model.fields.q)
        QGYBJplus.compute_qw!(
            qwave_split, BRk, BIk, context.grid, context.plans;
            f, Lmask=context.mask)
        @test parent(qwave_split) ≈ parent(qwave_complex) rtol=5e-13 atol=5e-11

        recombined = similar(model.fields.B)
        QGYBJplus.combine_real_imag_to_B!(recombined, BRk, BIk)
        @test parent(recombined) ≈ parent(model.fields.B) atol=2e-13

        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        q_saved = QGYBJplus.replace_q_with_wave_feedback_rhs!(
            model.fields, context.grid, options, context.plans, context.mask)
        @test parent(model.fields.q) ≈ -parent(qwave_complex) atol=5e-11
        QGYBJplus.restore_prognostic_q!(model.fields, q_saved)
        @test iszero(maximum(abs, parent(model.fields.q)))
    finally
        finalize_model!(model)
    end
end

@testset "Normal-YBJ complex tendency parity" begin
    grid = RectilinearGrid(
        size=(8, 8, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    plus_model = wave_feedback_test_model(
        grid, YBJPlus(); f=1.0, linear=LinearDynamics())
    normal_model = wave_feedback_test_model(
        grid, YBJ(); f=1.0, linear=LinearDynamics())

    try
        for model in (plus_model, normal_model)
            psi_physical = QGYBJplus.allocate_fft_backward_dst(
                model.fields.psi, model.runtime)
            wave_physical = QGYBJplus.allocate_fft_backward_dst(
                model.fields.B, model.runtime)
            psi = parent(psi_physical)
            wave = parent(wave_physical)
            vertical_sign = (-1.0, 1.0, -1.0, 1.0)
            @inbounds for k in axes(psi, 1), i in axes(psi, 2), j in axes(psi, 3)
                psi[k, i, j] = cos(grid.x[i]) + 0.3 * cos(grid.y[j])
                wave[k, i, j] = vertical_sign[k] * (
                    sin(grid.x[i]) + im * cos(grid.y[j]))
            end
            QGYBJplus.fft_forward!(
                model.fields.psi, psi_physical, model.runtime.plans)
            QGYBJplus.fft_forward!(
                model.fields.B, wave_physical, model.runtime.plans)
        end

        function wave_rhs(model)
            context = QGYBJplus._operator_context(model)
            options = QGYBJplus.ETDModelOptions(
                model.physics, model.numerics)
            rhsq = similar(model.fields.q)
            rhsB = similar(model.fields.B)
            QGYBJplus._compute_etdrk2_rhs!(
                rhsq, rhsB, model.fields, context.grid, options, context.plans;
                a=context.a,
                dealias_mask=context.mask,
                workspace=context.workspace,
                N2_profile=context.N2,
            )
            return copy(parent(rhsB))
        end

        @test wave_rhs(normal_model) ≈ wave_rhs(plus_model) atol=2e-12
    finally
        finalize_model!(plus_model)
        finalize_model!(normal_model)
    end
end

@testset "Normal-YBJ complex recovery parity" begin
    f = 7.5e-5
    grid = RectilinearGrid(
        size=(8, 8, 5),
        x=(-π, π),
        y=(-π, π),
        z=(-400.0, 0.0),
    )
    model = wave_feedback_test_model(grid, YBJ(); f)

    try
        context = QGYBJplus._operator_context(model)
        component_arrays = ntuple(_ -> similar(model.fields.B), 6)
        BRk, BIk, nBRk, nBIk, rBRk, rBIk = component_arrays

        arrays = map(parent, component_arrays)
        @inbounds for k in axes(arrays[1], 1),
                      i in axes(arrays[1], 2),
                      j in axes(arrays[1], 3)
            arrays[1][k, i, j] = complex(0.11k + 0.03i, -0.07j + 0.02k)
            arrays[2][k, i, j] = complex(-0.05k + 0.01j, 0.09i - 0.02j)
            arrays[3][k, i, j] = complex(0.04k - 0.02i, 0.03j + 0.01k)
            arrays[4][k, i, j] = complex(-0.06j + 0.01i, 0.05k - 0.02i)
            arrays[5][k, i, j] = complex(0.08i - 0.03k, -0.04j + 0.01i)
            arrays[6][k, i, j] = complex(0.02j + 0.01k, 0.07i - 0.03k)
        end

        for component in (BRk, BIk)
            values = parent(component)
            @inbounds for i in axes(values, 2), j in axes(values, 3)
                component_mean = sum(@view values[:, i, j]) / size(values, 1)
                @views values[:, i, j] .-= component_mean
            end
            values .*= 1e-8
        end

        B = similar(model.fields.B)
        nB = similar(model.fields.B)
        rB = similar(model.fields.B)
        parent(B) .= parent(BRk) .+ im .* parent(BIk)
        parent(nB) .= parent(nBRk) .+ im .* parent(nBIk)
        parent(rB) .= parent(rBRk) .+ im .* parent(rBIk)

        N2_profile = [1.0e-4, 1.2e-4, 1.5e-4, 1.1e-4, 0.9e-4]
        sigma_complex = QGYBJplus.compute_sigma(
            f, context.grid, nB, rB;
            Lmask=context.mask)
        sigma_components = QGYBJplus.compute_sigma(
            f, context.grid, nBRk, nBIk, rBRk, rBIk;
            Lmask=context.mask)
        @test sigma_complex ≈ sigma_components rtol=5e-15 atol=5e-12

        nB_values = parent(nB)
        rB_values = parent(rB)
        @inbounds for i in axes(nB_values, 2), j in axes(nB_values, 3)
            i_global = QGYBJplus.local_to_global(i, 2, nB)
            j_global = QGYBJplus.local_to_global(j, 3, nB)
            kh² = context.grid.kx[i_global]^2 + context.grid.ky[j_global]^2
            expected_sigma = if context.mask[i_global, j_global] && kh² > 0
                sum(@view(rB_values[:, i, j]) .-
                    2im .* @view(nB_values[:, i, j])) / (f * kh²)
            else
                0.0im
            end
            @test sigma_complex[i, j] ≈ expected_sigma atol=2e-10
        end

        A_complex = similar(model.fields.A)
        C_complex = similar(model.fields.C)
        A_components = similar(model.fields.A)
        C_components = similar(model.fields.C)
        QGYBJplus.compute_A!(
            A_complex, C_complex, B, sigma_complex, context.grid;
            f, Lmask=context.mask, N2_profile)
        QGYBJplus.compute_A!(
            A_components, C_components, BRk, BIk,
            sigma_components, context.grid;
            f, Lmask=context.mask, N2_profile)

        @test parent(A_complex) ≈ parent(A_components) atol=2e-12
        @test parent(C_complex) ≈ parent(C_components) atol=2e-12

        A_values = parent(A_complex)
        B_values = parent(B)
        C_values = parent(C_complex)
        Δz = grid.z[2] - grid.z[1]
        @inbounds for i in axes(A_values, 2), j in axes(A_values, 3)
            i_global = QGYBJplus.local_to_global(i, 2, A_complex)
            j_global = QGYBJplus.local_to_global(j, 3, A_complex)
            kh² = context.grid.kx[i_global]^2 + context.grid.ky[j_global]^2
            @test sum(@view A_values[:, i, j]) ≈
                  sigma_complex[i, j] rtol=5e-15 atol=1e-11
            @test iszero(C_values[end, i, j])

            if context.mask[i_global, j_global] && kh² > 0
                cumulative_B = 0.0im
                for k in 1:(context.grid.nz-1)
                    cumulative_B += B_values[k, i, j]
                    expected_C = cumulative_B * N2_profile[k] * Δz / f^2
                    @test C_values[k, i, j] ≈ expected_C atol=2e-8
                end

                rhs_vertical_sum = sum(
                    -nB_values[k, i, j] +
                    0.5im * f * kh² * A_values[k, i, j] -
                    0.5im * rB_values[k, i, j]
                    for k in axes(A_values, 1))
                @test abs(rhs_vertical_sum) < 2e-10
            end
        end
    finally
        finalize_model!(model)
    end
end
