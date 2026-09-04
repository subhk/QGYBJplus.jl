using Test
using NCDatasets
using QGYBJplus

function wave_feedback_test_model(grid, formulation; f=2e-4, linear=NonlinearDynamics())
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=f),
        stratification=ConstantStratification(N²=1e-4),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=FixedFlow(),
        feedback=NoFeedback(),
        formulation=formulation,
        linear=linear,
        no_dispersion=NoDispersion(),
        topology=(1, 1),
        verbose=false,
    )
end

function coupled_feedback_test_model(grid; f=1.0, formulation=YBJPlus())
    return QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=f),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=EvolvingFlow(),
        feedback=WaveMeanFeedback(),
        formulation=formulation,
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
                N2_face_profile=context.N2_face,
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

#=
Structural properties the wave PV must satisfy, independently of how it is
discretised:

    q^w = (i/2f₀) J(B*, B) + (1/4f₀) ∇²_h |B|²

The testset above pins the value against a hand calculation for one B. These
pin the properties, so a change that happens to preserve that one case but
breaks the form still fails.
=#
@testset "Wave PV structural properties" begin
    f = 2e-4
    grid = RectilinearGrid(size=(16, 16, 4), x=(-π, π), y=(-π, π), z=(-1.0, 0.0))

    function feedback_model(coriolis)
        return QGYBJModel(
            grid=grid, coriolis=FPlane(f=coriolis),
            stratification=ConstantStratification(N²=1e-4),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0)),
            flow=FixedFlow(), feedback=WaveMeanFeedback(),
            formulation=YBJPlus(), topology=(1, 1), verbose=false)
    end

    """Physical-space q^w for an envelope given as a function of (x, y, z)."""
    function wave_pv(coriolis, envelope)
        model = feedback_model(coriolis)
        try
            context = QGYBJplus._operator_context(model)
            physical = QGYBJplus.allocate_fft_backward_dst(
                model.fields.B, model.runtime)
            values = parent(physical)
            @inbounds for k in axes(values, 1), i in axes(values, 2),
                          j in axes(values, 3)
                values[k, i, j] = envelope(grid.x[i], grid.y[j], grid.z[k])
            end
            QGYBJplus.fft_forward!(model.fields.B, physical, model.runtime.plans)

            spectral = similar(model.fields.q)
            QGYBJplus.compute_qw_complex!(spectral, model.fields.B,
                context.grid, context.plans; f=coriolis, Lmask=context.mask)
            result = QGYBJplus.allocate_fft_backward_dst(spectral, model.runtime)
            QGYBJplus.fft_backward!(result, spectral, context.plans)
            return copy(parent(result))
        finally
            finalize_model!(model)
        end
    end

    largest(array) = maximum(abs, array)
    mixed(x, y, z) = sin(x) + im * sin(y)

    baseline = wave_pv(f, mixed)
    @test largest(real.(baseline)) > 0

    # (i/2)J(B*,B) is real because J(B*,B) is purely imaginary, and ∇²|B|² is
    # real, so the wave PV carries no imaginary part.
    @test largest(imag.(baseline)) / largest(real.(baseline)) < 1e-12

    # Both terms are horizontal derivatives: a horizontally uniform envelope
    # contributes no PV, however large it is.
    @test largest(wave_pv(f, (x, y, z) -> 1 + 2im)) == 0

    # A single plane wave has |B|² constant and J(B*,B) = 0.
    @test largest(wave_pv(f, (x, y, z) -> 0.7 * exp(3im * x))) < 1e-9

    # Quadratic in the envelope.
    doubled = wave_pv(f, (x, y, z) -> 2 * mixed(x, y, z))
    @test largest(real.(doubled)) / largest(real.(baseline)) ≈ 4 rtol = 1e-10

    # Inversely proportional to f₀.
    halved_f = wave_pv(f / 2, mixed)
    @test largest(real.(halved_f)) / largest(real.(baseline)) ≈ 2 rtol = 1e-10

    # The signed Coriolis frequency matters: the full wave-PV contribution
    # reverses between hemispheres.
    negative_f = wave_pv(-f, mixed)
    @test negative_f ≈ -baseline rtol = 1e-10 atol = 1e-9

    # A constant envelope phase cancels from both B* derivatives and |B|².
    phase = cis(0.37)
    phase_shifted = wave_pv(f, (x, y, z) -> phase * mixed(x, y, z))
    @test phase_shifted ≈ baseline rtol = 1e-10 atol = 1e-9

    # Both terms are horizontal divergences on the periodic domain.
    @test abs(sum(real, baseline)) < 1e-11 * length(baseline) * largest(baseline)

    # A purely real envelope kills the Jacobian, isolating the |B|² term:
    # (1/4f) ∂ₓ²sin²x = (1/2f) cos 2x.
    real_envelope = wave_pv(f, (x, y, z) -> sin(x) + 0im)
    expected = [0.5 * cos(2 * grid.x[i]) / f
                for k in axes(real_envelope, 1), i in axes(real_envelope, 2),
                    j in axes(real_envelope, 3)]
    @test real.(real_envelope) ≈ expected rtol = 1e-10

    # A conjugated envelope flips the Jacobian term and leaves |B|² alone, so
    # q^w(B*) is the |B|² part minus the Jacobian part.
    conjugated = wave_pv(f, (x, y, z) -> conj(mixed(x, y, z)))
    jacobian_part = (real.(baseline) .- real.(conjugated)) ./ 2
    magnitude_part = (real.(baseline) .+ real.(conjugated)) ./ 2
    @test largest(jacobian_part) > 0
    # For B = sin x + i sin y the Jacobian part is -cos x cos y / f.
    jacobian_expected = [-cos(grid.x[i]) * cos(grid.y[j]) / f
                         for k in axes(baseline, 1), i in axes(baseline, 2),
                             j in axes(baseline, 3)]
    @test jacobian_part ≈ jacobian_expected rtol = 1e-10
    # ...and the remainder is the |B|² part, 0.5(cos 2x + cos 2y)/f.
    magnitude_expected = [0.5 * (cos(2grid.x[i]) + cos(2grid.y[j])) / f
                          for k in axes(baseline, 1), i in axes(baseline, 2),
                              j in axes(baseline, 3)]
    @test magnitude_part ≈ magnitude_expected rtol = 1e-10
end

@testset "Wave feedback owns total PV throughout the model API" begin
    grid = RectilinearGrid(
        size=(16, 16, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = coupled_feedback_test_model(grid)

    try
        wave = [
            (1 + 0.1k) * (sin(grid.x[i]) + im * sin(grid.y[j]))
            for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]
        ]
        set!(model;
            ψ=(x, y, z) -> cos(x) + 0.3sin(y),
            B=FieldArray(wave),
            verbose=false)

        context = QGYBJplus._operator_context(model)
        expected_psi = copy(parent(model.fields.psi))
        qg = similar(model.fields.q)
        dz = grid.z[2] - grid.z[1]
        QGYBJplus.compute_q_from_psi!(
            qg, model.fields.psi, context.grid, context.a, dz;
            workspace=context.workspace)
        qw = similar(model.fields.q)
        QGYBJplus.compute_qw_complex!(
            qw, model.fields.B, context.grid, context.plans;
            f=context.f, Lmask=context.mask, workspace=context.workspace)
        expected_total_q = parent(qg) .+ parent(qw)

        # `q` is prognostic total generalized PV, including immediately after
        # initialization. A first diagnostic pass must not move a prescribed ψ.
        @test parent(model.fields.q) ≈ expected_total_q atol=2e-11
        QGYBJplus._diagnose_flow!(model.fields, context.grid,
            QGYBJplus.ETDModelOptions(model.physics, model.numerics),
            context.plans, context.a, context.mask;
            workspace=context.workspace,
            N2_profile=context.N2)
        @test parent(model.fields.psi) ≈ expected_psi atol=2e-11
        @test parent(model.fields.q) ≈ expected_total_q atol=2e-11

        # The exported model-level inversion has the same total-PV semantics as
        # the time-stepper, and leaves the prognostic field untouched.
        fill!(parent(model.fields.psi), 0)
        invert_q_to_psi!(model)
        @test parent(model.fields.psi) ≈ expected_psi atol=2e-11
        @test parent(model.fields.q) ≈ expected_total_q atol=2e-11

        # Temporary q-qʷ replacement is transactional even if inversion fails.
        q_before_failure = copy(parent(model.fields.q))
        @test_throws AssertionError QGYBJplus._diagnose_flow!(
            model.fields, context.grid,
            QGYBJplus.ETDModelOptions(model.physics, model.numerics),
            context.plans, ones(1), context.mask;
            workspace=context.workspace,
            N2_profile=context.N2)
        @test parent(model.fields.q) == q_before_failure
    finally
        finalize_model!(model)
    end
end

@testset "Normal YBJ wave initialization preserves prescribed flow" begin
    grid = RectilinearGrid(
        size=(8, 8, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = coupled_feedback_test_model(grid; formulation=YBJ())

    try
        set!(model; ψ=(x, y, z) -> cos(x) + 0.2sin(y), verbose=false)
        prescribed_psi = copy(parent(model.fields.psi))
        vertical_mode = (-1.5, -0.5, 0.5, 1.5)
        wave = [
            vertical_mode[k] * (sin(grid.x[i]) + im * sin(grid.y[j]))
            for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]
        ]
        set!(model; B=FieldArray(wave), verbose=false)

        context = QGYBJplus._operator_context(model)
        qg = similar(model.fields.q)
        qw = similar(model.fields.q)
        QGYBJplus.compute_q_from_psi!(qg, model.fields.psi,
            context.grid, context.a, grid.z[2] - grid.z[1];
            workspace=context.workspace)
        QGYBJplus.compute_qw_complex!(qw, model.fields.B,
            context.grid, context.plans;
            f=context.f, Lmask=context.mask, workspace=context.workspace)

        @test parent(model.fields.psi) ≈ prescribed_psi atol=2e-11
        @test parent(model.fields.q) ≈
              parent(qg) .+ parent(qw) atol=2e-11
    finally
        finalize_model!(model)
    end
end

@testset "Coupled output, energy, and restart use balanced PV" begin
    grid = RectilinearGrid(
        size=(8, 8, 4),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = coupled_feedback_test_model(grid)

    try
        wave = [
            (1 + 0.15k) * (sin(grid.x[i]) + im * sin(grid.y[j]))
            for k in 1:grid.size[3], i in 1:grid.size[1], j in 1:grid.size[2]
        ]
        set!(model;
            ψ=(x, y, z) -> cos(x) + 0.2sin(y),
            B=FieldArray(wave),
            verbose=false)

        context = QGYBJplus._operator_context(model)
        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        expected_psi = copy(parent(model.fields.psi))
        qg = similar(model.fields.q)
        qw = similar(model.fields.q)
        QGYBJplus.compute_q_from_psi!(qg, model.fields.psi,
            context.grid, context.a, grid.z[2] - grid.z[1];
            workspace=context.workspace)
        QGYBJplus.compute_qw_complex!(qw, model.fields.B,
            context.grid, context.plans;
            f=context.f, Lmask=context.mask, workspace=context.workspace)
        parent(model.fields.q) .= parent(qg) .+ parent(qw)
        expected_q = copy(parent(model.fields.q))
        expected_B = copy(parent(model.fields.B))

        # Scheduled energy diagnostics rebuild a private field snapshot. Its
        # flow energy must be based on q-qʷ, and observing must not mutate the
        # live model.
        reference = QGYBJplus.copy_fields(model.fields)
        QGYBJplus._diagnose_flow!(reference, context.grid, options,
            context.plans, context.a, context.mask;
            workspace=context.workspace,
            N2_profile=context.N2)
        QGYBJplus._refresh_wave_diagnostics!(reference, model)
        reference_energy = QGYBJplus._energy_components(model, reference)

        simulation = Simulation(model;
            Δt=0.1,
            stop_iteration=1,
            output=false,
            diagnostics=false,
            verbose=false)
        diagnostic_specification = EnergyDiagnosticsOutput(
            path="unused", schedule=IterationInterval(1))
        diagnostic_manager = QGYBJplus.EnergyDiagnosticsManager(
            diagnostic_specification, Float64)
        QGYBJplus._record_energy_diagnostics!(
            diagnostic_manager, simulation)
        @test only(diagnostic_manager.mean_flow_KE) ≈ reference_energy[4]
        @test only(diagnostic_manager.mean_flow_PE) ≈ reference_energy[5]
        @test parent(model.fields.psi) == expected_psi
        @test parent(model.fields.q) == expected_q

        mktempdir() do directory
            specification = NetCDFOutput(
                path=directory,
                schedule=IterationInterval(1),
                fields=(:ψ, :waves),
                velocities=true)
            manager = QGYBJplus.ModelOutputManager(
                specification, Float64)
            QGYBJplus._write_model_state_file!(manager, simulation)
            path = joinpath(directory, "state0001.nc")

            expected_spectral = similar(model.fields.psi)
            parent(expected_spectral) .= expected_psi
            expected_physical = QGYBJplus.allocate_fft_backward_dst(
                model.fields.psi, model.runtime)
            QGYBJplus.fft_backward!(expected_physical,
                expected_spectral, model.runtime.plans)
            NCDataset(path, "r") do dataset
                @test dataset["psi"][:, :, :] ≈
                      permutedims(real.(parent(expected_physical)), (2, 3, 1))
                @test dataset.attrib["feedback_mode"] == "WaveMeanFeedback"
                @test dataset.attrib["generalized_pv"] == "total_with_wave_pv"
            end

            uncoupled = wave_feedback_test_model(
                grid, YBJPlus(); f=context.f)
            try
                @test_throws ErrorException restore!(uncoupled, path)
            finally
                finalize_model!(uncoupled)
            end
            normal_ybj = coupled_feedback_test_model(
                grid; f=context.f, formulation=YBJ())
            try
                @test_throws ErrorException restore!(normal_ybj, path)
            finally
                finalize_model!(normal_ybj)
            end

            fill!(parent(model.fields.psi), 0)
            fill!(parent(model.fields.A), 0)
            fill!(parent(model.fields.C), 0)
            restore!(model, path)
            @test parent(model.fields.psi) ≈ expected_psi atol=2e-11
            @test parent(model.fields.q) ≈ expected_q atol=2e-11
            @test parent(model.fields.B) ≈ expected_B atol=2e-11
        end
    finally
        finalize_model!(model)
    end
end

@testset "Wave feedback excludes the aliased two-thirds endpoint" begin
    grid = RectilinearGrid(
        size=(12, 12, 1),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = wave_feedback_test_model(grid, YBJPlus(); f=1.0)

    try
        context = QGYBJplus._operator_context(model)
        wave = QGYBJplus.allocate_fft_backward_dst(
            model.fields.B, model.runtime)
        values = parent(wave)
        @inbounds for k in axes(values, 1), i in axes(values, 2),
                      j in axes(values, 3)
            values[k, i, j] = cos(4grid.x[i])
        end
        QGYBJplus.fft_forward!(
            model.fields.B, wave, model.runtime.plans)

        qw = similar(model.fields.q)
        QGYBJplus.compute_qw_complex!(
            qw, model.fields.B, context.grid, context.plans;
            f=1.0, Lmask=context.mask, workspace=context.workspace)

        @test !is_dealiased(5, 1, grid) # kx = N/3 is not alias-safe.
        @test is_dealiased(4, 3, grid)  # (kx, ky) = (3, 2) remains inside.
        @test maximum(abs, parent(qw)) < 1e-12
    finally
        finalize_model!(model)
    end
end

@testset "Coupled wave-feedback ETD-RK2 is second order" begin
    grid = RectilinearGrid(
        size=(8, 8, 4),
        extent=(2π, 2π, 1.0),
    )
    horizon = 0.1

    function evolve_coupled(Δt)
        model = coupled_feedback_test_model(grid)
        try
            wave = [
                (0.10 + 0.02k) *
                (sin(grid.x[i]) + 0.7im * cos(grid.y[j]) +
                 0.3sin(grid.x[i] + grid.y[j]))
                for k in 1:grid.size[3], i in 1:grid.size[1],
                    j in 1:grid.size[2]
            ]
            set!(model;
                ψ=(x, y, z) ->
                    0.8sin(x) * (1 + 0.4z) +
                    0.35sin(x + y) * (1 + 0.3z^2),
                B=FieldArray(wave),
                verbose=false,
            )

            timestepper = ExponentialRungeKutta2(Δt=Δt)
            for _ in 1:round(Int, horizon / Δt)
                step!(model, timestepper)
            end

            context = QGYBJplus._operator_context(model)
            qg = similar(model.fields.q)
            qw = similar(model.fields.q)
            QGYBJplus.compute_q_from_psi!(
                qg, model.fields.psi, context.grid, context.a, grid.dz;
                workspace=context.workspace,
            )
            QGYBJplus.compute_qw_complex!(
                qw, model.fields.B, context.grid, context.plans;
                f=context.f, Lmask=context.mask, workspace=context.workspace,
            )
            q_values = copy(parent(model.fields.q))
            consistency = maximum(abs,
                q_values .- parent(qg) .- parent(qw)) /
                max(maximum(abs, q_values), eps())
            return (
                q=q_values,
                B=copy(parent(model.fields.B)),
                consistency,
            )
        finally
            finalize_model!(model)
        end
    end

    coarse = evolve_coupled(0.01)
    medium = evolve_coupled(0.005)
    fine = evolve_coupled(0.0025)
    q_ratio = maximum(abs, coarse.q .- medium.q) /
              maximum(abs, medium.q .- fine.q)
    B_ratio = maximum(abs, coarse.B .- medium.B) /
              maximum(abs, medium.B .- fine.B)

    @test 3.7 < q_ratio < 4.3
    @test 3.7 < B_ratio < 4.3
    @test fine.consistency < 2e-12
end

@testset "Coupled-energy spectral coefficients" begin
    grid = RectilinearGrid(
        size=(8, 8, 3),
        x=(-π, π),
        y=(-π, π),
        z=(-1.0, 0.0),
    )
    model = coupled_feedback_test_model(grid; f=2.0)

    try
        fields = model.fields
        a = only(unique(model.runtime.coefficients.a_ell))
        dz = grid.z[2] - grid.z[1]
        normalization = 0.5 / ((grid.size[1] * grid.size[2])^2 * grid.size[3])

        # These are full complex FFTs, so the horizontal zero mode has the same
        # Parseval weight as every other mode.
        fill!(parent(fields.A), 0)
        fill!(parent(fields.C), 0)
        parent(fields.C)[1, 1, 1] = 2
        LA_bottom = a * 2 / dz
        LA_above = -LA_bottom
        expected_wke = normalization * (abs2(LA_bottom) + abs2(LA_above))
        energies = QGYBJplus._local_energy_components(model, fields)
        @test energies[1] ≈ expected_wke rtol=2e-14

        # Asselin--Young (3.7): a/4 |∇A_z|² and 1/16 |ΔA|².
        # `_local_energy_components` applies the common outer factor 1/2.
        fill!(parent(fields.A), 0)
        fill!(parent(fields.C), 0)
        parent(fields.C)[1, 2, 1] = 3
        parent(fields.A)[1, 2, 1] = 5
        energies = QGYBJplus._local_energy_components(model, fields)
        @test energies[2] ≈ normalization * (0.5a * 9) rtol=2e-14
        @test energies[3] ≈ normalization * ((1 / 8) * 25) rtol=2e-14
    finally
        finalize_model!(model)
    end

    ybj_model = coupled_feedback_test_model(grid; f=2.0, formulation=YBJ())
    try
        parent(ybj_model.fields.A)[1, 2, 1] = 5
        @test iszero(QGYBJplus._local_energy_components(
            ybj_model, ybj_model.fields)[3])
    finally
        finalize_model!(ybj_model)
    end
end
