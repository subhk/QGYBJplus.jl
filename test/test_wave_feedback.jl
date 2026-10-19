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

        # The wave PV reverses with the sign of f (hemisphere).
        opposite = similar(model.fields.q)
        QGYBJplus.compute_qw_complex!(
            opposite, model.fields.B, context.grid, context.plans;
            f=-f, Lmask=context.mask)
        @test parent(opposite) ≈ -parent(qwave_complex) rtol=1e-12

        # The inversion reads the balanced PV q - qʷ; prognostic q is untouched.
        options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
        balanced = similar(model.fields.q)
        QGYBJplus._balanced_pv_rhs!(balanced, model.fields, context.grid,
            options, context.plans, context.mask)
        @test parent(balanced) ≈ -parent(qwave_complex) atol=5e-11
        @test iszero(maximum(abs, parent(model.fields.q)))
    finally
        finalize_model!(model)
    end
end

@testset "Normal-YBJ amplitude recovery satisfies B = LA and solvability" begin
    f = 7.5e-5
    grid = RectilinearGrid(size=(8, 8, 5), x=(-π, π), y=(-π, π), z=(-400.0, 0.0))
    model = wave_feedback_test_model(grid, YBJ(); f)
    try
        context = QGYBJplus._operator_context(model)
        nz = grid.size[3]
        B, nB, rB = (similar(model.fields.B) for _ in 1:3)
        for (array, (a, b, c)) in zip((B, nB, rB),
                                      ((0.11, 0.03, -0.07), (0.04, -0.02, 0.03),
                                       (0.08, -0.03, -0.04)))
            parent(array) .= [complex(a * k + b * i, c * j + 0.02k)
                              for k in 1:nz, i in 1:8, j in 1:8]
        end
        parent(B) .-= sum(parent(B); dims=1) ./ nz   # B = LA has zero depth mean
        parent(B) .*= 1e-8
        N2_profile = [1.0e-4, 1.2e-4, 1.5e-4, 1.1e-4, 0.9e-4]
        σ = QGYBJplus.compute_sigma(f, context.grid, nB, rB; Lmask=context.mask)
        A, C = similar(model.fields.A), similar(model.fields.C)
        QGYBJplus.compute_A!(A, C, B, σ, context.grid;
                             f, Lmask=context.mask, N2_profile)

        kh² = reshape(grid.kx .^ 2, 1, :, 1) .+ reshape(grid.ky .^ 2, 1, 1, :)
        kept = reshape(Array(context.mask), 1, 8, 8)
        # B = LA, with the flux (f²/N²) A_z = a C on the upper faces.
        a = f^2 ./ N2_profile
        LA = map(1:nz) do k
            up = k < nz ? a[k] .* parent(C)[k:k, :, :] : zero(parent(C)[k:k, :, :])
            down = k > 1 ? a[k - 1] .* parent(C)[k - 1:k - 1, :, :] : zero(up)
            (up .- down) ./ grid.dz
        end
        @test maximum(abs, (reduce(vcat, LA) .- parent(B)) .* kept) <
              1e-10 * maximum(abs, parent(B))
        # Solvability: the YBJ tendency has zero depth sum wherever kₕ > 0.
        tendency = .-parent(nB) .- 0.5im .* parent(rB) .+
                   0.5im * f .* kh² .* parent(A)
        @test maximum(abs, sum(tendency; dims=1) .* kept .* (kh² .> 0)) <
              1e-10 * maximum(abs, tendency)
    finally
        finalize_model!(model)
    end
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

        # Scheduled energy diagnostics rebuild ψ from q-qʷ (and A from B) in
        # their own storage, and observing must not mutate the live model.
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
                @test_throws ArgumentError restore!(uncoupled, path)
            finally
                finalize_model!(uncoupled)
            end
            normal_ybj = coupled_feedback_test_model(
                grid; f=context.f, formulation=YBJ())
            try
                @test_throws ArgumentError restore!(normal_ybj, path)
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

"""Coupled model with generic low-mode ψ and B (zero vertical mean per B column)."""
function generic_coupled_model(formulation; nx=16, nz=6)
    grid = RectilinearGrid(size=(nx, nx, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> 1.0 + 0.5cos(3z); returns=:N²),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0)),
        flow=EvolvingFlow(), feedback=WaveMeanFeedback(), formulation=formulation,
        inviscid=Inviscid(), verbose=false)
    wave = zeros(ComplexF64, nz, nx, nx)
    flow = zeros(ComplexF64, nz, nx, nx)
    for (index, (m, n)) in enumerate(((1, 0), (0, 1), (1, 1), (2, -1), (1, -2), (2, 1)))
        vertical_B = [cis(0.7index * k) * (1 + 0.2k) for k in 1:nz]
        vertical_ψ = [cos(0.9index + 0.5k) for k in 1:nz]
        for k in 1:nz, i in 1:nx, j in 1:nx
            phase = m * grid.x[i] + n * grid.y[j]
            wave[k, i, j] += 0.15 * vertical_B[k] * cis(phase)
            flow[k, i, j] += 0.3 * vertical_ψ[k] * cos(phase + index) / (m^2 + n^2)
        end
    end
    for i in 1:nx, j in 1:nx
        wave[:, i, j] .-= sum(wave[:, i, j]) / nz
    end
    set!(model; ψ=FieldArray(flow), B=FieldArray(wave), verbose=false)
    return model
end

"""Flow and wave parts of the coupled energy of the state (q, B)."""
function coupled_energy_parts(model, q, B)
    fields = copy_fields(model.fields)
    copyto!(fields.q, q)
    copyto!(fields.B, B)
    runtime = model.runtime
    options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
    QGYBJplus._invert_total_q_to_psi!(fields, runtime.geometry, options,
        runtime.plans, runtime.device_coefficients.a_ell, runtime.dealias_mask;
        workspace=runtime.workspace)
    QGYBJplus._refresh_wave_diagnostics!(fields, model)
    energy = QGYBJplus._energy_components(model, fields)
    return energy[4] + energy[5], energy[2] + energy[3]
end

@testset "Wave feedback closes the semi-discrete coupled energy budget" begin
    # dE/dt along the model's own tendencies, by central differences with
    # Richardson extrapolation (exact for this polynomial energy). The flow
    # and wave parts exchange energy at an O(1) rate; with the XV15 / A&Y
    # wave PV their sum is conserved. Flipping the sign of either term of
    # qʷ, or scaling it, leaves a residual comparable to the exchange.
    for formulation in (YBJPlus(), YBJ())
        model = generic_coupled_model(formulation)
        try
            context = QGYBJplus._operator_context(model)
            options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
            state = copy_fields(model.fields)
            q_t, B_t = similar(state.q), similar(state.B)
            QGYBJplus._compute_etdrk2_rhs!(q_t, B_t, state, context.grid, options,
                context.plans; a=context.a, dealias_mask=context.mask,
                workspace=context.workspace, N2_profile=context.N2,
                N2_face_profile=context.N2_face)
            q, B = model.fields.q, model.fields.B
            function rates(ε)
                plus = coupled_energy_parts(model, q .+ ε .* q_t, B .+ ε .* B_t)
                minus = coupled_energy_parts(model, q .- ε .* q_t, B .- ε .* B_t)
                return (plus .- minus) ./ 2ε
            end
            coarse, fine = rates(1e-3), rates(5e-4)
            flow_rate, wave_rate = (4 .* fine .- coarse) ./ 3
            # A wrong sign of either qʷ term leaves a residual about as large
            # as the exchange for these fields.
            energy = sum(coupled_energy_parts(model, q, B))
            @test abs(flow_rate) > 1e-4 * energy
            @test abs(flow_rate + wave_rate) < 1e-6 * abs(flow_rate)
            # Wave action ∫|B|² is conserved too.
            @test abs(real(sum(conj.(B) .* B_t))) < 1e-12 * sum(abs2, B)
        finally
            finalize_model!(model)
        end
    end
end

@testset "Coupled mean-flow w closes the coupled buoyancy equation" begin
    # With wave feedback D_t(q_g + qʷ) = 0, so the omega equation gains the
    # forcing f ∂z(D_t qʷ). The resulting w must close the Lagrangian-mean
    # buoyancy equation b_t + J(ψ, b) + N² w = 0 with the coupled ψ_t.
    for formulation in (YBJPlus(), YBJ())
        model = generic_coupled_model(formulation)
        try
            context = QGYBJplus._operator_context(model)
            options = QGYBJplus.ETDModelOptions(model.physics, model.numerics)
            runtime = model.runtime
            G, plans, mask = runtime.geometry, runtime.plans, runtime.dealias_mask
            compute_velocities!(model)
            w = copy(model.fields.w)

            state = copy_fields(model.fields)
            q_t, B_t = similar(state.q), similar(state.B)
            QGYBJplus._compute_etdrk2_rhs!(q_t, B_t, state, context.grid, options,
                plans; a=context.a, dealias_mask=mask, workspace=context.workspace,
                N2_profile=context.N2, N2_face_profile=context.N2_face)
            B = model.fields.B
            qw(field) = QGYBJplus.compute_qw_complex!(similar(field), field, G, plans;
                                                      f=options.f, Lmask=mask)
            ε = maximum(abs, B) / maximum(abs, B_t)
            qw_t = (qw(B .+ ε .* B_t) .- qw(B .- ε .* B_t)) ./ 2ε
            tendency = copy_fields(model.fields)
            tendency.q .= q_t .- qw_t
            QGYBJplus.invert_q_to_psi!(tendency, G; a=runtime.coefficients.a_ell)

            ψ, ψ_t = model.fields.psi, tendency.psi
            f, Δz, nz = options.f, model.grid.dz, model.grid.size[3]
            ψ_above = cat(ψ[2:end, :, :], zero(ψ[1:1, :, :]); dims=1)
            J_faces = QGYBJplus.jacobian_spectral!(similar(ψ), ψ, ψ_above, G, plans;
                                                    Lmask=mask)
            w_face_hat = zero(ψ)
            for k in 1:(nz - 1)
                b_t = f .* (ψ_t[k + 1, :, :] .- ψ_t[k, :, :]) ./ Δz
                w_face_hat[k, :, :] = .-(b_t .+ (f / Δz) .* J_faces[k, :, :]) ./
                                      runtime.coefficients.N²_face[k]
            end
            w_face = real.(fft_backward!(similar(w_face_hat), w_face_hat, plans))
            centres = (cat(zero(w_face[1:1, :, :]), w_face[1:end-1, :, :]; dims=1) .+ w_face) ./ 2
            @test maximum(abs, w .- centres) < 1e-11 * maximum(abs, w_face)
        finally
            finalize_model!(model)
        end
    end
end

@testset "Flow damping acts on the balanced PV, not the wave PV" begin
    n, nz = 16, 8
    grid = RectilinearGrid(size=(n, n, nz), extent=(2π, 2π, 1.0))
    wave = [complex(0.3cospi((k - 0.5) / nz)) *
            (1 + 0.6cos(grid.x[i]) + 0.4im * sin(2grid.y[j]))
            for k in 1:nz, i in 1:n, j in 1:n]
    function damped_model(; ν=0.05, νz=0.0, dispersion=Dispersive())
        return QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0), flow=EvolvingFlow(),
            feedback=WaveMeanFeedback(), formulation=YBJPlus(), no_dispersion=dispersion,
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=ν, order=4),
                wave=WaveHyperdiffusivity(coefficient=0)),
            vertical_diffusion=VerticalDiffusivity(νz), verbose=false)
    end
    advance!(model, T, steps) = foreach(_ -> step!(model, ExponentialRungeKutta2(Δt=T / steps)), 1:steps)

    # Frozen waves and no balanced flow: damping the total PV would turn the
    # wave PV into a balanced flow; damping the balanced PV leaves none.
    for (ν, νz) in ((0.05, 0.0), (0.0, 0.01))
        model = damped_model(; ν, νz, dispersion=NoDispersion())
        try
            set!(model; B=FieldArray(wave), verbose=false)
            advance!(model, 1.0, 20)
            compute_velocities!(model; compute_w=false)
            @test maximum(abs, model.fields.u) < 1e-14
        finally
            finalize_model!(model)
        end
    end

    # The coupled, damped system stays second order in Δt.
    final_q = map((10, 20, 40)) do steps
        model = damped_model(; νz=0.002)
        try
            set!(model; ψ=(x, y, z) -> 0.4sin(x + 0.2) * cos(2y) * (1 + 0.5z),
                 B=FieldArray(wave), verbose=false)
            advance!(model, 0.2, steps)
            copy(Array(model.fields.q))
        finally
            finalize_model!(model)
        end
    end
    ratio = maximum(abs, final_q[1] .- final_q[2]) / maximum(abs, final_q[2] .- final_q[3])
    @test 3.6 < ratio < 4.4
end
