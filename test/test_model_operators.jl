using Test
using QGYBJplus

@testset "Model-owned elliptic and velocity operators" begin
    grid = RectilinearGrid(
        size=(8, 8, 8),
        extent=(2π, 2π, 2π),
    )
    N2_function = z -> 1.0 + 0.1cos(z)
    profile = AnalyticalProfile{Float64, typeof(N2_function)}(
        N2_function,
        true,
    )
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=profile,
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0),
        ),
        topology=(1, 1),
        verbose=false,
    )

    try
        fields = model.fields
        runtime = model.runtime
        coefficients = runtime.coefficients

        @test length(coefficients.N²) == grid.size[3]
        @test length(coefficients.N²_face) == grid.size[3]
        @test !all(==(first(coefficients.N²)), coefficients.N²)
        @test coefficients.N² ≈ N2_function.(grid.z)
        @test coefficients.N²_face ≈ N2_function.(grid.z_faces[2:end])
        @test coefficients.a_ell ≈
              model.physics.coriolis.f^2 ./ coefficients.N²_face

        fields.q[4, 2, 3] = 1.0 - 0.25im
        q_reference = copy_fields(fields)
        invert_q_to_psi!(model)
        QGYBJplus._invert_total_q_to_psi!(
            q_reference, runtime.geometry,
            QGYBJplus.ETDModelOptions(model.physics, model.numerics),
            runtime.plans, coefficients.a_ell, runtime.dealias_mask;
            workspace=runtime.workspace,
        )
        @test parent(fields.psi) ≈ parent(q_reference.psi)
        @test all(isfinite, parent(fields.psi))

        fields.B[5, 3, 2] = 0.75 + 0.5im
        B_reference = copy_fields(fields)
        invert_B_to_A!(model)
        invert_B_to_A!(
            B_reference,
            runtime.geometry,
            coefficients.a_ell;
            workspace=runtime.workspace,
        )
        @test parent(fields.A) ≈ parent(B_reference.A)
        @test parent(fields.C) ≈ parent(B_reference.C)

        velocity_reference = copy_fields(fields)
        compute_velocities!(model; compute_w=false)
        compute_velocities!(
            velocity_reference,
            runtime.geometry;
            plans=runtime.plans,
            f=model.physics.coriolis.f,
            N2_profile=coefficients.N²,
            compute_w=false,
            workspace=runtime.workspace,
            dealias_mask=runtime.dealias_mask,
        )
        @test parent(fields.u) ≈ parent(velocity_reference.u)
        @test parent(fields.v) ≈ parent(velocity_reference.v)

        ybj_reference = copy_fields(fields)
        compute_ybj_vertical_velocity!(model; skip_inversion=false, t=0.0)
        compute_ybj_vertical_velocity!(
            ybj_reference,
            runtime.geometry,
            runtime.plans;
            f=model.physics.coriolis.f,
            N2_profile=coefficients.N²,
            N2_face_profile=coefficients.N²_face,
            workspace=runtime.workspace,
            skip_inversion=false,
            t=0.0,
        )
        @test parent(fields.A) ≈ parent(ybj_reference.A)
        @test parent(fields.C) ≈ parent(ybj_reference.C)
        @test parent(fields.w) ≈ parent(ybj_reference.w)
        @test all(isfinite, parent(fields.w))

        # Both variable-coefficient elliptic solvers must invert the same
        # upper-face flux stencil used by their forward operators.
        fill!(parent(fields.psi), 0)
        psi_initial = @. complex(cos(0.7 * grid.z), sin(0.4 * grid.z))
        parent(fields.psi)[:, 2, 2] .= psi_initial
        QGYBJplus.compute_q_from_psi!(
            fields.q, fields.psi, runtime.geometry,
            coefficients.a_ell, grid.dz;
            workspace=runtime.workspace,
        )
        fill!(parent(fields.psi), 0)
        invert_q_to_psi!(model)
        @test parent(fields.psi)[:, 2, 2] ≈ psi_initial rtol=2e-13 atol=2e-13

        fill!(parent(fields.A), 0)
        fill!(parent(fields.B), 0)
        A_initial = @. complex(sin(0.8 * grid.z), cos(0.3 * grid.z))
        parent(fields.A)[:, 2, 1] .= A_initial
        A_values = parent(fields.A)
        B_values = parent(fields.B)
        a = coefficients.a_ell
        dz² = grid.dz^2
        kh² = grid.kx[2]^2
        B_values[1, 2, 1] =
            a[1] * (A_values[2, 2, 1] - A_values[1, 2, 1]) / dz² -
            0.25kh² * A_values[1, 2, 1]
        for k in 2:(grid.size[3] - 1)
            B_values[k, 2, 1] =
                (a[k] * (A_values[k + 1, 2, 1] - A_values[k, 2, 1]) -
                 a[k - 1] * (A_values[k, 2, 1] - A_values[k - 1, 2, 1])) / dz² -
                0.25kh² * A_values[k, 2, 1]
        end
        B_values[end, 2, 1] =
            a[end - 1] * (A_values[end - 1, 2, 1] - A_values[end, 2, 1]) / dz² -
            0.25kh² * A_values[end, 2, 1]
        fill!(parent(fields.A), 0)
        invert_B_to_A!(model)
        @test parent(fields.A)[:, 2, 1] ≈ A_initial rtol=2e-13 atol=2e-13
    finally
        finalize_model!(model)
    end
end

@testset "Horizontal-mean QG PV inversion retains vertical structure" begin
    grid = RectilinearGrid(size=(4, 4, 8), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> inv(2 + z); returns=:N²),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0),
        ),
        topology=(1, 1),
        verbose=false,
    )

    try
        fields = model.fields
        runtime = model.runtime
        ψ_initial = @. cos(2π * grid.z) + 0.3sin(3π * grid.z)
        ψ_initial .-= sum(ψ_initial) / length(ψ_initial)
        parent(fields.psi)[:, 1, 1] .= ψ_initial

        QGYBJplus.compute_q_from_psi!(
            fields.q, fields.psi, runtime.geometry,
            runtime.coefficients.a_ell, grid.dz;
            workspace=runtime.workspace,
        )
        @test abs(sum(@view parent(fields.q)[:, 1, 1])) < 2e-13

        fill!(parent(fields.psi), 0)
        invert_q_to_psi!(model)
        @test parent(fields.psi)[:, 1, 1] ≈ ψ_initial rtol=2e-13 atol=2e-13
    finally
        finalize_model!(model)
    end
end

@testset "Public QG inversion does not apply the nonlinear stage cutoff" begin
    grid = RectilinearGrid(size=(8, 8, 1), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        topology=(1, 1),
        verbose=false,
    )

    try
        fields = model.fields
        runtime = model.runtime
        @test !is_dealiased(4, 1, grid) # kx=3 is resolved but outside N/3.
        parent(fields.psi)[1, 4, 1] = 0.7 - 0.2im
        parent(fields.psi)[1, 6, 1] = 0.7 + 0.2im
        ψ_initial = copy(parent(fields.psi))
        QGYBJplus.compute_q_from_psi!(
            fields.q, fields.psi, runtime.geometry,
            runtime.coefficients.a_ell, grid.dz;
            workspace=runtime.workspace,
        )

        fill!(parent(fields.psi), 0)
        invert_q_to_psi!(model)
        @test parent(fields.psi) ≈ ψ_initial rtol=2e-13 atol=2e-13
    finally
        finalize_model!(model)
    end
end

@testset "Variable-stratification PV stencil is second order" begin
    function manufactured_pv_error(nz)
        grid = RectilinearGrid(
            size=(4, 4, nz),
            extent=(2π, 2π, 1.0),
        )
        # Choose N² so that a=f²/N²=2+z has a nonzero vertical gradient.
        N²_function = z -> inv(2 + z)
        model = QGYBJModel(
            grid=grid,
            coriolis=FPlane(f=1.0),
            stratification=AnalyticalProfile(N²_function; returns=:N²),
            closure=HorizontalHyperdiffusivity(
                flow=FlowHyperdiffusivity(coefficient=0),
                wave=WaveHyperdiffusivity(coefficient=0),
            ),
            topology=(1, 1),
            verbose=false,
        )

        try
            fields = model.fields
            runtime = model.runtime
            z = grid.z
            φ = @. cos(π * (z + 1))
            parent(fields.psi)[:, 2, 1] .= φ
            QGYBJplus.compute_q_from_psi!(
                fields.q, fields.psi, runtime.geometry,
                runtime.coefficients.a_ell, grid.dz;
                workspace=runtime.workspace,
            )

            # For kx=1, q=-φ + ∂z[(2+z)φz].
            exact = @. -φ - π * sin(π * (z + 1)) -
                       (2 + z) * π^2 * φ
            return maximum(abs,
                real.(@view(parent(fields.q)[:, 2, 1])) .- exact)
        finally
            finalize_model!(model)
        end
    end

    errors = manufactured_pv_error.((16, 32, 64))
    @test all(>(0), errors)
    @test 3.5 < errors[1] / errors[2] < 4.5
    @test 3.5 < errors[2] / errors[3] < 4.5
end

@testset "Model-level energy diagnostics" begin
    # These two entry points are the only callers of the MPI-aware
    # *_global energy wrappers, so without them the wrappers look dead.
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        topology=(1, 1),
        verbose=false,
    )

    try
        set!(model;
             ψ=(x, y, z) -> sinpi(x / π) * cospi(y / π),
             waves=SurfaceWave(amplitude=0.1, scale=0.2),
             verbose=false)

        kinetic_energy = flow_kinetic_energy(model)
        @test kinetic_energy isa Real
        @test isfinite(kinetic_energy)
        @test kinetic_energy > 0

        envelope_energy, amplitude_energy = wave_energy(model)
        @test isfinite(envelope_energy) && envelope_energy > 0
        @test isfinite(amplitude_energy) && amplitude_energy > 0

        # A quiescent model carries no balanced kinetic energy.
        fill!(parent(model.fields.q), 0)
        fill!(parent(model.fields.psi), 0)
        @test flow_kinetic_energy(model) ≈ 0 atol=1e-20
    finally
        finalize_model!(model)
    end
end

@testset "Vertical Stokes drift differentiates A_z once" begin
    grid = RectilinearGrid(
        size=(8, 4, 5),
        extent=(2π, 2π, 1.0),
    )
    model = QGYBJModel(
        grid=grid,
        coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0),
        closure=HorizontalHyperdiffusivity(
            flow=FlowHyperdiffusivity(coefficient=0),
            wave=WaveHyperdiffusivity(coefficient=0),
        ),
        topology=(1, 1),
        verbose=false,
    )

    try
        fields = model.fields
        runtime = model.runtime
        physical_Az = QGYBJplus.allocate_fft_backward_dst(fields.C, runtime)
        Az = parent(physical_Az)

        A_mode_one = (grid.z .+ 1.5) .^ 2
        A_mode_two = (grid.z .+ 1.25) .^ 3
        p = zeros(length(grid.z))
        q = zeros(length(grid.z))
        for k in 1:(length(grid.z) - 1)
            p[k] = (A_mode_one[k + 1] - A_mode_one[k]) / grid.dz
            q[k] = (A_mode_two[k + 1] - A_mode_two[k]) / grid.dz
        end
        @test p[end] == q[end] == 0
        @inbounds for k in axes(Az, 1), i in axes(Az, 2), j in axes(Az, 3)
            x = grid.x[i]
            Az[k, i, j] = p[k] * cis(x) + im * q[k] * cis(2x)
        end
        QGYBJplus.fft_forward!(fields.C, physical_Az, runtime.plans)

        fill!(parent(fields.A), 0)
        fill!(parent(fields.B), 0)
        fill!(parent(fields.u), 0)
        fill!(parent(fields.v), 0)
        fill!(parent(fields.w), 0)

        compute_wave_velocities!(
            fields,
            runtime.geometry;
            plans=runtime.plans,
            f=1.0,
            N2_profile=ones(length(grid.z)),
            compute_w=true,
            include_wave_velocity=false,
            workspace=runtime.workspace,
        )

        dp = similar(p)
        dq = similar(q)
        # C[k] = A_z is stored at the upper face of cell k, with the top
        # Neumann value in C[end] and an implicit zero bottom-face value.
        dp[1] = p[1] / grid.dz
        dq[1] = q[1] / grid.dz
        for k in 2:(length(p) - 1)
            dp[k] = (p[k] - p[k - 1]) / grid.dz
            dq[k] = (q[k] - q[k - 1]) / grid.dz
        end
        dp[end] = -p[end - 1] / grid.dz
        dq[end] = -q[end - 1] / grid.dz

        expected_w = similar(parent(fields.w))
        @inbounds for k in axes(expected_w, 1), i in axes(expected_w, 2), j in axes(expected_w, 3)
            expected_w[k, i, j] = (dp[k] * q[k] + 0.5p[k] * dq[k]) * cos(grid.x[i])
        end

        @test parent(fields.w) ≈ expected_w rtol=2e-12 atol=2e-12
    finally
        finalize_model!(model)
    end
end
