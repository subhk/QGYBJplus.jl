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
        @test !all(==(first(coefficients.N²)), coefficients.N²)
        @test coefficients.a_ell ≈ model.physics.coriolis.f^2 ./ coefficients.N²

        fields.q[4, 2, 3] = 1.0 - 0.25im
        q_reference = copy_fields(fields)
        invert_q_to_psi!(model)
        invert_q_to_psi!(
            q_reference,
            runtime.geometry;
            a=coefficients.a_ell,
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
        compute_ybj_vertical_velocity!(model; skip_inversion=true, t=0.0)
        compute_ybj_vertical_velocity!(
            ybj_reference,
            runtime.geometry,
            runtime.plans;
            f=model.physics.coriolis.f,
            N2_profile=coefficients.N²,
            workspace=runtime.workspace,
            skip_inversion=true,
            t=0.0,
        )
        @test parent(fields.w) ≈ parent(ybj_reference.w)
        @test all(isfinite, parent(fields.w))
    finally
        finalize_model!(model)
    end
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
