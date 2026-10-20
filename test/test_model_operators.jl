using Test
using QGYBJplus

@testset "Variable-N² coefficients and elliptic round trips" begin
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
        verbose=false,
    )

    try
        fields = model.fields
        runtime = model.runtime
        coefficients = runtime.coefficients

        @test coefficients.N² ≈ N2_function.(grid.z)
        @test coefficients.N²_face ≈ N2_function.(grid.z_faces[2:end])
        @test coefficients.a_ell ≈
              model.physics.coriolis.f^2 ./ coefficients.N²_face

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

# Wagner & Young (2016) Stokes drift for A = F(x, y) G(z) with a = f²/N² = 1 + z/2
# and G = cos(πz), so M = a A_z = g₁F with g₁ = aG', M_z = g₂F, M_zz = g₃F.
stokes_g₁(z) = -π * (1 + z / 2) * sin(π * z)
stokes_g₂(z) = -(π / 2) * sin(π * z) - π^2 * (1 + z / 2) * cos(π * z)
stokes_g₃(z) = -π^2 * cos(π * z) + π^3 * (1 + z / 2) * sin(π * z)
const STOKES_MODES = ((1, 0, 1.0 + 0.0im), (1, 2, 0.6im), (0, -2, 0.3 + 0.2im))
stokes_F(x, y, factor) = sum(c * factor(m, n) * cis(m * x + n * y) for (m, n, c) in STOKES_MODES)

"""Analytic (u^S, v^S, w^S) with ∂_s = (∂ₓ - i∂ᵧ)/2 and f = 1."""
function analytic_stokes_drift(x, y, z)
    F = stokes_F(x, y, (m, n) -> 1)
    F_s = stokes_F(x, y, (m, n) -> 0.5 * (im * m + n))
    F_sc = stokes_F(x, y, (m, n) -> 0.5 * (im * m - n))
    F_ssc = stokes_F(x, y, (m, n) -> -0.25 * (m^2 + n^2))
    g₁, g₂, g₃ = stokes_g₁(z), stokes_g₂(z), stokes_g₃(z)
    J₀ = g₂^2 * conj(F) * F_sc - g₁ * g₃ * conj(F_s) * F
    K₀ = g₁ * g₂ * (conj(F) * F_ssc - abs2(F_s))
    return imag(J₀), -real(J₀), -2imag(K₀)
end

"""Relative errors of the model's Stokes drift on an `nz`-level grid."""
function stokes_drift_errors(nz; nh=16)
    grid = RectilinearGrid(size=(nh, nh, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> 1 / (1 + z / 2); returns=:N²),
        formulation=YBJPlus(), verbose=false)
    try
        fields, runtime = model.fields, model.runtime
        A = ComplexF64[stokes_F(grid.x[i], grid.y[j], (m, n) -> 1) * cos(π * grid.z[k])
                       for k in 1:nz, i in 1:nh, j in 1:nh]
        Ak = fft_forward!(similar(fields.A), A, runtime.plans)
        # B = L⁺A with the model's flux-form operator, inverted back to A and A_z.
        a = runtime.coefficients.a_ell
        for j in 1:nh, i in 1:nh
            kh² = grid.kx[i]^2 + grid.ky[j]^2
            for k in 1:nz
                up = k < nz ? a[k] * (Ak[k + 1, i, j] - Ak[k, i, j]) : 0
                down = k > 1 ? a[k - 1] * (Ak[k, i, j] - Ak[k - 1, i, j]) : 0
                fields.B[k, i, j] = (up - down) / grid.dz^2 - kh² / 4 * Ak[k, i, j]
            end
        end
        invert_B_to_A!(model)
        # From a zero Lagrangian mean, the Eulerian mean is minus the drift.
        fill!(fields.u, 0); fill!(fields.v, 0); fill!(fields.w, 0)
        compute_wave_velocities!(model; compute_w=true, include_wave_velocity=false)

        exact = [analytic_stokes_drift(grid.x[i], grid.y[j], grid.z[k])
                 for k in 1:nz, i in 1:nh, j in 1:nh]
        return map(enumerate((fields.u, fields.v, fields.w))) do (c, component)
            expected = getindex.(exact, c)
            maximum(abs, component .+ expected) / maximum(abs, expected)
        end
    finally
        finalize_model!(model)
    end
end

@testset "Stokes drift converges at second order to Wagner & Young (2016)" begin
    coarse, fine = stokes_drift_errors(16), stokes_drift_errors(32)
    # Each component, including w^S from K₀, is centred on the levels: a
    # half-cell offset between face and centre quantities would make these
    # ratios approach 2.
    for c in 1:3
        @test fine[c] < 2e-2
        @test 3.3 < coarse[c] / fine[c] < 4.7
    end
end

"""
The spectral wave velocity LA of `model`, by the formulation's relation:
`B = LA` for YBJ, `B = L⁺A = LA - kₕ²A/4` for YBJ⁺.
"""
function expected_wave_velocity(model)
    fields = model.fields
    model.physics.formulation isa YBJ && return copy(fields.B)
    grid = model.grid
    kh² = [grid.kx[i]^2 + grid.ky[j]^2 for k in 1:1, i in 1:grid.size[1], j in 1:grid.size[2]]
    return fields.B .+ kh² .* fields.A ./ 4
end

@testset "Wave velocity follows B for each formulation" begin
    grid = RectilinearGrid(size=(8, 8, 6), extent=(2π, 2π, 1.0))
    for formulation in (YBJPlus(), YBJ())
        model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
            stratification=ConstantStratification(N²=1.0), formulation=formulation,
            verbose=false)
        try
            fields = model.fields
            B = ComplexF64[(1 + 0.2grid.z[k]) * (cos(grid.x[i]) + 0.5im * sin(grid.y[j]))
                           for k in 1:6, i in 1:8, j in 1:8]
            set!(model; B=FieldArray(B), verbose=false)
            invert_B_to_A!(model)   # any A ≠ 0 tells the two relations apart
            # u + iv = e^{-ift} LA: LA = B for YBJ and B + kₕ²A/4 for YBJ⁺,
            # written out here rather than taken from the model's own helper.
            LA = expected_wave_velocity(model)
            LA_physical = fft_backward!(similar(LA), LA, model.runtime.plans)
            t = 0.7
            fill!(fields.u, 0); fill!(fields.v, 0)
            compute_wave_velocities!(model; t, compute_w=false, include_wave_velocity=true)
            with_waves_u, with_waves_v = copy(fields.u), copy(fields.v)
            fill!(fields.u, 0); fill!(fields.v, 0)
            compute_wave_velocities!(model; compute_w=false, include_wave_velocity=false)
            orbital = cis(-t) .* LA_physical
            @test with_waves_u .- fields.u ≈ real.(orbital) atol=1e-12
            @test with_waves_v .- fields.v ≈ imag.(orbital) atol=1e-12
        finally
            finalize_model!(model)
        end
    end
end

"""Apply the `invert_helmholtz!` stencil to `φ`, returning the matching rhs."""
function helmholtz_rhs(φ, grid, a, b, α, bot_bc, top_bc)
    nz, nx, ny = size(φ)
    Δz = grid.dz
    rhs = similar(φ)
    for j in 1:ny, i in 1:nx
        shift = α * (grid.kx[i]^2 + grid.ky[j]^2) * Δz^2
        for k in 1:nz
            if k == 1
                row = (-a[1] - 0.5b[1] * Δz - shift) * φ[1, i, j] +
                      (a[1] + 0.5b[1] * Δz) * φ[2, i, j] -
                      (a[1] - 0.5b[1] * Δz) * Δz * bot_bc[i, j]
            elseif k == nz
                row = (a[nz] - 0.5b[nz] * Δz) * φ[nz - 1, i, j] +
                      (-a[nz] + 0.5b[nz] * Δz - shift) * φ[nz, i, j] +
                      (a[nz] + 0.5b[nz] * Δz) * Δz * top_bc[i, j]
            else
                row = (a[k] - 0.5b[k] * Δz) * φ[k - 1, i, j] +
                      (-2a[k] - shift) * φ[k, i, j] +
                      (a[k] + 0.5b[k] * Δz) * φ[k + 1, i, j]
            end
            rhs[k, i, j] = row / Δz^2
        end
    end
    return rhs
end

remove_column_means(φ) = φ .- sum(φ; dims=1) ./ size(φ, 1)

@testset "General Helmholtz inversion" begin
    grid = RectilinearGrid(size=(8, 6, 8), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), verbose=false)
    try
        G = model.runtime.geometry
        nz, nx, ny = grid.size[3], grid.size[1], grid.size[2]
        a = 1.0 .+ 0.3cos.(2π .* grid.z)
        b = 0.4sin.(2π .* grid.z)
        bot_bc = ComplexF64[0.2cis(i + 2j) for i in 1:nx, j in 1:ny]
        top_bc = ComplexF64[0.1cis(2i - j) for i in 1:nx, j in 1:ny]
        φ = ComplexF64[cis(0.7k + 1.3i - 0.4j) * (1 + 0.1k)
                       for k in 1:nz, i in 1:nx, j in 1:ny]
        # The kₕ = 0 column is defined up to a constant.
        φ[:, 1, 1] .-= sum(φ[:, 1, 1]) / nz

        for α in (2.5, 1)
            rhs = helmholtz_rhs(φ, grid, a, b, α, bot_bc, top_bc)
            solution = similar(rhs)
            invert_helmholtz!(solution, rhs, G; a, b, scale_kh2=α,
                              bot_bc, top_bc, workspace=model.runtime.workspace)
            @test solution ≈ φ rtol=1e-12
            # In place, without boundary fluxes.
            zero_bc = zeros(ComplexF64, nx, ny)
            in_place = helmholtz_rhs(φ, grid, a, b, α, zero_bc, zero_bc)
            invert_helmholtz!(in_place, in_place, G; a, b, scale_kh2=α)
            @test in_place ≈ φ rtol=1e-12
        end

        # With scale_kh2 = 0 every column is singular: each solution is the
        # zero-mean member of its null-space family.
        rhs = helmholtz_rhs(φ, grid, a, b, 0, bot_bc, top_bc)
        solution = similar(rhs)
        invert_helmholtz!(solution, rhs, G; a, b, scale_kh2=0, bot_bc, top_bc)
        @test all(isfinite, solution)
        @test solution ≈ remove_column_means(φ) rtol=1e-12
    finally
        finalize_model!(model)
    end

    # One level on a dimensional grid: kₕ² ~ 1e-11 m⁻² is small but valid.
    flat = RectilinearGrid(size=(8, 8, 1), extent=(1e6, 1e6, 100.0))
    flat_model = QGYBJModel(grid=flat, coriolis=FPlane(f=1e-4),
        stratification=ConstantStratification(N²=1e-5), verbose=false)
    try
        G = flat_model.runtime.geometry
        kh² = [flat.kx[i]^2 + flat.ky[j]^2 for k in 1:1, i in 1:8, j in 1:8]
        φ = ComplexF64[cis(i - 2j) for k in 1:1, i in 1:8, j in 1:8]
        φ[1, 1, 1] = 0
        rhs = -kh² .* φ
        solution = similar(rhs)
        invert_helmholtz!(solution, rhs, G; a=ones(1))
        @test solution ≈ φ rtol=1e-12
        rhs[1, 1, 1] = 1
        @test_logs (:warn, r"scale_kh2\*kₕ² = 0") invert_helmholtz!(
            solution, rhs, G; a=ones(1))
        @test solution[1, 1, 1] == 0
    finally
        finalize_model!(flat_model)
    end
end

"""Average values on the faces above each level onto cell centres (w = 0 below)."""
face_average(w_face) =
    (cat(zero(w_face[1:1, :, :]), w_face[1:end-1, :, :]; dims=1) .+ w_face) ./ 2

"""Invert centre values of w back to face values, integrating up from the bottom."""
function faces_from_centres(w)
    faces = similar(w)
    below = zero(w[1, :, :])
    for k in axes(w, 1)
        faces[k, :, :] = 2 .* w[k, :, :] .- below
        below = faces[k, :, :]
    end
    return faces
end

@testset "Omega-equation w satisfies the model's vorticity and buoyancy equations" begin
    nx, ny, nz = 24, 24, 10
    grid = RectilinearGrid(size=(nx, ny, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.3),
        stratification=AnalyticalProfile(z -> 1.0 + 0.5cos(3z); returns=:N²),
        verbose=false)
    try
        # Modes with different vertical structures, so the deformation part
        # of the Q-vector forcing is active.
        set!(model; ψ=(x, y, z) -> 0.3 * (sin(x) * cos(y) * (1 + 0.3z) +
                                         0.5sin(2x + y) * cos(π * z) +
                                         0.4cos(x - 2y) * z^2),
             pv_method=:qg, verbose=false)
        compute_vertical_velocity!(model)

        runtime = model.runtime
        G, plans, mask = runtime.geometry, runtime.plans, runtime.dealias_mask
        f, Δz = model.physics.coriolis.f, grid.dz
        N²_face = runtime.coefficients.N²_face
        ψ = copy(model.fields.psi)
        jacobian(a, b) = QGYBJplus.jacobian_spectral!(similar(a), a, b, G, plans;
                                                        Lmask=mask)

        # ψ_t from the model's own PV tendency q_t = -J(ψ, q).
        tendency = copy_fields(model.fields)
        tendency.q .= .-jacobian(ψ, model.fields.q)
        QGYBJplus.invert_q_to_psi!(tendency, G; a=runtime.coefficients.a_ell)
        ψ_t = tendency.psi

        # Buoyancy equation on the face above level k: with b = f δzψ and the
        # face-averaged ψ, J(ψ, b) = (f/Δz) J(ψ[k], ψ[k+1]).
        ψ_above = cat(ψ[2:end, :, :], zero(ψ[1:1, :, :]); dims=1)
        J_faces = jacobian(ψ, ψ_above)
        w_face_hat = zero(ψ)
        for k in 1:(nz - 1)
            b_t = f .* (ψ_t[k + 1, :, :] .- ψ_t[k, :, :]) ./ Δz
            w_face_hat[k, :, :] = .-(b_t .+ (f / Δz) .* J_faces[k, :, :]) ./ N²_face[k]
        end
        w_face = real.(fft_backward!(similar(w_face_hat), w_face_hat, plans))

        w = model.fields.w
        scale = maximum(abs, w_face)
        @test scale > 1e-2
        @test maximum(abs, w .- face_average(w_face)) < 1e-12 * scale
        @test maximum(abs, faces_from_centres(w)[end, :, :]) < 1e-12 * scale

        # The same w closes the vorticity equation f w_z = ζ_t + J(ψ, ζ).
        kₕ² = [grid.kx[i]^2 + grid.ky[j]^2 for k in 1:1, i in 1:nx, j in 1:ny]
        vorticity_rhs = .-kₕ² .* ψ_t .+ jacobian(ψ, .-kₕ² .* ψ)
        vorticity_rhs = real.(fft_backward!(similar(ψ), vorticity_rhs, plans))
        padded = cat(zero(w_face[1:1, :, :]), w_face; dims=1)
        f_w_z = f .* (padded[2:end, :, :] .- padded[1:end-1, :, :]) ./ Δz
        @test maximum(abs, f_w_z .- vorticity_rhs) < 1e-11 * maximum(abs, vorticity_rhs)
    finally
        finalize_model!(model)
    end
end

@testset "YBJ w satisfies discrete continuity with the wave velocity ($(nameof(typeof(formulation))))" for formulation in (YBJPlus(), YBJ())
    # For YBJ the amplitude comes from the solvability-constrained B = LA,
    # not from the YBJ⁺ inversion of B = L⁺A.
    nx, ny, nz = 16, 16, 8
    grid = RectilinearGrid(size=(nx, ny, nz), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=AnalyticalProfile(z -> 1.0 + 0.5cos(3z); returns=:N²),
        formulation=formulation, verbose=false)
    try
        wave = ComplexF64[(1 + 0.3cos(π * (grid.z[k] + 1))) *
                          (cos(grid.x[i]) + 0.6im * sin(grid.y[j]) +
                           0.4cis(grid.x[i] + grid.y[j]))
                          for k in 1:nz, i in 1:nx, j in 1:ny]
        set!(model; B=FieldArray(wave), verbose=false)
        t = 0.3
        compute_ybj_vertical_velocity!(model; t)

        # Wave velocity u + iv = M = e^{-ift} LA; differentiate u and v.
        plans = model.runtime.plans
        LA = expected_wave_velocity(model)
        M = fft_backward!(similar(LA), cis(-t) .* LA, plans)
        u_hat = fft_forward!(similar(LA), complex.(real.(M)), plans)
        v_hat = fft_forward!(similar(LA), complex.(imag.(M)), plans)
        kx = [grid.kx[i] for k in 1:1, i in 1:nx, j in 1:ny]
        ky = [grid.ky[j] for k in 1:1, i in 1:nx, j in 1:ny]
        divergence = real.(fft_backward!(similar(LA),
                                         im .* kx .* u_hat .+ im .* ky .* v_hat, plans))

        faces = faces_from_centres(model.fields.w)
        padded = cat(zero(faces[1:1, :, :]), faces; dims=1)
        w_z = (padded[2:end, :, :] .- padded[1:end-1, :, :]) ./ grid.dz
        @test maximum(abs, divergence) > 1e-2
        @test maximum(abs, w_z .+ divergence) < 1e-11 * maximum(abs, divergence)
        @test maximum(abs, faces[end, :, :]) < 1e-12 * maximum(abs, faces)
    finally
        finalize_model!(model)
    end
end

@testset "Velocity API requires the wave phase" begin
    grid = RectilinearGrid(size=(8, 8, 4), extent=(2π, 2π, 1.0))
    model = QGYBJModel(grid=grid, coriolis=FPlane(f=1.0),
        stratification=ConstantStratification(N²=1.0), verbose=false)
    try
        @test_throws ArgumentError compute_total_velocities!(model)
        @test_throws ArgumentError compute_wave_velocities!(model)
        @test compute_wave_velocities!(model; include_wave_velocity=false) === model
        @test_throws ArgumentError compute_velocities!(model; use_ybj_w=true)
        @test compute_total_velocities!(model; t=0.0) === model
    finally
        finalize_model!(model)
    end
end
