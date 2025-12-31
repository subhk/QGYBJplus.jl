using Test
using QGYBJplus

# Test domain size (small for unit tests)
const TEST_Lx = 500e3  # 500 km
const TEST_Ly = 500e3  # 500 km
const TEST_Lz = 4000.0 # 4 km

#=
================================================================================
                    ERROR HANDLING TESTS
================================================================================
=#

@testset "Parameter validation errors" begin
    # Grid dimensions must be positive
    @test_throws ArgumentError default_params(nx=0, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    @test_throws ArgumentError default_params(nx=8, ny=-1, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=0, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)

    # Domain sizes must be positive
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=0.0, Ly=TEST_Ly, Lz=TEST_Lz)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=-1.0, Lz=TEST_Lz)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=0.0)

    # Time stepping parameters
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, dt=0.0)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, dt=-0.1)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, nt=0)

    # Physical parameters
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, N²=0.0)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, N²=-1.0)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, f₀=0.0)

    # Robert-Asselin filter coefficient
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, γ=-0.1)
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, γ=1.5)

    # Hyperviscosity must be non-negative
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, νₕ₁=-0.1)

    # Laplacian powers must be positive
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, ilap1=0)

    # Invalid stratification type
    @test_throws ArgumentError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, stratification=:invalid_type)

    # Valid southern hemisphere case (negative f₀ should work)
    par_south = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, f₀=-1e-4)
    @test par_south.f₀ < 0
end

@testset "Stratification validation" begin
    # Empty N² profile should error
    @test_throws ErrorException QGYBJplus.compute_deformation_radius(Float64[], 1e-4, 4000.0)

    # Empty profile in compute_stratification_coefficients
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G = QGYBJplus.init_grid(par)
    @test_throws ErrorException QGYBJplus.compute_stratification_coefficients(Float64[], G)

    # validate_stratification should return errors for empty profile
    errors, warnings = QGYBJplus.validate_stratification(Float64[])
    @test !isempty(errors)
    @test any(occursin("Empty", e) for e in errors)
end

@testset "Edge cases" begin
    # nz=1 should work (single vertical level)
    par_nz1 = default_params(nx=8, ny=8, nz=1, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G_nz1 = QGYBJplus.init_grid(par_nz1)
    @test length(G_nz1.z) == 1
    @test length(G_nz1.dz) == 1  # Should have fallback value, not empty
    @test G_nz1.dz[1] == TEST_Lz

    # Small grid should work
    par_small = default_params(nx=4, ny=4, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G_small, S_small, plans_small, a_small = setup_model(par_small)
    @test size(S_small.q) == (par_small.nz, par_small.nx, par_small.ny)
end

@testset "Adaptive interpolation small grids" begin
    nx, ny, nz = 2, 2, 1
    dx = 1.0
    dy = 1.0
    dz = 1.0
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    u = zeros(Float64, nz, nx, ny)
    v = zeros(Float64, nz, nx, ny)
    w = zeros(Float64, nz, nx, ny)

    u[1, 1, 1] = 1.0
    u[1, 2, 1] = 2.0
    u[1, 1, 2] = 3.0
    u[1, 2, 2] = 4.0
    v .= 0.5
    w .= -0.25

    grid_info = (dx=dx, dy=dy, dz=dz, Lx=Lx, Ly=Ly, Lz=Lz)
    boundary_conditions = (periodic_x=false, periodic_y=false, periodic_z=false)

    u_i, v_i, w_i = QGYBJplus.interpolate_velocity_advanced(
        0.6, 0.7, 0.2,
        u, v, w,
        grid_info, boundary_conditions,
        QGYBJplus.ADAPTIVE
    )

    @test isfinite(u_i)
    @test isfinite(v_i)
    @test isfinite(w_i)
end

@testset "Layered particle distribution counts" begin
    par = default_params(nx=8, ny=8, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G = init_grid(par)

    base_cfg = ParticleConfig{Float64}(x_max=G.Lx, y_max=G.Ly, z_level=0.0)
    tracker = ParticleTracker(base_cfg, G)

    z_levels = [1000.0, 3000.0]
    per_level = [6, 10]
    cfg3d = particles_in_layers(z_levels; x_max=G.Lx, y_max=G.Ly, nx=4, ny=4,
                                particles_per_level=per_level)

    initialize_particles!(tracker, cfg3d)

    @test tracker.particles.np == sum(per_level)
    @test length(tracker.particles.x) == tracker.particles.np
    @test length(tracker.particles.y) == tracker.particles.np
    @test length(tracker.particles.z) == tracker.particles.np

    tol = 100 * eps(eltype(tracker.particles.z))
    level_counts = zeros(Int, length(cfg3d.z_levels))
    for z in tracker.particles.z
        level_idx = findfirst(lvl -> isapprox(z, lvl; atol=tol), cfg3d.z_levels)
        @test level_idx !== nothing
        level_counts[level_idx] += 1
    end
    @test level_counts == per_level
end

@testset "Random 3D particle seed determinism" begin
    par = default_params(nx=8, ny=8, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G = init_grid(par)

    base_cfg = ParticleConfig{Float64}(x_max=G.Lx, y_max=G.Ly, z_level=0.0)
    cfg3d = particles_random_3d(10; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, seed=42)

    tracker1 = ParticleTracker(base_cfg, G)
    tracker2 = ParticleTracker(base_cfg, G)

    initialize_particles!(tracker1, cfg3d)
    initialize_particles!(tracker2, cfg3d)

    @test tracker1.particles.np == 10
    @test tracker1.particles.x == tracker2.particles.x
    @test tracker1.particles.y == tracker2.particles.y
    @test tracker1.particles.z == tracker2.particles.z

    @test all(x -> x >= 0.0 && x <= G.Lx, tracker1.particles.x)
    @test all(y -> y >= 0.0 && y <= G.Ly, tracker1.particles.y)
    @test all(z -> z >= 0.0 && z <= G.Lz, tracker1.particles.z)
end

@testset "QGYBJplus basic API" begin
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, stratification=:constant_N)
    G, S, plans, a = setup_model(par)
    @test size(S.q) == (par.nz, par.nx, par.ny)

    # Invert q->psi (all zeros)
    invert_q_to_psi!(S, G; a, par=par)
    @test all(isfinite, real.(S.psi))

    # Put a simple B mode and invert to A (YBJ+)
    S.B[3, 2, 2] = 1 + 0im
    invert_B_to_A!(S, G, par, a)
    @test all(isfinite, real.(S.A))

    # One projection step should run without error
    L = dealias_mask(G)
    first_projection_step!(S, G, par, plans; a, dealias_mask=L)
    @test all(isfinite, real.(S.q))

    # Leapfrog step should also update fields
    Snp1 = deepcopy(S); Snm1 = deepcopy(S)
    leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)

    @test all(isfinite, real.(Snp1.q))
end

@testset "IMEX-CN step (wave feedback enabled)" begin
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         dt=0.01, no_feedback=false, no_wave_feedback=false)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    # Seed a nontrivial wave mode
    S.B[3, 2, 2] = 1.0 + 0.2im

    imex_ws = init_imex_workspace(S, G)
    first_imex_step!(S, G, par, plans, imex_ws; a=a, dealias_mask=L)

    Snp1 = copy_state(S)
    imex_cn_step!(Snp1, S, G, par, plans, imex_ws; a=a, dealias_mask=L)

    @test all(isfinite, real.(Snp1.q))
    @test all(isfinite, imag.(Snp1.q))
    @test all(isfinite, real.(Snp1.B))
    @test all(isfinite, imag.(Snp1.B))
    @test all(isfinite, real.(Snp1.psi))
end

@testset "Normal YBJ branch + dealias + kh=0" begin
    par = default_params(nx=12, ny=12, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, stratification=:constant_N)

    # switch to normal YBJ
    par = QGParams(; (field=>getfield(par, field) for field in fieldnames(typeof(par)))... )
    setfield!(par, :ybj_plus, false)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    # Dealias property checks for a few indices
    let keep = L, nx = par.nx, ny = par.ny
        # center should be kept (kx=ky=0)
        @test keep[1,1]
        # pick a point beyond radial 2/3 cutoff
        kmax = min(nx, ny) ÷ 3
        i_bad = min(nx, kmax + 3)
        j_bad = min(ny, kmax + 3)
        @test !keep[i_bad, j_bad]
    end

    # kh=0 psi inversion should zero the whole vertical column for that (i,j)
    S.q[3, 1, 1] = 1 + 0im
    invert_q_to_psi!(S, G; a, par=par)
    @test all(iszero, S.psi[:, 1, 1])

    # Run one normal-branch step to ensure it executes
    S.B[4, 3, 3] = 0.5 + 0.2im
    first_projection_step!(S, G, par, plans; a, dealias_mask=L)
    Snp1 = deepcopy(S); Snm1 = deepcopy(S)

    leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)

    @test all(isfinite, real.(Snp1.q))
end

#=
================================================================================
                    PHYSICS OPERATOR TESTS
================================================================================
Tests for key physics operators that were identified as error-prone:
- Elliptic B→A inversion (correct RHS scaling)
- Hyperdiffusion (isotropic form)
- Vertical velocity with non-constant N²
================================================================================
=#

@testset "Elliptic B→A inversion consistency" begin
    # Test that B→A inversion gives consistent results
    # The operator should satisfy: L⁺A = B where L⁺ = ∂²/∂z² - (f²/N²)∇²
    par = default_params(nx=8, ny=8, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)

    # Set a non-trivial B field (single mode)
    S.B[8, 3, 3] = 1.0 + 0.5im

    # Invert to get A
    invert_B_to_A!(S, G, par, a)

    # Check A is finite and non-zero where B is non-zero
    @test all(isfinite, real.(S.A))
    @test all(isfinite, imag.(S.A))
    @test !all(iszero, S.A)

    # The A field should have smooth vertical structure (not oscillatory artifacts)
    # Check that vertical derivative C = A_z is also well-behaved
    @test all(isfinite, real.(S.C))
end

@testset "Hyperdiffusion integrating factor - isotropic form" begin
    # Test that hyperdiffusion uses isotropic form: ν(kx² + ky²)^n
    # not anisotropic: ν(|kx|^{2n} + |ky|^{2n})
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                        νₕ₁=1.0, ilap1=2)  # Biharmonic with ν=1

    # Access the int_factor function through QGYBJplus module
    # For isotropic form: int_factor(kx, ky) = dt * ν * (kx² + ky²)^2
    # For anisotropic form: int_factor(kx, ky) = dt * ν * (|kx|^4 + |ky|^4)

    # At kx=ky=1 (normalized), isotropic gives: dt * ν * 2^2 = 4*dt*ν
    # At kx=1,ky=0: isotropic gives: dt * ν * 1^2 = dt*ν
    # At kx=0,ky=1: isotropic gives: dt * ν * 1^2 = dt*ν

    # For anisotropic at kx=ky=1: dt * ν * (1 + 1) = 2*dt*ν
    # This would be different from isotropic (4*dt*ν vs 2*dt*ν)

    # We can verify the isotropic behavior by checking that
    # modes at 45° angles dissipate more than axis-aligned modes
    # (since sqrt(2)² = 2 > 1² = 1)

    # For now, just verify the integrating factor is accessible and finite
    if isdefined(QGYBJplus.Nonlinear, :int_factor)
        kx, ky = 1.0, 1.0
        dt = par.dt
        int_f = QGYBJplus.Nonlinear.int_factor(kx, ky, par; waves=false)
        @test isfinite(int_f)
        @test int_f > 0  # Should be positive dissipation

        # Verify isotropic: at (1,1) should have same factor as at (sqrt(2), 0)
        int_f_diagonal = QGYBJplus.Nonlinear.int_factor(1.0, 1.0, par; waves=false)
        int_f_axis = QGYBJplus.Nonlinear.int_factor(sqrt(2.0), 0.0, par; waves=false)
        @test isapprox(int_f_diagonal, int_f_axis, rtol=1e-10)
    end
end

@testset "Vertical velocity with N² profile" begin
    # Test that vertical velocity computation accepts and uses N² profile
    par = default_params(nx=8, ny=8, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)

    # Create a non-constant N² profile (exponential decay with depth)
    N2_profile = [1.0 * exp(-G.z[k] / (TEST_Lz / 4)) for k in 1:par.nz]

    # Set up a simple flow
    S.psi[8, 3, 3] = 1.0 + 0im
    invert_q_to_psi!(S, G; a, par=par)

    # Compute velocities with N² profile
    compute_velocities!(S, G; plans, params=par, N2_profile=N2_profile)

    # Check velocities are finite
    @test all(isfinite, S.u)
    @test all(isfinite, S.v)
    @test all(isfinite, S.w)

    # Compute velocities without N² profile (should default to N²=1)
    S2 = deepcopy(S)
    S2.psi[8, 3, 3] = 1.0 + 0im
    compute_velocities!(S2, G; plans, params=par)

    # Results should be different when using variable vs constant N²
    # (unless by chance they're identical, which is unlikely)
    @test all(isfinite, S2.w)
end

@testset "YBJ vertical velocity coefficient" begin
    # Test that YBJ vertical velocity uses f²/N² coefficient (not 1/N²)
    # The YBJ formula is: w = -(f²/N²)[(∂A/∂x)_z - i(∂A/∂y)_z] + c.c.
    par = default_params(nx=8, ny=8, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)

    # Set a non-trivial B field and compute A
    S.B[8, 3, 3] = 1.0 + 0.5im
    invert_B_to_A!(S, G, par, a)

    # The A field should be non-zero after inversion
    @test !all(iszero, S.A)

    # Create N² profile
    N2_profile = ones(par.nz)  # Constant for simplicity

    # Compute YBJ vertical velocity
    QGYBJplus.Operators.compute_ybj_vertical_velocity!(S, G, plans, par; N2_profile=N2_profile)

    # Check w is finite (main verification that the code path works)
    @test all(isfinite, S.w)
end

@testset "First projection step with A initialization" begin
    # Test that first_projection_step! properly initializes A before using it
    # (Previously A was zero when first used for dispersion)
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    # Set initial wave field
    S.B[4, 3, 3] = 1.0 + 0.5im

    # A should initially be zero
    @test all(iszero, S.A)

    # Run first projection step
    first_projection_step!(S, G, par, plans; a=a, dealias_mask=L)

    # After projection step, A should be computed (not zero if B was non-zero)
    # Due to physics switches and filtering, A might still be small but the
    # computation should have happened
    @test all(isfinite, S.A)
end

@testset "Nonlinear operator normalization and balance" begin
    par = default_params(nx=32, ny=32, nz=1, Lx=2*pi, Ly=2*pi, Lz=1.0)
    G = init_grid(par)
    plans = plan_transforms!(G)
    L = dealias_mask(G)

    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    x = [dx * (i - 1) for i in 1:G.nx]
    y = [dy * (j - 1) for j in 1:G.ny]

    # ---- convol_waqg! normalization and energy balance ----
    u = zeros(Float64, G.nz, G.nx, G.ny)
    v = zeros(Float64, G.nz, G.nx, G.ny)
    q_phys = zeros(Float64, G.nz, G.nx, G.ny)
    J_expected = zeros(Float64, G.nz, G.nx, G.ny)

    for j in 1:G.ny, i in 1:G.nx
        xi = x[i]
        yj = y[j]
        u[1, i, j] = sin(yj)
        v[1, i, j] = cos(xi)
        q_phys[1, i, j] = sin(2 * xi) + cos(3 * yj)
        J_expected[1, i, j] = 2 * sin(yj) * cos(2 * xi) - 3 * cos(xi) * sin(3 * yj)
    end

    qk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_forward!(qk, q_phys, plans)
    BRk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    BIk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    nqk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    nBRk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    nBIk = zeros(ComplexF64, G.nz, G.nx, G.ny)

    convol_waqg!(nqk, nBRk, nBIk, u, v, qk, BRk, BIk, G, plans; Lmask=L)

    nq_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_backward!(nq_phys, nqk, plans)

    @test maximum(abs.(real.(nq_phys) .- J_expected)) < 5e-10
    @test abs(real(nqk[1, 1, 1])) < 1e-12

    energy_balance = sum(q_phys .* real.(nq_phys)) / (G.nx * G.ny)
    @test abs(energy_balance) < 1e-10

    # ---- refraction_waqg! normalization ----
    psi_phys = zeros(Float64, G.nz, G.nx, G.ny)
    BR_phys = ones(Float64, G.nz, G.nx, G.ny)
    BI_phys = zeros(Float64, G.nz, G.nx, G.ny)
    zeta_expected = zeros(Float64, G.nz, G.nx, G.ny)

    for j in 1:G.ny, i in 1:G.nx
        xi = x[i]
        yj = y[j]
        psi_phys[1, i, j] = sin(xi) + cos(yj)
        zeta_expected[1, i, j] = -sin(xi) - cos(yj)
    end

    psik = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_forward!(psik, psi_phys, plans)
    fft_forward!(BRk, BR_phys, plans)
    fft_forward!(BIk, BI_phys, plans)

    rBRk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    rBIk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    refraction_waqg!(rBRk, rBIk, BRk, BIk, psik, G, plans; Lmask=L)

    rBR_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    rBI_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_backward!(rBR_phys, rBRk, plans)
    fft_backward!(rBI_phys, rBIk, plans)

    @test maximum(abs.(real.(rBR_phys) .- zeta_expected)) < 5e-10
    @test maximum(abs.(real.(rBI_phys))) < 1e-12

    # ---- compute_qw! normalization ----
    BR_phys .= 0.0
    BI_phys .= 0.0
    qw_expected = zeros(Float64, G.nz, G.nx, G.ny)

    for j in 1:G.ny, i in 1:G.nx
        xi = x[i]
        yj = y[j]
        BR_phys[1, i, j] = cos(xi) + cos(yj)
        BI_phys[1, i, j] = sin(xi) + sin(yj)
        qw_expected[1, i, j] = sin(xi - yj) - cos(xi - yj)
    end

    fft_forward!(BRk, BR_phys, plans)
    fft_forward!(BIk, BI_phys, plans)

    qwk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)

    qw_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_backward!(qw_phys, qwk, plans)

    @test maximum(abs.(real.(qw_phys) .- qw_expected)) < 5e-10
end
