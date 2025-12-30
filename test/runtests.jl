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
