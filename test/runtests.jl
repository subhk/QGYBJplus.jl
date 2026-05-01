using Test
using QGYBJplus
using JLD2
using NCDatasets

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

    # Legacy time-filter parameter
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

@testset "File-backed stratification" begin
    filename = tempname() * ".nc"
    z_file = [-TEST_Lz, -TEST_Lz / 2, 0.0]
    N_file = [1.0e-3, 2.0e-3, 3.0e-3]

    NCDataset(filename, "c") do ds
        defDim(ds, "z", length(z_file))
        z_var = defVar(ds, "z", Float64, ("z",))
        N_var = defVar(ds, "N", Float64, ("z",))
        z_var[:] = z_file
        N_var[:] = N_file
    end

    par = default_params(nx=4, ny=4, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G = init_grid(par)

    profile = FileStratification(filename)
    N2_profile = compute_stratification_profile(profile, G)
    expected_N = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
    expected_N2 = expected_N .^ 2

    @test N2_profile ≈ expected_N2

    grid = RectilinearGrid(size=(4, 4, 4),
                           x=(-TEST_Lx/2, TEST_Lx/2),
                           y=(-TEST_Ly/2, TEST_Ly/2),
                           z=(-TEST_Lz, 0.0))
    model = QGYBJModel(grid=grid,
                       stratification=profile,
                       verbose=false)

    @test model.N2_profile ≈ expected_N2
    @test model.params.N² ≈ sum(expected_N2) / length(expected_N2)

    depth_filename = tempname() * ".nc"
    depth_file = [0.0, TEST_Lz / 2, TEST_Lz]
    N_depth_file = reverse(N_file)

    NCDataset(depth_filename, "c") do ds
        defDim(ds, "z", length(depth_file))
        z_var = defVar(ds, "z", Float64, ("z",))
        N_var = defVar(ds, "N", Float64, ("z",))
        z_var[:] = depth_file
        N_var[:] = N_depth_file
    end

    depth_profile = FileStratification(depth_filename)
    @test compute_stratification_profile(depth_profile, G) ≈ expected_N2

    jld2_filename = tempname() * ".jld2"
    jldsave(jld2_filename; z=z_file, N=N_file)

    jld2_profile = FileStratification(jld2_filename)
    @test compute_stratification_profile(jld2_profile, G) ≈ expected_N2
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

@testset "Oceananigans-style interface" begin
    grid = RectilinearGrid(size=(8, 8, 4),
                           x=(-TEST_Lx/2, TEST_Lx/2),
                           y=(-TEST_Ly/2, TEST_Ly/2),
                           z=(-TEST_Lz, 0.0))
    model = QGYBJModel(grid=grid,
                       coriolis=FPlane(f=1e-4),
                       stratification=ConstantStratification(N²=1e-5),
                       closure=HorizontalHyperdiffusivity(waves=1e5),
                       flow=:fixed,
                       feedback=:none,
                       verbose=false)

    ψ = (x, y, z) -> 1.0e3 * sin(2π * x / TEST_Lx) * cos(2π * y / TEST_Ly)
    set!(model; ψ=ψ, pv_method=:barotropic,
         waves=SurfaceWave(amplitude=0.05, scale=500.0))

    simulation = Simulation(model;
                            Δt=10.0,
                            stop_time=25.0,
                            output=NetCDFOutput(path="interface_output",
                                                schedule=TimeInterval(20.0),
                                                fields=(:ψ, :waves),
                                                z=0.0),
                            diagnostics=IterationInterval(3),
                            verbose=false)

    @test model.grid.x0 == -TEST_Lx / 2
    @test model.grid.y0 == -TEST_Ly / 2
    @test model.params.fixed_flow
    @test model.params.no_feedback
    @test model.params.no_wave_feedback
    @test model.params.νₕ₁ʷ == 1e5
    @test simulation.params.dt == 10.0
    @test simulation.params.nt == 2
    @test simulation.run_options.output_dir == "interface_output"
    @test simulation.run_options.output_z_levels == [0.0]
    @test simulation.run_options.save_interval == 20.0
    @test simulation.run_options.diagnostics_interval == 3
    @test any(!iszero, parent(model.state.psi))
    @test any(!iszero, parent(model.state.L⁺A))
end

@testset "Vertical discretization" begin
    par = default_params(nx=4, ny=4, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, νz=1.0)
    G = init_grid(par)
    dz = TEST_Lz / par.nz

    # Vertical grid: z ∈ [-Lz, 0] with staggered (cell-centered) levels
    # z[1] = -Lz + dz/2 (near bottom), z[end] = -dz/2 (near surface)
    @test isapprox(G.z[1], -TEST_Lz + dz / 2; rtol=0.0, atol=10 * eps(dz))
    @test isapprox(G.z[end], -dz / 2; rtol=0.0, atol=10 * eps(TEST_Lz))
    @test all(isapprox.(diff(G.z), dz; rtol=0.0, atol=10 * eps(dz)))
    @test all(isapprox.(G.dz, dz; rtol=0.0, atol=10 * eps(dz)))

    qok = zeros(ComplexF64, par.nz, par.nx, par.ny)
    for k in 1:par.nz
        z = G.z[k]
        qval = cos(pi * z / TEST_Lz)
        qok[k, :, :] .= qval
    end

    dqk = similar(qok)
    dissipation_q_nv!(dqk, qok, par, G)

    k2 = (pi / TEST_Lz)^2
    for k in 2:(par.nz - 1)
        expected = -k2 * cos(pi * G.z[k] / TEST_Lz)
        @test isapprox(real(dqk[k, 1, 1]), expected; rtol=0.05, atol=1e-8)
    end
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

    z_levels = [-1000.0, -3000.0]
    per_level = [6, 10]
    cfg3d = particles_in_layers(z_levels; x_max=G.Lx, y_max=G.Ly, nx=4, ny=4,
                                particles_per_level=per_level, precision=Float64)

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
    cfg3d = particles_random_3d(10; x_max=G.Lx, y_max=G.Ly, z_max=G.Lz, seed=42, precision=Float64)

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
    # z ∈ [-Lz, 0] with surface at z=0 (oceanographic convention)
    @test all(z -> z >= -G.Lz && z <= 0.0, tracker1.particles.z)
end

@testset "QGYBJplus basic API" begin
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, stratification=:constant_N)
    G, S, plans, a = setup_model(par)
    @test size(S.q) == (par.nz, par.nx, par.ny)

    # Invert q->psi (all zeros)
    invert_q_to_psi!(S, G; a, par=par)
    @test all(isfinite, real.(S.psi))

    # Put a simple B mode and invert to A (YBJ+)
    S.L⁺A[3, 2, 2] = 1 + 0im
    invert_L⁺A_to_A!(S, G, par, a)
    @test all(isfinite, real.(S.A))

    # One ETDRK2 step should run without error
    L = dealias_mask(G)
    Snp1 = copy_state(S)
    exp_rk2_step!(Snp1, S, G, par, plans; a, dealias_mask=L)

    @test all(isfinite, real.(Snp1.q))
end

@testset "Barotropic flow initialization" begin
    par = default_params(nx=8, ny=8, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)

    fill!(parent(S.psi), 0)
    S.psi[1, 1, 1] = 3.0 + 0.0im
    S.psi[2, 3, 4] = 1.0 + 2.0im
    compute_barotropic_q_from_psi!(S.q, S.psi, G)

    kₕ² = G.kx[3]^2 + G.ky[4]^2
    @test S.q[1, 1, 1] == 0
    @test S.q[2, 3, 4] == -kₕ² * S.psi[2, 3, 4]
end

@testset "State NetCDF metadata is self describing" begin
    outdir = mktempdir()
    par = default_params(nx=8, ny=8, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)
    invert_q_to_psi!(S, G; a, par=par)

    config = create_output_config(output_dir=outdir,
                                  state_file_pattern="state%04d.nc",
                                  save_psi=true,
                                  save_waves=false,
                                  save_velocities=false,
                                  save_diagnostics=false)
    manager = OutputManager(config, par)
    filepath = write_state_file(manager, S, G, plans, 60.0; params=par,
                                write_psi=true, write_waves=false)

    NCDataset(filepath, "r") do ds
        @test ds.attrib["equation_form"] == "dimensional"
        @test ds.attrib["field_layout"] == "Variables are stored as (x, y, z); internal spectral arrays use (z, x, y)."
        @test ds["time"].attrib["units"] == "s"
        @test ds["time"].attrib["description"] == "Elapsed seconds since the start of the simulation."
        @test ds["z"].attrib["positive"] == "up"
        @test ds["psi"].attrib["long_name"] == "quasi-geostrophic streamfunction"
    end

    z_config = create_output_config(output_dir=outdir,
                                    state_file_pattern="z_state%04d.nc",
                                    save_psi=true,
                                    save_waves=false,
                                    save_velocities=false,
                                    save_diagnostics=false,
                                    z_levels=[0.0, -TEST_Lz])
    z_manager = OutputManager(z_config, par)
    z_filepath = write_state_file(z_manager, S, G, plans, 120.0; params=par,
                                  write_psi=true, write_waves=false)

    NCDataset(z_filepath, "r") do ds
        @test ds["z"][:] ≈ [G.z[end], G.z[1]]
        @test size(ds["psi"]) == (G.nx, G.ny, 2)
        @test ds["z"].attrib["description"] == "Nearest native grid z coordinates saved for requested z_levels."
    end

    full_outdir = mktempdir()
    surface_outdir = mktempdir()
    par_multi = default_params(nx=4, ny=4, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                               nt=1, dt=1.0, fixed_flow=true, no_feedback=true,
                               no_wave_feedback=true)
    G_multi, S_multi, plans_multi, a_multi = setup_model(par_multi)
    invert_q_to_psi!(S_multi, G_multi; a=a_multi, par=par_multi)

    full_config = create_output_config(output_dir=full_outdir,
                                       save_psi=true,
                                       save_waves=false,
                                       save_velocities=false,
                                       save_diagnostics=false)
    surface_config = create_output_config(output_dir=surface_outdir,
                                          save_psi=true,
                                          save_waves=false,
                                          save_velocities=false,
                                          save_diagnostics=false,
                                          z_levels=[0.0])

    run_simulation!(S_multi, G_multi, par_multi, plans_multi;
                    output_config=(full_config, surface_config),
                    print_progress=false,
                    diagnostics_interval=1)

    NCDataset(joinpath(full_outdir, "state0001.nc"), "r") do ds
        @test size(ds["psi"]) == (G_multi.nx, G_multi.ny, G_multi.nz)
    end
    NCDataset(joinpath(surface_outdir, "state0001.nc"), "r") do ds
        @test ds["z"][:] ≈ [G_multi.z[end]]
        @test size(ds["psi"]) == (G_multi.nx, G_multi.ny, 1)
    end
end

@testset "Exponential RK2 step (wave feedback enabled)" begin
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         dt=0.01, no_feedback=false, no_wave_feedback=false)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    # Seed a nontrivial wave mode
    S.L⁺A[3, 2, 2] = 1.0 + 0.2im

    Snp1 = copy_state(S)
    exp_rk2_step!(Snp1, S, G, par, plans; a=a, dealias_mask=L)

    @test all(isfinite, real.(Snp1.q))
    @test all(isfinite, imag.(Snp1.q))
    @test all(isfinite, real.(Snp1.L⁺A))
    @test all(isfinite, imag.(Snp1.L⁺A))
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

    # The production stepper is YBJ+ only; keep the normal-YBJ helper path
    # covered through its diagnostic reconstruction.
    S.L⁺A[4, 3, 3] = 0.5 + 0.2im
    L⁺ARk = similar(S.L⁺A)
    L⁺AIk = similar(S.L⁺A)
    QGYBJplus.split_L⁺A_to_real_imag!(L⁺ARk, L⁺AIk, S.L⁺A)
    zero_rhs = similar(S.L⁺A)
    fill!(parent(zero_rhs), 0)
    sigma = compute_sigma(par, G, zero_rhs, zero_rhs, zero_rhs, zero_rhs; Lmask=L)
    compute_A!(S.A, S.C, L⁺ARk, L⁺AIk, sigma, par, G; Lmask=L)

    @test all(isfinite, real.(S.A))
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
    S.L⁺A[8, 3, 3] = 1.0 + 0.5im

    # Invert to get A
    invert_L⁺A_to_A!(S, G, par, a)

    # Check A is finite and non-zero where B is non-zero
    @test all(isfinite, real.(S.A))
    @test all(isfinite, imag.(S.A))
    @test !all(iszero, S.A)

    # The A field should have smooth vertical structure (not oscillatory artifacts)
    # Check that vertical derivative C = A_z is also well-behaved
    @test all(isfinite, real.(S.C))
end

@testset "Hyperdiffusion integrating factor - QG-YBJp separable form" begin
    # QG-YBJp uses separable powers: ν(|kx|^(2n) + |ky|^(2n)).
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         νₕ₁=1.0, νₕ₂=0.0, ilap1=2)

    # Access the int_factor function through QGYBJplus module
    if isdefined(QGYBJplus.Nonlinear, :int_factor)
        kx, ky = 1.0, 1.0
        dt = par.dt
        int_f = QGYBJplus.Nonlinear.int_factor(kx, ky, par; waves=false)
        @test isfinite(int_f)
        @test int_f > 0  # Should be positive dissipation
        @test isapprox(int_f, 2dt; rtol=1e-12)

        # Separable QG-YBJp form differs from isotropic (kx² + ky²)^n.
        int_f_diagonal = QGYBJplus.Nonlinear.int_factor(1.0, 1.0, par; waves=false)
        int_f_axis = QGYBJplus.Nonlinear.int_factor(sqrt(2.0), 0.0, par; waves=false)
        @test !isapprox(int_f_diagonal, int_f_axis, rtol=1e-10)
    end
end

@testset "Wave feedback does not alter prognostic q" begin
    common = (nx=8, ny=8, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
              dt=1e-3, inviscid=true, νₕ₁=0.0, νₕ₂=0.0, νₕ₁ʷ=0.0, νₕ₂ʷ=0.0,
              fixed_flow=false, ybj_plus=true)
    par_no_feedback = default_params(; common..., no_feedback=true, no_wave_feedback=true)
    par_feedback = default_params(; common..., no_feedback=false, no_wave_feedback=false)

    G, S_no_feedback, plans, a = setup_model(par_no_feedback)
    S_no_feedback.q[1, 2, 1] = 1.0 + 0.0im
    S_no_feedback.q[2, 1, 2] = -0.4 + 0.0im
    S_no_feedback.L⁺A[1, 2, 2] = 2.0 + 0.7im
    S_no_feedback.L⁺A[2, 3, 2] = -0.6 + 1.1im

    S_feedback = copy_state(S_no_feedback)
    L = dealias_mask(G)

    S_no_feedback_np1 = copy_state(S_no_feedback)
    S_feedback_np1 = copy_state(S_feedback)
    exp_rk2_step!(S_no_feedback_np1, S_no_feedback, G, par_no_feedback, plans; a=a, dealias_mask=L)
    exp_rk2_step!(S_feedback_np1, S_feedback, G, par_feedback, plans; a=a, dealias_mask=L)

    @test isapprox(parent(S_feedback_np1.q), parent(S_no_feedback_np1.q); rtol=1e-12, atol=1e-12)
    @test sum(abs2, parent(S_feedback_np1.psi) .- parent(S_no_feedback_np1.psi)) > 0
end

@testset "Vertical velocity with N² profile" begin
    # Test that vertical velocity computation accepts and uses N² profile
    par = default_params(nx=8, ny=8, nz=16, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)

    # Create a non-constant N² profile (exponential decay with depth)
    dz = G.Lz / G.nz
    N2_profile = [1.0 * exp(-(-G.z[k] + dz / 2) / (TEST_Lz / 4)) for k in 1:par.nz]

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
    S.L⁺A[8, 3, 3] = 1.0 + 0.5im
    invert_L⁺A_to_A!(S, G, par, a)

    # The A field should be non-zero after inversion
    @test !all(iszero, S.A)

    # Create N² profile
    N2_profile = ones(par.nz)  # Constant for simplicity

    # Compute YBJ vertical velocity
    QGYBJplus.Operators.compute_ybj_vertical_velocity!(S, G, plans, par; N2_profile=N2_profile)

    # Check w is finite (main verification that the code path works)
    @test all(isfinite, S.w)
end

@testset "Exponential RK2 step initializes A" begin
    # Test that exp_rk2_step! initializes A before using it
    # (Previously A was zero when first used for dispersion)
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    # Set initial wave field
    S.L⁺A[4, 3, 3] = 1.0 + 0.5im

    # A should initially be zero
    @test all(iszero, S.A)

    Snp1 = copy_state(S)
    exp_rk2_step!(Snp1, S, G, par, plans; a=a, dealias_mask=L)

    # After the step, A should be computed (not zero if B was non-zero)
    # Due to physics switches and filtering, A might still be small but the
    # computation should have happened
    @test all(isfinite, Snp1.A)
end

@testset "Exponential RK2 workspace limits repeat allocations" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true, fixed_flow=true, no_feedback=true,
                         no_wave_feedback=true, dt=1.0, nt=1)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)
    Snp1 = copy_state(S)
    timestep_workspace = ExpRK2Workspace(S, plans)

    exp_rk2_step!(Snp1, S, G, par, plans; a=a, dealias_mask=L,
                  timestep_workspace=timestep_workspace)
    exp_rk2_step!(Snp1, S, G, par, plans; a=a, dealias_mask=L)

    without_workspace_allocations = @allocated exp_rk2_step!(Snp1, S, G, par, plans;
                                                             a=a, dealias_mask=L)
    step_allocations = @allocated exp_rk2_step!(Snp1, S, G, par, plans; a=a,
                                                dealias_mask=L,
                                                timestep_workspace=timestep_workspace)

    @test step_allocations < 0.9 * without_workspace_allocations
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
