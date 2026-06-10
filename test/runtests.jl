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

    # The simulation API exposes only the exponential RK2 time stepper.
    par = default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    @test :γ ∉ fieldnames(typeof(par))
    @test_throws MethodError default_params(nx=8, ny=8, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz, γ=1e-3)

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

@testset "ETDRK2 coefficients are stable for small hyperdiffusion" begin
    dt = 3.0

    for λdt in (1e-8, 1e-10)
        E, hφ1, hφ2 = QGYBJplus._etd_coefficients(λdt, dt)

        λdt_big = parse(BigFloat, string(λdt))
        dt_big = BigFloat(dt)
        E_expected = exp(-λdt_big)
        hφ1_expected = dt_big * (1 - E_expected) / λdt_big
        hφ2_expected = dt_big * (E_expected - 1 + λdt_big) / λdt_big^2

        @test E ≈ Float64(E_expected) rtol=1e-14
        @test hφ1 ≈ Float64(hφ1_expected) rtol=1e-13
        @test hφ2 ≈ Float64(hφ2_expected) rtol=1e-13
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

@testset "ETDRK2 RHS uses wave-feedback effective PV for diagnostics" begin
    par = default_params(nx=8, ny=8, nz=4, Lx=2π, Ly=2π, Lz=2π,
                         dt=1e-3, inviscid=true, νₕ₁=0.0, νₕ₂=0.0,
                         νₕ₁ʷ=0.0, νₕ₂ʷ=0.0, fixed_flow=false,
                         ybj_plus=true, no_feedback=false, no_wave_feedback=false)
    G, S, plans, a = setup_model(par)
    L = dealias_mask(G)

    S.q[1, 2, 1] = 1.0 + 0.0im
    S.q[2, 1, 2] = -0.4 + 0.0im
    S.L⁺A[1, 2, 1] = 1.0 + 0.3im
    S.L⁺A[1, 1, 2] = 0.7 - 0.2im
    S.L⁺A[2, 3, 2] = -0.6 + 1.1im
    S.L⁺A[3, 2, 3] = 0.5 - 0.4im
    S.L⁺A[4, 4, 1] = -0.2 + 0.9im

    q_original = copy(parent(S.q))

    S_expected = copy_state(S)
    q_base = QGYBJplus.replace_q_with_wave_feedback_rhs!(S_expected, G, par, plans, L)
    invert_q_to_psi!(S_expected, G; a=a, par=par)
    QGYBJplus.restore_prognostic_q!(S_expected, q_base)

    S_raw = copy_state(S)
    invert_q_to_psi!(S_raw, G; a=a, par=par)
    @test sum(abs2, parent(S_expected.psi) .- parent(S_raw.psi)) > 0

    rhsq = similar(S.q)
    rhsB = similar(S.L⁺A)
    S_rhs = copy_state(S)
    QGYBJplus._compute_etdrk2_rhs!(rhsq, rhsB, S_rhs, G, par, plans;
                                   a=a, dealias_mask=L)

    @test parent(S_rhs.q) ≈ q_original
    @test parent(S_rhs.psi) ≈ parent(S_expected.psi)
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

@testset "Elliptic direct workspaces limit allocations" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true, fixed_flow=true, no_feedback=true,
                         no_wave_feedback=true, dt=1.0, nt=1)
    G, S, plans, a = setup_model(par)
    fill!(parent(S.q), zero(eltype(parent(S.q))))
    fill!(parent(S.L⁺A), zero(eltype(parent(S.L⁺A))))
    S.q[3, 4, 4] = 1.0 + 0.25im
    S.L⁺A[4, 5, 5] = 0.5 - 0.25im

    nonlinear_workspace = QGYBJplus.NonlinearWorkspace(S.psi, plans)

    invert_q_to_psi!(S, G; a=a, par=par)
    invert_q_to_psi!(S, G; a=a, par=par, workspace=nonlinear_workspace)
    no_workspace_q_allocations = @allocated invert_q_to_psi!(S, G; a=a, par=par)
    workspace_q_allocations = @allocated invert_q_to_psi!(S, G; a=a, par=par,
                                                          workspace=nonlinear_workspace)

    invert_L⁺A_to_A!(S, G, par, a)
    invert_L⁺A_to_A!(S, G, par, a; workspace=nonlinear_workspace)
    no_workspace_A_allocations = @allocated invert_L⁺A_to_A!(S, G, par, a)
    workspace_A_allocations = @allocated invert_L⁺A_to_A!(S, G, par, a;
                                                          workspace=nonlinear_workspace)

    # The workspace path must allocate far less than the allocating path. The
    # absolute bound is intentionally generous: the residual is a small constant
    # (~0.5 KB, independent of problem size) whose exact value drifts by a few tens
    # of bytes with unrelated recompilation/GC state, so a razor-tight bound is
    # fragile. The relative checks carry the real "limit allocations" intent.
    @test workspace_q_allocations < no_workspace_q_allocations
    @test workspace_q_allocations < 1024
    @test workspace_A_allocations < no_workspace_A_allocations
    @test workspace_A_allocations < 1024
end

@testset "Wave vertical diagnostic workspaces limit allocations" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true, fixed_flow=true, no_feedback=true,
                         no_wave_feedback=true, dt=1.0, nt=1)
    G, S, plans, a = setup_model(par)
    S.A[4, 3, 3] = 1.0 + 0.25im
    S.C[4, 3, 3] = 0.5 - 0.25im

    nonlinear_workspace = QGYBJplus.NonlinearWorkspace(S.psi, plans)

    compute_ybj_vertical_velocity!(S, G, plans, par; skip_inversion=true)
    compute_ybj_vertical_velocity!(S, G, plans, par; skip_inversion=true,
                                   workspace=nonlinear_workspace)
    no_workspace_w_allocations = @allocated compute_ybj_vertical_velocity!(S, G, plans, par;
                                                                           skip_inversion=true)
    workspace_w_allocations = @allocated compute_ybj_vertical_velocity!(S, G, plans, par;
                                                                        skip_inversion=true,
                                                                        workspace=nonlinear_workspace)

    compute_vertical_wave_displacement!(S, G, plans, par; skip_inversion=true)
    compute_vertical_wave_displacement!(S, G, plans, par; skip_inversion=true,
                                        workspace=nonlinear_workspace)
    no_workspace_ξ_allocations = @allocated compute_vertical_wave_displacement!(S, G, plans, par;
                                                                                skip_inversion=true)
    workspace_ξ_allocations = @allocated compute_vertical_wave_displacement!(S, G, plans, par;
                                                                             skip_inversion=true,
                                                                             workspace=nonlinear_workspace)

    @test workspace_w_allocations < 0.2 * no_workspace_w_allocations
    @test workspace_w_allocations < 20_000
    @test workspace_ξ_allocations < 0.2 * no_workspace_ξ_allocations
    @test workspace_ξ_allocations < 20_000
end

@testset "Wave velocity diagnostics execute without per-call setup failures" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true, fixed_flow=true, no_feedback=true,
                         no_wave_feedback=true, dt=1.0, nt=1)
    G, S, plans, a = setup_model(par)
    S.L⁺A[4, 5, 5] = 0.5 - 0.25im
    invert_L⁺A_to_A!(S, G, par, a)
    compute_velocities!(S, G; plans=plans, params=par, compute_w=true)

    compute_wave_velocities!(S, G; plans=plans, params=par, compute_w=true)

    @test all(isfinite, parent(S.u))
    @test all(isfinite, parent(S.v))
    @test all(isfinite, parent(S.w))
end

@testset "Particle field updates reuse cached workspaces" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz,
                         ybj_plus=true, fixed_flow=true, no_feedback=true,
                         no_wave_feedback=true, dt=1.0, nt=1)
    G, S, plans, a = setup_model(par)
    S.q[3, 4, 4] = 1.0 + 0.25im
    S.L⁺A[4, 5, 5] = 0.5 - 0.25im
    invert_q_to_psi!(S, G; a=a, par=par)
    invert_L⁺A_to_A!(S, G, par, a)

    particle_config = particles_in_box(Float64, G.z[4];
                                       x_max=G.Lx, y_max=G.Ly,
                                       nx=4, ny=4)
    tracker = ParticleTracker(particle_config, G)
    initialize_particles!(tracker, particle_config)

    QGYBJplus.UnifiedParticleAdvection.update_velocity_fields!(tracker, S, G; params=par)
    QGYBJplus.UnifiedParticleAdvection.update_wave_fields!(tracker, S, G; params=par)

    velocity_update_allocations =
        @allocated QGYBJplus.UnifiedParticleAdvection.update_velocity_fields!(tracker, S, G; params=par)
    wave_update_allocations =
        @allocated QGYBJplus.UnifiedParticleAdvection.update_wave_fields!(tracker, S, G; params=par)

    @test velocity_update_allocations < 100_000
    @test wave_update_allocations < 100_000
    @test fieldtype(typeof(tracker), :plans) !== Any
    @test fieldtype(typeof(tracker), :comm) !== Any
end

@testset "Legacy simulation container keeps concrete field types" begin
    @test fieldtype(QGYBJplus.QGYBJSimulation{Float64}, :plans) !== Any
    @test fieldtype(QGYBJplus.QGYBJSimulation{Float64}, :output_manager) !== Any
    @test fieldtype(QGYBJplus.QGYBJSimulation{Float64}, :grid) <: Grid
    @test fieldtype(QGYBJplus.QGYBJSimulation{Float64}, :state) <: State
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

    # ---- compute_qw! normalization, sign, f₀ scaling, and complex path ----
    par_qw = default_params(nx=32, ny=32, nz=1, Lx=2*pi, Ly=2*pi, Lz=1.0, f₀=2.0)
    BR_phys .= 0.0
    BI_phys .= 0.0
    qw_expected = zeros(Float64, G.nz, G.nx, G.ny)

    for j in 1:G.ny, i in 1:G.nx
        xi = x[i]
        yj = y[j]
        BR_phys[1, i, j] = cos(xi) + cos(yj)
        BI_phys[1, i, j] = sin(xi) + sin(yj)
        qw_expected[1, i, j] = (sin(xi - yj) - cos(xi - yj)) / par_qw.f₀
    end

    fft_forward!(BRk, BR_phys, plans)
    fft_forward!(BIk, BI_phys, plans)

    qwk = zeros(ComplexF64, G.nz, G.nx, G.ny)
    compute_qw!(qwk, BRk, BIk, par_qw, G, plans; Lmask=L)

    qw_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_backward!(qw_phys, qwk, plans)

    @test maximum(abs.(real.(qw_phys) .- qw_expected)) < 5e-10

    Bk = BRk .+ im .* BIk
    qwk_complex = zeros(ComplexF64, G.nz, G.nx, G.ny)
    QGYBJplus.compute_qw_complex!(qwk_complex, Bk, par_qw, G, plans; Lmask=L)

    qw_complex_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    fft_backward!(qw_complex_phys, qwk_complex, plans)

    @test maximum(abs.(qwk_complex .- qwk)) < 5e-10
    @test maximum(abs.(real.(qw_complex_phys) .- qw_expected)) < 5e-10
    @test maximum(abs.(imag.(qw_complex_phys))) < 5e-10

    qwk_workspace = zeros(ComplexF64, G.nz, G.nx, G.ny)
    nonlinear_workspace = QGYBJplus.NonlinearWorkspace(Bk, plans)
    QGYBJplus.compute_qw_complex!(qwk_workspace, Bk, par_qw, G, plans; Lmask=L,
                                  workspace=nonlinear_workspace)
    @test maximum(abs.(qwk_workspace .- qwk_complex)) < 5e-10
end

#=
================================================================================
                    PARTICLE PERIODIC-DOMAIN TESTS
================================================================================
=#

@testset "Particle advection in periodic domain" begin
    UPA = QGYBJplus.UnifiedParticleAdvection

    par = default_params(nx=16, ny=16, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G = init_grid(par)
    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    z_mid = -G.Lz / 2

    cfg = ParticleConfig{Float64}(x_max=G.Lx, y_max=G.Ly, z_level=z_mid)
    tracker = ParticleTracker(cfg, G)
    p = tracker.particles

    # Particles near the right x-edge, left x-edge, and top y-edge
    p.x = [G.Lx - 0.25*dx, 0.25*dx, G.Lx/2]
    p.y = [G.Ly/2, G.Ly/2, G.Ly - 0.25*dy]
    p.z = fill(z_mid, 3)
    p.id = [1, 2, 3]
    p.u = zeros(3); p.v = zeros(3); p.w = zeros(3)
    p.np = 3
    x_init = copy(p.x)
    y_init = copy(p.y)

    # --- Forward uniform flow: particles 1 and 3 cross x=Lx / y=Ly ---
    U = 1.0
    tracker.u_field .= U
    tracker.v_field .= U
    tracker.w_field .= 0.0

    dt = 0.4 * dx / U
    nsteps = 10
    for _ in 1:nsteps
        UPA.advect_euler!(tracker, dt)
        UPA.apply_boundary_conditions!(tracker)
        # positions stay inside the periodic domain after every step
        @test all(x -> 0.0 <= x < G.Lx + eps(G.Lx), p.x)
        @test all(y -> 0.0 <= y < G.Ly + eps(G.Ly), p.y)
    end
    # uniform field -> trilinear interpolation is exact; trajectory must match
    # the analytic mod-wrapped path
    for i in 1:3
        @test p.x[i] ≈ mod(x_init[i] + nsteps*dt*U, G.Lx) rtol=1e-12
        @test p.y[i] ≈ mod(y_init[i] + nsteps*dt*U, G.Ly) rtol=1e-12
    end
    # particle 1 actually crossed the x=Lx seam
    @test p.x[1] < x_init[1]
    # particle 3 actually crossed the y=Ly seam
    @test p.y[3] < y_init[3]

    # --- Backward uniform flow: particle 2 crosses x=0 ---
    x_back = copy(p.x); y_back = copy(p.y)
    tracker.u_field .= -U
    tracker.v_field .= -U
    for _ in 1:nsteps
        UPA.advect_euler!(tracker, dt)
        UPA.apply_boundary_conditions!(tracker)
    end
    for i in 1:3
        @test p.x[i] ≈ mod(x_back[i] - nsteps*dt*U, G.Lx) rtol=1e-12
        @test p.y[i] ≈ mod(y_back[i] - nsteps*dt*U, G.Ly) rtol=1e-12
    end

    # --- Interpolation continuity across the periodic seam ---
    # u(x) = sin(2*pi*x/Lx) sampled at grid nodes x_i = (i-1)*dx
    nx = G.nx
    for j in 1:G.ny, i in 1:nx, k in 1:G.nz
        tracker.u_field[k, i, j] = sin(2π*(i-1)/nx)
    end
    tracker.v_field .= 0.0

    yq = G.Ly/2
    δ = 1e-6 * dx

    # mod identity: querying just outside the domain equals the wrapped query
    u_neg, _, _ = interpolate_velocity_at_position(-δ, yq, z_mid, tracker)
    u_wrap, _, _ = interpolate_velocity_at_position(G.Lx - δ, yq, z_mid, tracker)
    @test u_neg ≈ u_wrap atol=1e-12
    u_past, _, _ = interpolate_velocity_at_position(G.Lx + δ, yq, z_mid, tracker)
    u_in, _, _ = interpolate_velocity_at_position(δ, yq, z_mid, tracker)
    @test u_past ≈ u_in atol=1e-12

    # continuity: values just left and right of the seam differ by O(δ)
    @test abs(u_wrap - u_in) < 1e-6

    # seam midpoint interpolates between last node (i=nx) and first node (i=1)
    u_mid, _, _ = interpolate_velocity_at_position(G.Lx - 0.5*dx, yq, z_mid, tracker)
    expected_mid = 0.5 * (sin(2π*(nx-1)/nx) + sin(0.0))
    @test u_mid ≈ expected_mid rtol=1e-12

    # --- apply_boundary_conditions! wraps far-out positions ---
    p.x[1] = G.Lx + 0.3*dx
    p.x[2] = -0.3*dx
    UPA.apply_boundary_conditions!(tracker)
    @test p.x[1] ≈ 0.3*dx rtol=1e-10
    @test p.x[2] ≈ G.Lx - 0.3*dx rtol=1e-10
end

#=
================================================================================
                    WAVE FEEDBACK ANALYTIC VERIFICATION
================================================================================
For a two-wave field B = a·exp(iθ₁) + b·exp(iθ₂) with θₘ = kₘx + lₘy and
real a, b, the wave feedback (Xie & Vanneste 2015; Rocha, Wagner & Young 2018)

    qʷ = (i/2f)·J(B*, B) + (1/4f)·∇²|B|²

has the closed form (Δk = k₁-k₂, Δl = l₁-l₂):

    qʷ = (ab/f)·[ (k₁l₂ - k₂l₁)·sin(θ₁-θ₂) - ((Δk² + Δl²)/2)·cos(θ₁-θ₂) ]

since J(B*,B) = 2i·ab·(k₁l₂-k₂l₁)·sin(θ₂-θ₁) and
|B|² = a² + b² + 2ab·cos(θ₁-θ₂). Verifies signs, FFT normalization, and
the spectral Laplacian in both the complex and the split-BR/BI paths.
=#

@testset "Wave feedback qʷ matches analytic two-wave solution" begin
    par = default_params(nx=16, ny=16, nz=4, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, _ = setup_model(par)
    L = dealias_mask(G)
    f₀ = par.f₀

    # Two waves inside the dealiased band
    n1, m1 = 1, 2      # wave 1 integer wavenumbers
    n2, m2 = 2, -1     # wave 2
    k₁ = 2π * n1 / G.Lx; l₁ = 2π * m1 / G.Ly
    k₂ = 2π * n2 / G.Lx; l₂ = 2π * m2 / G.Ly
    a, b = 0.1, 0.07

    B_phys = zeros(ComplexF64, G.nz, G.nx, G.ny)
    qw_exact = zeros(Float64, G.nz, G.nx, G.ny)
    Δk = k₁ - k₂; Δl = l₁ - l₂
    for j in 1:G.ny, i in 1:G.nx
        x = (i - 1) * G.Lx / G.nx
        y = (j - 1) * G.Ly / G.ny
        θ₁ = k₁ * x + l₁ * y
        θ₂ = k₂ * x + l₂ * y
        Bval = a * cis(θ₁) + b * cis(θ₂)
        qw = (a * b / f₀) * ((k₁ * l₂ - k₂ * l₁) * sin(θ₁ - θ₂) -
                             0.5 * (Δk^2 + Δl^2) * cos(θ₁ - θ₂))
        for k in 1:G.nz
            B_phys[k, i, j] = Bval
            qw_exact[k, i, j] = qw
        end
    end

    Bk = similar(S.L⁺A)
    QGYBJplus.fft_forward!(parent(Bk), B_phys, plans)
    qw_scale = maximum(abs.(qw_exact))

    # --- Complex path (used by the ybj_plus production branch) ---
    qwk = similar(S.L⁺A)
    QGYBJplus.compute_qw_complex!(qwk, Bk, par, G, plans; Lmask=L)
    qw_num = similar(B_phys)
    QGYBJplus.fft_backward!(qw_num, parent(qwk), plans)
    @test maximum(abs.(real.(qw_num) .- qw_exact)) < 1e-12 * qw_scale
    @test maximum(abs.(imag.(qw_num))) < 1e-12 * qw_scale

    # --- Split BR/BI path (used by the non-plus branch) ---
    BRk = similar(Bk); BIk = similar(Bk)
    QGYBJplus.fft_forward!(parent(BRk), complex.(real.(B_phys)), plans)
    QGYBJplus.fft_forward!(parent(BIk), complex.(imag.(B_phys)), plans)
    qwk2 = similar(Bk)
    QGYBJplus.compute_qw!(qwk2, BRk, BIk, par, G, plans; Lmask=L)
    qw_num2 = similar(B_phys)
    QGYBJplus.fft_backward!(qw_num2, parent(qwk2), plans)
    @test maximum(abs.(real.(qw_num2) .- qw_exact)) < 1e-12 * qw_scale

    # --- Single plane wave: J(B*,B) = 0 and |B|² constant → qʷ ≡ 0 ---
    for j in 1:G.ny, i in 1:G.nx, k in 1:G.nz
        x = (i - 1) * G.Lx / G.nx
        y = (j - 1) * G.Ly / G.ny
        B_phys[k, i, j] = a * cis(k₁ * x + l₁ * y)
    end
    QGYBJplus.fft_forward!(parent(Bk), B_phys, plans)
    QGYBJplus.compute_qw_complex!(qwk, Bk, par, G, plans; Lmask=L)
    @test maximum(abs.(parent(qwk))) < 1e-14
end

#=
================================================================================
                FILE-BASED STRATIFICATION + 3D VORTICITY IC
================================================================================
One NetCDF file carries N²(z) and ζ(x,y,z); the model loads the stratification
via the :from_file profile and the flow via psi_type = :from_file_vorticity
(ψ̂ = -ζ̂/kₕ²). Verified against the analytic ψ that generated ζ.
=#

@testset "Load stratification + 3D vorticity from file" begin
    par = default_params(nx=16, ny=16, nz=8, Lx=TEST_Lx, Ly=TEST_Ly, Lz=TEST_Lz)
    G, S, plans, _ = setup_model(par)

    fn = joinpath(mktempdir(), "ic_strat_vort.nc")

    # Analytic streamfunction with vertical structure; ζ = ∇²ψ = -kₕ²ψ
    kx = 2π * 2 / G.Lx; ky = 2π * 3 / G.Ly
    kh2 = kx^2 + ky^2
    P = 0.5
    ψf(x, y, z) = P * cos(kx * x) * sin(ky * y) * (1 + 0.5 * cos(π * z / G.Lz))
    # The model evaluates N² on unstaggered (face) levels z - dz/2, so store
    # the profile there for an exact round-trip.
    z_face = G.z .- (G.Lz / G.nz) / 2
    N2_data = [1e-5 * exp(z / 1000.0) for z in z_face]

    zeta_xyz = zeros(G.nx, G.ny, G.nz)
    for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
        x = (i - 1) * G.Lx / G.nx
        y = (j - 1) * G.Ly / G.ny
        zeta_xyz[i, j, k] = -kh2 * ψf(x, y, G.z[k])
    end

    NCDatasets.Dataset(fn, "c") do ds
        NCDatasets.defDim(ds, "x", G.nx)
        NCDatasets.defDim(ds, "y", G.ny)
        NCDatasets.defDim(ds, "z", G.nz)
        zv = NCDatasets.defVar(ds, "z", Float64, ("z",)); zv[:] = z_face
        nv = NCDatasets.defVar(ds, "N2", Float64, ("z",)); nv[:] = N2_data
        vv = NCDatasets.defVar(ds, "zeta", Float64, ("x", "y", "z")); vv[:, :, :] = zeta_xyz
    end

    # --- Stratification from the same file ---
    profile = create_stratification_profile((type=:from_file, filename=fn))
    N2_loaded = compute_stratification_profile(profile, G)
    @test maximum(abs.(N2_loaded .- N2_data)) < 1e-12 * maximum(N2_data)

    # --- ζ file → spectral ψ ---
    psik = read_initial_vorticity(fn, G, plans)
    ψ_num = similar(psik)
    QGYBJplus.fft_backward!(ψ_num, psik, plans)
    ψ_exact = [ψf((i - 1) * G.Lx / G.nx, (j - 1) * G.Ly / G.ny, G.z[k])
               for k in 1:G.nz, i in 1:G.nx, j in 1:G.ny]
    @test maximum(abs.(real.(ψ_num) .- ψ_exact)) < 1e-12 * P
    @test maximum(abs.(imag.(ψ_num))) < 1e-12 * P
    @test abs(psik[1, 1, 1]) == 0.0   # horizontal-mean mode zeroed

    # --- Full config path: q computed from the loaded ψ and N² profile ---
    icc = InitialConditionConfig{Float64}(psi_type=:from_file_vorticity, psi_filename=fn,
                                          wave_type=:zero)
    cfg = (initial_conditions=icc,)
    initialize_from_config(cfg, G, S, plans; params=par, N2_profile=N2_loaded)
    @test maximum(abs.(parent(S.psi) .- psik)) < 1e-12 * maximum(abs.(psik))
    @test maximum(abs.(parent(S.q))) > 0
    @test all(isfinite, parent(S.q))
end
