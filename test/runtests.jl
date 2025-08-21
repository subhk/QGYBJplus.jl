using Test
using QGYBJ

@testset "QGYBJ basic API" begin
    par = default_params(nx=8, ny=8, nz=8, stratification=:constant_N)
    G, S, plans, a = setup_model(; par)
    @test size(S.q) == (par.nx, par.ny, par.nz)
    # Invert q->psi (all zeros)
    invert_q_to_psi!(S, G; a)
    @test all(isfinite, real.(S.psi))
    # Put a simple B mode and invert to A (YBJ+)
    S.B[2,2,3] = 1 + 0im
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

@testset "Normal YBJ branch + dealias + kh=0" begin
    par = default_params(nx=12, ny=12, nz=8, stratification=:constant_N)
    # switch to normal YBJ
    par = QGParams(; (field=>getfield(par, field) for field in fieldnames(typeof(par)))... )
    setfield!(par, :ybj_plus, false)
    G, S, plans, a = setup_model(; par)
    L = dealias_mask(G)
    # Dealias property checks for a few indices
    let keep = L, nx = par.nx, ny = par.ny
        # center should be kept
        @test keep[1,1]
        # a far corner should be dropped (heuristic)
        @test !keep[nx, ny]
    end
    # kh=0 psi inversion should zero the whole vertical column for that (i,j)
    S.q[1,1,3] = 1 + 0im
    invert_q_to_psi!(S, G; a)
    @test all(iszero, S.psi[1,1,:])
    # Run one normal-branch step to ensure it executes
    S.B[3,3,4] = 0.5 + 0.2im
    first_projection_step!(S, G, par, plans; a, dealias_mask=L)
    Snp1 = deepcopy(S); Snm1 = deepcopy(S)
    leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)
    @test all(isfinite, real.(Snp1.q))
end
