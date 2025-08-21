using QGYBJ

# Normal YBJ demo: set ybj_plus=false and run
par = default_params(nx=32, ny=32, nz=16, stratification=:constant_N)
par = QGParams(; (field=>getfield(par, field) for field in fieldnames(typeof(par)))... )
setfield!(par, :ybj_plus, false)

G, S, plans, a = setup_model(; par)
L = dealias_mask(G)

S.B[4,4,6] = 0.8 + 0.3im

first_projection_step!(S, G, par, plans; a, dealias_mask=L)
Snp1 = deepcopy(S); Snm1 = deepcopy(S)
leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)

try
    ncdump_psi(Snp1, G, plans; path="psi_normal.out.nc")
    ncdump_la(Snp1, G, plans; path="la_normal.out.nc")
    @info "Wrote psi_normal.out.nc and la_normal.out.nc"
catch err
    @warn "Skipping NetCDF output: $(err)"
end

