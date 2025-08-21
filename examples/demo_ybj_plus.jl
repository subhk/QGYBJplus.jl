using QGYBJ

# YBJ+ demo: initialize, step a couple of times, write NetCDF outputs
par = default_params(nx=32, ny=32, nz=16, stratification=:constant_N)
G, S, plans, a = setup_model(; par)
L = dealias_mask(G)

# Simple initial B spectrum: one mode
S.B[3,3,5] = 1 + 0im

# First projection and one leapfrog step
first_projection_step!(S, G, par, plans; a, dealias_mask=L)
Snp1 = deepcopy(S); Snm1 = deepcopy(S)
leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)

# NetCDF outputs (requires NCDatasets.jl)
try
    ncdump_psi(Snp1, G, plans; path="psi.out.nc")
    ncdump_la(Snp1, G, plans; path="la.out.nc")
    @info "Wrote psi.out.nc and la.out.nc"
catch err
    @warn "Skipping NetCDF output: $(err)"
end

