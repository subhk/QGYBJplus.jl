using MPI
using QGYBJ

# MPI + PencilArrays/PencilFFTs demo (requires MPI, PencilArrays, PencilFFTs)
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

par = default_params(nx=64, ny=64, nz=32, stratification=:constant_N)
G, S, plans, a = setup_model(; par)
init_pencil_decomposition!(G)
plans = plan_transforms!(G)

if rank == 0
    @info "Pencil decomposition", G.decomp
end

L = dealias_mask(G)
S.B[3,3,5] = 1 + 0im

first_projection_step!(S, G, par, plans; a, dealias_mask=L)
Snp1 = deepcopy(S); Snm1 = deepcopy(S)
leapfrog_step!(Snp1, S, Snm1, G, par, plans; a, dealias_mask=L)

if rank == 0
    @info "MPI demo complete on rank 0"
end

MPI.Barrier(comm)
MPI.Finalize()

# Launch like:
#   mpiexec -n 4 julia --project examples/demo_mpi.jl

