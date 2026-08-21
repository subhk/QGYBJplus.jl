using Test
using MPI
using QGYBJplus
using NCDatasets

# Keep MPI externally owned for the complete suite. Individual models can then
# be constructed and finalized independently without finalizing MPI between
# test files.
MPI.Initialized() || MPI.Init()

try
    include("test_core_components.jl")
    include("test_model_fields.jl")
    include("test_core_architecture.jl")
    include("test_model_ownership.jl")
    include("test_model_operators.jl")
    include("test_model_etdrk2.jl")
    include("test_model_initialization.jl")
    include("test_simulation_lifecycle.jl")
    include("test_model_io.jl")
    include("test_model_particles.jl")
    include("test_asselin_smoke.jl")
finally
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Barrier(MPI.COMM_WORLD)
        MPI.Finalize()
    end
end
