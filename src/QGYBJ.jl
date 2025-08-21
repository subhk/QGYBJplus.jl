module QGYBJ

using LinearAlgebra

# External backends (declared here; the user should add them to the project)
try
    import MPI
    using PencilArrays
    using PencilFFTs
catch
    @info "MPI/PencilArrays/PencilFFTs not loaded yet. You can still use serial mode."
end

# Public API
export QGParams, Grid, State,
       init_grid, init_state,
       plan_transforms!, fft_forward!, fft_backward!,
       compute_wavenumbers!,
       invert_q_to_psi!, compute_velocities!,
       default_params, setup_model,
       a_ell_ut

include("parameters.jl")
include("grid.jl")
include("transforms.jl")
include("operators.jl")
include("elliptic.jl")
include("physics.jl")
include("runtime.jl")

end # module
