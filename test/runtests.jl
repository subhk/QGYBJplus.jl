using Test
using QGYBJplus
using NCDatasets

include("test_core_components.jl")
include("test_stratification.jl")
include("test_model_operators.jl")
include("test_wave_feedback.jl")
include("test_ybj_plus_physics.jl")
include("test_model_switches.jl")
include("test_qg_dynamics.jl")
include("test_model_etdrk2.jl")
include("test_model_initialization.jl")
include("test_simulation_lifecycle.jl")
include("test_model_io.jl")
include("test_model_particles.jl")
include("test_allocations.jl")
include("test_architectures.jl")
