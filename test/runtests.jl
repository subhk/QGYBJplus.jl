using Test
using QGYBJplus
using NCDatasets

# The emulated-GPU tests need JLArrays, a test dependency that the
# lowest-compatible-versions CI job leaves out: it requires a newer
# KernelAbstractions than the package's own compat floor.
const RUN_ARCHITECTURE_TESTS = Base.find_package("JLArrays") !== nothing

# One enclosing test set, so that a failing file does not stop the others.
@testset "QGYBJplus" begin
    include("test_core_components.jl")
    include("test_stratification.jl")
    include("test_input_conventions.jl")
    include("test_model_operators.jl")
    include("test_operator_entry_points.jl")
    include("test_wave_feedback.jl")
    include("test_ybj_plus_physics.jl")
    include("test_model_switches.jl")
    include("test_qg_dynamics.jl")
    include("test_model_etdrk2.jl")
    include("test_model_initialization.jl")
    include("test_initial_conditions.jl")
    include("test_simulation_lifecycle.jl")
    include("test_model_io.jl")
    include("test_time_and_output.jl")
    include("test_model_particles.jl")
    include("test_particle_timing.jl")
    include("test_public_api.jl")
    include("test_allocations.jl")
    if RUN_ARCHITECTURE_TESTS
        include("test_architectures.jl")
    else
        @info "JLArrays is not installed: skipping the emulated-GPU tests"
    end
end
