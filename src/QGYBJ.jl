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

# Public API - Core functionality
export QGParams, Grid, State,
       init_grid, init_state, init_pencil_decomposition!,
       plan_transforms!, fft_forward!, fft_backward!,
       compute_wavenumbers!,
       invert_q_to_psi!, compute_velocities!,
       default_params, setup_model,
       a_ell_ut, dealias_mask,
       invert_B_to_A!,
       jacobian_spectral!, convol_waqg!, refraction_waqg!, compute_qw!, dissipation_q_nv!, int_factor,
       init_random_psi!,
        first_projection_step!, leapfrog_step!,
        sumB!, compute_sigma, compute_A!,
        ncdump_psi, ncdump_la, ncread_psi!, ncread_la!,
        omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg, slice_horizontal, slice_vertical_xz

# Public API - New user interface
export DomainConfig, StratificationConfig, InitialConditionConfig, OutputConfig, ModelConfig,
       create_domain_config, create_stratification_config, create_initial_condition_config, 
       create_output_config, create_model_config,
       QGYBJSimulation, setup_simulation, run_simulation!,
       create_simple_config, run_simple_simulation,
       OutputManager, write_state_file, read_initial_psi, read_initial_waves, read_stratification_profile,
       StratificationProfile, ConstantN, SkewedGaussian, TanhProfile,
       create_stratification_profile, compute_stratification_profile,
       # Parallel interface
       ParallelConfig, setup_parallel_environment, init_parallel_grid, init_parallel_state,
       setup_parallel_transforms, ParallelOutputManager, write_parallel_state_file

include("parameters.jl")
include("grid.jl")
include("transforms.jl")
include("operators.jl")
include("elliptic.jl")
include("physics.jl")
include("runtime.jl")
include("nonlinear.jl")
include("timestep.jl")
include("initconds.jl")
include("ybj_normal.jl")
include("io.jl")
include("diagnostics.jl")

# New user interface modules
include("model_interface.jl")

end # module
