"""
Main module for `QGYBJplus.jl`, a Julia implementation of the QG-YBJ+ model for
near-inertial waves interacting with balanced quasi-geostrophic flow.
"""
module QGYBJplus

export
    # Core data structures
    QGParams, Grid, State,

    # Grid, state, and transforms
    init_grid, init_state, copy_state,
    plan_transforms!, setup_parallel_transforms, fft_forward!, fft_backward!,
    compute_wavenumbers!,
    get_local_range, local_to_global, get_kx, get_ky, get_kh2,
    get_local_dims, is_parallel_array, z_is_local,

    # Physics and numerics
    invert_q_to_psi!, invert_L⁺A_to_A!, invert_helmholtz!,
    compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!,
    compute_total_velocities!, compute_wave_velocities!,
    compute_wave_displacement!, compute_vertical_wave_displacement!,
    default_params, setup_model, setup_model_with_profile,
    a_ell_ut, a_ell_from_N2, dealias_mask, is_dealiased,
    compute_hyperdiff_coeff, compute_hyperdiff_params, dimensional_hyperdiff_params,
    jacobian_spectral!, convol_waqg!, refraction_waqg!, compute_qw!,
    dissipation_q_nv!, int_factor,
    init_random_psi!, init_analytical_psi!, init_analytical_waves!, init_surface_waves!,
    add_balanced_component!, compute_q_from_psi!, compute_barotropic_q_from_psi!,
    initialize_from_config,

    # Time stepping
    exp_rk2_step!, ExpRK2Workspace,
    sumL⁺A!, compute_sigma, compute_A!,

    # Diagnostics
    omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg,
    slice_horizontal, slice_vertical_xz,
    flow_kinetic_energy_global, wave_energy_global,
    EnergyDiagnosticsManager, record_energies!, write_all_energy_files!,

    # User interface
    DomainConfig, StratificationConfig, InitialConditionConfig, OutputConfig, ModelConfig,
    create_domain_config, create_stratification_config, create_initial_condition_config,
    create_output_config, create_model_config,
    QGYBJSimulation, setup_simulation, run_simulation!,
    create_simple_config, run_simple_simulation, setup_model_with_config,
    RectilinearGrid, RectilinearGridSpec,
    FPlane, ConstantStratification, HorizontalHyperdiffusivity, SurfaceWave,
    TimeInterval, IterationInterval, NetCDFOutput,
    QGYBJModel, Simulation, initialize_simulation, run!, set!,
    set_mean_flow!, set_surface_waves!, set_exponential_surface_waves!, set_wave_packet!,
    get_inertial_period, inertial_period, get_duration, get_duration_ip,

    # I/O
    OutputManager, write_state_file,
    read_initial_psi, read_initial_waves,
    read_stratification_profile, read_stratification_raw,
    ncdump_psi, ncdump_la, ncread_psi!, ncread_la!,

    # Stratification
    StratificationProfile, ConstantN, SkewedGaussian, TanhProfile, AnalyticalProfile,
    FileProfile, FileStratification,
    create_stratification_profile, compute_stratification_profile,

    # MPI parallel interface
    MPIConfig, ParallelConfig, PencilDecomp, MPIPlans, MPIWorkspace,
    setup_mpi_environment, setup_parallel_environment,
    init_mpi_grid, init_parallel_grid, init_mpi_state, init_parallel_state,
    init_mpi_workspace, plan_mpi_transforms,
    gather_to_root, scatter_from_root, mpi_barrier, mpi_reduce_sum, local_indices,
    write_mpi_field, init_mpi_random_field!, parallel_initialize_fields!,
    clear_mpi_transpose_buffer_cache!,
    reduce_sum_if_mpi, reduce_min_if_mpi, reduce_max_if_mpi,
    transpose_to_z_pencil!, transpose_to_xy_pencil!,
    get_local_range_xy, get_local_range_z, local_to_global_xy, local_to_global_z,
    get_local_range_physical, get_local_range_spectral,
    allocate_z_pencil, allocate_xy_pencil, allocate_xz_pencil, allocate_fft_backward_dst,
    is_root, nprocs, finalize_simulation!,

    # Lagrangian particle tracking
    ParticleConfig, ParticleState, ParticleTracker, create_particle_config,
    initialize_particles!, advect_particles!, interpolate_velocity_at_position,
    write_particle_trajectories, read_particle_trajectories, write_particle_snapshot,
    create_particle_output_file, write_particle_trajectories_by_zlevel,
    enable_auto_file_splitting!, finalize_trajectory_files!,
    InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
    interpolate_velocity_advanced,
    particles_in_box, particles_in_circle, particles_in_grid_3d, particles_in_layers,
    particles_random_3d, particles_custom,
    ParticleConfig3D, ParticleDistribution, initialize_particles_3d!,
    UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM

using LinearAlgebra
using MPI
using PencilArrays
using PencilFFTs
using SpecialFunctions: erf

#####
##### Include submodules
#####

# Basics
include("parameters.jl")
include("grid.jl")
include("transforms.jl")
include("parallel_mpi.jl")
include("loop_macros.jl")

# Physics, solvers, and time stepping
include("physics.jl")
include("elliptic.jl")
include("operators.jl")
include("runtime.jl")
include("nonlinear.jl")
include("timestep.jl")
include("initconds.jl")
include("ybj_normal.jl")

# Diagnostics
include("diagnostics.jl")
include("energy_diagnostics.jl")

using .EnergyDiagnostics: EnergyDiagnosticsManager, should_output, record_energies!
using .EnergyDiagnostics: write_all_energy_files!, append_and_write!, finalize!

# Configuration, I/O, and user interfaces
include("config.jl")
include("netcdf_io.jl")
include("initialization.jl")
include("stratification.jl")
include("model_interface.jl")
include("simulation.jl")

# Lagrangian particle tracking
include("particles/particle_advection.jl")
include("particles/particle_io.jl")

# Display methods
include("pretty_printing.jl")

end # module
