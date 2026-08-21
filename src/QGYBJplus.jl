module QGYBJplus

using LinearAlgebra
using Random
using SpecialFunctions: erf
using NCDatasets
using MPI
using PencilArrays
using PencilFFTs

export RectilinearGrid, ModelFields, allocate_fields, copy_fields,
       QGYBJModel, ModelPhysics, ModelNumerics, ModelRuntime,
       OperatorCoefficients, Simulation, Clock, SimulationState,
       Ready, Running, Stopped, Failed, Finalized,
       AbstractCoriolis, FPlane,
       AbstractStratification, ConstantStratification,
       StratificationProfile, ConstantN, SkewedGaussian, TanhProfile,
       ExponentialProfile, PiecewiseProfile, FileProfile, AnalyticalProfile,
       evaluate_N2, compute_stratification_profile,
       load_stratification_from_file, validate_stratification,
       AbstractClosure, HorizontalHyperdiffusivity, VerticalDiffusivity,
       FlowEvolution, FixedFlow, EvolvingFlow,
       FeedbackMode, NoFeedback, WaveMeanFeedback, NoWaveFeedback,
       WaveFormulation, YBJPlus, YBJ, PassiveWave,
       DissipationMode, Dissipative, Inviscid,
       DynamicsMode, NonlinearDynamics, LinearDynamics,
       DispersionMode, Dispersive, NoDispersion,
       SurfaceWave, RandomStreamfunction,
       AbstractSchedule, TimeInterval, IterationInterval, NetCDFOutput,
       ExponentialRungeKutta2, ExponentialRungeKutta2Workspace, step!,
       set!, set_mean_flow!, set_surface_waves!,
       set_exponential_surface_waves!, set_wave_packet!,
       run!, restore!, finalize_model!, finalize_simulation!,
       inertial_period, get_inertial_period, get_time, is_root, nprocs,
       invert_q_to_psi!, invert_B_to_A!, invert_helmholtz!,
       compute_velocities!, compute_vertical_velocity!,
       compute_ybj_vertical_velocity!, compute_total_velocities!,
       compute_wave_velocities!, flow_kinetic_energy, wave_energy,
       a_ell_from_N2, dealias_mask, is_dealiased,
       compute_hyperdiff_coeff, compute_hyperdiff_params,
       dimensional_hyperdiff_params,
       gather_to_root, scatter_from_root,
       get_local_range, get_local_range_physical, get_local_range_spectral,
       local_to_global, get_kh2, fft_forward!, fft_backward!,
       ParticleConfig, ParticleConfig3D, ParticleDistribution,
       ParticleState, ParticleTracker,
       InterpolationMethod, TRILINEAR, TRICUBIC, ADAPTIVE, QUINTIC,
       UNIFORM_GRID, LAYERED, RANDOM_3D, CUSTOM,
       initialize_particles!, advect_particles!,
       interpolate_velocity_at_position, interpolate_velocity_advanced,
       particles_in_box, particles_in_circle, particles_in_grid_3d,
       particles_in_layers, particles_random_3d, particles_custom,
       ParticleOutputManager, write_particle_trajectories,
       read_particle_trajectories, write_particle_snapshot,
       create_particle_output_file, write_particle_trajectories_by_zlevel

include("core/components.jl")
include("core/grid.jl")
include("core/fields.jl")
include("stratification.jl")

include("transforms.jl")
include("parallel_mpi.jl")
include("loop_macros.jl")

include("physics.jl")
include("elliptic.jl")
include("diagnostics.jl")
include("operators.jl")
include("nonlinear.jl")
include("timestep.jl")
include("ybj_normal.jl")
include("initialization.jl")

include("core/model.jl")
include("core/runtime.jl")
include("core/simulation.jl")
include("simulation.jl")
include("core/io.jl")

include("particles/particle_advection.jl")
include("particles/particle_io.jl")
include("core/particles.jl")

end
