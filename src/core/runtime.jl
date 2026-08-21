"""Precomputed vertical coefficients used by model operators."""
struct OperatorCoefficients{T, C}
    N²::Vector{T}
    a_ell::Vector{T}
    stratification::C
end

"""
    ModelRuntime

Ephemeral execution resources associated with a model: MPI ownership,
decomposition metadata, transform plans and destinations, operator
coefficients, dealiasing metadata, and reusable workspaces.

`computational_grid` and `parameters` are temporary migration internals. They
are deliberately contained here, never exposed as model ownership, and are
removed as numerical subsystems adopt `RectilinearGrid` and typed components.
"""
mutable struct ModelRuntime{M, D, P, X, W, C, L, G, Q}
    mpi::M
    decomposition::D
    plans::P
    transform_destinations::X
    workspace::W
    coefficients::C
    dealias_mask::L
    computational_grid::G
    parameters::Q
    owns_mpi::Bool
    finalized::Bool
end

_coriolis_frequency(coriolis::FPlane) = coriolis.f
_stratification_N²(stratification::ConstantStratification) = stratification.N²
_stratification_N²(stratification::StratificationProfile) = evaluate_N2(stratification, 0.0)

_runtime_profile(stratification::ConstantStratification) =
    ConstantN(sqrt(float(stratification.N²)))
_runtime_profile(stratification::StratificationProfile) = stratification

function _feedback_flags(feedback::FeedbackMode)
    feedback isa NoFeedback && return true, true
    feedback isa WaveMeanFeedback && return false, false
    feedback isa NoWaveFeedback && return false, true
    error("unsupported feedback component $(typeof(feedback))")
end

function _legacy_parameters(grid::RectilinearGrid, physics::ModelPhysics,
    numerics::ModelNumerics)

    nx, ny, nz = grid.size
    Lx, Ly, Lz = grid.extent
    x0, y0 = grid.origin
    no_feedback, no_wave_feedback = _feedback_flags(physics.feedback)
    closure = numerics.closure
    formulation = physics.formulation

    return default_params(
        nx=nx, ny=ny, nz=nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        x0=x0, y0=y0,
        dt=1.0, nt=1,
        f₀=_coriolis_frequency(physics.coriolis),
        N²=_stratification_N²(physics.stratification),
        ybj_plus=formulation isa YBJPlus || formulation isa PassiveWave,
        passive_scalar=formulation isa PassiveWave,
        fixed_flow=physics.flow isa FixedFlow,
        no_feedback=no_feedback,
        no_wave_feedback=no_wave_feedback,
        inviscid=numerics.dissipation isa Inviscid,
        linear=numerics.dynamics isa LinearDynamics,
        no_dispersion=numerics.dispersion isa NoDispersion,
        νz=numerics.vertical_diffusion.coefficient,
        νₕ₁=closure.flow,
        νₕ₂=closure.flow2,
        ilap1=closure.flow_laplacian_order,
        ilap2=closure.flow_laplacian_order2,
        νₕ₁ʷ=closure.waves,
        νₕ₂ʷ=closure.waves2,
        ilap1w=closure.wave_laplacian_order,
        ilap2w=closure.wave_laplacian_order2,
    )
end

"""Construct execution resources for `grid` transactionally."""
function build_runtime(grid::RectilinearGrid, physics::ModelPhysics,
    numerics::ModelNumerics; topology=nothing, parallel_io::Bool=false,
    verbose::Bool=true)

    mpi_was_initialized = MPI.Initialized()
    runtime = nothing
    try
        parameters = _legacy_parameters(grid, physics, numerics)
        mpi = setup_mpi_environment(; topology, parallel_io)
        computational_grid = init_mpi_grid(parameters, mpi)
        plans = plan_mpi_transforms(computational_grid, mpi)
        workspace = init_mpi_workspace(computational_grid, mpi)

        profile = _runtime_profile(physics.stratification)
        N² = Float64.(compute_stratification_profile(profile, computational_grid))
        parameters.N² = sum(N²) / length(N²)
        a_ell = a_ell_from_N2(N², parameters)
        stratification = compute_stratification_coefficients(
            N², computational_grid; f0_sq=physics.coriolis.f^2)
        coefficients = OperatorCoefficients(N², a_ell, stratification)
        mask = dealias_mask(computational_grid)
        destinations = hasproperty(plans, :work_arrays) ? plans.work_arrays : nothing

        runtime = ModelRuntime(
            mpi, computational_grid.decomp, plans, destinations, workspace,
            coefficients, mask, computational_grid, parameters,
            !mpi_was_initialized, false)

        if mpi.is_root && verbose
            @info "QGYBJModel runtime initialized" size=grid.size ranks=mpi.nprocs
        end
        MPI.Barrier(mpi.comm)
        return runtime
    catch
        if !mpi_was_initialized && MPI.Initialized() && !MPI.Finalized()
            GC.gc(true)
            MPI.Finalize()
        end
        rethrow()
    end
end

"""Allocate distributed model fields using a runtime's transform pencils."""
function allocate_fields(grid::RectilinearGrid, runtime::ModelRuntime;
    T::Type{<:AbstractFloat}=Float64)

    runtime.finalized && error("cannot allocate fields from a finalized runtime")
    grid.size == (runtime.computational_grid.nx,
                  runtime.computational_grid.ny,
                  runtime.computational_grid.nz) ||
        throw(ArgumentError("runtime geometry does not match the public grid"))
    return init_mpi_state(runtime.computational_grid, runtime.plans, runtime.mpi; T)
end

"""Release runtime-owned MPI resources without touching externally owned MPI."""
function finalize_runtime!(runtime::ModelRuntime; synchronize::Bool=true)
    runtime.finalized && return runtime

    if runtime.owns_mpi && MPI.Initialized() && !MPI.Finalized()
        synchronize && MPI.Barrier(runtime.mpi.comm)
        GC.gc(true)
        MPI.Finalize()
    end
    runtime.finalized = true
    return runtime
end
