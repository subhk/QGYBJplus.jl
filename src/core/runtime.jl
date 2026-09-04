"""Precomputed vertical coefficients used by model operators."""
struct OperatorCoefficients{T}
    "Pointwise N² sampled at cell centers (`grid.z`)."
    N²::Vector{T}
    "N² sampled at each cell's upper face (`grid.z_faces[2:end]`)."
    N²_face::Vector{T}
    "Flux coefficient f²/N² sampled at the same upper faces as `N²_face`."
    a_ell::Vector{T}
end

"""
    ModelRuntime

Ephemeral execution resources associated with a model: MPI ownership,
decomposition metadata, transform plans, operator coefficients, dealiasing
metadata, and reusable workspaces.
"""
mutable struct ModelRuntime{M, D, P, W, C, L, G}
    mpi::M
    decomposition::D
    plans::P
    workspace::W
    coefficients::C
    dealias_mask::L
    geometry::G
    owns_mpi::Bool
    finalized::Bool
end

_coriolis_frequency(coriolis::FPlane) = coriolis.f
_stratification_N²(stratification::ConstantStratification) = stratification.N²
_runtime_profile(stratification::ConstantStratification) =
    ConstantN(sqrt(float(stratification.N²)))
_runtime_profile(stratification::StratificationProfile) = stratification

"""Construct execution resources for `grid` transactionally."""
function build_runtime(grid::RectilinearGrid, physics::ModelPhysics,
    numerics::ModelNumerics; topology=nothing, parallel_io::Bool=false,
    verbose::Bool=true)

    mpi_was_initialized = MPI.Initialized()
    runtime = nothing
    try
        mpi = setup_mpi_environment(; topology, parallel_io)
        geometry = build_runtime_geometry(grid, mpi)
        plans = plan_distributed_transforms(geometry, mpi)
        workspace = allocate_distributed_workspace(geometry, mpi)

        profile = _runtime_profile(physics.stratification)
        N² = Float64.(compute_stratification_profile(profile, grid))
        N²_face = Float64.(_compute_stratification_face_profile(profile, grid))
        a_ell = a_ell_from_N2(N²_face, physics.coriolis)
        coefficients = OperatorCoefficients(N², N²_face, a_ell)
        mask = dealias_mask(grid)

        runtime = ModelRuntime(
            mpi, geometry.decomposition, plans, workspace,
            coefficients, mask, geometry,
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
    grid.size == (runtime.geometry.nx,
                  runtime.geometry.ny,
                  runtime.geometry.nz) ||
        throw(ArgumentError("runtime geometry does not match the public grid"))
    return allocate_distributed_fields(
        runtime.geometry, runtime.plans, runtime.mpi; T)
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

# Runtime-local layout and index mapping. Spectral layout is the default
# because prognostic model fields live on the FFT output pencil.
get_local_range_spectral(runtime::ModelRuntime) =
    get_local_range_spectral(runtime.plans)
get_local_range_physical(runtime::ModelRuntime) =
    get_local_range_physical(runtime.plans)
get_local_range(runtime::ModelRuntime) = get_local_range_spectral(runtime)
get_local_range(model::QGYBJModel) = get_local_range(model.runtime)
get_local_range_spectral(model::QGYBJModel) =
    get_local_range_spectral(model.runtime)
get_local_range_physical(model::QGYBJModel) =
    get_local_range_physical(model.runtime)

function local_to_global(local_index::Int, dimension::Int,
    runtime::ModelRuntime)
    return get_local_range_spectral(runtime)[dimension][local_index]
end

local_to_global(local_index::Int, dimension::Int, model::QGYBJModel) =
    local_to_global(local_index, dimension, model.runtime)

get_local_range_xy(runtime::ModelRuntime) = runtime.decomposition.local_range_xy
get_local_range_z(runtime::ModelRuntime) = runtime.decomposition.local_range_z
local_to_global_xy(local_index::Int, dimension::Int, runtime::ModelRuntime) =
    get_local_range_xy(runtime)[dimension][local_index]
local_to_global_z(local_index::Int, dimension::Int, runtime::ModelRuntime) =
    get_local_range_z(runtime)[dimension][local_index]

function z_is_local(runtime::ModelRuntime)
    return get_local_range(runtime)[1] == 1:runtime.decomposition.global_dims[1]
end
z_is_local(model::QGYBJModel) = z_is_local(model.runtime)

function get_kh2(i_local::Int, j_local::Int, k_local::Int, array,
    model::QGYBJModel)
    i_global = local_to_global(i_local, 2, array)
    j_global = local_to_global(j_local, 3, array)
    return model.grid.kh2[i_global, j_global]
end

get_kx(i_local::Int, array, model::QGYBJModel) =
    model.grid.kx[local_to_global(i_local, 2, array)]
get_ky(j_local::Int, array, model::QGYBJModel) =
    model.grid.ky[local_to_global(j_local, 3, array)]

transpose_to_z_pencil!(destination, source, runtime::ModelRuntime) =
    transpose_to_z_pencil!(destination, source, runtime.decomposition)
transpose_to_xy_pencil!(destination, source, runtime::ModelRuntime) =
    transpose_to_xy_pencil!(destination, source, runtime.decomposition)

allocate_fft_backward_dst(array, runtime::ModelRuntime) =
    allocate_fft_backward_dst(array, runtime.plans)

# Model-owned physical operator entry points use global geometry from the model
# and distributed spectral metadata from its runtime.
function _operator_context(model::QGYBJModel)
    runtime = model.runtime
    runtime.finalized && error("cannot use operators after model finalization")
    coefficients = runtime.coefficients
    return (
        fields=model.fields,
        grid=runtime.geometry,
        plans=runtime.plans,
        workspace=runtime.workspace,
        mask=runtime.dealias_mask,
        f=model.physics.coriolis.f,
        N2=coefficients.N²,
        N2_face=coefficients.N²_face,
        a=coefficients.a_ell,
    )
end

function Elliptic.invert_q_to_psi!(model::QGYBJModel)
    context = _operator_context(model)
    options = ETDModelOptions(model.physics, model.numerics)
    _invert_total_q_to_psi!(context.fields, context.grid, options,
        context.plans, context.a, context.mask;
        workspace=context.workspace)
    return model
end

function Elliptic.invert_B_to_A!(model::QGYBJModel)
    context = _operator_context(model)
    invert_B_to_A!(context.fields, context.grid, context.a;
        workspace=context.workspace)
    return model
end

function Elliptic.invert_helmholtz!(destination, rhs, model::QGYBJModel; kwargs...)
    context = _operator_context(model)
    invert_helmholtz!(destination, rhs, context.grid;
        workspace=context.workspace, kwargs...)
    return destination
end

function Operators.compute_velocities!(model::QGYBJModel;
    compute_w::Bool=true, use_ybj_w::Bool=false)
    context = _operator_context(model)
    compute_velocities!(context.fields, context.grid;
        plans=context.plans, f=context.f, N2=first(context.N2),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, use_ybj_w, workspace=context.workspace,
        dealias_mask=context.mask)
    return model
end

function Operators.compute_vertical_velocity!(model::QGYBJModel)
    context = _operator_context(model)
    compute_vertical_velocity!(context.fields, context.grid, context.plans;
        f=context.f, N2=first(context.N2), N2_profile=context.N2,
        workspace=context.workspace, dealias_mask=context.mask)
    return model
end


function Operators.compute_ybj_vertical_velocity!(model::QGYBJModel;
    skip_inversion::Bool=false, t=nothing)
    context = _operator_context(model)
    compute_ybj_vertical_velocity!(context.fields, context.grid, context.plans;
        f=context.f, N2=first(context.N2), N2_profile=context.N2,
        N2_face_profile=context.N2_face,
        workspace=context.workspace, skip_inversion, t)
    return model
end

function Operators.compute_total_velocities!(model::QGYBJModel;
    compute_w::Bool=true, use_ybj_w::Bool=false,
    include_wave_velocity::Bool=true)
    context = _operator_context(model)
    compute_total_velocities!(context.fields, context.grid;
        plans=context.plans, f=context.f, N2=first(context.N2),
        N2_profile=context.N2, N2_face_profile=context.N2_face,
        compute_w, use_ybj_w, include_wave_velocity,
        workspace=context.workspace, dealias_mask=context.mask)
    return model
end

function Operators.compute_wave_velocities!(model::QGYBJModel;
    compute_w::Bool=true, include_wave_velocity::Bool=true)
    context = _operator_context(model)
    compute_wave_velocities!(context.fields, context.grid;
        plans=context.plans, f=context.f, N2=first(context.N2),
        N2_profile=context.N2, compute_w, include_wave_velocity,
        workspace=context.workspace)
    return model
end

"""Advance `model` by one model-owned exponential Runge-Kutta step."""
function step!(model::QGYBJModel, timestepper::ExponentialRungeKutta2)
    Δt = timestepper.Δt
    isfinite(Δt) && Δt > zero(Δt) ||
        throw(ArgumentError("Δt must be finite and positive (got $Δt)"))

    context = _operator_context(model)
    timestep_workspace = timestepper.workspace
    owner_changed = timestepper.workspace_owner !== nothing &&
                    timestepper.workspace_owner !== model.runtime
    if timestep_workspace === nothing || owner_changed
        timestep_workspace = ExponentialRungeKutta2Workspace(
            model.fields, context.plans; G=context.grid)
    elseif !(timestep_workspace isa ExponentialRungeKutta2Workspace)
        throw(ArgumentError("timestepper workspace has an incompatible type"))
    elseif !_etdrk2_workspace_matches(timestep_workspace, model.fields)
        timestep_workspace = ExponentialRungeKutta2Workspace(
            model.fields, context.plans; G=context.grid)
    end
    timestepper.workspace = timestep_workspace
    timestep_workspace isa ExponentialRungeKutta2Workspace ||
        throw(ArgumentError("timestepper workspace has an incompatible type"))
    timestepper.workspace_owner = model.runtime

    options = ETDModelOptions(model.physics, model.numerics)
    next_fields = timestep_workspace.next
    _advance_etdrk2!(next_fields, model.fields, context.grid, options,
        context.plans;
        Δt,
        a=context.a,
        dealias_mask=context.mask,
        workspace=context.workspace,
        N2_profile=context.N2,
        N2_face_profile=context.N2_face,
        timestep_workspace=timestep_workspace)

    # Keep the public `model.fields` container stable for callbacks and cached
    # handles while rotating its double-buffered array storage at zero copy.
    _swap_field_storage!(model.fields, next_fields)
    return model
end
