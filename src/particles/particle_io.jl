"""
Particle I/O module for saving and loading particle trajectories.

Handles NetCDF output of particle trajectories with proper metadata.
Supports saving to a dedicated `particles/` directory with configurable
iteration-based intervals.

Key Features:
- ParticleOutputManager: Manages automatic saving at regular intervals
- Separate `particles/` directory for organized output
- Multiple output modes: snapshots, trajectories, streaming
- Support for parallel (MPI) particle tracking
- Automatic file creation and directory management
"""

module ParticleIO

using Dates
using NCDatasets
using MPI
using ..UnifiedParticleAdvection: ParticleTracker, ParticleState, ParticleConfig

# NCDatasets is always available since it's a required dependency (using NCDatasets above)
const HAS_NCDS = true

export write_particle_trajectories, read_particle_trajectories,
       write_particle_snapshot, create_particle_output_file,
       write_particle_trajectories_by_zlevel,
       # New exports for managed output
       ParticleOutputManager, setup_particle_output!,
       should_save_particles, save_particle_positions!,
       finalize_particle_output!

function _dim_length(ds, name::String)
    dim = ds.dim[name]
    return length(dim)
end

function _gatherv_vector(sendbuf, counts::Vector{Cint}, comm::MPI.Comm; root::Int=0)
    if MPI.Comm_rank(comm) == root
        recvbuf = similar(sendbuf, Int(sum(counts)))
        MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), comm; root=root)
        return recvbuf
    end
    MPI.Gatherv!(sendbuf, nothing, comm; root=root)
    return nothing
end

function _gather_particle_snapshot(tracker::ParticleTracker{T}) where T
    particles = tracker.particles
    if !tracker.is_parallel || tracker.comm === nothing
        return (
            ids = copy(particles.id),
            x = copy(particles.x),
            y = copy(particles.y),
            z = copy(particles.z),
            u = copy(particles.u),
            v = copy(particles.v),
            w = copy(particles.w)
        )
    end

    comm = tracker.comm
    root = 0
    counts = MPI.gather(particles.np, comm; root=root)
    counts_c = counts === nothing ? Cint[] : Cint.(counts)

    ids = _gatherv_vector(particles.id, counts_c, comm; root=root)
    x = _gatherv_vector(particles.x, counts_c, comm; root=root)
    y = _gatherv_vector(particles.y, counts_c, comm; root=root)
    z = _gatherv_vector(particles.z, counts_c, comm; root=root)
    u = _gatherv_vector(particles.u, counts_c, comm; root=root)
    v = _gatherv_vector(particles.v, counts_c, comm; root=root)
    w = _gatherv_vector(particles.w, counts_c, comm; root=root)

    if MPI.Comm_rank(comm) != root
        return nothing
    end

    perm = sortperm(ids)
    return (
        ids = ids[perm],
        x = x[perm],
        y = y[perm],
        z = z[perm],
        u = u[perm],
        v = v[perm],
        w = w[perm]
    )
end

function _gather_particle_positions(tracker::ParticleTracker{T},
                                    ids_local::Vector{Int},
                                    x_local::Vector{T},
                                    y_local::Vector{T},
                                    z_local::Vector{T}) where T
    if !tracker.is_parallel || tracker.comm === nothing
        return (
            ids = copy(ids_local),
            x = copy(x_local),
            y = copy(y_local),
            z = copy(z_local)
        )
    end

    comm = tracker.comm
    root = 0
    counts = MPI.gather(length(ids_local), comm; root=root)
    counts_c = counts === nothing ? Cint[] : Cint.(counts)

    ids = _gatherv_vector(ids_local, counts_c, comm; root=root)
    x = _gatherv_vector(x_local, counts_c, comm; root=root)
    y = _gatherv_vector(y_local, counts_c, comm; root=root)
    z = _gatherv_vector(z_local, counts_c, comm; root=root)

    if MPI.Comm_rank(comm) != root
        return nothing
    end

    perm = sortperm(ids)
    return (
        ids = ids[perm],
        x = x[perm],
        y = y[perm],
        z = z[perm]
    )
end

#=
================================================================================
                    PARTICLE OUTPUT MANAGER
================================================================================
Manages automatic particle position saving at regular iteration intervals.
Creates a dedicated `particles/` subdirectory for organized output.
================================================================================
=#

"""
    ParticleOutputManager{T}

Manages particle output to NetCDF files in a dedicated `particles/` directory.

# Features
- Automatic directory creation (`output_dir/particles/`)
- Configurable save interval (by iteration or time)
- Multiple output modes: snapshots, trajectories, or streaming
- Handles both serial and parallel execution
- Accumulates trajectory data in memory or streams to disk

# Fields
- `output_dir`: Base output directory
- `particle_dir`: Particle output subdirectory (`output_dir/particles/`)
- `save_interval_iter`: Save every N iterations (0 = disabled)
- `save_interval_time`: Save every T time units (0.0 = disabled)
- `output_mode`: `:snapshots`, `:trajectory`, or `:streaming`
- `file_prefix`: Prefix for output files (default: "particles")
- `current_file`: Path to current output file (for streaming mode)
- `save_count`: Number of saves performed
- `last_save_iter`: Last iteration when particles were saved
- `last_save_time`: Last time when particles were saved
- `time_series`: Accumulated time values
- `x_series`, `y_series`, `z_series`: Accumulated position data
- `u_series`, `v_series`, `w_series`: Accumulated velocity data
- `initialized`: Whether manager has been initialized
"""
mutable struct ParticleOutputManager{T<:AbstractFloat}
    # Directory configuration
    output_dir::String
    particle_dir::String

    # Save interval configuration
    save_interval_iter::Int          # Save every N iterations (0 = use time)
    save_interval_time::T            # Save every T time units (0 = use iterations)

    # Output mode
    output_mode::Symbol              # :snapshots, :trajectory, :streaming
    file_prefix::String              # Prefix for output files

    # Current state
    current_file::String             # Current output file path
    save_count::Int                  # Number of saves performed
    last_save_iter::Int              # Last iteration saved
    last_save_time::T                # Last time saved

    # Accumulated data (for trajectory mode)
    time_series::Vector{T}
    x_series::Vector{Vector{T}}
    y_series::Vector{Vector{T}}
    z_series::Vector{Vector{T}}
    u_series::Vector{Vector{T}}
    v_series::Vector{Vector{T}}
    w_series::Vector{Vector{T}}
    particle_ids::Vector{Int}

    # State flags
    initialized::Bool
    is_io_rank::Bool                 # Only rank 0 writes in parallel

    function ParticleOutputManager{T}(output_dir::String;
                                      save_interval_iter::Int=0,
                                      save_interval_time::T=T(0.1),
                                      output_mode::Symbol=:trajectory,
                                      file_prefix::String="particles") where T
        # Create particles subdirectory
        particle_dir = joinpath(output_dir, "particles")

        new{T}(
            output_dir,
            particle_dir,
            save_interval_iter,
            save_interval_time,
            output_mode,
            file_prefix,
            "",                      # current_file (set during initialization)
            0,                       # save_count
            0,                       # last_save_iter
            T(0),                    # last_save_time
            T[],                     # time_series
            Vector{T}[],             # x_series
            Vector{T}[],             # y_series
            Vector{T}[],             # z_series
            Vector{T}[],             # u_series
            Vector{T}[],             # v_series
            Vector{T}[],             # w_series
            Int[],                   # particle_ids
            false,                   # initialized
            true                     # is_io_rank (default true for serial)
        )
    end
end

# Convenience constructor
ParticleOutputManager(output_dir::String; kwargs...) =
    ParticleOutputManager{Float64}(output_dir; kwargs...)

"""
    setup_particle_output!(manager, tracker; rank=0)

Initialize the particle output manager and create the output directory.

# Arguments
- `manager`: ParticleOutputManager instance
- `tracker`: ParticleTracker instance
- `rank`: MPI rank (default: 0, only rank 0 does I/O)
"""
function setup_particle_output!(manager::ParticleOutputManager{T},
                                tracker::ParticleTracker{T};
                                rank::Int=0) where T

    manager.is_io_rank = (rank == 0)
    np_global = tracker.is_parallel && tracker.comm !== nothing ?
                MPI.Allreduce(tracker.particles.np, +, tracker.comm) :
                tracker.particles.np

    if manager.is_io_rank
        # Create particles directory
        if !isdir(manager.particle_dir)
            mkpath(manager.particle_dir)
            println("Created particle output directory: $(manager.particle_dir)")
        end

        # Set up initial file based on output mode
        if manager.output_mode == :streaming
            # Create streaming file with unlimited time dimension
            manager.current_file = joinpath(manager.particle_dir,
                                           "$(manager.file_prefix)_stream.nc")
            create_streaming_particle_file!(manager, tracker; np_global=np_global)
        elseif manager.output_mode == :trajectory
            # Trajectory mode accumulates in memory, writes at end
            manager.current_file = joinpath(manager.particle_dir,
                                           "$(manager.file_prefix)_trajectory.nc")
        end
        # Snapshot mode creates individual files per save
    end

    manager.initialized = true
    return manager
end

"""
    should_save_particles(manager, iteration, time) -> Bool

Check if particles should be saved at current iteration/time.
"""
function should_save_particles(manager::ParticleOutputManager{T},
                               iteration::Int, time::T) where T
    if !manager.initialized
        return false
    end

    # Check iteration-based interval
    if manager.save_interval_iter > 0
        return (iteration - manager.last_save_iter) >= manager.save_interval_iter
    end

    # Check time-based interval
    if manager.save_interval_time > 0
        return (time - manager.last_save_time) >= manager.save_interval_time
    end

    return false
end

"""
    save_particle_positions!(manager, tracker, iteration, time)

Save current particle positions based on the output mode.

# Arguments
- `manager`: ParticleOutputManager instance
- `tracker`: ParticleTracker with current particle positions
- `iteration`: Current simulation iteration
- `time`: Current simulation time
"""
function save_particle_positions!(manager::ParticleOutputManager{T},
                                  tracker::ParticleTracker{T},
                                  iteration::Int, time::T) where T
    data = _gather_particle_snapshot(tracker)
    if !manager.is_io_rank
        return manager
    end
    if data === nothing
        return manager
    end

    if isempty(manager.particle_ids)
        manager.particle_ids = data.ids
    elseif manager.particle_ids != data.ids
        @warn "Particle ID ordering changed between saves; updating output ordering"
        manager.particle_ids = data.ids
    end

    if manager.output_mode == :snapshots
        # Save individual snapshot file
        save_particle_snapshot!(manager, tracker, iteration, time; data=data)

    elseif manager.output_mode == :trajectory
        # Accumulate in memory
        push!(manager.time_series, time)
        push!(manager.x_series, data.x)
        push!(manager.y_series, data.y)
        push!(manager.z_series, data.z)
        push!(manager.u_series, data.u)
        push!(manager.v_series, data.v)
        push!(manager.w_series, data.w)

    elseif manager.output_mode == :streaming
        # Append to streaming file
        append_to_streaming_file!(manager, tracker, time; data=data)
    end

    # Update counters
    manager.save_count += 1
    manager.last_save_iter = iteration
    manager.last_save_time = time

    return manager
end

"""
    save_particle_snapshot!(manager, tracker, iteration, time)

Save a single snapshot file with current particle positions.
"""
function save_particle_snapshot!(manager::ParticleOutputManager{T},
                                 tracker::ParticleTracker{T},
                                 iteration::Int, time::T;
                                 data=nothing) where T
    if data === nothing
        particles = tracker.particles
        ids = particles.id
        x = particles.x
        y = particles.y
        z = particles.z
        u = particles.u
        v = particles.v
        w = particles.w
    else
        ids = data.ids
        x = data.x
        y = data.y
        z = data.z
        u = data.u
        v = data.v
        w = data.w
    end

    # Create filename with iteration number
    filename = joinpath(manager.particle_dir,
                       "$(manager.file_prefix)_$(lpad(iteration, 6, '0')).nc")

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = length(ids)

            # Create variables
            defVar_particle = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            defVar_x = NCDatasets.defVar(ds, "x", Float64, ("particle",))
            defVar_y = NCDatasets.defVar(ds, "y", Float64, ("particle",))
            defVar_z = NCDatasets.defVar(ds, "z", Float64, ("particle",))
            defVar_u = NCDatasets.defVar(ds, "u", Float64, ("particle",))
            defVar_v = NCDatasets.defVar(ds, "v", Float64, ("particle",))
            defVar_w = NCDatasets.defVar(ds, "w", Float64, ("particle",))

            # Fill data
            defVar_particle[:] = ids
            defVar_x[:] = x
            defVar_y[:] = y
            defVar_z[:] = z
            defVar_u[:] = u
            defVar_v[:] = v
            defVar_w[:] = w

            # Set variable attributes
            defVar_x.attrib["units"] = "nondimensional"
            defVar_x.attrib["long_name"] = "x position"
            defVar_y.attrib["units"] = "nondimensional"
            defVar_y.attrib["long_name"] = "y position"
            defVar_z.attrib["units"] = "nondimensional"
            defVar_z.attrib["long_name"] = "z position"
            defVar_u.attrib["units"] = "nondimensional/time"
            defVar_u.attrib["long_name"] = "x velocity"
            defVar_v.attrib["units"] = "nondimensional/time"
            defVar_v.attrib["long_name"] = "y velocity"
            defVar_w.attrib["units"] = "nondimensional/time"
            defVar_w.attrib["long_name"] = "z velocity"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Snapshot"
            ds.attrib["created_at"] = string(now())
            ds.attrib["time"] = time
            ds.attrib["iteration"] = iteration
            ds.attrib["number_of_particles"] = length(ids)
            ds.attrib["integration_method"] = string(tracker.config.integration_method)
            ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
            ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection
        end

    catch e
        @warn "Cannot write particle snapshot: $e"
    end

    return filename
end

"""
    create_streaming_particle_file!(manager, tracker)

Create NetCDF file for streaming particle output with unlimited time dimension.
"""
function create_streaming_particle_file!(manager::ParticleOutputManager{T},
                                         tracker::ParticleTracker{T};
                                         np_global::Union{Nothing,Int}=nothing) where T
    particles = tracker.particles
    filename = manager.current_file
    if np_global === nothing
        np_global = particles.np
    end

    if isempty(manager.particle_ids)
        manager.particle_ids = collect(1:np_global)
    end

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions (time is unlimited)
            ds.dim["particle"] = np_global
            ds.dim["time"] = NCDatasets.Unlimited()

            # Create coordinate variables
            defVar_particle = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            defVar_time = NCDatasets.defVar(ds, "time", Float64, ("time",))

            # Create trajectory variables
            defVar_x = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
            defVar_y = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
            defVar_z = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
            defVar_u = NCDatasets.defVar(ds, "u", Float64, ("particle", "time"))
            defVar_v = NCDatasets.defVar(ds, "v", Float64, ("particle", "time"))
            defVar_w = NCDatasets.defVar(ds, "w", Float64, ("particle", "time"))

            # Fill particle IDs
            defVar_particle[:] = manager.particle_ids

            # Set variable attributes
            defVar_particle.attrib["long_name"] = "particle identifier"
            defVar_time.attrib["units"] = "model time units"
            defVar_time.attrib["long_name"] = "time"

            defVar_x.attrib["units"] = "nondimensional"
            defVar_x.attrib["long_name"] = "x position"
            defVar_y.attrib["units"] = "nondimensional"
            defVar_y.attrib["long_name"] = "y position"
            defVar_z.attrib["units"] = "nondimensional"
            defVar_z.attrib["long_name"] = "z position"
            defVar_u.attrib["units"] = "nondimensional/time"
            defVar_u.attrib["long_name"] = "x velocity"
            defVar_v.attrib["units"] = "nondimensional/time"
            defVar_v.attrib["long_name"] = "y velocity"
            defVar_w.attrib["units"] = "nondimensional/time"
            defVar_w.attrib["long_name"] = "z velocity"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Trajectories (Streaming)"
            ds.attrib["created_at"] = string(now())
            ds.attrib["number_of_particles"] = np_global
            ds.attrib["output_mode"] = "streaming"
            ds.attrib["integration_method"] = string(tracker.config.integration_method)
            ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
            ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection
        end

        println("Created streaming particle file: $filename")

    catch e
        @warn "Cannot create streaming particle file: $e"
    end

    return filename
end

"""
    append_to_streaming_file!(manager, tracker, time)

Append current particle positions to streaming NetCDF file.
"""
function append_to_streaming_file!(manager::ParticleOutputManager{T},
                                   tracker::ParticleTracker{T}, time::T;
                                   data=nothing) where T
    if data === nothing
        particles = tracker.particles
        ids = particles.id
        x = particles.x
        y = particles.y
        z = particles.z
        u = particles.u
        v = particles.v
        w = particles.w
    else
        ids = data.ids
        x = data.x
        y = data.y
        z = data.z
        u = data.u
        v = data.v
        w = data.w
    end
    if isempty(manager.particle_ids)
        manager.particle_ids = ids
    end
    filename = manager.current_file

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "a") do ds
            # Get current time index
            current_size = length(ds["time"])
            new_idx = current_size + 1

            # Append data
            ds["time"][new_idx] = time
            if haskey(ds, "particle")
                ds["particle"][:] = manager.particle_ids
            end
            ds["x"][:, new_idx] = x
            ds["y"][:, new_idx] = y
            ds["z"][:, new_idx] = z
            ds["u"][:, new_idx] = u
            ds["v"][:, new_idx] = v
            ds["w"][:, new_idx] = w
        end

    catch e
        @warn "Cannot append to streaming file: $e"
    end

    return manager
end

"""
    finalize_particle_output!(manager, tracker; metadata=Dict())

Finalize particle output and write any accumulated data.

For trajectory mode, writes all accumulated positions to the output file.
For streaming mode, updates file metadata.
For snapshot mode, writes a summary file.
"""
function finalize_particle_output!(manager::ParticleOutputManager{T},
                                   tracker::ParticleTracker{T};
                                   metadata::Dict=Dict()) where T
    if !manager.is_io_rank || !manager.initialized
        return manager
    end

    println("\nFinalizing particle output...")
    println("  Output directory: $(manager.particle_dir)")
    println("  Output mode: $(manager.output_mode)")
    println("  Total saves: $(manager.save_count)")

    if manager.output_mode == :trajectory && !isempty(manager.time_series)
        # Write accumulated trajectory data
        write_accumulated_trajectories!(manager, tracker; metadata=metadata)

    elseif manager.output_mode == :streaming
        # Update streaming file metadata
        update_streaming_metadata!(manager)

    elseif manager.output_mode == :snapshots
        # Write summary file for snapshots
        write_snapshot_summary!(manager, tracker)
    end

    println("Particle output finalized successfully")

    return manager
end

"""
    write_accumulated_trajectories!(manager, tracker; metadata=Dict())

Write accumulated trajectory data to NetCDF file.
"""
function write_accumulated_trajectories!(manager::ParticleOutputManager{T},
                                         tracker::ParticleTracker{T};
                                         metadata::Dict=Dict()) where T
    filename = manager.current_file
    np = length(manager.x_series[1])
    nt = length(manager.time_series)
    ids = isempty(manager.particle_ids) ? collect(1:np) : manager.particle_ids
    if length(ids) != np
        @warn "Particle ID length ($(length(ids))) does not match trajectory data size ($np); using sequential IDs"
        ids = collect(1:np)
    end

    println("  Writing $(nt) time points for $(np) particles...")

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = np
            ds.dim["time"] = nt

            # Create coordinate variables
            defVar_particle = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            defVar_time = NCDatasets.defVar(ds, "time", Float64, ("time",))

            # Create trajectory variables
            defVar_x = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
            defVar_y = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
            defVar_z = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
            defVar_u = NCDatasets.defVar(ds, "u", Float64, ("particle", "time"))
            defVar_v = NCDatasets.defVar(ds, "v", Float64, ("particle", "time"))
            defVar_w = NCDatasets.defVar(ds, "w", Float64, ("particle", "time"))

            # Fill coordinate data
            defVar_particle[:] = ids
            defVar_time[:] = manager.time_series

            # Fill trajectory data
            for (t_idx, _) in enumerate(manager.time_series)
                defVar_x[:, t_idx] = manager.x_series[t_idx]
                defVar_y[:, t_idx] = manager.y_series[t_idx]
                defVar_z[:, t_idx] = manager.z_series[t_idx]
                defVar_u[:, t_idx] = manager.u_series[t_idx]
                defVar_v[:, t_idx] = manager.v_series[t_idx]
                defVar_w[:, t_idx] = manager.w_series[t_idx]
            end

            # Set variable attributes
            defVar_particle.attrib["long_name"] = "particle identifier"
            defVar_time.attrib["units"] = "model time units"
            defVar_time.attrib["long_name"] = "time"

            defVar_x.attrib["units"] = "nondimensional"
            defVar_x.attrib["long_name"] = "x position"
            defVar_y.attrib["units"] = "nondimensional"
            defVar_y.attrib["long_name"] = "y position"
            defVar_z.attrib["units"] = "nondimensional"
            defVar_z.attrib["long_name"] = "z position"
            defVar_u.attrib["units"] = "nondimensional/time"
            defVar_u.attrib["long_name"] = "x velocity"
            defVar_v.attrib["units"] = "nondimensional/time"
            defVar_v.attrib["long_name"] = "y velocity"
            defVar_w.attrib["units"] = "nondimensional/time"
            defVar_w.attrib["long_name"] = "z velocity"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Trajectories"
            ds.attrib["created_at"] = string(now())
            ds.attrib["number_of_particles"] = np
            ds.attrib["number_of_timesteps"] = nt
            ds.attrib["start_time"] = manager.time_series[1]
            ds.attrib["end_time"] = manager.time_series[end]
            ds.attrib["output_mode"] = "trajectory"
            ds.attrib["integration_method"] = string(tracker.config.integration_method)
            ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
            ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection

            # Add user metadata
            for (key, value) in metadata
                ds.attrib[string(key)] = value
            end
        end

        println("  Trajectory file written: $filename")

    catch e
        @warn "Cannot write accumulated trajectories: $e"
    end

    return manager
end

"""
    update_streaming_metadata!(manager)

Update metadata in streaming file after simulation.
"""
function update_streaming_metadata!(manager::ParticleOutputManager)
    filename = manager.current_file

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "a") do ds
            nt = length(ds["time"])
            times = Array(ds["time"][:])

            ds.attrib["finalized_at"] = string(now())
            ds.attrib["number_of_timesteps"] = nt
            if nt > 0
                ds.attrib["start_time"] = times[1]
                ds.attrib["end_time"] = times[end]
            end
        end

        println("  Streaming file metadata updated: $filename")

    catch e
        @warn "Cannot update streaming metadata: $e"
    end

    return manager
end

"""
    write_snapshot_summary!(manager, tracker)

Write summary file listing all snapshot files.
"""
function write_snapshot_summary!(manager::ParticleOutputManager, tracker::ParticleTracker)
    summary_file = joinpath(manager.particle_dir, "$(manager.file_prefix)_summary.txt")
    np_global = tracker.is_parallel && tracker.comm !== nothing ?
                MPI.Allreduce(tracker.particles.np, +, tracker.comm) :
                tracker.particles.np

    try
        open(summary_file, "w") do f
            println(f, "QG-YBJ Particle Snapshot Summary")
            println(f, "================================")
            println(f, "Created: $(now())")
            println(f, "Number of particles: $np_global")
            println(f, "Number of snapshots: $(manager.save_count)")
            println(f, "Output directory: $(manager.particle_dir)")
            println(f, "")
            println(f, "Integration method: $(tracker.config.integration_method)")
            println(f, "Use YBJ w: $(tracker.config.use_ybj_w)")
            println(f, "Use 3D advection: $(tracker.config.use_3d_advection)")
            println(f, "")
            println(f, "Files:")

            # List snapshot files
            files = filter(f -> startswith(f, manager.file_prefix) && endswith(f, ".nc"),
                          readdir(manager.particle_dir))
            for file in sort(files)
                println(f, "  $file")
            end
        end

        println("  Snapshot summary written: $summary_file")

    catch e
        @warn "Cannot write snapshot summary: $e"
    end

    return manager
end

#=
================================================================================
                    ORIGINAL FUNCTIONS (preserved for compatibility)
================================================================================
=#

"""
    write_particle_trajectories(filename, tracker; metadata=Dict())

Write complete particle trajectory history to NetCDF file.
"""
function write_particle_trajectories(filename::String, tracker::ParticleTracker;
                                    metadata::Dict=Dict())
    if tracker.is_parallel && tracker.comm !== nothing
        return _write_particle_trajectories_parallel(filename, tracker; metadata=metadata)
    end
    return _write_particle_trajectories(filename, tracker.particles, tracker.config; metadata=metadata)
end

function write_particle_trajectories(filename::String, tracker::NamedTuple;
                                    metadata::Dict=Dict())
    particles = getproperty(tracker, :particles)
    config = getproperty(tracker, :config)
    return _write_particle_trajectories(filename, particles, config; metadata=metadata)
end

function _write_particle_trajectories(filename::String, particles::ParticleState, config::ParticleConfig;
                                     metadata::Dict=Dict())
    if isempty(particles.time_history)
        @warn "No particle history to save"
        return filename
    end
    ids = isempty(particles.id) ? collect(1:particles.np) : particles.id
    return _write_particle_trajectories_arrays(
        filename,
        ids,
        particles.time_history,
        particles.x_history,
        particles.y_history,
        particles.z_history,
        config;
        metadata=metadata
    )
end

function _write_particle_trajectories_parallel(filename::String, tracker::ParticleTracker{T};
                                              metadata::Dict=Dict()) where T
    particles = tracker.particles
    if isempty(particles.time_history)
        @warn "No particle history to save"
        return filename
    end
    if length(particles.id_history) != length(particles.time_history)
        @warn "Particle id history is missing; global trajectory reconstruction may be incorrect"
    end

    ntime = length(particles.time_history)
    x_series = Vector{Vector{T}}(undef, ntime)
    y_series = Vector{Vector{T}}(undef, ntime)
    z_series = Vector{Vector{T}}(undef, ntime)
    ids_global = nothing

    for t_idx in 1:ntime
        ids_local = t_idx <= length(particles.id_history) ? particles.id_history[t_idx] : particles.id
        data = _gather_particle_positions(tracker, ids_local,
                                          particles.x_history[t_idx],
                                          particles.y_history[t_idx],
                                          particles.z_history[t_idx])
        if tracker.rank == 0
            if ids_global === nothing
                ids_global = data.ids
            elseif ids_global != data.ids
                @warn "Particle ID ordering changed at time index $t_idx; updating output ordering"
                ids_global = data.ids
            end
            x_series[t_idx] = data.x
            y_series[t_idx] = data.y
            z_series[t_idx] = data.z
        end
    end

    if tracker.rank != 0
        return filename
    end

    return _write_particle_trajectories_arrays(
        filename,
        ids_global === nothing ? Int[] : ids_global,
        particles.time_history,
        x_series,
        y_series,
        z_series,
        tracker.config;
        metadata=metadata
    )
end

function _write_particle_trajectories_arrays(filename::String,
                                             ids::Vector{Int},
                                             time_series::Vector,
                                             x_series::Vector,
                                             y_series::Vector,
                                             z_series::Vector,
                                             config::ParticleConfig;
                                             metadata::Dict=Dict())
    np = length(ids)
    nt = length(time_series)

    # Check if NCDatasets is available
    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = np
            ds.dim["time"] = nt

            # Create coordinate variables
            particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))

            # Create trajectory variables
            x_var = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))

            # Fill coordinate data
            particle_var[:] = ids
            time_var[:] = time_series

            # Fill trajectory data
            for (t_idx, _) in enumerate(time_series)
                x_var[:, t_idx] = x_series[t_idx]
                y_var[:, t_idx] = y_series[t_idx]
                z_var[:, t_idx] = z_series[t_idx]
            end

            # Set attributes
            particle_var.attrib["long_name"] = "particle identifier"
            time_var.attrib["units"] = "model time units"
            time_var.attrib["long_name"] = "time"

            x_var.attrib["units"] = "nondimensional"
            x_var.attrib["long_name"] = "x position (horizontal)"
            y_var.attrib["units"] = "nondimensional"
            y_var.attrib["long_name"] = "y position (horizontal)"
            z_var.attrib["units"] = "nondimensional"
            z_var.attrib["long_name"] = "z position (vertical)"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Trajectories"
            ds.attrib["created_at"] = string(now())
            ds.attrib["number_of_particles"] = np
            ds.attrib["integration_method"] = string(config.integration_method)
            ds.attrib["use_ybj_w"] = config.use_ybj_w
            ds.attrib["use_3d_advection"] = config.use_3d_advection
            ds.attrib["periodic_x"] = config.periodic_x
            ds.attrib["periodic_y"] = config.periodic_y
            ds.attrib["reflect_z"] = config.reflect_z

            # Add user metadata
            for (key, value) in metadata
                ds.attrib[string(key)] = value
            end
        end

        @info "Particle trajectories written to: $filename"

    catch e
        @warn "NCDatasets not available, cannot write particle trajectories: $e"
    end

    return filename
end

"""
    read_particle_trajectories(filename) -> NamedTuple

Read particle trajectory history from NetCDF file.

Returns a NamedTuple with fields:
- `x`: Matrix of x positions (np × ntime)
- `y`: Matrix of y positions (np × ntime)
- `z`: Matrix of z positions (np × ntime)
- `time`: Vector of time values (ntime)
- `particle_ids`: Vector of particle identifiers (np)
- `attributes`: Dict of global attributes from the file

This is the inverse of `write_particle_trajectories`.

# Example
```julia
# Write trajectories
write_particle_trajectories("particles.nc", tracker)

# Read them back
traj = read_particle_trajectories("particles.nc")
println("Number of particles: ", size(traj.x, 1))
println("Number of time steps: ", length(traj.time))
println("Initial x positions: ", traj.x[:, 1])
```
"""
function read_particle_trajectories(filename::String)
    try
        HAS_NCDS || error("NCDatasets not available")

        x = nothing
        y = nothing
        z = nothing
        time = nothing
        particle_ids = nothing
        attributes = Dict{String,Any}()

        NCDatasets.Dataset(filename, "r") do ds
            # Read dimensions
            np = _dim_length(ds, "particle")
            ntime = _dim_length(ds, "time")

            # Read coordinate variables
            if haskey(ds, "particle")
                particle_ids = Array(ds["particle"][:])
            else
                particle_ids = collect(1:np)
            end

            if haskey(ds, "time")
                time = Array(ds["time"][:])
            else
                time = collect(1:ntime)
            end

            # Read trajectory variables
            if haskey(ds, "x")
                x = Array(ds["x"][:, :])
            else
                error("Variable 'x' not found in $filename")
            end

            if haskey(ds, "y")
                y = Array(ds["y"][:, :])
            else
                error("Variable 'y' not found in $filename")
            end

            if haskey(ds, "z")
                z = Array(ds["z"][:, :])
            else
                error("Variable 'z' not found in $filename")
            end

            # Read global attributes
            for (name, value) in ds.attrib
                attributes[name] = value
            end
        end

        @info "Read particle trajectories from: $filename ($(size(x, 1)) particles, $(length(time)) time steps)"

        return (
            x = x,
            y = y,
            z = z,
            time = time,
            particle_ids = particle_ids,
            attributes = attributes
        )

    catch e
        @warn "Cannot read particle trajectories: $e"
        rethrow(e)
    end
end

"""
    write_particle_snapshot(filename, tracker, time)

Write current particle positions to NetCDF file (single time snapshot).
"""
function write_particle_snapshot(filename::String, tracker::ParticleTracker, time::Real)
    data = _gather_particle_snapshot(tracker)
    if tracker.is_parallel && tracker.comm !== nothing && tracker.rank != 0
        return filename
    end
    if data === nothing
        data = (
            ids = tracker.particles.id,
            x = tracker.particles.x,
            y = tracker.particles.y,
            z = tracker.particles.z,
            u = tracker.particles.u,
            v = tracker.particles.v,
            w = tracker.particles.w
        )
    end

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = length(data.ids)

            # Create variables
            particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            x_var = NCDatasets.defVar(ds, "x", Float64, ("particle",))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("particle",))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("particle",))
            u_var = NCDatasets.defVar(ds, "u", Float64, ("particle",))
            v_var = NCDatasets.defVar(ds, "v", Float64, ("particle",))
            w_var = NCDatasets.defVar(ds, "w", Float64, ("particle",))

            # Fill data
            particle_var[:] = data.ids
            x_var[:] = data.x
            y_var[:] = data.y
            z_var[:] = data.z
            u_var[:] = data.u
            v_var[:] = data.v
            w_var[:] = data.w

            # Set attributes
            particle_var.attrib["long_name"] = "particle identifier"

            x_var.attrib["units"] = "nondimensional"
            x_var.attrib["long_name"] = "x position (horizontal)"
            y_var.attrib["units"] = "nondimensional"
            y_var.attrib["long_name"] = "y position (horizontal)"
            z_var.attrib["units"] = "nondimensional"
            z_var.attrib["long_name"] = "z position (vertical)"

            u_var.attrib["units"] = "nondimensional/time"
            u_var.attrib["long_name"] = "x velocity"
            v_var.attrib["units"] = "nondimensional/time"
            v_var.attrib["long_name"] = "y velocity"
            w_var.attrib["units"] = "nondimensional/time"
            w_var.attrib["long_name"] = "z velocity"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Snapshot"
            ds.attrib["created_at"] = string(now())
            ds.attrib["time"] = time
            ds.attrib["number_of_particles"] = length(data.ids)
            ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
            ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection
        end

    catch e
        @warn "Cannot write particle snapshot: $e"
    end

    return filename
end

"""
    create_particle_output_file(filename, tracker; append_mode=false)

Create NetCDF file for time series particle output.
"""
function create_particle_output_file(filename::String, tracker::ParticleTracker;
                                    append_mode::Bool=false)
    particles = tracker.particles
    mode = append_mode ? "a" : "c"
    np_global = tracker.is_parallel && tracker.comm !== nothing ?
                MPI.Allreduce(particles.np, +, tracker.comm) :
                particles.np
    if tracker.is_parallel && tracker.comm !== nothing && tracker.rank != 0
        return filename
    end

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, mode) do ds
            if !append_mode
                # Define dimensions
                ds.dim["particle"] = np_global
                ds.dim["time"] = NCDatasets.Unlimited()

                # Create coordinate variables
                particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
                time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))

                # Create trajectory variables
                x_var = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
                y_var = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
                z_var = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
                u_var = NCDatasets.defVar(ds, "u", Float64, ("particle", "time"))
                v_var = NCDatasets.defVar(ds, "v", Float64, ("particle", "time"))
                w_var = NCDatasets.defVar(ds, "w", Float64, ("particle", "time"))

                # Fill particle IDs
                particle_var[:] = collect(1:np_global)

                # Set attributes
                particle_var.attrib["long_name"] = "particle identifier"
                time_var.attrib["units"] = "model time units"
                time_var.attrib["long_name"] = "time"

                x_var.attrib["units"] = "nondimensional"
                x_var.attrib["long_name"] = "x position (horizontal)"
                y_var.attrib["units"] = "nondimensional"
                y_var.attrib["long_name"] = "y position (horizontal)"
                z_var.attrib["units"] = "nondimensional"
                z_var.attrib["long_name"] = "z position (vertical)"

                u_var.attrib["units"] = "nondimensional/time"
                u_var.attrib["long_name"] = "x velocity"
                v_var.attrib["units"] = "nondimensional/time"
                v_var.attrib["long_name"] = "y velocity"
                w_var.attrib["units"] = "nondimensional/time"
                w_var.attrib["long_name"] = "z velocity"

                # Global attributes
                ds.attrib["title"] = "QG-YBJ Particle Trajectories (Time Series)"
                ds.attrib["created_at"] = string(now())
                ds.attrib["number_of_particles"] = np_global
                ds.attrib["integration_method"] = string(tracker.config.integration_method)
                ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
                ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection
            end
        end

    catch e
        @warn "Cannot create particle output file: $e"
    end

    return filename
end

"""
    append_particle_data!(filename, tracker, time_index)

Append current particle state to existing NetCDF file.
"""
function append_particle_data!(filename::String, tracker::ParticleTracker, time_index::Int)
    data = _gather_particle_snapshot(tracker)
    if tracker.is_parallel && tracker.comm !== nothing && tracker.rank != 0
        return filename
    end
    if data === nothing
        data = (
            ids = tracker.particles.id,
            x = tracker.particles.x,
            y = tracker.particles.y,
            z = tracker.particles.z,
            u = tracker.particles.u,
            v = tracker.particles.v,
            w = tracker.particles.w
        )
    end

    try
        HAS_NCDS || error("NCDatasets not available")

        NCDatasets.Dataset(filename, "a") do ds
            # Get current time dimension size
            current_time_size = length(ds["time"])
            new_time_index = time_index > 0 ? time_index : current_time_size + 1
            if new_time_index <= current_time_size
                new_time_index = current_time_size + 1
            end

            # Append time
            ds["time"][new_time_index] = tracker.particles.time

            # Append particle data
            if haskey(ds, "particle")
                ds["particle"][:] = data.ids
            end
            ds["x"][:, new_time_index] = data.x
            ds["y"][:, new_time_index] = data.y
            ds["z"][:, new_time_index] = data.z
            ds["u"][:, new_time_index] = data.u
            ds["v"][:, new_time_index] = data.v
            ds["w"][:, new_time_index] = data.w
        end

    catch e
        @warn "Cannot append particle data: $e"
    end

    return filename
end

"""
    write_particle_trajectories_by_zlevel(base_filename, tracker;
                                          z_tolerance=1e-6, metadata=Dict())

Write particle trajectories to separate files based on z-level.

This function groups particles by their initial z-level and saves each group
to a separate NetCDF file. Useful for analyzing particles initialized at
different depths independently.

Parameters:
- base_filename: Base name for output files (e.g., "particles" -> "particles_z1.23.nc")
- tracker: ParticleTracker instance with trajectory history
- z_tolerance: Tolerance for grouping particles by z-level (default: 1e-6)
- metadata: Additional metadata to include in all files

Returns: Dictionary mapping z-levels to filenames

Example:
```julia
# Initialize particles at multiple z-levels
config = create_layered_distribution(0.0, 2π, 0.0, 2π, [π/4, π/2, 3π/4], 4, 4)
tracker = ParticleTracker(config, grid, parallel_config)

# Run simulation...

# Save each z-level to separate file
files = write_particle_trajectories_by_zlevel("particles", tracker)
# Creates: particles_z0.785.nc, particles_z1.571.nc, particles_z2.356.nc
```
"""
function write_particle_trajectories_by_zlevel(base_filename::String, tracker::ParticleTracker;
                                              z_tolerance::Float64=1e-6,
                                              metadata::Dict=Dict())
    particles = tracker.particles

    if tracker.is_parallel && tracker.comm !== nothing
        return _write_particle_trajectories_by_zlevel_parallel(
            base_filename, tracker; z_tolerance=z_tolerance, metadata=metadata
        )
    end

    if isempty(particles.x_history)
        @warn "No particle history to save"
        return Dict()
    end

    # Get initial z positions (first time in history)
    initial_z = particles.z_history[1]

    # Group particles by z-level
    z_groups = Dict{Float64, Vector{Int}}()

    for (i, z) in enumerate(initial_z)
        # Find existing group or create new one
        group_z = nothing
        for existing_z in keys(z_groups)
            if abs(z - existing_z) <= z_tolerance
                group_z = existing_z
                break
            end
        end

        if group_z === nothing
            group_z = z
            z_groups[group_z] = Int[]
        end

        push!(z_groups[group_z], i)
    end

    println("Found $(length(z_groups)) distinct z-levels:")
    for (z, indices) in z_groups
        println("  z = $(round(z, digits=3)): $(length(indices)) particles")
    end

    # Save each group to separate file
    output_files = Dict{Float64, String}()

    for (z_level, particle_indices) in z_groups
        # Create filename with z-level
        z_str = replace(string(round(z_level, digits=3)), "." => "p")
        filename = "$(base_filename)_z$(z_str).nc"

        println("Saving z=$(round(z_level, digits=3)) to $filename")

        # Create subset tracker for this z-level
        subset_tracker = create_zlevel_subset(tracker, particle_indices)

        # Add z-level info to metadata
        level_metadata = copy(metadata)
        level_metadata["z_level"] = z_level
        level_metadata["num_particles"] = length(particle_indices)
        level_metadata["particle_indices"] = particle_indices

        # Write trajectories for this z-level
        write_particle_trajectories(filename, subset_tracker; metadata=level_metadata)

        output_files[z_level] = filename
    end

    println("Successfully saved $(length(output_files)) z-level files")

    return output_files
end

function _write_particle_trajectories_by_zlevel_parallel(base_filename::String,
                                                         tracker::ParticleTracker{T};
                                                         z_tolerance::Float64=1e-6,
                                                         metadata::Dict=Dict()) where T
    particles = tracker.particles
    comm = tracker.comm

    has_history = !isempty(particles.x_history)
    has_any = MPI.Allreduce(has_history ? 1 : 0, +, comm)
    if has_any == 0
        if tracker.rank == 0
            @warn "No particle history to save"
        end
        return Dict{Float64,String}()
    end

    ntime = length(particles.time_history)
    x_series = Vector{Vector{T}}(undef, tracker.rank == 0 ? ntime : 0)
    y_series = Vector{Vector{T}}(undef, tracker.rank == 0 ? ntime : 0)
    z_series = Vector{Vector{T}}(undef, tracker.rank == 0 ? ntime : 0)
    ids_global = nothing

    for t_idx in 1:ntime
        ids_local = t_idx <= length(particles.id_history) ? particles.id_history[t_idx] : particles.id
        data = _gather_particle_positions(tracker, ids_local,
                                          particles.x_history[t_idx],
                                          particles.y_history[t_idx],
                                          particles.z_history[t_idx])
        if tracker.rank == 0
            if ids_global === nothing
                ids_global = data.ids
            elseif ids_global != data.ids
                @warn "Particle ID ordering changed at time index $t_idx; updating output ordering"
                ids_global = data.ids
            end
            x_series[t_idx] = data.x
            y_series[t_idx] = data.y
            z_series[t_idx] = data.z
        end
    end

    if tracker.rank != 0
        return Dict{Float64,String}()
    end

    initial_z = z_series[1]
    z_groups = Dict{Float64, Vector{Int}}()

    for (i, z) in enumerate(initial_z)
        group_z = nothing
        for existing_z in keys(z_groups)
            if abs(z - existing_z) <= z_tolerance
                group_z = existing_z
                break
            end
        end

        if group_z === nothing
            group_z = z
            z_groups[group_z] = Int[]
        end

        push!(z_groups[group_z], i)
    end

    println("Found $(length(z_groups)) distinct z-levels:")
    for (z, indices) in z_groups
        println("  z = $(round(z, digits=3)): $(length(indices)) particles")
    end

    output_files = Dict{Float64, String}()
    ids_all = ids_global === nothing ? Int[] : ids_global

    for (z_level, particle_indices) in z_groups
        z_str = replace(string(round(z_level, digits=3)), "." => "p")
        filename = "$(base_filename)_z$(z_str).nc"

        println("Saving z=$(round(z_level, digits=3)) to $filename")

        ids_subset = ids_all[particle_indices]
        x_subset = [x_series[t_idx][particle_indices] for t_idx in 1:ntime]
        y_subset = [y_series[t_idx][particle_indices] for t_idx in 1:ntime]
        z_subset = [z_series[t_idx][particle_indices] for t_idx in 1:ntime]

        level_metadata = copy(metadata)
        level_metadata["z_level"] = z_level
        level_metadata["num_particles"] = length(particle_indices)
        level_metadata["particle_indices"] = particle_indices

        _write_particle_trajectories_arrays(
            filename,
            ids_subset,
            particles.time_history,
            x_subset,
            y_subset,
            z_subset,
            tracker.config;
            metadata=level_metadata
        )

        output_files[z_level] = filename
    end

    println("Successfully saved $(length(output_files)) z-level files")

    return output_files
end

"""
    create_zlevel_subset(tracker, particle_indices)

Create a subset ParticleTracker containing only specified particles.
Used internally by write_particle_trajectories_by_zlevel.
"""
function create_zlevel_subset(tracker::ParticleTracker{T}, particle_indices::Vector{Int}) where T
    original = tracker.particles

    # Create new particle state with subset of particles
    subset_particles = ParticleState{T}(length(particle_indices))

    # Copy current positions and velocities for selected particles
    subset_particles.x = original.x[particle_indices]
    subset_particles.y = original.y[particle_indices]
    subset_particles.z = original.z[particle_indices]
    subset_particles.id = original.id[particle_indices]
    subset_particles.u = original.u[particle_indices]
    subset_particles.v = original.v[particle_indices]
    subset_particles.w = original.w[particle_indices]

    # Copy time information
    subset_particles.time = original.time
    subset_particles.np = length(particle_indices)

    # Copy trajectory history for selected particles
    subset_particles.x_history = [x_t[particle_indices] for x_t in original.x_history]
    subset_particles.y_history = [y_t[particle_indices] for y_t in original.y_history]
    subset_particles.z_history = [z_t[particle_indices] for z_t in original.z_history]
    subset_particles.id_history = isempty(original.id_history) ?
                                  Vector{Int}[] :
                                  [id_t[particle_indices] for id_t in original.id_history]
    subset_particles.time_history = copy(original.time_history)

    # Create a minimal subset tracker (only fields needed for I/O)
    # We use duck-typing here - return a NamedTuple that works with write_particle_trajectories
    return (
        particles = subset_particles,
        config = tracker.config,
        nx = tracker.nx, ny = tracker.ny, nz = tracker.nz,
        Lx = tracker.Lx, Ly = tracker.Ly, Lz = tracker.Lz,
        is_parallel = tracker.is_parallel,
        rank = tracker.rank,
        nprocs = tracker.nprocs
    )
end

end # module ParticleIO

using .ParticleIO
