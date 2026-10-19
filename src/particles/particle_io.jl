"""
Particle I/O module for saving and loading particle trajectories.

Handles NetCDF output of particle trajectories with proper metadata.
Supports saving to a dedicated `particles/` directory with configurable
iteration-based intervals.

Key Features:
- ParticleOutputManager: Manages automatic saving at regular intervals
- Separate `particles/` directory for organized output
- Multiple output modes: snapshots, trajectories, streaming
- Automatic file creation and directory management
"""

module ParticleIO

using Dates
using NCDatasets
using ..UnifiedParticleAdvection: ParticleTracker, ParticleState, ParticleConfig

# NCDatasets is always available since it's a required dependency (using NCDatasets above)
const HAS_NCDS = true

const _UNITS_CONVENTION = parentmodule(@__MODULE__)._UNITS_CONVENTION

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

"""Host copies of the particle IDs, positions, and horizontal velocities."""
function _particle_snapshot(tracker::ParticleTracker)
    particles = tracker.particles
    return (
        ids = copy(particles.id),
        x = Array(particles.x),
        y = Array(particles.y),
        z = Array(particles.z),
        u = Array(particles.u),
        v = Array(particles.v)
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
    ParticleOutputManager(output_dir; save_interval_iter=0,
                          save_interval_time=0,
                          output_mode=:trajectory,
                          file_prefix="particles",
                          overwrite=false)

Schedule particle NetCDF output under `output_dir/particles`, every
`save_interval_iter` iterations or every `save_interval_time` in the model's
time units: give one of them; the iteration interval wins when both are
positive. `output_mode` may be:

- `:trajectory`: one file, `<prefix>_trajectory.nc`, written when the run ends;
- `:streaming`: one file, `<prefix>_stream.nc`, extended at every save, so a
  run that is killed keeps the trajectories saved so far;
- `:snapshots`: one file per save, `<prefix>_<iteration>.nc`.

Each record holds the particle positions and their velocities at the
recorded time. Existing files of the same mode and prefix raise an error
unless `overwrite=true`, which removes them. Particle positions are not part
of the model restart files, so a restored simulation starts new trajectories.

Pass the manager as `particle_output` when constructing a
[`QGYBJplus.Simulation`](@ref).
The simulation writes initial, scheduled, and final particle states and closes
the manager during finalization.
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
    overwrite::Bool                  # Remove existing files instead of failing

    # Current state
    current_file::String             # Current output file path
    save_count::Int                  # Number of saves performed
    last_save_iter::Int              # Last iteration saved
    last_save_time::T                # Last time saved
    next_save_time::T                # Next nominal time-based deadline

    # Accumulated data (for trajectory mode)
    time_series::Vector{T}
    x_series::Vector{Vector{T}}
    y_series::Vector{Vector{T}}
    z_series::Vector{Vector{T}}
    u_series::Vector{Vector{T}}
    v_series::Vector{Vector{T}}
    particle_ids::Vector{Int}

    # ModelFields flags
    initialized::Bool
    global_particle_count::Int
    closed::Bool

    function ParticleOutputManager{T}(output_dir::AbstractString;
                                      save_interval_iter::Integer=0,
                                      save_interval_time::Real=0,
                                      output_mode::Symbol=:trajectory,
                                      file_prefix::AbstractString="particles",
                                      overwrite::Bool=false) where T
        save_interval_iter >= 0 ||
            throw(ArgumentError("save_interval_iter must be non-negative"))
        isfinite(save_interval_time) && save_interval_time >= 0 ||
            throw(ArgumentError("save_interval_time must be finite and non-negative"))
        # No default schedule suits every run: a time interval means seconds
        # in a dimensional run and model units in a nondimensional one, and
        # saving every step can exhaust memory in :trajectory mode.
        save_interval_iter > 0 || save_interval_time > 0 ||
            throw(ArgumentError("give a positive save_interval_iter or save_interval_time"))
        output_dir = String(output_dir)
        file_prefix = String(file_prefix)
        save_interval_time = T(save_interval_time)
        output_mode in (:snapshots, :trajectory, :streaming) ||
            throw(ArgumentError(
                "output_mode must be :snapshots, :trajectory, or :streaming"))
        isempty(output_dir) &&
            throw(ArgumentError("particle output directory cannot be empty"))
        isempty(file_prefix) &&
            throw(ArgumentError("particle output prefix cannot be empty"))
        # Create particles subdirectory
        particle_dir = joinpath(output_dir, "particles")

        new{T}(
            output_dir,
            particle_dir,
            save_interval_iter,
            save_interval_time,
            output_mode,
            file_prefix,
            overwrite,
            "",                      # current_file (set during initialization)
            0,                       # save_count
            0,                       # last_save_iter
            T(0),                    # last_save_time
            save_interval_time,      # next_save_time
            T[],                     # time_series
            Vector{T}[],             # x_series
            Vector{T}[],             # y_series
            Vector{T}[],             # z_series
            Vector{T}[],             # u_series
            Vector{T}[],             # v_series
            Int[],                   # particle_ids
            false,                   # initialized
            0,                       # global_particle_count
            false                    # closed
        )
    end
end

# Convenience constructor
ParticleOutputManager(output_dir::AbstractString; kwargs...) =
    ParticleOutputManager{Float64}(output_dir; kwargs...)

"""
    setup_particle_output!(manager, tracker)

Initialize the particle output manager and create the output directory.

# Arguments
- `manager`: ParticleOutputManager instance
- `tracker`: ParticleTracker instance
"""
function setup_particle_output!(manager::ParticleOutputManager{T},
                                tracker::ParticleTracker{S}) where {T,S}

    manager.closed && error("cannot reuse a finalized particle output manager")
    manager.initialized && return manager
    np_global = tracker.particles.np
    manager.global_particle_count = np_global

    # Create particles directory
    if !isdir(manager.particle_dir)
        mkpath(manager.particle_dir)
        println("Created particle output directory: $(manager.particle_dir)")
    end
    _clear_existing_particle_files!(manager)

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

    manager.initialized = true
    return manager
end

"""Whether `name` is a snapshot file `<prefix>_<digits>.nc`."""
function _is_snapshot_name(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix * "_") && endswith(name, ".nc") || return false
    number = chop(name; head=length(prefix) + 1, tail=3)
    return !isempty(number) && all(isdigit, number)
end

"""
Files this manager's mode and prefix would write that already exist: an error
unless `overwrite=true`, which removes them.
"""
function _clear_existing_particle_files!(manager::ParticleOutputManager)
    prefix = manager.file_prefix
    names = if manager.output_mode == :trajectory
        ["$(prefix)_trajectory.nc"]
    elseif manager.output_mode == :streaming
        ["$(prefix)_stream.nc"]
    else
        [filter(name -> _is_snapshot_name(name, prefix),
                readdir(manager.particle_dir));
         "$(prefix)_summary.txt"]
    end
    existing = filter(isfile, joinpath.(manager.particle_dir, names))
    isempty(existing) && return manager
    manager.overwrite || throw(ArgumentError(
        "particle output $(join(basename.(existing), ", ")) already exists in " *
        "$(manager.particle_dir); choose another directory or prefix, or pass " *
        "overwrite=true to replace it"))
    foreach(rm, existing)
    return manager
end

"""
    should_save_particles(manager, iteration, time) -> Bool

Check if particles should be saved at current iteration/time.
"""
function should_save_particles(manager::ParticleOutputManager{T},
                               iteration::Int, time::Real) where T
    if !manager.initialized
        return false
    end
    manager.closed && return false

    # Check iteration-based interval
    if manager.save_interval_iter > 0
        return (iteration - manager.last_save_iter) >= manager.save_interval_iter
    end

    # Check time-based interval
    if manager.save_interval_time > 0
        manager_time = T(time)
        tolerance = 8eps(max(abs(manager_time), abs(manager.save_interval_time), one(T)))
        return manager_time >= manager.next_save_time - tolerance
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
                                  tracker::ParticleTracker{S},
                                  iteration::Int, time::Real) where {T,S}
    manager_time = T(time)
    data = _particle_snapshot(tracker)

    if isempty(manager.particle_ids)
        manager.particle_ids = data.ids
    elseif manager.particle_ids != data.ids
        @warn "Particle ID ordering changed between saves; updating output ordering"
        manager.particle_ids = data.ids
    end

    if manager.output_mode == :snapshots
        # Save individual snapshot file
        save_particle_snapshot!(manager, tracker, iteration, manager_time; data=data)

    elseif manager.output_mode == :trajectory
        # Accumulate in memory
        push!(manager.time_series, manager_time)
        push!(manager.x_series, T.(data.x))
        push!(manager.y_series, T.(data.y))
        push!(manager.z_series, T.(data.z))
        push!(manager.u_series, T.(data.u))
        push!(manager.v_series, T.(data.v))

    elseif manager.output_mode == :streaming
        # Append to streaming file
        append_to_streaming_file!(manager, tracker, manager_time; data=data)
    end

    manager.save_count += 1
    manager.last_save_iter = iteration
    manager.last_save_time = manager_time
    if manager.save_interval_iter == 0 && manager.save_interval_time > 0
        tolerance = 8eps(max(abs(manager_time), abs(manager.save_interval_time), one(T)))
        while manager.next_save_time <= manager_time + tolerance
            manager.next_save_time += manager.save_interval_time
        end
    end

    return manager
end

"""
    save_particle_snapshot!(manager, tracker, iteration, time)

Save a single snapshot file with current particle positions.
"""
function save_particle_snapshot!(manager::ParticleOutputManager{T},
                                 tracker::ParticleTracker{S},
                                 iteration::Int, time::Real;
                                 data=nothing) where {T,S}
    data === nothing && (data = _particle_snapshot(tracker))
    ids, x, y, z, u, v = data.ids, data.x, data.y, data.z, data.u, data.v

    # Create filename with iteration number
    filename = joinpath(manager.particle_dir,
                       "$(manager.file_prefix)_$(lpad(iteration, 6, '0')).nc")

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

        # Fill data
        defVar_particle[:] = ids
        defVar_x[:] = x
        defVar_y[:] = y
        defVar_z[:] = z
        defVar_u[:] = u
        defVar_v[:] = v

        # Set variable attributes
        defVar_x.attrib["long_name"] = "x position"
        defVar_y.attrib["long_name"] = "y position"
        defVar_z.attrib["long_name"] = "z position"
        defVar_u.attrib["long_name"] = "x velocity at the recorded time"
        defVar_v.attrib["long_name"] = "y velocity at the recorded time"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Particle Snapshot"
        ds.attrib["units_convention"] = _UNITS_CONVENTION
        ds.attrib["created_at"] = string(now())
        ds.attrib["time"] = time
        ds.attrib["iteration"] = iteration
        ds.attrib["number_of_particles"] = length(ids)
        ds.attrib["integration_method"] = string(tracker.config.integration_method)
        ds.attrib["wave_motion"] = String(tracker.config.wave_motion)
    end

    return filename
end

"""
    create_streaming_particle_file!(manager, tracker)

Create NetCDF file for streaming particle output with unlimited time dimension.
"""
function create_streaming_particle_file!(manager::ParticleOutputManager{T},
                                         tracker::ParticleTracker{S};
                                         np_global::Union{Nothing,Int}=nothing) where {T,S}
    particles = tracker.particles
    filename = manager.current_file
    if np_global === nothing
        np_global = particles.np
    end

    if isempty(manager.particle_ids)
        manager.particle_ids = collect(1:np_global)
    end

    HAS_NCDS || error("NCDatasets not available")

    NCDatasets.Dataset(filename, "c") do ds
        # Define dimensions (time is unlimited)
        ds.dim["particle"] = np_global
        ds.dim["time"] = Inf

        # Create coordinate variables
        defVar_particle = NCDatasets.defVar(ds, "particle", Int, ("particle",))
        defVar_time = NCDatasets.defVar(ds, "time", Float64, ("time",))

        # Create trajectory variables
        defVar_x = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
        defVar_y = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
        defVar_z = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
        defVar_u = NCDatasets.defVar(ds, "u", Float64, ("particle", "time"))
        defVar_v = NCDatasets.defVar(ds, "v", Float64, ("particle", "time"))

        # Fill particle IDs
        defVar_particle[:] = manager.particle_ids

        # Set variable attributes
        defVar_particle.attrib["long_name"] = "particle identifier"
        defVar_time.attrib["long_name"] = "time"

        defVar_x.attrib["long_name"] = "x position"
        defVar_y.attrib["long_name"] = "y position"
        defVar_z.attrib["long_name"] = "z position"
        defVar_u.attrib["long_name"] = "x velocity at the recorded time"
        defVar_v.attrib["long_name"] = "y velocity at the recorded time"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Particle Trajectories (Streaming)"
        ds.attrib["units_convention"] = _UNITS_CONVENTION
        ds.attrib["created_at"] = string(now())
        ds.attrib["number_of_particles"] = np_global
        ds.attrib["output_mode"] = "streaming"
        ds.attrib["integration_method"] = string(tracker.config.integration_method)
        ds.attrib["wave_motion"] = String(tracker.config.wave_motion)
    end

    println("Created streaming particle file: $filename")

    return filename
end

"""
    append_to_streaming_file!(manager, tracker, time)

Append current particle positions to streaming NetCDF file.
"""
function append_to_streaming_file!(manager::ParticleOutputManager{T},
                                   tracker::ParticleTracker{S}, time::Real;
                                   data=nothing) where {T,S}
    data === nothing && (data = _particle_snapshot(tracker))
    ids, x, y, z, u, v = data.ids, data.x, data.y, data.z, data.u, data.v
    if isempty(manager.particle_ids)
        manager.particle_ids = ids
    end
    filename = manager.current_file

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
                                   tracker::ParticleTracker{S};
                                   metadata::Dict=Dict()) where {T,S}
    manager.closed && return manager
    if !manager.initialized
        manager.closed = true
        return manager
    end

    try
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
    finally
        manager.closed = true
    end

    return manager
end

"""
    write_accumulated_trajectories!(manager, tracker; metadata=Dict())

Write accumulated trajectory data to NetCDF file.
"""
function write_accumulated_trajectories!(manager::ParticleOutputManager{T},
                                         tracker::ParticleTracker{S};
                                         metadata::Dict=Dict()) where {T,S}
    filename = manager.current_file
    np = length(manager.x_series[1])
    nt = length(manager.time_series)
    ids = isempty(manager.particle_ids) ? collect(1:np) : manager.particle_ids
    if length(ids) != np
        @warn "Particle ID length ($(length(ids))) does not match trajectory data size ($np); using sequential IDs"
        ids = collect(1:np)
    end

    println("  Writing $(nt) time points for $(np) particles...")

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
        end

        # Set variable attributes
        defVar_particle.attrib["long_name"] = "particle identifier"
        defVar_time.attrib["long_name"] = "time"

        defVar_x.attrib["long_name"] = "x position"
        defVar_y.attrib["long_name"] = "y position"
        defVar_z.attrib["long_name"] = "z position"
        defVar_u.attrib["long_name"] = "x velocity at the recorded time"
        defVar_v.attrib["long_name"] = "y velocity at the recorded time"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Particle Trajectories"
        ds.attrib["units_convention"] = _UNITS_CONVENTION
        ds.attrib["created_at"] = string(now())
        ds.attrib["number_of_particles"] = np
        ds.attrib["number_of_timesteps"] = nt
        ds.attrib["start_time"] = manager.time_series[1]
        ds.attrib["end_time"] = manager.time_series[end]
        ds.attrib["output_mode"] = "trajectory"
        ds.attrib["integration_method"] = string(tracker.config.integration_method)
        ds.attrib["wave_motion"] = String(tracker.config.wave_motion)

        # Add user metadata
        for (key, value) in metadata
            ds.attrib[string(key)] = value
        end
    end

    println("  Trajectory file written: $filename")

    return manager
end

"""
    update_streaming_metadata!(manager)

Update metadata in streaming file after simulation.
"""
function update_streaming_metadata!(manager::ParticleOutputManager)
    filename = manager.current_file

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

    return manager
end

"""
    write_snapshot_summary!(manager, tracker)

Write summary file listing all snapshot files.
"""
function write_snapshot_summary!(manager::ParticleOutputManager, tracker::ParticleTracker)
    summary_file = joinpath(manager.particle_dir, "$(manager.file_prefix)_summary.txt")
    np_global = manager.global_particle_count

    open(summary_file, "w") do f
        println(f, "QG-YBJ Particle Snapshot Summary")
        println(f, "================================")
        println(f, "Created: $(now())")
        println(f, "Number of particles: $np_global")
        println(f, "Number of snapshots: $(manager.save_count)")
        println(f, "Output directory: $(manager.particle_dir)")
        println(f, "")
        println(f, "Integration method: $(tracker.config.integration_method)")
        println(f, "Wave motion: $(tracker.config.wave_motion)")
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

    return manager
end

# Complete-trajectory output

"""
    write_particle_trajectories(filename, tracker; metadata=Dict())

Write the tracker's in-memory trajectory history to a NetCDF file. The
tracker keeps that history only when its configuration sets `save_interval`.
During a simulation a [`ParticleOutputManager`](@ref) writes trajectories
instead; its `:streaming` mode does so without holding them in memory.
"""
function write_particle_trajectories(filename::String, tracker::ParticleTracker;
                                    metadata::Dict=Dict())
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
        @warn "No particle history to save: the tracker records one only when " *
              "its configuration sets save_interval (see ParticleConfig)"
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
        time_var.attrib["long_name"] = "time"

        x_var.attrib["long_name"] = "x position (horizontal)"
        y_var.attrib["long_name"] = "y position (horizontal)"
        z_var.attrib["long_name"] = "z position (vertical)"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Particle Trajectories"
        ds.attrib["units_convention"] = _UNITS_CONVENTION
        ds.attrib["created_at"] = string(now())
        ds.attrib["number_of_particles"] = np
        ds.attrib["integration_method"] = string(config.integration_method)
        ds.attrib["wave_motion"] = String(config.wave_motion)
        ds.attrib["periodic_x"] = Int(config.periodic_x)
        ds.attrib["periodic_y"] = Int(config.periodic_y)

        # Add user metadata
        for (key, value) in metadata
            ds.attrib[string(key)] = value isa Bool ? Int(value) : value
        end
    end

    @info "Particle trajectories written to: $filename"

    return filename
end

"""
    read_particle_trajectories(filename) -> NamedTuple

Read a trajectory file written by [`write_particle_trajectories`](@ref).
The returned named tuple contains `x`, `y`, and `z` matrices in
`(particle, time)` order, plus `time`, `particle_ids`, and file `attributes`.

```julia
trajectories = read_particle_trajectories("particles.nc")
initial_x = trajectories.x[:, 1]
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
    data = _particle_snapshot(tracker)

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

        # Fill data
        particle_var[:] = data.ids
        x_var[:] = data.x
        y_var[:] = data.y
        z_var[:] = data.z
        u_var[:] = data.u
        v_var[:] = data.v

        # Set attributes
        particle_var.attrib["long_name"] = "particle identifier"

        x_var.attrib["long_name"] = "x position (horizontal)"
        y_var.attrib["long_name"] = "y position (horizontal)"
        z_var.attrib["long_name"] = "z position (vertical)"

        u_var.attrib["long_name"] = "x velocity at the recorded time"
        v_var.attrib["long_name"] = "y velocity at the recorded time"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Particle Snapshot"
        ds.attrib["units_convention"] = _UNITS_CONVENTION
        ds.attrib["created_at"] = string(now())
        ds.attrib["time"] = time
        ds.attrib["number_of_particles"] = length(data.ids)
        ds.attrib["wave_motion"] = String(tracker.config.wave_motion)
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
    np_global = particles.np

    HAS_NCDS || error("NCDatasets not available")

    NCDatasets.Dataset(filename, mode) do ds
        if !append_mode
            # Define dimensions
            ds.dim["particle"] = np_global
            ds.dim["time"] = Inf

            # Create coordinate variables
            particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))

            # Create trajectory variables
            x_var = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
            u_var = NCDatasets.defVar(ds, "u", Float64, ("particle", "time"))
            v_var = NCDatasets.defVar(ds, "v", Float64, ("particle", "time"))

            # Fill particle IDs
            particle_var[:] = collect(1:np_global)

            # Set attributes
            particle_var.attrib["long_name"] = "particle identifier"
            time_var.attrib["long_name"] = "time"

            x_var.attrib["long_name"] = "x position (horizontal)"
            y_var.attrib["long_name"] = "y position (horizontal)"
            z_var.attrib["long_name"] = "z position (vertical)"

            u_var.attrib["long_name"] = "x velocity at the recorded time"
            v_var.attrib["long_name"] = "y velocity at the recorded time"

            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Trajectories (Time Series)"
            ds.attrib["units_convention"] = _UNITS_CONVENTION
            ds.attrib["created_at"] = string(now())
            ds.attrib["number_of_particles"] = np_global
            ds.attrib["integration_method"] = string(tracker.config.integration_method)
            ds.attrib["wave_motion"] = String(tracker.config.wave_motion)
        end
    end

    return filename
end

"""
    append_particle_data!(filename, tracker, time_index)

Append current particle state to existing NetCDF file.
"""
function append_particle_data!(filename::String, tracker::ParticleTracker, time_index::Int)
    data = _particle_snapshot(tracker)

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
# Initialize particles at multiple depths
config = particles_in_layers(
    [-250.0, -500.0, -750.0];
    x_min=first(model.grid.x_faces), x_max=last(model.grid.x_faces),
    y_min=first(model.grid.y_faces), y_max=last(model.grid.y_faces),
    nx=4, ny=4,
)
tracker = initialize_particles!(model, config)

# Run simulation...

# Save each z-level to separate file
files = write_particle_trajectories_by_zlevel("particles", tracker)
# Creates one file for each initial depth.
```
"""
function write_particle_trajectories_by_zlevel(base_filename::String, tracker::ParticleTracker;
                                              z_tolerance::Float64=1e-6,
                                              metadata::Dict=Dict())
    particles = tracker.particles

    if isempty(particles.x_history)
        @warn "No particle history to save: the tracker records one only when " *
              "its configuration sets save_interval (see ParticleConfig)"
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
    subset_particles.x = Array(original.x)[particle_indices]
    subset_particles.y = Array(original.y)[particle_indices]
    subset_particles.z = Array(original.z)[particle_indices]
    subset_particles.id = original.id[particle_indices]
    subset_particles.u = Array(original.u)[particle_indices]
    subset_particles.v = Array(original.v)[particle_indices]

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
    )
end

end # module ParticleIO

using .ParticleIO
