"""
Particle I/O module for saving and loading particle trajectories.

Handles NetCDF output of particle trajectories with proper metadata.
"""

module ParticleIO

using ..UnifiedParticleAdvection: ParticleTracker, ParticleState, ParticleConfig

export write_particle_trajectories, read_particle_trajectories,
       write_particle_snapshot, create_particle_output_file, 
       write_particle_trajectories_by_zlevel

"""
    write_particle_trajectories(filename, tracker; metadata=Dict())

Write complete particle trajectory history to NetCDF file.
"""
function write_particle_trajectories(filename::String, tracker::ParticleTracker; 
                                    metadata::Dict=Dict())
    particles = tracker.particles
    
    # Check if NCDatasets is available
    try
        import NCDatasets
        
        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = particles.np
            ds.dim["time"] = length(particles.time_history)
            
            # Create coordinate variables
            particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))
            
            # Create trajectory variables
            x_var = NCDatasets.defVar(ds, "x", Float64, ("particle", "time"))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("particle", "time"))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("particle", "time"))
            
            # Fill coordinate data
            particle_var[:] = 1:particles.np
            time_var[:] = particles.time_history
            
            # Fill trajectory data
            for (t_idx, t) in enumerate(particles.time_history)
                x_var[:, t_idx] = particles.x_history[t_idx]
                y_var[:, t_idx] = particles.y_history[t_idx]
                z_var[:, t_idx] = particles.z_history[t_idx]
            end
            
            # Set attributes
            particle_var.attrib["long_name"] = "particle identifier"
            time_var.attrib["units"] = "model time units"
            time_var.attrib["long_name"] = "time"
            
            x_var.attrib["units"] = "m"
            x_var.attrib["long_name"] = "x position"
            y_var.attrib["units"] = "m"
            y_var.attrib["long_name"] = "y position"
            z_var.attrib["units"] = "m"
            z_var.attrib["long_name"] = "z position"
            
            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Trajectories"
            ds.attrib["created_at"] = string(now())
            ds.attrib["number_of_particles"] = particles.np
            ds.attrib["integration_method"] = string(tracker.config.integration_method)
            ds.attrib["use_ybj_w"] = tracker.config.use_ybj_w
            ds.attrib["use_3d_advection"] = tracker.config.use_3d_advection
            ds.attrib["periodic_x"] = tracker.config.periodic_x
            ds.attrib["periodic_y"] = tracker.config.periodic_y
            ds.attrib["reflect_z"] = tracker.config.reflect_z
            
            # Add user metadata
            for (key, value) in metadata
                ds.attrib[key] = value
            end
        end
        
        @info "Particle trajectories written to: $filename"
        
    catch e
        @warn "NCDatasets not available, cannot write particle trajectories: $e"
    end
    
    return filename
end

"""
    write_particle_snapshot(filename, tracker, time)

Write current particle positions to NetCDF file (single time snapshot).
"""
function write_particle_snapshot(filename::String, tracker::ParticleTracker, time::Real)
    particles = tracker.particles
    
    try
        import NCDatasets
        
        NCDatasets.Dataset(filename, "c") do ds
            # Define dimensions
            ds.dim["particle"] = particles.np
            
            # Create variables
            particle_var = NCDatasets.defVar(ds, "particle", Int, ("particle",))
            x_var = NCDatasets.defVar(ds, "x", Float64, ("particle",))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("particle",))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("particle",))
            u_var = NCDatasets.defVar(ds, "u", Float64, ("particle",))
            v_var = NCDatasets.defVar(ds, "v", Float64, ("particle",))
            w_var = NCDatasets.defVar(ds, "w", Float64, ("particle",))
            
            # Fill data
            particle_var[:] = 1:particles.np
            x_var[:] = particles.x
            y_var[:] = particles.y
            z_var[:] = particles.z
            u_var[:] = particles.u
            v_var[:] = particles.v
            w_var[:] = particles.w
            
            # Set attributes
            particle_var.attrib["long_name"] = "particle identifier"
            
            x_var.attrib["units"] = "m"
            x_var.attrib["long_name"] = "x position"
            y_var.attrib["units"] = "m" 
            y_var.attrib["long_name"] = "y position"
            z_var.attrib["units"] = "m"
            z_var.attrib["long_name"] = "z position"
            
            u_var.attrib["units"] = "m/s"
            u_var.attrib["long_name"] = "x velocity"
            v_var.attrib["units"] = "m/s"
            v_var.attrib["long_name"] = "y velocity"  
            w_var.attrib["units"] = "m/s"
            w_var.attrib["long_name"] = "z velocity"
            
            # Global attributes
            ds.attrib["title"] = "QG-YBJ Particle Snapshot"
            ds.attrib["created_at"] = string(now())
            ds.attrib["time"] = time
            ds.attrib["number_of_particles"] = particles.np
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
    
    try
        import NCDatasets
        
        NCDatasets.Dataset(filename, mode) do ds
            if !append_mode
                # Define dimensions
                ds.dim["particle"] = particles.np
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
                particle_var[:] = 1:particles.np
                
                # Set attributes
                particle_var.attrib["long_name"] = "particle identifier"
                time_var.attrib["units"] = "model time units"
                time_var.attrib["long_name"] = "time"
                
                x_var.attrib["units"] = "m"
                x_var.attrib["long_name"] = "x position"
                y_var.attrib["units"] = "m"
                y_var.attrib["long_name"] = "y position"
                z_var.attrib["units"] = "m"
                z_var.attrib["long_name"] = "z position"
                
                u_var.attrib["units"] = "m/s"
                u_var.attrib["long_name"] = "x velocity"
                v_var.attrib["units"] = "m/s"
                v_var.attrib["long_name"] = "y velocity"
                w_var.attrib["units"] = "m/s"
                w_var.attrib["long_name"] = "z velocity"
                
                # Global attributes
                ds.attrib["title"] = "QG-YBJ Particle Trajectories (Time Series)"
                ds.attrib["created_at"] = string(now())
                ds.attrib["number_of_particles"] = particles.np
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
    particles = tracker.particles
    
    try
        import NCDatasets
        
        NCDatasets.Dataset(filename, "a") do ds
            # Get current time dimension size
            current_time_size = length(ds["time"])
            new_time_index = current_time_size + 1
            
            # Append time
            ds["time"][new_time_index] = particles.time
            
            # Append particle data
            ds["x"][:, new_time_index] = particles.x
            ds["y"][:, new_time_index] = particles.y
            ds["z"][:, new_time_index] = particles.z
            ds["u"][:, new_time_index] = particles.u
            ds["v"][:, new_time_index] = particles.v
            ds["w"][:, new_time_index] = particles.w
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

"""
    create_zlevel_subset(tracker, particle_indices)

Create a subset ParticleTracker containing only specified particles.
Used internally by write_particle_trajectories_by_zlevel.
"""
function create_zlevel_subset(tracker::ParticleTracker{T}, particle_indices::Vector{Int}) where T
    original = tracker.particles
    
    # Create new particle state with subset of particles
    subset_particles = ParticleState{T}()
    subset_particles.np = length(particle_indices)
    
    # Copy current positions and velocities for selected particles
    subset_particles.x = original.x[particle_indices]
    subset_particles.y = original.y[particle_indices]
    subset_particles.z = original.z[particle_indices]
    subset_particles.u = original.u[particle_indices]
    subset_particles.v = original.v[particle_indices]
    subset_particles.w = original.w[particle_indices]
    
    # Copy time information
    subset_particles.time = original.time
    subset_particles.time_history = copy(original.time_history)
    
    # Copy trajectory history for selected particles
    subset_particles.x_history = [x_t[particle_indices] for x_t in original.x_history]
    subset_particles.y_history = [y_t[particle_indices] for y_t in original.y_history]
    subset_particles.z_history = [z_t[particle_indices] for z_t in original.z_history]
    
    # Create new tracker with subset particles
    subset_tracker = ParticleTracker{T}()
    subset_tracker.particles = subset_particles
    subset_tracker.config = tracker.config
    subset_tracker.grid = tracker.grid
    subset_tracker.is_parallel = tracker.is_parallel
    subset_tracker.rank = tracker.rank
    subset_tracker.nprocs = tracker.nprocs
    subset_tracker.last_save_time = tracker.last_save_time
    
    return subset_tracker
end

end # module ParticleIO

using .ParticleIO