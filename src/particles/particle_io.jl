"""
Particle I/O module for saving and loading particle trajectories.

Handles NetCDF output of particle trajectories with proper metadata.
"""

module ParticleIO

using ..UnifiedParticleAdvection: ParticleTracker, ParticleState, ParticleConfig

export write_particle_trajectories, read_particle_trajectories,
       write_particle_snapshot, create_particle_output_file

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

end # module ParticleIO

using .ParticleIO