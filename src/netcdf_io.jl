"""
Enhanced NetCDF I/O functionality for QG-YBJ model.

This module provides comprehensive NetCDF input/output capabilities including:
- State file output with time series (state0001.nc, state0002.nc, ...)
- Initial condition reading from NetCDF files  
- Stratification profile I/O
- Flexible variable selection and metadata handling
"""

const HAS_NCDS = Base.find_package("NCDatasets") !== nothing
function ensure_ncds_loaded()
    if HAS_NCDS
        try
            @eval import NCDatasets
        catch
            # ignore, caller will error if missing
        end
    end
end
using Printf
using Dates
using ..QGYBJ: Grid, State, QGParams
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!

"""
    OutputManager

Manages NetCDF output for model state and diagnostics.
"""
mutable struct OutputManager{T}
    output_dir::String
    state_file_pattern::String
    
    # Output intervals (in model time units)
    psi_interval::T
    wave_interval::T
    diagnostics_interval::T
    
    # Counters and timers
    psi_counter::Int
    wave_counter::Int
    diagnostics_counter::Int
    
    last_psi_output::T
    last_wave_output::T
    last_diagnostics_output::T
    
    # Variable selections
    save_psi::Bool
    save_waves::Bool
    save_velocities::Bool
    save_vertical_velocity::Bool
    save_vorticity::Bool
    save_diagnostics::Bool
    
    # Parallel configuration (optional)
    parallel_config
    
    # Metadata
    run_info::Dict{String,Any}
end

"""
    OutputManager(config::OutputConfig, params::QGParams, parallel_config=nothing)

Create output manager from configuration with optional parallel support.
"""
function OutputManager(config, params::QGParams{T}, parallel_config=nothing) where T
    # Create output directory if it doesn't exist
    mkpath(config.output_dir)
    
    # Initialize run info
    run_info = Dict{String,Any}(
        "created_at" => string(now()),
        "julia_version" => string(VERSION),
        "nx" => params.nx,
        "ny" => params.ny, 
        "nz" => params.nz,
        "Lx" => params.Lx,
        "Ly" => params.Ly,
        "dt" => params.dt,
        "f0" => params.f0
    )
    
    return OutputManager{T}(
        config.output_dir,
        config.state_file_pattern,
        config.psi_interval,
        config.wave_interval, 
        config.diagnostics_interval,
        1, 1, 1,  # counters
        T(0), T(0), T(0),  # last output times
        config.save_psi,
        config.save_waves,
        config.save_velocities,
        hasfield(typeof(config), :save_vertical_velocity) ? config.save_vertical_velocity : false,
        config.save_vorticity,
        config.save_diagnostics,
        parallel_config,
        run_info
    )
end

"""
    should_output_psi(manager::OutputManager, time::Real)

Check if it's time to output psi field.
"""
function should_output_psi(manager::OutputManager, time::Real)
    return manager.save_psi && (time - manager.last_psi_output >= manager.psi_interval)
end

"""
    should_output_waves(manager::OutputManager, time::Real)

Check if it's time to output wave fields.
"""
function should_output_waves(manager::OutputManager, time::Real)
    return manager.save_waves && (time - manager.last_wave_output >= manager.wave_interval)
end

"""
    should_output_diagnostics(manager::OutputManager, time::Real)

Check if it's time to output diagnostics.
"""
function should_output_diagnostics(manager::OutputManager, time::Real)
    return manager.save_diagnostics && (time - manager.last_diagnostics_output >= manager.diagnostics_interval)
end

"""
    write_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real, parallel_config=nothing; params=nothing)

Write complete state to NetCDF file with standardized naming.
Unified interface for both serial and parallel I/O.
"""
function write_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real, parallel_config=nothing; params=nothing)
    if !HAS_NCDS
        error("NCDatasets not available. Install NCDatasets.jl or skip NetCDF I/O.")
    end
    ensure_ncds_loaded()
    # Use parallel_config from manager if not provided
    if parallel_config === nothing
        parallel_config = manager.parallel_config
    end
    
    # Route to appropriate I/O method
    if parallel_config !== nothing && parallel_config.use_mpi && G.decomp !== nothing
        return write_parallel_state_file(manager, S, G, plans, time, parallel_config; params=params)
    else
        return write_serial_state_file(manager, S, G, plans, time; params=params)
    end
end

"""
    write_serial_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real; params=nothing)

Write state file in serial mode.
"""
function write_serial_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real; params=nothing)
    ensure_ncds_loaded()
    # Generate filename (printf-style pattern supported)
    io = IOBuffer()
    Printf.format(io, Printf.Format(manager.state_file_pattern), manager.psi_counter)
    filename = String(take!(io))
    filepath = joinpath(manager.output_dir, filename)
    
    # Convert spectral fields to real space
    psir = similar(S.psi)
    fft_backward!(psir, S.psi, plans)
    
    BRr = similar(S.B)
    BIr = similar(S.B) 
    fft_backward!(BRr, real.(S.B), plans)
    fft_backward!(BIr, imag.(S.B), plans)
    
    # Normalization factor for IFFT
    norm_factor = G.nx * G.ny
    
    NCDatasets.Dataset(filepath, "c") do ds
        # Define dimensions
        ds.dim["x"] = G.nx
        ds.dim["y"] = G.ny
        ds.dim["z"] = G.nz
        ds.dim["time"] = 1
        
        # Create coordinate variables
        x_var = defVar(ds, "x", Float64, ("x",))
        y_var = defVar(ds, "y", Float64, ("y",))
        z_var = defVar(ds, "z", Float64, ("z",))
        time_var = defVar(ds, "time", Float64, ("time",))
        
        # Set coordinate values
        dx = 2π / G.nx  # Assuming domain [0, 2π]
        dy = 2π / G.ny
        dz = 2π / G.nz
        
        x_var[:] = collect(0:dx:(2π-dx))
        y_var[:] = collect(0:dy:(2π-dy))
        z_var[:] = collect(0:dz:(2π-dz))
        time_var[1] = time
        
        # Add coordinate attributes
        x_var.attrib["units"] = "radians"
        x_var.attrib["long_name"] = "x coordinate"
        y_var.attrib["units"] = "radians" 
        y_var.attrib["long_name"] = "y coordinate"
        z_var.attrib["units"] = "radians"
        z_var.attrib["long_name"] = "z coordinate"
        time_var.attrib["units"] = "model time units"
        time_var.attrib["long_name"] = "time"
        
        # Stream function
        if manager.save_psi
            psi_var = defVar(ds, "psi", Float64, ("x", "y", "z"))
            psi_var[:,:,:] = real.(psir) / norm_factor
            psi_var.attrib["units"] = "m²/s"
            psi_var.attrib["long_name"] = "stream function"
        end
        
        # Wave fields (L+A real and imaginary parts)
        if manager.save_waves
            LAr_var = defVar(ds, "LAr", Float64, ("x", "y", "z"))
            LAi_var = defVar(ds, "LAi", Float64, ("x", "y", "z"))
            
            LAr_var[:,:,:] = real.(BRr) / norm_factor
            LAi_var[:,:,:] = real.(BIr) / norm_factor
            
            LAr_var.attrib["units"] = "wave amplitude"
            LAr_var.attrib["long_name"] = "L+A real part"
            LAi_var.attrib["units"] = "wave amplitude"
            LAi_var.attrib["long_name"] = "L+A imaginary part"
        end
        
        # Horizontal velocities (if requested)
        if manager.save_velocities && hasfield(typeof(S), :u) && hasfield(typeof(S), :v)
            ur = similar(S.u)
            vr = similar(S.v)
            fft_backward!(ur, S.u, plans)
            fft_backward!(vr, S.v, plans)
            
            u_var = defVar(ds, "u", Float64, ("x", "y", "z"))
            v_var = defVar(ds, "v", Float64, ("x", "y", "z"))
            
            u_var[:,:,:] = real.(ur) / norm_factor
            v_var[:,:,:] = real.(vr) / norm_factor
            
            u_var.attrib["units"] = "m/s"
            u_var.attrib["long_name"] = "zonal velocity"
            v_var.attrib["units"] = "m/s" 
            v_var.attrib["long_name"] = "meridional velocity"
        end
        
        # Vertical velocity (if requested)
        if manager.save_vertical_velocity && hasfield(typeof(S), :w)
            w_var = defVar(ds, "w", Float64, ("x", "y", "z"))
            w_var[:,:,:] = S.w  # w is already in real space
            
            w_var.attrib["units"] = "m/s"
            w_var.attrib["long_name"] = "vertical velocity (QG ageostrophic)"
            w_var.attrib["description"] = "Diagnostic vertical velocity from omega equation"
        end
        
        # Global attributes
        ds.attrib["title"] = "QG-YBJ Model State"
        ds.attrib["created_at"] = string(now())
        ds.attrib["model_time"] = time
        ds.attrib["file_counter"] = manager.psi_counter
        
        # Add parameter information if provided
        if !isnothing(params)
            ds.attrib["nx"] = params.nx
            ds.attrib["ny"] = params.ny
            ds.attrib["nz"] = params.nz
            ds.attrib["f0"] = params.f0
            ds.attrib["dt"] = params.dt
        end
        
        # Add run information
        for (key, value) in manager.run_info
            ds.attrib[key] = value
        end
    end
    
    @info "Wrote state file: $filename (t=$time)"
    manager.psi_counter += 1
    manager.last_psi_output = time
    
    return filepath
end

"""
    write_parallel_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real, parallel_config; params=nothing)

Write state file using parallel NetCDF I/O.
"""
function write_parallel_state_file(manager::OutputManager, S::State, G::Grid, plans, time::Real, parallel_config; params=nothing)
    # Generate filename
    io = IOBuffer()
    Printf.format(io, Printf.Format(manager.state_file_pattern), manager.psi_counter)
    filename = String(take!(io))
    filepath = joinpath(manager.output_dir, filename)
    
    if parallel_config.use_mpi && G.decomp !== nothing
        rank = MPI.Comm_rank(parallel_config.comm)
        
        if parallel_config.parallel_io
            # Use parallel NetCDF I/O
            write_parallel_netcdf_file(filepath, S, G, plans, time, parallel_config; params=params)
        else
            # Gather to rank 0 and write
            if rank == 0
                gathered_state = gather_state_for_io(S, G, parallel_config)
                write_gathered_state_file(filepath, gathered_state, G, plans, time; params=params)
            end
            MPI.Barrier(parallel_config.comm)
        end
    else
        # Fallback to serial
        return write_serial_state_file(manager, S, G, plans, time; params=params)
    end
    
    manager.psi_counter += 1
    manager.last_psi_output = time
    
    if parallel_config.use_mpi
        rank = MPI.Comm_rank(parallel_config.comm)
        if rank == 0
            @info "Wrote parallel state file: $filename (t=$time)"
        end
    else
        @info "Wrote state file: $filename (t=$time)"
    end
    
    return filepath
end

"""
    write_parallel_netcdf_file(filepath, S::State, G::Grid, plans, time, parallel_config; params=nothing)

Write NetCDF file with 2D decomposition support.
Uses gather-to-root approach: all data is gathered to rank 0, which writes the file.
This is simpler and more reliable than parallel NetCDF.
"""
function write_parallel_netcdf_file(filepath, S::State, G::Grid, plans, time, parallel_config; params=nothing)
    ensure_ncds_loaded()

    # Import MPI
    MPI = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))

    rank = MPI.Comm_rank(parallel_config.comm)
    nprocs = MPI.Comm_size(parallel_config.comm)

    # Gather fields to rank 0 using QGYBJ's gather function
    gathered_psi = QGYBJ.gather_to_root(S.psi, G, parallel_config)
    gathered_B = QGYBJ.gather_to_root(S.B, G, parallel_config)

    # Only rank 0 writes
    if rank == 0
        if gathered_psi === nothing || gathered_B === nothing
            error("Failed to gather fields to rank 0")
        end

        # Create serial FFT plans for the full domain
        temp_plans = plan_transforms!(G)

        # Convert to real space
        psir = zeros(Float64, G.nx, G.ny, G.nz)
        BRr = zeros(Float64, G.nx, G.ny, G.nz)
        BIr = zeros(Float64, G.nx, G.ny, G.nz)

        fft_backward!(psir, gathered_psi, temp_plans)
        fft_backward!(BRr, real.(gathered_B), temp_plans)
        fft_backward!(BIr, imag.(gathered_B), temp_plans)

        norm_factor = G.nx * G.ny

        NCDatasets.Dataset(filepath, "c") do ds
            # Define dimensions
            ds.dim["x"] = G.nx
            ds.dim["y"] = G.ny
            ds.dim["z"] = G.nz
            ds.dim["time"] = 1

            # Create coordinate variables
            x_var = NCDatasets.defVar(ds, "x", Float64, ("x",))
            y_var = NCDatasets.defVar(ds, "y", Float64, ("y",))
            z_var = NCDatasets.defVar(ds, "z", Float64, ("z",))
            time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))

            # Set coordinate values
            dx = 2π / G.nx
            dy = 2π / G.ny
            dz = 2π / G.nz

            x_var[:] = collect(0:dx:(2π-dx))
            y_var[:] = collect(0:dy:(2π-dy))
            z_var[:] = collect(0:dz:(2π-dz))
            time_var[1] = time

            # Create and write data variables
            psi_var = NCDatasets.defVar(ds, "psi", Float64, ("x", "y", "z"))
            LAr_var = NCDatasets.defVar(ds, "LAr", Float64, ("x", "y", "z"))
            LAi_var = NCDatasets.defVar(ds, "LAi", Float64, ("x", "y", "z"))

            psi_var[:,:,:] = real.(psir) / norm_factor
            LAr_var[:,:,:] = real.(BRr) / norm_factor
            LAi_var[:,:,:] = real.(BIr) / norm_factor

            # Add attributes
            ds.attrib["title"] = "QG-YBJ Model State"
            ds.attrib["created_at"] = string(Dates.now())
            ds.attrib["model_time"] = time
            ds.attrib["n_processes"] = nprocs

            # Add parameter info if provided
            if params !== nothing
                ds.attrib["nx"] = params.nx
                ds.attrib["ny"] = params.ny
                ds.attrib["nz"] = params.nz
                ds.attrib["f0"] = params.f0
                ds.attrib["dt"] = params.dt
            end
        end

        @info "Rank 0 wrote state file: $filepath (t=$time)"
    end

    # Synchronize all processes
    MPI.Barrier(parallel_config.comm)
end

"""
    gather_state_for_io(S::State, G::Grid, parallel_config)

Gather distributed state to rank 0 for I/O.
Uses QGYBJ.gather_to_root which handles 2D decomposition properly.
"""
function gather_state_for_io(S::State, G::Grid, parallel_config)
    if G.decomp === nothing
        return S
    end

    # Use QGYBJ's gather function which handles 2D decomposition
    gathered_psi = QGYBJ.gather_to_root(S.psi, G, parallel_config)
    gathered_B = QGYBJ.gather_to_root(S.B, G, parallel_config)

    # Create tuple with gathered arrays (only meaningful on rank 0)
    return (psi=gathered_psi, B=gathered_B)
end

"""
    write_gathered_state_file(filepath, gathered_state, G::Grid, plans, time; params=nothing)

Write gathered state from rank 0.
"""
function write_gathered_state_file(filepath, gathered_state, G::Grid, plans, time; params=nothing)
    ensure_ncds_loaded()
    # Simple implementation - would need full state reconstruction
    @warn "Gathered state writing not fully implemented - using simplified version"
    
    NCDatasets.Dataset(filepath, "c") do ds
        ds.dim["x"] = G.nx
        ds.dim["y"] = G.ny  
        ds.dim["z"] = G.nz
        ds.dim["time"] = 1
        
        # Minimal implementation
        ds.attrib["title"] = "QG-YBJ Model State (Gathered)"
        ds.attrib["created_at"] = string(Dates.now())
        ds.attrib["model_time"] = time
    end
end

"""
    write_diagnostics_file(manager::OutputManager, diagnostics::Dict, time::Real)

Write diagnostic quantities to NetCDF file.
"""
function write_diagnostics_file(manager::OutputManager, diagnostics::Dict, time::Real)
    ensure_ncds_loaded()
    filename = "diagnostics_$(lpad(manager.diagnostics_counter, 4, '0')).nc"
    filepath = joinpath(manager.output_dir, filename)
    
    NCDatasets.Dataset(filepath, "c") do ds
        ds.dim["time"] = 1
        
        time_var = defVar(ds, "time", Float64, ("time",))
        time_var[1] = time
        time_var.attrib["units"] = "model time units"
        
        # Write diagnostic quantities
        for (name, value) in diagnostics
            if isa(value, Real)
                var = defVar(ds, string(name), Float64, ("time",))
                var[1] = value
            elseif isa(value, AbstractArray) && ndims(value) == 1
                ds.dim[string(name)*"_dim"] = length(value)
                var = defVar(ds, string(name), Float64, (string(name)*"_dim",))
                var[:] = value
            end
        end
        
        ds.attrib["title"] = "QG-YBJ Model Diagnostics"
        ds.attrib["created_at"] = string(now())
        ds.attrib["model_time"] = time
    end
    
    manager.diagnostics_counter += 1
    manager.last_diagnostics_output = time
    
    return filepath
end

"""
    read_initial_psi(filename::String, G::Grid, plans; parallel_config=nothing)

Read initial stream function from NetCDF file.
Supports both serial and parallel (2D decomposition) modes.

In parallel mode:
- Rank 0 reads the full array
- Data is scattered to all processes
"""
function read_initial_psi(filename::String, G::Grid, plans; parallel_config=nothing)
    ensure_ncds_loaded()

    # Determine if running in parallel
    is_parallel = parallel_config !== nothing && G.decomp !== nothing

    if is_parallel
        return _read_initial_psi_parallel(filename, G, plans, parallel_config)
    else
        return _read_initial_psi_serial(filename, G, plans)
    end
end

"""
    _read_initial_psi_serial(filename, G, plans)

Serial implementation of psi reading.
"""
function _read_initial_psi_serial(filename::String, G::Grid, plans)
    @info "Reading initial psi from: $filename (serial)"

    psir = zeros(Float64, G.nx, G.ny, G.nz)

    NCDatasets.Dataset(filename, "r") do ds
        # Check dimensions
        if haskey(ds.dim, "x") && haskey(ds.dim, "y") && haskey(ds.dim, "z")
            nx_file = ds.dim["x"]
            ny_file = ds.dim["y"]
            nz_file = ds.dim["z"]

            if nx_file != G.nx || ny_file != G.ny || nz_file != G.nz
                error("Grid mismatch: file ($nx_file×$ny_file×$nz_file) vs model ($(G.nx)×$(G.ny)×$(G.nz))")
            end
        end

        # Read psi variable
        if haskey(ds, "psi")
            psir[:,:,:] = ds["psi"][:,:,:]
        else
            error("Variable 'psi' not found in $filename")
        end
    end

    # Convert to spectral space
    psik = similar(psir, Complex{Float64})
    fft_forward!(psik, psir, plans)

    return psik
end

"""
    _read_initial_psi_parallel(filename, G, plans, parallel_config)

Parallel implementation: rank 0 reads, then scatters to all processes.
"""
function _read_initial_psi_parallel(filename::String, G::Grid, plans, parallel_config)
    MPI = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))

    rank = MPI.Comm_rank(parallel_config.comm)

    # Only rank 0 reads
    psik_global = nothing
    if rank == 0
        @info "Reading initial psi from: $filename (parallel, rank 0)"
        psik_global = _read_initial_psi_serial(filename, G, plans)
    end

    MPI.Barrier(parallel_config.comm)

    # Scatter from rank 0 to all processes
    psik_local = QGYBJ.scatter_from_root(psik_global, G, parallel_config)

    return psik_local
end

"""
    read_initial_waves(filename::String, G::Grid, plans; parallel_config=nothing)

Read initial wave field (L+A) from NetCDF file.
Supports both serial and parallel (2D decomposition) modes.

In parallel mode:
- Rank 0 reads the full array
- Data is scattered to all processes
"""
function read_initial_waves(filename::String, G::Grid, plans; parallel_config=nothing)
    ensure_ncds_loaded()

    # Determine if running in parallel
    is_parallel = parallel_config !== nothing && G.decomp !== nothing

    if is_parallel
        return _read_initial_waves_parallel(filename, G, plans, parallel_config)
    else
        return _read_initial_waves_serial(filename, G, plans)
    end
end

"""
    _read_initial_waves_serial(filename, G, plans)

Serial implementation of wave field reading.
"""
function _read_initial_waves_serial(filename::String, G::Grid, plans)
    @info "Reading initial wave field from: $filename (serial)"

    BRr = zeros(Float64, G.nx, G.ny, G.nz)
    BIr = zeros(Float64, G.nx, G.ny, G.nz)

    NCDatasets.Dataset(filename, "r") do ds
        # Check dimensions
        if haskey(ds.dim, "x") && haskey(ds.dim, "y") && haskey(ds.dim, "z")
            nx_file = ds.dim["x"]
            ny_file = ds.dim["y"]
            nz_file = ds.dim["z"]

            if nx_file != G.nx || ny_file != G.ny || nz_file != G.nz
                error("Grid mismatch: file ($nx_file×$ny_file×$nz_file) vs model ($(G.nx)×$(G.ny)×$(G.nz))")
            end
        end

        # Read L+A real and imaginary parts
        if haskey(ds, "LAr") && haskey(ds, "LAi")
            BRr[:,:,:] = ds["LAr"][:,:,:]
            BIr[:,:,:] = ds["LAi"][:,:,:]
        else
            error("Variables 'LAr' and 'LAi' not found in $filename")
        end
    end

    # Convert to spectral space
    BRk = similar(BRr, Complex{Float64})
    BIk = similar(BIr, Complex{Float64})
    fft_forward!(BRk, BRr, plans)
    fft_forward!(BIk, BIr, plans)

    # Combine into complex field B = BR + i*BI
    Bk = BRk + im * BIk

    return Bk
end

"""
    _read_initial_waves_parallel(filename, G, plans, parallel_config)

Parallel implementation: rank 0 reads, then scatters to all processes.
"""
function _read_initial_waves_parallel(filename::String, G::Grid, plans, parallel_config)
    MPI = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))

    rank = MPI.Comm_rank(parallel_config.comm)

    # Only rank 0 reads
    Bk_global = nothing
    if rank == 0
        @info "Reading initial waves from: $filename (parallel, rank 0)"
        Bk_global = _read_initial_waves_serial(filename, G, plans)
    end

    MPI.Barrier(parallel_config.comm)

    # Scatter from rank 0 to all processes
    Bk_local = QGYBJ.scatter_from_root(Bk_global, G, parallel_config)

    return Bk_local
end

"""
    read_stratification_profile(filename::String, nz::Int)

Read stratification profile (N²) from NetCDF file.
"""
function read_stratification_profile(filename::String, nz::Int)
    @info "Reading stratification profile from: $filename"
    ensure_ncds_loaded()
    
    N2_profile = zeros(Float64, nz)
    
    NCDatasets.Dataset(filename, "r") do ds
        if haskey(ds.dim, "z")
            nz_file = ds.dim["z"]
            if nz_file != nz
                @warn "Stratification profile length mismatch: file ($nz_file) vs model ($nz). Interpolating..."
                # TODO: Add interpolation logic
            end
        end
        
        # Look for common variable names
        var_names = ["N2", "N_squared", "buoyancy_frequency_squared", "brunt_vaisala_frequency_squared"]
        var_found = false
        
        for name in var_names
            if haskey(ds, name)
                N2_profile[:] = ds[name][1:min(nz, length(ds[name]))]
                var_found = true
                break
            end
        end
        
        if !var_found
            error("No recognized stratification variable found in $filename. Expected one of: $var_names")
        end
    end
    
    return N2_profile
end

"""
    write_stratification_profile(filename::String, z_coords, N2_profile)

Write stratification profile to NetCDF file.
"""
function write_stratification_profile(filename::String, z_coords, N2_profile)
    @info "Writing stratification profile to: $filename"
    
    NCDatasets.Dataset(filename, "c") do ds
        ds.dim["z"] = length(z_coords)
        
        z_var = defVar(ds, "z", Float64, ("z",))
        N2_var = defVar(ds, "N2", Float64, ("z",))
        
        z_var[:] = z_coords
        N2_var[:] = N2_profile
        
        z_var.attrib["units"] = "m"
        z_var.attrib["long_name"] = "height"
        N2_var.attrib["units"] = "s⁻²"
        N2_var.attrib["long_name"] = "buoyancy frequency squared"
        
        ds.attrib["title"] = "Stratification Profile"
        ds.attrib["created_at"] = string(now())
    end
end

"""
    create_empty_state_file(filepath::String, G::Grid, time::Real; metadata::Dict=Dict())

Create an empty state file template with proper structure.
"""
function create_empty_state_file(filepath::String, G::Grid, time::Real; metadata::Dict=Dict())
    NCDatasets.Dataset(filepath, "c") do ds
        # Define dimensions
        ds.dim["x"] = G.nx
        ds.dim["y"] = G.ny
        ds.dim["z"] = G.nz
        ds.dim["time"] = NCDatasets.Unlimited()
        
        # Create coordinate variables
        x_var = defVar(ds, "x", Float64, ("x",))
        y_var = defVar(ds, "y", Float64, ("y",))
        z_var = defVar(ds, "z", Float64, ("z",))
        time_var = defVar(ds, "time", Float64, ("time",))
        
        # Create data variables (initially empty)
        psi_var = defVar(ds, "psi", Float64, ("x", "y", "z", "time"))
        LAr_var = defVar(ds, "LAr", Float64, ("x", "y", "z", "time"))
        LAi_var = defVar(ds, "LAi", Float64, ("x", "y", "z", "time"))
        
        # Set attributes
        x_var.attrib["units"] = "radians"
        y_var.attrib["units"] = "radians"
        z_var.attrib["units"] = "radians"
        time_var.attrib["units"] = "model time units"
        
        psi_var.attrib["units"] = "m²/s"
        psi_var.attrib["long_name"] = "stream function"
        LAr_var.attrib["units"] = "wave amplitude"
        LAr_var.attrib["long_name"] = "L+A real part"
        LAi_var.attrib["units"] = "wave amplitude"
        LAi_var.attrib["long_name"] = "L+A imaginary part"
        
        # Global attributes
        ds.attrib["title"] = "QG-YBJ Model State Time Series"
        ds.attrib["created_at"] = string(now())
        
        for (key, value) in metadata
            ds.attrib[key] = value
        end
    end
end

# =============================================================================
# Compatibility wrappers for legacy io.jl functions
# These provide backward compatibility while using the enhanced NetCDF I/O system
# =============================================================================

"""
    ncdump_psi(S, G, plans; path="psi.out.nc")

Legacy compatibility wrapper for writing stream function to NetCDF.
Uses the enhanced I/O system internally.
"""
function ncdump_psi(S::State, G::Grid, plans; path="psi.out.nc")
    @info "Using legacy ncdump_psi (compatibility mode)"
    
    # Create a temporary output manager for this operation
    output_dir = dirname(path)
    filename = basename(path)
    
    manager = OutputManager(
        output_dir=output_dir,
        state_file_pattern=filename,
        save_psi=true,
        save_waves=false,
        save_velocities=false,
        save_vertical_velocity=false,
        save_vorticity=false
    )
    
    # Write the state file
    write_state_file(manager, S, G, plans, 0.0)
    
    return path
end

"""
    ncdump_la(S, G, plans; path="la.out.nc")

Legacy compatibility wrapper for writing L+A wave field to NetCDF.
Uses the enhanced I/O system internally.
"""
function ncdump_la(S::State, G::Grid, plans; path="la.out.nc")
    @info "Using legacy ncdump_la (compatibility mode)"
    
    # Create a temporary output manager for this operation
    output_dir = dirname(path)
    filename = basename(path)
    
    manager = OutputManager(
        output_dir=output_dir,
        state_file_pattern=filename,
        save_psi=false,
        save_waves=true,
        save_velocities=false,
        save_vertical_velocity=false,
        save_vorticity=false
    )
    
    # Write the state file
    write_state_file(manager, S, G, plans, 0.0)
    
    return path
end

"""
    ncread_psi!(S, G, plans; path="psi000.in.nc", parallel_config=nothing)

Legacy compatibility wrapper for reading stream function from NetCDF.
Uses the enhanced I/O system internally.
Supports both serial and parallel (2D decomposition) modes.
"""
function ncread_psi!(S::State, G::Grid, plans; path="psi000.in.nc", parallel_config=nothing)
    @info "Using legacy ncread_psi! (compatibility mode)"

    # Use the enhanced read function with parallel support
    psi_data = read_initial_psi(path, G, plans; parallel_config=parallel_config)

    # Copy to state (handles both Array and PencilArray)
    parent(S.psi) .= parent(psi_data)

    return S
end

"""
    ncread_la!(S, G, plans; path="la000.in.nc", parallel_config=nothing)

Legacy compatibility wrapper for reading L+A wave field from NetCDF.
Uses the enhanced I/O system internally.
Supports both serial and parallel (2D decomposition) modes.
"""
function ncread_la!(S::State, G::Grid, plans; path="la000.in.nc", parallel_config=nothing)
    @info "Using legacy ncread_la! (compatibility mode)"

    # Use the enhanced read function with parallel support
    B_data = read_initial_waves(path, G, plans; parallel_config=parallel_config)

    # Copy to state (handles both Array and PencilArray)
    parent(S.B) .= parent(B_data)

    return S
end
