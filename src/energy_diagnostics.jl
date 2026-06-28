#=
================================================================================
                    energy_diagnostics.jl - Separate Energy Output Files
================================================================================

This module provides functionality to save energy diagnostics to separate files
in a dedicated "diagnostic" folder, following the structure:

    output_dir/
    └── diagnostic/
        ├── wave_KE.nc        # Wave kinetic energy time series
        ├── wave_PE.nc        # Wave potential energy time series
        ├── mean_flow_KE.nc   # Mean flow kinetic energy time series
        └── mean_flow_PE.nc   # Mean flow potential energy time series

Each file contains:
- time: Time coordinate (unlimited dimension)
- energy: The energy value at each time
- Metadata about the computation

This matches the QG_YBJp Fortran code organization where wave_energy and
diag_zentrum output to separate files.

================================================================================
=#

module EnergyDiagnostics

using Printf
using Dates
using NCDatasets

# NCDatasets is always available since it's a required dependency (using NCDatasets above)
const HAS_NCDS = true

"""
    EnergyDiagnosticsManager

Manages separate energy diagnostic output files.
"""
mutable struct EnergyDiagnosticsManager{T}
    diagnostic_dir::String

    # File paths
    wave_KE_file::String
    wave_PE_file::String
    wave_CE_file::String      # Wave correction energy (YBJ+)
    mean_flow_KE_file::String
    mean_flow_PE_file::String
    total_energy_file::String

    # Output interval
    output_interval::T
    last_output_time::T

    # Data storage for time series (appended each output)
    time_series::Vector{T}
    wave_KE_series::Vector{T}
    wave_PE_series::Vector{T}
    wave_CE_series::Vector{T}
    mean_flow_KE_series::Vector{T}
    mean_flow_PE_series::Vector{T}

    # Flags
    initialized::Bool
end

"""
    EnergyDiagnosticsManager(output_dir::String; output_interval=1.0)

Create an energy diagnostics manager that saves to output_dir/diagnostic/.

# Arguments
- `output_dir::String`: Base output directory
- `output_interval::T`: Time interval between diagnostic outputs (default: 1.0)

# Returns
EnergyDiagnosticsManager that handles separate energy files.
"""
function EnergyDiagnosticsManager(output_dir::String; output_interval::T=1.0) where T<:Real
    # Create diagnostic subdirectory
    diagnostic_dir = joinpath(output_dir, "diagnostic")
    mkpath(diagnostic_dir)

    return EnergyDiagnosticsManager{T}(
        diagnostic_dir,
        joinpath(diagnostic_dir, "wave_KE.nc"),
        joinpath(diagnostic_dir, "wave_PE.nc"),
        joinpath(diagnostic_dir, "wave_CE.nc"),
        joinpath(diagnostic_dir, "mean_flow_KE.nc"),
        joinpath(diagnostic_dir, "mean_flow_PE.nc"),
        joinpath(diagnostic_dir, "total_energy.nc"),
        T(output_interval),
        T(-Inf),  # last_output_time (start with -Inf to trigger first output)
        T[],      # time_series
        T[],      # wave_KE_series
        T[],      # wave_PE_series
        T[],      # wave_CE_series
        T[],      # mean_flow_KE_series
        T[],      # mean_flow_PE_series
        false     # initialized
    )
end

"""
    should_output(manager::EnergyDiagnosticsManager, time::Real)

Check if it's time to output energy diagnostics.
"""
function should_output(manager::EnergyDiagnosticsManager, time::Real)
    return (time - manager.last_output_time) >= manager.output_interval
end

"""
    record_energies!(manager, time, wave_KE, wave_PE, wave_CE, mean_flow_KE, mean_flow_PE)

Record energy values to the internal time series.
"""
function record_energies!(manager::EnergyDiagnosticsManager{T},
                          time::Real,
                          wave_KE::Real,
                          wave_PE::Real,
                          wave_CE::Real,
                          mean_flow_KE::Real,
                          mean_flow_PE::Real) where T
    push!(manager.time_series, T(time))
    push!(manager.wave_KE_series, T(wave_KE))
    push!(manager.wave_PE_series, T(wave_PE))
    push!(manager.wave_CE_series, T(wave_CE))
    push!(manager.mean_flow_KE_series, T(mean_flow_KE))
    push!(manager.mean_flow_PE_series, T(mean_flow_PE))

    manager.last_output_time = T(time)
    manager.initialized = true
end

"""
    write_energy_file(filepath, varname, times, values;
                      units="nondimensional", long_name="energy")

Write a single energy time series to a NetCDF file.
"""
function write_energy_file(filepath::String, varname::String,
                           times::Vector{T}, values::Vector{T};
                           units::String="nondimensional",
                           long_name::String="energy") where T
    if !HAS_NCDS
        error("NCDatasets not available. Install NCDatasets.jl or skip NetCDF I/O.")
    end

    NCDatasets.Dataset(filepath, "c") do ds
        # Define time dimension (unlimited for appending)
        ds.dim["time"] = length(times)

        # Time coordinate
        time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))
        time_var[:] = times
        time_var.attrib["units"] = "model time units"
        time_var.attrib["long_name"] = "simulation time"

        # Energy variable
        energy_var = NCDatasets.defVar(ds, varname, Float64, ("time",))
        energy_var[:] = values
        energy_var.attrib["units"] = units
        energy_var.attrib["long_name"] = long_name

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Energy Diagnostic: $long_name"
        ds.attrib["created_at"] = string(Dates.now())
        ds.attrib["n_timesteps"] = length(times)
        if !isempty(times)
            ds.attrib["time_start"] = times[1]
            ds.attrib["time_end"] = times[end]
        end
    end

    return filepath
end

"""
    write_all_energy_files!(manager::EnergyDiagnosticsManager)

Write all energy time series to their respective files.
"""
function write_all_energy_files!(manager::EnergyDiagnosticsManager)
    if isempty(manager.time_series)
        @warn "No energy data to write"
        return
    end

    # Write wave KE
    write_energy_file(
        manager.wave_KE_file, "wave_KE",
        manager.time_series, manager.wave_KE_series;
        units="nondimensional",
        long_name="wave kinetic energy"
    )
    @info "Wrote wave KE to: $(manager.wave_KE_file)"

    # Write wave PE
    write_energy_file(
        manager.wave_PE_file, "wave_PE",
        manager.time_series, manager.wave_PE_series;
        units="nondimensional",
        long_name="wave potential energy"
    )
    @info "Wrote wave PE to: $(manager.wave_PE_file)"

    # Write wave CE (correction energy from YBJ+)
    write_energy_file(
        manager.wave_CE_file, "wave_CE",
        manager.time_series, manager.wave_CE_series;
        units="nondimensional",
        long_name="wave correction energy (YBJ+)"
    )
    @info "Wrote wave CE to: $(manager.wave_CE_file)"

    # Write mean flow KE
    write_energy_file(
        manager.mean_flow_KE_file, "mean_flow_KE",
        manager.time_series, manager.mean_flow_KE_series;
        units="nondimensional",
        long_name="mean flow kinetic energy"
    )
    @info "Wrote mean flow KE to: $(manager.mean_flow_KE_file)"

    # Write mean flow PE
    write_energy_file(
        manager.mean_flow_PE_file, "mean_flow_PE",
        manager.time_series, manager.mean_flow_PE_series;
        units="nondimensional",
        long_name="mean flow potential energy"
    )
    @info "Wrote mean flow PE to: $(manager.mean_flow_PE_file)"

    # Write total energy summary file
    write_total_energy_file!(manager)
end

"""
    write_total_energy_file!(manager::EnergyDiagnosticsManager)

Write a summary file containing all energies in one place.
"""
function write_total_energy_file!(manager::EnergyDiagnosticsManager{T}) where T
    if !HAS_NCDS
        return
    end

    NCDatasets.Dataset(manager.total_energy_file, "c") do ds
        nt = length(manager.time_series)
        ds.dim["time"] = nt

        # Time coordinate
        time_var = NCDatasets.defVar(ds, "time", Float64, ("time",))
        time_var[:] = manager.time_series
        time_var.attrib["units"] = "model time units"
        time_var.attrib["long_name"] = "simulation time"

        # All energy variables
        wke_var = NCDatasets.defVar(ds, "wave_KE", Float64, ("time",))
        wke_var[:] = manager.wave_KE_series
        wke_var.attrib["long_name"] = "wave kinetic energy"

        wpe_var = NCDatasets.defVar(ds, "wave_PE", Float64, ("time",))
        wpe_var[:] = manager.wave_PE_series
        wpe_var.attrib["long_name"] = "wave potential energy"

        wce_var = NCDatasets.defVar(ds, "wave_CE", Float64, ("time",))
        wce_var[:] = manager.wave_CE_series
        wce_var.attrib["long_name"] = "wave correction energy"

        mke_var = NCDatasets.defVar(ds, "mean_flow_KE", Float64, ("time",))
        mke_var[:] = manager.mean_flow_KE_series
        mke_var.attrib["long_name"] = "mean flow kinetic energy"

        mpe_var = NCDatasets.defVar(ds, "mean_flow_PE", Float64, ("time",))
        mpe_var[:] = manager.mean_flow_PE_series
        mpe_var.attrib["long_name"] = "mean flow potential energy"

        # Computed totals
        total_wave = manager.wave_KE_series .+ manager.wave_PE_series .+ manager.wave_CE_series
        total_flow = manager.mean_flow_KE_series .+ manager.mean_flow_PE_series
        total_all = total_wave .+ total_flow

        twave_var = NCDatasets.defVar(ds, "total_wave_energy", Float64, ("time",))
        twave_var[:] = total_wave
        twave_var.attrib["long_name"] = "total wave energy (KE + PE + CE)"

        tflow_var = NCDatasets.defVar(ds, "total_flow_energy", Float64, ("time",))
        tflow_var[:] = total_flow
        tflow_var.attrib["long_name"] = "total mean flow energy (KE + PE)"

        ttotal_var = NCDatasets.defVar(ds, "total_energy", Float64, ("time",))
        ttotal_var[:] = total_all
        ttotal_var.attrib["long_name"] = "total system energy"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Total Energy Summary"
        ds.attrib["created_at"] = string(Dates.now())
        ds.attrib["n_timesteps"] = nt
        ds.attrib["description"] = "All energy components in one file for convenience"
    end

    @info "Wrote total energy summary to: $(manager.total_energy_file)"
end

"""
    append_and_write!(manager, time, wave_KE, wave_PE, wave_CE, mean_flow_KE, mean_flow_PE)

Record energies and immediately write all files (overwrites previous).
Use this for real-time monitoring of energy evolution.
"""
function append_and_write!(manager::EnergyDiagnosticsManager{T},
                           time::Real,
                           wave_KE::Real,
                           wave_PE::Real,
                           wave_CE::Real,
                           mean_flow_KE::Real,
                           mean_flow_PE::Real) where T
    record_energies!(manager, time, wave_KE, wave_PE, wave_CE, mean_flow_KE, mean_flow_PE)
    write_all_energy_files!(manager)
end

"""
    finalize!(manager::EnergyDiagnosticsManager)

Write all accumulated energy data to files. Call this at the end of simulation.
"""
function finalize!(manager::EnergyDiagnosticsManager)
    if manager.initialized && !isempty(manager.time_series)
        write_all_energy_files!(manager)
        @info "Energy diagnostics finalized: $(length(manager.time_series)) time steps written"
    end
end

export EnergyDiagnosticsManager, should_output, record_energies!
export write_all_energy_files!, append_and_write!, finalize!

end # module
