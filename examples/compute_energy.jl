#=
================================================================================
    Compute Spatial Flow and Wave Kinetic Energy from NetCDF Output
================================================================================

This script reads NetCDF state files and computes spatial KE fields:
  - Flow KE(x,y,z): (1/2)(u² + v²) where u = -∂ψ/∂y, v = ∂ψ/∂x
  - Wave KE(x,y,z): (1/2)(LAr² + LAi²)

USAGE:
------
    julia --project examples/compute_energy.jl output_dir/

OUTPUT:
-------
    - energy0001.nc, energy0002.nc, ... : Spatial KE fields for each timestep

================================================================================
=#

using NCDatasets
using FFTW
using Printf

"""
    compute_gradient_spectral(field, kx, ky)

Compute horizontal gradient using spectral derivatives.
Returns (∂f/∂x, ∂f/∂y) in physical space.
"""
function compute_gradient_spectral(field::Array{T,3}, kx::Vector, ky::Vector) where T
    nz, nx, ny = size(field)

    # FFT to spectral space (2D horizontal)
    field_hat = fft(field, (2,3))

    # Compute spectral derivatives
    dfdx_hat = similar(field_hat)
    dfdy_hat = similar(field_hat)

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                dfdx_hat[k, i, j] = im * kx[i] * field_hat[k, i, j]
                dfdy_hat[k, i, j] = im * ky[j] * field_hat[k, i, j]
            end
        end
    end

    # Back to physical space
    dfdx = real.(ifft(dfdx_hat, (2,3)))
    dfdy = real.(ifft(dfdy_hat, (2,3)))

    return dfdx, dfdy
end

"""
    compute_wavenumbers(nx, ny, Lx, Ly)

Compute wavenumber arrays for spectral derivatives.
"""
function compute_wavenumbers(nx::Int, ny::Int, Lx::Real=2π, Ly::Real=2π)
    # kx wavenumbers
    kx = zeros(nx)
    for i in 1:nx
        if i <= nx÷2 + 1
            kx[i] = (i-1) * 2π / Lx
        else
            kx[i] = (i - nx - 1) * 2π / Lx
        end
    end

    # ky wavenumbers
    ky = zeros(ny)
    for j in 1:ny
        if j <= ny÷2 + 1
            ky[j] = (j-1) * 2π / Ly
        else
            ky[j] = (j - ny - 1) * 2π / Ly
        end
    end

    return kx, ky
end

"""
    compute_flow_ke(psi, kx, ky)

Compute spatial flow kinetic energy field from streamfunction.

Flow KE(x,y,z) = (1/2)(u² + v²)
where u = -∂ψ/∂y, v = ∂ψ/∂x (geostrophic balance)

Returns: KE field, u field, v field
"""
function compute_flow_ke(psi::Array{T,3}, kx::Vector, ky::Vector) where T
    # Compute velocity from streamfunction
    dpsidx, dpsidy = compute_gradient_spectral(psi, kx, ky)

    u = -dpsidy  # u = -∂ψ/∂y
    v = dpsidx   # v = ∂ψ/∂x

    # KE(x,y,z) = (1/2)(u² + v²)
    ke = 0.5 .* (u.^2 .+ v.^2)

    return ke, u, v
end

"""
    compute_wave_ke(LAr, LAi)

Compute spatial wave kinetic energy field from wave envelope.

Wave KE(x,y,z) = (1/2)|B|² = (1/2)(LAr² + LAi²)
where B = LAr + i*LAi is the wave envelope
"""
function compute_wave_ke(LAr::Array{T,3}, LAi::Array{T,3}) where T
    ke = 0.5 .* (LAr.^2 .+ LAi.^2)
    return ke
end

"""
    process_and_save(input_file, output_file, kx, ky)

Process a single state file and save spatial KE fields.
"""
function process_and_save(input_file::String, output_file::String, kx::Vector, ky::Vector)
    # Read input
    ds_in = NCDataset(input_file, "r")

    time = haskey(ds_in, "time") ? (length(ds_in["time"]) > 0 ? ds_in["time"][1] : ds_in["time"][]) : 0.0
    x = Array(ds_in["x"][:])
    y = Array(ds_in["y"][:])
    z = Array(ds_in["z"][:])
    psi_xyz = Array(ds_in["psi"][:,:,:])
    LAr_xyz = Array(ds_in["LAr"][:,:,:])
    LAi_xyz = Array(ds_in["LAi"][:,:,:])

    close(ds_in)

    # Convert to internal (z, x, y) order for spectral derivatives
    psi = permutedims(psi_xyz, (3, 1, 2))
    LAr = permutedims(LAr_xyz, (3, 1, 2))
    LAi = permutedims(LAi_xyz, (3, 1, 2))

    # Compute spatial KE fields
    flow_ke, u, v = compute_flow_ke(psi, kx, ky)
    wave_ke = compute_wave_ke(LAr, LAi)
    total_ke = flow_ke .+ wave_ke

    # Convert back to file order (x, y, z)
    flow_ke_xyz = permutedims(flow_ke, (2, 3, 1))
    wave_ke_xyz = permutedims(wave_ke, (2, 3, 1))
    total_ke_xyz = permutedims(total_ke, (2, 3, 1))
    u_xyz = permutedims(u, (2, 3, 1))
    v_xyz = permutedims(v, (2, 3, 1))

    # Write output
    NCDataset(output_file, "c") do ds
        # Dimensions
        nx, ny, nz = size(psi_xyz)
        ds.dim["x"] = nx
        ds.dim["y"] = ny
        ds.dim["z"] = nz

        # Coordinates
        x_var = defVar(ds, "x", Float64, ("x",))
        y_var = defVar(ds, "y", Float64, ("y",))
        z_var = defVar(ds, "z", Float64, ("z",))
        time_var = defVar(ds, "time", Float64, ())

        x_var[:] = x
        y_var[:] = y
        z_var[:] = z
        time_var[] = time

        x_var.attrib["units"] = "radians"
        y_var.attrib["units"] = "radians"
        z_var.attrib["units"] = "nondimensional"
        time_var.attrib["units"] = "seconds"

        # Flow KE field
        flow_ke_var = defVar(ds, "flow_KE", Float64, ("x", "y", "z"))
        flow_ke_var[:,:,:] = flow_ke_xyz
        flow_ke_var.attrib["units"] = "m²/s²"
        flow_ke_var.attrib["long_name"] = "flow kinetic energy"
        flow_ke_var.attrib["formula"] = "KE = (1/2)(u² + v²)"

        # Wave KE field
        wave_ke_var = defVar(ds, "wave_KE", Float64, ("x", "y", "z"))
        wave_ke_var[:,:,:] = wave_ke_xyz
        wave_ke_var.attrib["units"] = "m²/s²"
        wave_ke_var.attrib["long_name"] = "wave kinetic energy"
        wave_ke_var.attrib["formula"] = "KE = (1/2)(LAr² + LAi²)"

        # Total KE field
        total_ke_var = defVar(ds, "total_KE", Float64, ("x", "y", "z"))
        total_ke_var[:,:,:] = total_ke_xyz
        total_ke_var.attrib["units"] = "m²/s²"
        total_ke_var.attrib["long_name"] = "total kinetic energy"

        # Velocity fields
        u_var = defVar(ds, "u", Float64, ("x", "y", "z"))
        v_var = defVar(ds, "v", Float64, ("x", "y", "z"))
        u_var[:,:,:] = u_xyz
        v_var[:,:,:] = v_xyz
        u_var.attrib["units"] = "m/s"
        u_var.attrib["long_name"] = "zonal velocity (u = -∂ψ/∂y)"
        v_var.attrib["units"] = "m/s"
        v_var.attrib["long_name"] = "meridional velocity (v = ∂ψ/∂x)"

        # Global attributes
        ds.attrib["title"] = "QG-YBJ Kinetic Energy Fields"
        ds.attrib["source_file"] = basename(input_file)
        ds.attrib["flow_KE_formula"] = "KE_flow = (1/2)|∇ψ|² = (1/2)(u² + v²)"
        ds.attrib["wave_KE_formula"] = "KE_wave = (1/2)|B|² = (1/2)(LAr² + LAi²)"
    end
end

"""
    process_output_directory(output_dir)

Process all state files and create corresponding energy files.
"""
function process_output_directory(output_dir::String)
    # Find all state files
    files = filter(f -> startswith(f, "state") && endswith(f, ".nc"), readdir(output_dir))
    sort!(files)

    if isempty(files)
        error("No state files found in $output_dir")
    end

    println("Found $(length(files)) state files")

    # Read grid info from first file
    first_file = joinpath(output_dir, files[1])
    ds = NCDataset(first_file, "r")
    x = Array(ds["x"][:])
    y = Array(ds["y"][:])
    nx, ny = length(x), length(y)
    close(ds)

    Lx = x[end] - x[1] + (x[2] - x[1])
    Ly = y[end] - y[1] + (y[2] - y[1])

    println("Grid: $nx × $ny, Domain: Lx = $(round(Lx, digits=4)), Ly = $(round(Ly, digits=4))")

    # Compute wavenumbers
    kx, ky = compute_wavenumbers(nx, ny, Lx, Ly)

    # Process each file
    println("\nProcessing files...")
    println("-"^50)

    for (i, file) in enumerate(files)
        input_path = joinpath(output_dir, file)

        # Create output filename: state0001.nc -> energy0001.nc
        output_name = replace(file, "state" => "energy")
        output_path = joinpath(output_dir, output_name)

        process_and_save(input_path, output_path, kx, ky)

        @printf("  %s -> %s\n", file, output_name)
    end

    println("-"^50)
    println("Created $(length(files)) energy files")

    return length(files)
end

# ============================================================================
#                       MAIN
# ============================================================================

function main()
    if length(ARGS) < 1
        println("Usage: julia --project examples/compute_energy.jl <output_dir>")
        println("")
        println("Computes spatial KE fields from NetCDF state files.")
        println("")
        println("Input:  state0001.nc, state0002.nc, ... (containing psi, LAr, LAi)")
        println("Output: energy0001.nc, energy0002.nc, ... (containing flow_KE, wave_KE, u, v)")
        println("")
        println("Variables in output files:")
        println("  flow_KE(x,y,z) = (1/2)(u² + v²)    [geostrophic KE]")
        println("  wave_KE(x,y,z) = (1/2)(LAr² + LAi²) [wave KE]")
        println("  total_KE(x,y,z) = flow_KE + wave_KE")
        println("  u(x,y,z) = -∂ψ/∂y                  [zonal velocity]")
        println("  v(x,y,z) = ∂ψ/∂x                   [meridional velocity]")
        return
    end

    output_dir = ARGS[1]

    if !isdir(output_dir)
        error("Directory not found: $output_dir")
    end

    println("="^50)
    println("Computing Spatial KE Fields")
    println("="^50)
    println("Input directory: $output_dir")
    println("")

    n_files = process_output_directory(output_dir)

    println("\n" * "="^50)
    println("Done! Created $n_files energy files.")
    println("="^50)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
