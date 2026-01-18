#=
================================================================================
    Compute Spatial Flow and Wave Kinetic Energy from NetCDF Output
================================================================================

This script reads NetCDF state files and computes spatial KE fields:
  - Flow KE(x,y,z): (1/2)(u² + v²) where u = -∂ψ/∂y, v = ∂ψ/∂x
  - Wave KE(x,y,z): (1/2)|LA|² per YBJ+ equation (4.7)
                    where L = ∂_z(f²/N² × ∂_z) from equation (1.3)
                    and LA = ∂_z(a(z) × C) with a(z) = f²/N², C = ∂A/∂z

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
    compute_wave_ke(Ar, Ai, a_ell, Δz)

Compute spatial wave kinetic energy field per YBJ+ equation (4.7).

Wave KE(x,y,z) = (1/2)|LA|²

where L = ∂_z(f²/N² × ∂_z) from equation (1.3), computed as:
  LA = ∂_z(a(z) × C)  where a(z) = f²/N² and C = ∂A/∂z

This matches the discretization in src/diagnostics.jl:wave_energy_spectral:
  1. C[k] = (A[k+1] - A[k]) / Δz  for k = 1:nz-1, C[nz] = 0 (Neumann BC)
  2. LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
  3. Boundary: LA[1] = a[1] × C[1] / Δz, LA[nz] = -a[nz-1] × C[nz-1] / Δz

# Arguments
- `Ar, Ai`: Real and imaginary parts of wave amplitude A in physical space (nz, nx, ny)
- `a_ell`: Array of f²/N² values at cell centers (length nz). a_ell[k] is used for interface k.
- `Δz`: Vertical grid spacing

# Returns
Spatial wave kinetic energy field (1/2)|LA|²
"""
function compute_wave_ke(Ar::Array{T,3}, Ai::Array{T,3},
                         a_ell::Vector, Δz::Real) where T
    nz, nx, ny = size(Ar)

    # Initialize LA arrays
    LA_r = zeros(T, nz, nx, ny)
    LA_i = zeros(T, nz, nx, ny)

    if nz == 1
        # Single layer: LA = 0
        return 0.5 .* (LA_r.^2 .+ LA_i.^2)
    end

    # Step 1: Pre-compute C = ∂A/∂z at interfaces (matches elliptic.jl)
    # C[k] = (A[k+1] - A[k]) / Δz for k = 1:nz-1
    # C[nz] = 0 (Neumann BC at bottom)
    CR = zeros(T, nz, nx, ny)
    CI = zeros(T, nz, nx, ny)

    for j in 1:ny
        for i in 1:nx
            for k in 1:nz-1
                CR[k, i, j] = (Ar[k+1, i, j] - Ar[k, i, j]) / Δz
                CI[k, i, j] = (Ai[k+1, i, j] - Ai[k, i, j]) / Δz
            end
            CR[nz, i, j] = 0.0  # Neumann BC
            CI[nz, i, j] = 0.0
        end
    end

    # Step 2: Compute LA = ∂_z(a × C) (matches diagnostics.jl:wave_energy_spectral)
    # LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
    for j in 1:ny
        for i in 1:nx
            for k in 1:nz
                # Get a_ell values (with bounds checking matching main code)
                a_ell_k = k <= length(a_ell) ? a_ell[k] : a_ell[end]

                if k == 1
                    # Bottom boundary: LA[1] = a[1] × C[1] / Δz
                    # (assuming a[0] × C[0] = 0 from BC)
                    LA_r[k, i, j] = a_ell_k * CR[k, i, j] / Δz
                    LA_i[k, i, j] = a_ell_k * CI[k, i, j] / Δz
                elseif k == nz
                    # Top boundary: LA[nz] = -a[nz-1] × C[nz-1] / Δz
                    # (since C[nz] = 0 from BC)
                    a_ell_km1 = (k - 1) <= length(a_ell) ? a_ell[k - 1] : a_ell[end]
                    LA_r[k, i, j] = -a_ell_km1 * CR[k-1, i, j] / Δz
                    LA_i[k, i, j] = -a_ell_km1 * CI[k-1, i, j] / Δz
                else
                    # Interior: LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
                    a_ell_km1 = (k - 1) <= length(a_ell) ? a_ell[k - 1] : a_ell[end]
                    LA_r[k, i, j] = (a_ell_k * CR[k, i, j] - a_ell_km1 * CR[k-1, i, j]) / Δz
                    LA_i[k, i, j] = (a_ell_k * CI[k, i, j] - a_ell_km1 * CI[k-1, i, j]) / Δz
                end
            end
        end
    end

    # WKE = (1/2)|LA|²
    ke = 0.5 .* (LA_r.^2 .+ LA_i.^2)
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
    Br_xyz = Array(ds_in["LAr"][:,:,:])  # B = L⁺A (wave envelope)
    Bi_xyz = Array(ds_in["LAi"][:,:,:])

    # Read A (wave amplitude) if available - needed for correct WKE per equation (4.7)
    has_A = haskey(ds_in, "Ar") && haskey(ds_in, "Ai")
    Ar_xyz = has_A ? Array(ds_in["Ar"][:,:,:]) : nothing
    Ai_xyz = has_A ? Array(ds_in["Ai"][:,:,:]) : nothing

    # Read a_ell (f²/N² profile) if available - needed for L operator computation
    has_a_ell = haskey(ds_in, "a_ell")
    a_ell = has_a_ell ? Array(ds_in["a_ell"][:]) : nothing

    # Compute Δz from z coordinates
    nz = length(z)
    Δz = nz > 1 ? abs(z[2] - z[1]) : 1.0

    close(ds_in)

    # Convert to internal (z, x, y) order for spectral derivatives
    psi = permutedims(psi_xyz, (3, 1, 2))
    Br = permutedims(Br_xyz, (3, 1, 2))
    Bi = permutedims(Bi_xyz, (3, 1, 2))
    Ar = has_A ? permutedims(Ar_xyz, (3, 1, 2)) : nothing
    Ai = has_A ? permutedims(Ai_xyz, (3, 1, 2)) : nothing

    # Compute spatial KE fields
    flow_ke, u, v = compute_flow_ke(psi, kx, ky)

    # Compute wave KE per equation (4.7): WKE = (1/2)|LA|²
    # where L = ∂_z(f²/N² × ∂_z) from equation (1.3)
    if has_A && has_a_ell
        wave_ke = compute_wave_ke(Ar, Ai, a_ell, Δz)
    elseif has_A
        # Fallback: use constant a_ell = 1.0 (assumes f²/N² = 1)
        # a_ell has length nz (cell-centered values)
        @warn "a_ell not found in $input_file. Using constant a_ell=1.0 approximation for WKE." maxlog=1
        a_ell_approx = ones(size(Ar, 1))
        wave_ke = compute_wave_ke(Ar, Ai, a_ell_approx, Δz)
    else
        # Fallback for old files without A: use |B|² ≈ |LA|² (approximation)
        @warn "Wave amplitude A not found in $input_file. Using |B|² approximation for WKE." maxlog=1
        wave_ke = 0.5 .* (Br.^2 .+ Bi.^2)
    end
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
        wave_ke_var.attrib["long_name"] = "wave kinetic energy per YBJ+ equation (4.7)"
        wave_ke_var.attrib["formula"] = "KE = (1/2)|LA|² where L = ∂_z(f²/N² × ∂_z)"

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
        ds.attrib["title"] = "QG-YBJ+ Kinetic Energy Fields"
        ds.attrib["source_file"] = basename(input_file)
        ds.attrib["flow_KE_formula"] = "KE_flow = (1/2)|∇ψ|² = (1/2)(u² + v²)"
        ds.attrib["wave_KE_formula"] = "KE_wave = (1/2)|LA|² per YBJ+ equation (4.7)"
        ds.attrib["note"] = "LA computed using L = ∂_z(f²/N² × ∂_z) from equation (1.3)"
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
        println("Input:  state0001.nc, state0002.nc, ... (containing psi, LAr, LAi, Ar, Ai, a_ell)")
        println("        - a_ell: f²/N² at cell centers (length nz), used for L operator")
        println("Output: energy0001.nc, energy0002.nc, ... (containing flow_KE, wave_KE, u, v)")
        println("")
        println("Variables in output files:")
        println("  flow_KE(x,y,z) = (1/2)(u² + v²)    [geostrophic KE]")
        println("  wave_KE(x,y,z) = (1/2)|LA|²        [wave KE per YBJ+ eq. 4.7]")
        println("                   where L = ∂_z(f²/N² × ∂_z) from eq. (1.3)")
        println("                   LA[k] = (a[k]×C[k] - a[k-1]×C[k-1])/Δz, C = ∂A/∂z")
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
