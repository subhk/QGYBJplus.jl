"""
Model initialization module for QG-YBJ model.

Provides functions for initializing fields from various sources:
- Analytical expressions
- Random fields with specified spectra
- NetCDF files
- Combinations of the above
"""

using Random
using LinearAlgebra
using ..QGYBJ: Grid, State, QGParams
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!, compute_wavenumbers!
using ..QGYBJ: local_to_global

"""
    initialize_from_config(config::ModelConfig, G::Grid, S::State, plans)

Initialize model state from configuration.
"""
function initialize_from_config(config, G::Grid, S::State, plans)
    @info "Initializing model fields from configuration"
    
    # Set random seed for reproducibility
    Random.seed!(config.initial_conditions.random_seed)
    
    # Initialize stream function
    if config.initial_conditions.psi_type == :analytical
        init_analytical_psi!(S.psi, G, config.initial_conditions.psi_amplitude)
    elseif config.initial_conditions.psi_type == :random
        init_random_psi!(S.psi, G, config.initial_conditions.psi_amplitude)
    elseif config.initial_conditions.psi_type == :from_file
        S.psi .= read_initial_psi(config.initial_conditions.psi_filename, G, plans)
    else
        # Zero initialization
        S.psi .= 0.0
    end
    
    # Initialize wave field  
    if config.initial_conditions.wave_type == :analytical
        init_analytical_waves!(S.B, G, config.initial_conditions.wave_amplitude)
    elseif config.initial_conditions.wave_type == :random
        init_random_waves!(S.B, G, config.initial_conditions.wave_amplitude)
    elseif config.initial_conditions.wave_type == :from_file
        S.B .= read_initial_waves(config.initial_conditions.wave_filename, G, plans)
    else
        # Zero initialization
        S.B .= 0.0
    end
    
    @info "Model initialization complete"
end

"""
    init_analytical_psi!(psik, G::Grid, amplitude::Real)

Initialize stream function with analytical expression.
Based on the generate_fields_stag routine from Fortran code.
"""
function init_analytical_psi!(psik, G::Grid, amplitude::Real)
    @info "Initializing analytical stream function (amplitude=$amplitude)"
    
    # Create wavenumber arrays
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    # Initialize in real space first
    psir = zeros(Float64, G.nx, G.ny, G.nz)
    
    dx = 2π / G.nx
    dy = 2π / G.ny
    dz = 2π / G.nz
    
    for k in 1:G.nz
        z = (k - 1) * dz
        for j in 1:G.ny
            y = (j - 1) * dy
            for i in 1:G.nx
                x = (i - 1) * dx
                
                # Example: sum of Rossby waves with different modes
                # This mimics typical geostrophic turbulence patterns
                psir[i,j,k] = amplitude * (
                    sin(2*x) * cos(y) * cos(z) +
                    0.5 * cos(x) * sin(2*y) * sin(z) +
                    0.3 * sin(3*x) * sin(y) * cos(2*z) +
                    0.2 * cos(2*x) * cos(3*y) * sin(2*z)
                )
            end
        end
    end
    
    # Transform to spectral space
    fft_forward!(psik, psir, plans)
end

"""
    init_random_psi!(psik, G::Grid, amplitude::Real; slope::Real=-3.0)

Initialize stream function with random field having specified spectral slope.
"""
function init_random_psi!(psik, G::Grid, amplitude::Real; slope::Real=-3.0)
    @info "Initializing random stream function (amplitude=$amplitude, slope=$slope)"
    
    # Generate random phases
    phases = 2π * rand(Float64, G.nx÷2+1, G.ny, G.nz)
    
    # Create spectral field with desired slope
    fill!(psik, 0.0)
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:(G.nx÷2+1)
                kx = i-1
                
                if kx == 0 && ky == 0
                    continue  # Skip mean mode
                end
                
                # Total wavenumber
                k_total = sqrt(Float64(kx^2 + ky^2))
                
                if k_total > 0
                    # Energy spectrum E(k) ∝ k^slope
                    energy = amplitude * k_total^slope
                    
                    # Random amplitude with specified energy
                    amp = sqrt(2 * energy) * randn()
                    
                    # Set complex amplitude
                    phase = phases[i, j, k]
                    psik[i, j, k] = amp * cis(phase)
                end
            end
        end
    end
    
    # Apply dealiasing mask if needed
    apply_dealiasing_mask!(psik, G)
end

"""
    init_analytical_waves!(Bk, G::Grid, amplitude::Real)

Initialize wave field (L+A) with analytical expression.
"""
function init_analytical_waves!(Bk, G::Grid, amplitude::Real)
    @info "Initializing analytical wave field (amplitude=$amplitude)"
    
    # Initialize in real space
    Br = zeros(Float64, G.nx, G.ny, G.nz)
    Bi = zeros(Float64, G.nx, G.ny, G.nz)
    
    dx = 2π / G.nx
    dy = 2π / G.ny  
    dz = 2π / G.nz
    
    for k in 1:G.nz
        z = (k - 1) * dz
        for j in 1:G.ny
            y = (j - 1) * dy
            for i in 1:G.nx
                x = (i - 1) * dx
                
                # Example wave pattern - can be modified based on physics
                Br[i,j,k] = amplitude * (
                    sin(4*x + z) * cos(2*y) * exp(-((z-π)^2)/(2*0.5^2)) +
                    0.3 * cos(2*x) * sin(4*y + 2*z) * exp(-((z-π)^2)/(2*0.3^2))
                )
                
                Bi[i,j,k] = amplitude * 0.1 * (
                    cos(4*x + z) * sin(2*y) * exp(-((z-π)^2)/(2*0.5^2)) +
                    0.3 * sin(2*x) * cos(4*y + 2*z) * exp(-((z-π)^2)/(2*0.3^2))
                )
            end
        end
    end
    
    # Transform to spectral space
    Brk = similar(Bk)
    Bik = similar(Bk)
    fft_forward!(Brk, Br, plans)
    fft_forward!(Bik, Bi, plans)
    
    # Combine real and imaginary parts
    Bk .= Brk .+ im .* Bik
end

"""
    init_random_waves!(Bk, G::Grid, amplitude::Real; slope::Real=-2.0)

Initialize wave field with random amplitudes and phases.
"""
function init_random_waves!(Bk, G::Grid, amplitude::Real; slope::Real=-2.0)
    @info "Initializing random wave field (amplitude=$amplitude, slope=$slope)"
    
    # Generate random phases for real and imaginary parts
    phases_r = 2π * rand(Float64, G.nx÷2+1, G.ny, G.nz)
    phases_i = 2π * rand(Float64, G.nx÷2+1, G.ny, G.nz)
    
    fill!(Bk, 0.0)
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        # Add some vertical structure - stronger near middle depths
        z_factor = sin(π * k / G.nz)^2
        
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:(G.nx÷2+1)
                kx = i-1
                
                if kx == 0 && ky == 0
                    continue  # Skip mean mode
                end
                
                k_total = sqrt(Float64(kx^2 + ky^2))
                
                if k_total > 0
                    # Energy spectrum for waves
                    energy = amplitude^2 * k_total^slope * z_factor
                    
                    # Random amplitudes
                    amp_r = sqrt(energy) * randn()
                    amp_i = sqrt(energy) * randn()
                    
                    # Set complex field
                    Bk[i, j, k] = (amp_r * cis(phases_r[i, j, k])) + 
                                  im * (amp_i * cis(phases_i[i, j, k]))
                end
            end
        end
    end
    
    # Apply dealiasing mask
    apply_dealiasing_mask!(Bk, G)
end

"""
    init_zero_mean_flow!(psik)

Initialize with zero mean flow (fixed flow case).
"""
function init_zero_mean_flow!(psik)
    @info "Initializing zero mean flow"
    fill!(psik, 0.0)
end

"""
    apply_dealiasing_mask!(field, G::Grid)

Apply 2/3 dealiasing mask to spectral field.
"""
function apply_dealiasing_mask!(field, G::Grid)
    kx_max = G.nx ÷ 3  # 2/3 rule
    ky_max = G.ny ÷ 3
    
    for k in 1:G.nz
        for j in 1:G.ny
            ky = j <= G.ny÷2 ? j-1 : j-1-G.ny
            
            for i in 1:size(field, 1)
                kx = i-1
                
                if abs(kx) > kx_max || abs(ky) > ky_max
                    field[i, j, k] = 0.0
                end
            end
        end
    end
end

"""
    compute_energy_spectrum(field, G::Grid)

Compute horizontal energy spectrum E(k) from a spectral field.
"""
function compute_energy_spectrum(field, G::Grid)
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    k_max = min(kx_max, ky_max)
    
    spectrum = zeros(Float64, k_max)
    count = zeros(Int, k_max)
    
    for k in 1:G.nz
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:size(field, 1)
                kx = i-1
                
                k_total = round(Int, sqrt(Float64(kx^2 + ky^2)))
                
                if 1 <= k_total <= k_max
                    spectrum[k_total] += abs2(field[i, j, k])
                    count[k_total] += 1
                end
            end
        end
    end
    
    # Average over vertical levels and normalize
    for i in 1:k_max
        if count[i] > 0
            spectrum[i] /= count[i]
        end
    end
    
    return spectrum
end

"""
    normalize_field_energy!(field, G::Grid, target_energy::Real, plans)

Normalize field to have specified total energy.
"""
function normalize_field_energy!(field, G::Grid, target_energy::Real, plans)
    # Convert to real space to compute energy
    field_r = similar(field, Float64)
    fft_backward!(field_r, field, plans)
    
    # Compute current energy
    current_energy = 0.5 * sum(abs2, field_r) / (G.nx * G.ny * G.nz)
    
    if current_energy > 0
        scale_factor = sqrt(target_energy / current_energy)
        field .*= scale_factor
        @info "Field normalized: E_old=$current_energy → E_new=$target_energy (scale=$scale_factor)"
    else
        @warn "Cannot normalize zero field"
    end
end

"""
    create_wave_packet(G::Grid, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real)

Create a localized wave packet in spectral space.
"""
function create_wave_packet(G::Grid, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real)
    field = zeros(ComplexF64, G.nx÷2+1, G.ny, G.nz)
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        z_envelope = exp(-((k - G.nz/2)^2) / (2 * (G.nz/4)^2))
        
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:(G.nx÷2+1)
                kx = i-1
                
                # Gaussian envelope in wavenumber space
                k_dist2 = (kx - kx0)^2 + (ky - ky0)^2
                envelope = exp(-k_dist2 / (2 * sigma_k^2))
                
                if envelope > 1e-10
                    phase = 2π * rand()
                    field[i, j, k] = amplitude * envelope * z_envelope * cis(phase)
                end
            end
        end
    end
    
    return field
end

"""
    add_balanced_component!(S::State, G::Grid, params::QGParams, plans; N2_profile=nothing)

Add balanced component to the flow by computing geostrophically consistent fields.

This function:
1. Computes potential vorticity q from the streamfunction ψ
2. Computes geostrophically balanced velocities u = -∂ψ/∂y, v = ∂ψ/∂x
3. Computes buoyancy b = ∂ψ/∂z (from thermal wind balance)

Based on init_psi_generic and init_q from the Fortran implementation.

# Arguments
- `S::State`: Model state with streamfunction psi
- `G::Grid`: Grid structure
- `params::QGParams`: Model parameters (includes f0, N2)
- `plans`: FFT plans
- `N2_profile::Vector`: Optional N²(z) profile. If not provided, uses constant N²=1.

# Example
```julia
# With constant stratification
add_balanced_component!(state, grid, params, plans)

# With variable stratification
N2 = compute_stratification_profile(strat_profile, grid)
add_balanced_component!(state, grid, params, plans; N2_profile=N2)
```
"""
function add_balanced_component!(S::State, G::Grid, params::QGParams, plans; N2_profile=nothing)
    @info "Adding balanced component to initial state"

    nz = G.nz
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    dz2 = dz^2

    # Get elliptic coefficient a_ell = f²/N²
    # For constant N², a_ell = f₀²/N²
    # For variable N², a_ell[k] = f₀²/N²[k]
    f₀_sq = params.f₀^2
    if N2_profile === nothing || isempty(N2_profile)
        # Constant stratification N² = params.N²
        a_ell = fill(Float64(f₀_sq / params.N²), nz)
        @info "Using constant stratification (N² = $(params.N²))"
    else
        # Variable stratification from profile
        if length(N2_profile) != nz
            @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), interpolating..."
            # Simple linear interpolation if sizes don't match
            N2_interp = zeros(Float64, nz)
            for k in 1:nz
                # Map k to position in N2_profile
                pos = (k - 1) / (nz - 1) * (length(N2_profile) - 1) + 1
                k_low = max(1, floor(Int, pos))
                k_high = min(length(N2_profile), k_low + 1)
                w = pos - k_low
                N2_interp[k] = (1 - w) * N2_profile[k_low] + w * N2_profile[k_high]
            end
            a_ell = [f₀_sq / max(N2_interp[k], eps(Float64)) for k in 1:nz]
        else
            a_ell = [f₀_sq / max(N2_profile[k], eps(Float64)) for k in 1:nz]
        end
        @info "Using variable stratification from N² profile"
    end

    # Density weights (unity for Boussinesq)
    r_ut = ones(Float64, nz)  # rho at unstaggered (u) points
    r_st = ones(Float64, nz)  # rho at staggered (s) points

    # Get underlying arrays
    psi_arr = parent(S.psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    # Compute potential vorticity q from ψ
    # q = -kh² ψ + (1/ρ) ∂/∂z (ρ a_ell ∂ψ/∂z)
    if hasfield(typeof(S), :q)
        compute_q_from_psi!(S.q, S.psi, G, params, a_ell, r_ut, r_st, dz)
        @info "Computed potential vorticity q from streamfunction"
    end

    # Compute geostrophically balanced velocities
    # u = -∂ψ/∂y = -i*ky*ψ
    # v =  ∂ψ/∂x =  i*kx*ψ
    if hasfield(typeof(S), :u) && hasfield(typeof(S), :v)
        compute_geostrophic_velocities!(S.u, S.v, S.psi, G)
        @info "Computed geostrophic velocities u, v from streamfunction"
    end

    # Compute buoyancy from thermal wind balance
    # b = ∂ψ/∂z (in QG approximation with constant N²)
    if hasfield(typeof(S), :b)
        compute_buoyancy_from_psi!(S.b, S.psi, G, dz)
        @info "Computed buoyancy b from thermal wind balance"
    end
end

"""
    compute_q_from_psi!(q, psi, G, params, a_ell, r_ut, r_st, dz)

Compute QG potential vorticity from streamfunction.

The PV-streamfunction relationship is:
    q = ∇²ψ + (1/ρ) ∂/∂z (ρ a_ell ∂ψ/∂z)

In spectral space with finite differences in z:
    q = -kh² ψ + (1/dz²) [(ρ_u a_ell ρ_s⁻¹) (ψ[k+1] - 2ψ[k] + ψ[k-1])]

with Neumann BC ∂ψ/∂z = 0 at boundaries.
"""
function compute_q_from_psi!(q, psi, G::Grid, params, a_ell, r_ut, r_st, dz)
    nz = G.nz
    dz2 = dz^2

    q_arr = parent(q)
    psi_arr = parent(psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    for j_local in 1:ny_local, i_local in 1:nx_local
        # Get global wavenumber indices
        i_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(i_local, 1, G) : i_local
        j_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(j_local, 2, G) : j_local

        kx_val = G.kx[min(i_global, length(G.kx))]
        ky_val = G.ky[min(j_global, length(G.ky))]
        kh2 = kx_val^2 + ky_val^2

        # Interior points (k = 2, ..., nz-1)
        for k in 2:nz-1
            coeff_up = (r_ut[k] * a_ell[k]) / r_st[k]
            coeff_down = (r_ut[k-1] * a_ell[k-1]) / r_st[k]

            vert_term = coeff_up * psi_arr[i_local, j_local, k+1] -
                       (coeff_up + coeff_down) * psi_arr[i_local, j_local, k] +
                       coeff_down * psi_arr[i_local, j_local, k-1]

            q_arr[i_local, j_local, k] = -kh2 * psi_arr[i_local, j_local, k] + vert_term / dz2
        end

        # Bottom boundary (k=1): Neumann BC ψ_z = 0 ⟹ ψ[0] = ψ[1]
        if nz >= 1
            coeff_up = (r_ut[1] * a_ell[1]) / r_st[1]
            vert_term = coeff_up * (psi_arr[i_local, j_local, 2] - psi_arr[i_local, j_local, 1])
            q_arr[i_local, j_local, 1] = -kh2 * psi_arr[i_local, j_local, 1] + vert_term / dz2
        end

        # Top boundary (k=nz): Neumann BC ψ_z = 0 ⟹ ψ[nz+1] = ψ[nz]
        if nz >= 2
            coeff_down = (r_ut[nz-1] * a_ell[nz-1]) / r_st[nz]
            vert_term = coeff_down * (psi_arr[i_local, j_local, nz-1] - psi_arr[i_local, j_local, nz])
            q_arr[i_local, j_local, nz] = -kh2 * psi_arr[i_local, j_local, nz] + vert_term / dz2
        end
    end
end

"""
    compute_geostrophic_velocities!(u, v, psi, G)

Compute geostrophically balanced velocities from streamfunction.

Geostrophic balance:
    u = -∂ψ/∂y = -i*ky*ψ  (in spectral space)
    v =  ∂ψ/∂x =  i*kx*ψ  (in spectral space)
"""
function compute_geostrophic_velocities!(u, v, psi, G::Grid)
    u_arr = parent(u)
    v_arr = parent(v)
    psi_arr = parent(psi)

    nx_local, ny_local, nz_local = size(psi_arr)

    for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        # Get global wavenumber indices
        i_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(i_local, 1, G) : i_local
        j_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(j_local, 2, G) : j_local

        kx_val = G.kx[min(i_global, length(G.kx))]
        ky_val = G.ky[min(j_global, length(G.ky))]

        # Geostrophic velocities in spectral space
        u_arr[i_local, j_local, k] = -im * ky_val * psi_arr[i_local, j_local, k]
        v_arr[i_local, j_local, k] =  im * kx_val * psi_arr[i_local, j_local, k]
    end
end

"""
    compute_buoyancy_from_psi!(b, psi, G, dz)

Compute buoyancy from streamfunction using thermal wind balance.

In QG with thermal wind balance:
    b = f₀ ∂ψ/∂z / N²

For simplicity (and matching Fortran convention), we compute:
    b[k] = (ψ[k] - ψ[k-1]) / dz

at staggered (cell-face) points.
"""
function compute_buoyancy_from_psi!(b, psi, G::Grid, dz)
    b_arr = parent(b)
    psi_arr = parent(psi)

    nx_local, ny_local, nz_local = size(psi_arr)

    for j_local in 1:ny_local, i_local in 1:nx_local
        # Bottom boundary: b[1] from ψ[2] - ψ[1] (or extrapolation)
        if nz_local >= 2
            b_arr[i_local, j_local, 1] = (psi_arr[i_local, j_local, 2] - psi_arr[i_local, j_local, 1]) / dz
        else
            b_arr[i_local, j_local, 1] = 0
        end

        # Interior and top points
        for k in 2:nz_local
            b_arr[i_local, j_local, k] = (psi_arr[i_local, j_local, k] - psi_arr[i_local, j_local, k-1]) / dz
        end
    end
end

"""
    check_initial_conditions(S::State, G::Grid, plans)

Perform basic checks on initial conditions.
"""
function check_initial_conditions(S::State, G::Grid, plans)
    @info "Checking initial conditions..."
    
    # Check for NaNs or Infs
    if any(x -> !isfinite(x), S.psi)
        error("NaN or Inf detected in initial psi field")
    end
    
    if any(x -> !isfinite(x), S.B)
        error("NaN or Inf detected in initial wave field")
    end
    
    # Compute energy diagnostics
    psir = similar(S.psi, Float64)
    fft_backward!(psir, S.psi, plans)
    psi_energy = 0.5 * sum(abs2, psir) / (G.nx * G.ny * G.nz)
    
    Br = similar(S.B, Float64)
    fft_backward!(Br, real.(S.B), plans)
    wave_energy = 0.5 * sum(abs2, Br) / (G.nx * G.ny * G.nz)
    
    @info "Initial conditions summary:"
    @info "  Psi energy: $psi_energy"
    @info "  Wave energy: $wave_energy"
    @info "  Max |psi|: $(maximum(abs, psir))"
    @info "  Max |B|: $(maximum(abs, Br))"
    
    return Dict(
        "psi_energy" => psi_energy,
        "wave_energy" => wave_energy,
        "psi_max" => maximum(abs, psir),
        "wave_max" => maximum(abs, Br)
    )
end