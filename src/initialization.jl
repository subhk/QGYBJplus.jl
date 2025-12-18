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
    initialize_from_config(config::ModelConfig, G::Grid, S::State, plans; params=nothing, N2_profile=nothing)

Initialize model state from configuration.

This function initializes the streamfunction (ψ) and wave field (B), then computes
consistent potential vorticity (q) from ψ to ensure the first timestep doesn't
wipe out the user-provided initial conditions.

# Arguments
- `config`: Model configuration with initial condition settings
- `G::Grid`: Grid structure
- `S::State`: Model state to initialize
- `plans`: FFT plans
- `params`: QGParams (optional). If provided, q is computed from ψ for consistency.
- `N2_profile`: Optional N²(z) profile for variable stratification.
"""
function initialize_from_config(config, G::Grid, S::State, plans; params=nothing, N2_profile=nothing)
    @info "Initializing model fields from configuration"

    # Set random seed for reproducibility
    Random.seed!(config.initial_conditions.random_seed)

    # Initialize stream function
    if config.initial_conditions.psi_type == :analytical
        init_analytical_psi!(S.psi, G, config.initial_conditions.psi_amplitude, plans)
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
        init_analytical_waves!(S.B, G, config.initial_conditions.wave_amplitude, plans)
    elseif config.initial_conditions.wave_type == :random
        init_random_waves!(S.B, G, config.initial_conditions.wave_amplitude)
    elseif config.initial_conditions.wave_type == :from_file
        S.B .= read_initial_waves(config.initial_conditions.wave_filename, G, plans)
    else
        # Zero initialization
        S.B .= 0.0
    end

    # Compute q from ψ to ensure consistency
    # Without this, the first timestep's invert_q_to_psi! would recompute ψ from
    # the zero q field, wiping out the user's initial streamfunction.
    if params !== nothing && hasfield(typeof(S), :q)
        @info "Computing potential vorticity q from initialized streamfunction"
        add_balanced_component!(S, G, params, plans; N2_profile=N2_profile)
    elseif config.initial_conditions.psi_type != :zero && params === nothing
        @warn "params not provided to initialize_from_config. " *
              "Potential vorticity q will remain zero, and the first timestep " *
              "will recompute ψ from q=0, potentially wiping the initial ψ. " *
              "Pass params to compute consistent q from ψ." maxlog=1
    end

    @info "Model initialization complete"
end

"""
    init_analytical_psi!(psik, G::Grid, amplitude::Real, plans)

Initialize stream function with analytical expression.
Based on the generate_fields_stag routine from Fortran code.

# Arguments
- `psik`: Spectral field to populate (output)
- `G::Grid`: Grid structure
- `amplitude::Real`: Amplitude of the initial field
- `plans`: FFT plans for forward transform
"""
function init_analytical_psi!(psik, G::Grid, amplitude::Real, plans)
    @info "Initializing analytical stream function (amplitude=$amplitude)"

    # Get local dimensions from psik (works for both Array and PencilArray)
    psik_arr = parent(psik)
    nx_local, ny_local, nz_local = size(psik_arr)

    # Initialize in real space with LOCAL dimensions
    # Use similar() to get correct array type (Array or PencilArray)
    psir = similar(psik, Float64)
    psir_arr = parent(psir)

    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    dz = G.Lz / G.nz

    for k in 1:nz_local
        # Get global z-index for correct coordinate
        k_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(k, 3, G) : k
        z = (k_global - 1) * dz

        for j_local in 1:ny_local
            # Get global y-index
            j_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                       local_to_global(j_local, 2, G) : j_local
            y = (j_global - 1) * dy

            for i_local in 1:nx_local
                # Get global x-index
                i_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                           local_to_global(i_local, 1, G) : i_local
                x = (i_global - 1) * dx

                # Example: sum of Rossby waves with different modes
                # Use normalized coordinates for wave patterns: x̃ = 2πx/Lx, etc.
                x_norm = 2π * x / G.Lx
                y_norm = 2π * y / G.Ly
                z_norm = 2π * z / G.Lz

                # This mimics typical geostrophic turbulence patterns
                psir_arr[i_local, j_local, k] = amplitude * (
                    sin(2*x_norm) * cos(y_norm) * cos(z_norm) +
                    0.5 * cos(x_norm) * sin(2*y_norm) * sin(z_norm) +
                    0.3 * sin(3*x_norm) * sin(y_norm) * cos(2*z_norm) +
                    0.2 * cos(2*x_norm) * cos(3*y_norm) * sin(2*z_norm)
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

The streamfunction ψ is real-valued, so its Fourier transform must satisfy
Hermitian symmetry: ψ̂(-k) = conj(ψ̂(k)). For the rfft representation used here
(only kx ≥ 0 stored), this requires:
- For kx = 0: ψ̂(0, ky) = conj(ψ̂(0, -ky))
- For kx = nx/2 (if nx even): ψ̂(nx/2, ky) = conj(ψ̂(nx/2, -ky))
- ψ̂(0, 0) and ψ̂(0, ny/2) must be real

This function enforces these constraints to ensure IFFT produces real output.
"""
function init_random_psi!(psik, G::Grid, amplitude::Real; slope::Real=-3.0)
    @info "Initializing random stream function (amplitude=$amplitude, slope=$slope)"

    nx, ny, nz = G.nx, G.ny, G.nz
    kx_max = nx ÷ 2
    ky_max = ny ÷ 2

    # Create spectral field with desired slope
    fill!(psik, 0.0)

    for k in 1:nz
        for j in 1:ny
            ky = j <= ky_max ? j-1 : j-1-ny

            for i in 1:(nx÷2+1)
                kx = i-1

                if kx == 0 && ky == 0
                    continue  # Skip mean mode
                end

                # Total wavenumber
                k_total = sqrt(Float64(kx^2 + ky^2))

                if k_total > 0
                    # Energy spectrum E(k) ∝ k^slope
                    energy = amplitude * k_total^slope

                    # For Hermitian symmetry at kx=0 and kx=nx/2:
                    # Only set modes with ky >= 0, then set conjugate for ky < 0
                    if kx == 0 || (kx == kx_max && iseven(nx))
                        # These columns need Hermitian symmetry in ky
                        if ky > 0
                            # Set this mode with random phase
                            amp = sqrt(2 * energy) * randn()
                            phase = 2π * rand()
                            psik[i, j, k] = amp * cis(phase)
                            # Set conjugate at -ky (j_conj = ny - j + 2 for j > 1)
                            j_conj = ny - j + 2
                            psik[i, j_conj, k] = conj(psik[i, j, k])
                        elseif ky == 0
                            # ky=0 mode must be real
                            amp = sqrt(2 * energy) * randn()
                            psik[i, j, k] = amp  # Real value
                        elseif ky == -ky_max && iseven(ny)
                            # Nyquist mode in y (ky = -ny/2 = ny/2) must be real
                            amp = sqrt(2 * energy) * randn()
                            psik[i, j, k] = amp  # Real value
                        end
                        # ky < 0 modes are set as conjugates above, skip them here
                    else
                        # kx > 0 and kx < kx_max: no special symmetry needed
                        # (the kx < 0 modes are implicitly conjugates in rfft)
                        amp = sqrt(2 * energy) * randn()
                        phase = 2π * rand()
                        psik[i, j, k] = amp * cis(phase)
                    end
                end
            end
        end
    end

    # Apply dealiasing mask
    apply_dealiasing_mask!(psik, G)
end

"""
    init_analytical_waves!(Bk, G::Grid, amplitude::Real, plans)

Initialize wave field (L+A) with analytical expression.

# Arguments
- `Bk`: Spectral field to populate (output)
- `G::Grid`: Grid structure
- `amplitude::Real`: Amplitude of the initial field
- `plans`: FFT plans for forward transform
"""
function init_analytical_waves!(Bk, G::Grid, amplitude::Real, plans)
    @info "Initializing analytical wave field (amplitude=$amplitude)"

    # Initialize in real space
    Br = zeros(Float64, G.nx, G.ny, G.nz)
    Bi = zeros(Float64, G.nx, G.ny, G.nz)

    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    dz = G.Lz / G.nz

    # Mid-depth for vertical decay (in domain coordinates)
    z_mid = G.Lz / 2
    sigma_z = G.Lz / 10  # Decay scale

    for k in 1:G.nz
        z = (k - 1) * dz
        for j in 1:G.ny
            y = (j - 1) * dy
            for i in 1:G.nx
                x = (i - 1) * dx

                # Use normalized coordinates for wave patterns
                x_norm = 2π * x / G.Lx
                y_norm = 2π * y / G.Ly
                z_norm = 2π * z / G.Lz

                # Example wave pattern with vertical decay centered at mid-depth
                Br[i,j,k] = amplitude * (
                    sin(4*x_norm + z_norm) * cos(2*y_norm) * exp(-((z-z_mid)^2)/(2*sigma_z^2)) +
                    0.3 * cos(2*x_norm) * sin(4*y_norm + 2*z_norm) * exp(-((z-z_mid)^2)/(2*(0.6*sigma_z)^2))
                )

                Bi[i,j,k] = amplitude * 0.1 * (
                    cos(4*x_norm + z_norm) * sin(2*y_norm) * exp(-((z-z_mid)^2)/(2*sigma_z^2)) +
                    0.3 * sin(2*x_norm) * cos(4*y_norm + 2*z_norm) * exp(-((z-z_mid)^2)/(2*(0.6*sigma_z)^2))
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

    # Note: Geostrophic velocities (u, v) are NOT computed here.
    # The State struct has u, v as real-space arrays, and proper velocity computation
    # requires FFT plans and workspace. Velocities will be computed consistently by
    # compute_velocities! during the first projection step (first_projection_step!).

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

        # Handle boundary conditions based on nz
        if nz == 1
            # Single-layer case: No vertical derivatives, q = -kh² ψ (2D barotropic mode)
            q_arr[i_local, j_local, 1] = -kh2 * psi_arr[i_local, j_local, 1]
        else
            # Bottom boundary (k=1): Neumann BC ψ_z = 0 ⟹ ψ[0] = ψ[1]
            coeff_up = (r_ut[1] * a_ell[1]) / r_st[1]
            vert_term = coeff_up * (psi_arr[i_local, j_local, 2] - psi_arr[i_local, j_local, 1])
            q_arr[i_local, j_local, 1] = -kh2 * psi_arr[i_local, j_local, 1] + vert_term / dz2

            # Top boundary (k=nz): Neumann BC ψ_z = 0 ⟹ ψ[nz+1] = ψ[nz]
            coeff_down = (r_ut[nz-1] * a_ell[nz-1]) / r_st[nz]
            vert_term = coeff_down * (psi_arr[i_local, j_local, nz-1] - psi_arr[i_local, j_local, nz])
            q_arr[i_local, j_local, nz] = -kh2 * psi_arr[i_local, j_local, nz] + vert_term / dz2
        end
    end
end

"""
    compute_geostrophic_velocities!(u, v, psi, G, plans)

Compute geostrophically balanced velocities from streamfunction.

This function computes velocities in spectral space using geostrophic balance,
then transforms them to physical space.

Geostrophic balance:
    u = -∂ψ/∂y = -i*ky*ψ  (in spectral space)
    v =  ∂ψ/∂x =  i*kx*ψ  (in spectral space)

# Arguments
- `u`: Zonal velocity output (real-space, real array)
- `v`: Meridional velocity output (real-space, real array)
- `psi`: Streamfunction (spectral space, complex array)
- `G::Grid`: Grid structure
- `plans`: FFT plans for inverse transform

# Note
For typical use, velocities are computed by `compute_velocities!` in the main
timestepping loop. This function is provided for initialization or diagnostics.
"""
function compute_geostrophic_velocities!(u, v, psi, G::Grid, plans)
    psi_arr = parent(psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    # Allocate temporary spectral arrays for velocity derivatives
    uk_temp = similar(psi)
    vk_temp = similar(psi)
    uk_arr = parent(uk_temp)
    vk_arr = parent(vk_temp)

    for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        # Get global wavenumber indices
        i_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(i_local, 1, G) : i_local
        j_global = hasfield(typeof(G), :decomp) && G.decomp !== nothing ?
                   local_to_global(j_local, 2, G) : j_local

        kx_val = G.kx[min(i_global, length(G.kx))]
        ky_val = G.ky[min(j_global, length(G.ky))]

        # Geostrophic velocities in spectral space
        uk_arr[i_local, j_local, k] = -im * ky_val * psi_arr[i_local, j_local, k]
        vk_arr[i_local, j_local, k] =  im * kx_val * psi_arr[i_local, j_local, k]
    end

    # Transform to physical space
    fft_backward!(u, uk_temp, plans)
    fft_backward!(v, vk_temp, plans)
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