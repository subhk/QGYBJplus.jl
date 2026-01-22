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
using ..QGYBJplus: Grid, State, QGParams
using ..QGYBJplus: plan_transforms!, fft_forward!, fft_backward!, compute_wavenumbers!
using ..QGYBJplus: local_to_global
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
import PencilArrays: PencilArray

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

"""
    initialize_from_config(config::ModelConfig, G::Grid, S::State, plans;
                           params=nothing, N2_profile=nothing, parallel_config=nothing)

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
- `parallel_config`: MPIConfig for parallel file I/O when using `:from_file`.
"""
function initialize_from_config(config, G::Grid, S::State, plans;
                                params=nothing, N2_profile=nothing, parallel_config=nothing)
    @info "Initializing model fields from configuration"

    if G.decomp !== nothing
        parallel_config !== nothing || error("initialize_from_config requires parallel_config for MPI grids. " *
                                             "Use parallel_initialize_fields! or pass the MPIConfig used to create the grid.")
        parallel_initialize_fields!(S, G, plans, config, parallel_config; params=params, N2_profile=N2_profile)
        return
    end

    # Set random seed for reproducibility
    Random.seed!(config.initial_conditions.random_seed)

    # Initialize stream function
    if config.initial_conditions.psi_type == :analytical
        init_analytical_psi!(S.psi, G, config.initial_conditions.psi_amplitude, plans)
    elseif config.initial_conditions.psi_type == :random
        init_random_psi!(S.psi, G, config.initial_conditions.psi_amplitude)
    elseif config.initial_conditions.psi_type == :from_file
        S.psi .= read_initial_psi(config.initial_conditions.psi_filename, G, plans;
                                  parallel_config=parallel_config)
    else
        # Zero initialization (type-safe)
        fill!(S.psi, zero(eltype(S.psi)))
    end

    # Initialize wave field
    if config.initial_conditions.wave_type == :analytical
        init_analytical_waves!(S.B, G, config.initial_conditions.wave_amplitude, plans)
    elseif config.initial_conditions.wave_type == :random
        init_random_waves!(S.B, G, config.initial_conditions.wave_amplitude)
    elseif config.initial_conditions.wave_type == :from_file
        S.B .= read_initial_waves(config.initial_conditions.wave_filename, G, plans;
                                  parallel_config=parallel_config)
    elseif config.initial_conditions.wave_type == :surface_waves
        init_surface_waves!(
            S.B, G,
            config.initial_conditions.wave_amplitude,
            config.initial_conditions.wave_surface_depth,
            plans;
            uniform=config.initial_conditions.wave_uniform,
            profile=config.initial_conditions.wave_profile
        )
    elseif config.initial_conditions.wave_type == :surface_exponential
        init_surface_waves!(
            S.B, G,
            config.initial_conditions.wave_amplitude,
            config.initial_conditions.wave_surface_depth,
            plans;
            uniform=config.initial_conditions.wave_uniform,
            profile=:exponential
        )
    elseif config.initial_conditions.wave_type == :surface_gaussian
        init_surface_waves!(
            S.B, G,
            config.initial_conditions.wave_amplitude,
            config.initial_conditions.wave_surface_depth,
            plans;
            uniform=config.initial_conditions.wave_uniform,
            profile=:gaussian
        )
    else
        # Zero initialization (type-safe)
        fill!(S.B, zero(eltype(S.B)))
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

    # Initialize in real space with LOCAL dimensions (input pencil for MPI)
    psir = _allocate_fft_dst(psik, plans)
    psir_arr = parent(psir)
    nz_local, nx_local, ny_local = size(psir_arr)

    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    dz = G.Lz / G.nz

    for k in 1:nz_local
        # Get global z-index for correct coordinate
        k_global = local_to_global(k, 1, psir)
        z = (k_global - 1) * dz

        for j_local in 1:ny_local
            # Get global y-index
            j_global = local_to_global(j_local, 3, psir)
            y = (j_global - 1) * dy

            for i_local in 1:nx_local
                # Get global x-index
                i_global = local_to_global(i_local, 2, psir)
                x = (i_global - 1) * dx

                # Example: sum of Rossby waves with different modes
                # Use normalized coordinates for wave patterns: x̃ = 2πx/Lx, etc.
                x_norm = 2π * x / G.Lx
                y_norm = 2π * y / G.Ly
                z_norm = 2π * z / G.Lz

                # This mimics typical geostrophic turbulence patterns
                psir_arr[k, i_local, j_local] = amplitude * (
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
Hermitian symmetry: ψ̂(-k) = conj(ψ̂(k)). For complex-to-complex FFT (used by QGYBJplus),
this requires explicitly setting conjugate pairs:
- For kx = 0: ψ̂(0, ky) = conj(ψ̂(0, -ky))
- For kx = nx/2 (if nx even): ψ̂(nx/2, ky) = conj(ψ̂(nx/2, -ky))
- For 0 < kx < nx/2: ψ̂(kx, ky) and ψ̂(-kx, -ky) = conj(ψ̂(kx, ky))
- ψ̂(0, 0), ψ̂(0, ny/2), ψ̂(nx/2, 0), ψ̂(nx/2, ny/2) must be real

This function enforces these constraints to ensure IFFT produces real output.
"""
function init_random_psi!(psik, G::Grid, amplitude::Real; slope::Real=-3.0)
    @info "Initializing random stream function (amplitude=$amplitude, slope=$slope)"

    if psik isa PencilArray
        error("init_random_psi! does not support PencilArray. " *
              "Use init_mpi_random_psi! or parallel_initialize_fields! for MPI runs.")
    end

    nx, ny, nz = G.nx, G.ny, G.nz
    kx_max = nx ÷ 2
    ky_max = ny ÷ 2

    # Create spectral field with desired slope
    fill!(psik, zero(eltype(psik)))

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

                    # Handle Hermitian symmetry for real-valued output
                    if kx == 0
                        # kx = 0 column: need ψ̂(0, ky) = conj(ψ̂(0, -ky))
                        if ky > 0
                            # Set this mode with random phase
                            amp = sqrt(2 * energy) * randn()
                            phase = 2π * rand()
                            psik[k, i, j] = amp * cis(phase)
                            # Set conjugate at -ky
                            j_conj = ny - j + 2
                            psik[k, i, j_conj] = conj(psik[k, i, j])
                        elseif ky == 0
                            # ky=0 mode must be real (already skipped above)
                            continue
                        elseif ky == -ky_max && iseven(ny)
                            # Nyquist mode in y must be real
                            amp = sqrt(2 * energy) * randn()
                            psik[k, i, j] = amp
                        end
                        # ky < 0 modes (except Nyquist) are set as conjugates above

                    elseif kx == kx_max && iseven(nx)
                        # kx = nx/2 column (Nyquist in x): similar treatment
                        if ky > 0
                            amp = sqrt(2 * energy) * randn()
                            phase = 2π * rand()
                            psik[k, i, j] = amp * cis(phase)
                            # Set conjugate at -ky
                            j_conj = ny - j + 2
                            psik[k, i, j_conj] = conj(psik[k, i, j])
                        elseif ky == 0
                            # (kx=nx/2, ky=0) must be real
                            amp = sqrt(2 * energy) * randn()
                            psik[k, i, j] = amp
                        elseif ky == -ky_max && iseven(ny)
                            # (kx=nx/2, ky=ny/2) must be real
                            amp = sqrt(2 * energy) * randn()
                            psik[k, i, j] = amp
                        end
                        # ky < 0 modes (except Nyquist) are set as conjugates above

                    else
                        # 0 < kx < kx_max: set mode and its conjugate at (-kx, -ky)
                        amp = sqrt(2 * energy) * randn()
                        phase = 2π * rand()
                        psik[k, i, j] = amp * cis(phase)

                        # Set conjugate at (-kx, -ky)
                        # For wavenumber kx at index i, -kx is at index nx - i + 2
                        # For wavenumber ky at index j, -ky is at index:
                        #   - j=1 (ky=0): j_conj=1 (ky=0)
                        #   - j>1: j_conj = ny - j + 2
                        i_conj = nx - i + 2
                        j_conj = j == 1 ? 1 : ny - j + 2
                        psik[k, i_conj, j_conj] = conj(psik[k, i, j])
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

    # Initialize in real space with LOCAL dimensions (input pencil for MPI)
    Br = _allocate_fft_dst(Bk, plans)
    Bi = _allocate_fft_dst(Bk, plans)
    Br_arr = parent(Br)
    Bi_arr = parent(Bi)
    nz_local, nx_local, ny_local = size(Br_arr)

    dx = G.Lx / G.nx
    dy = G.Ly / G.ny
    dz = G.Lz / G.nz

    # Mid-depth for vertical decay (depth coordinates)
    z_mid = G.Lz / 2
    sigma_z = G.Lz / 10  # Decay scale

    for k in 1:nz_local
        # Get global z-index for correct coordinate
        k_global = local_to_global(k, 1, Br)
        z = -G.Lz + (k_global - 0.5) * dz
        depth = -z

        for j_local in 1:ny_local
            # Get global y-index
            j_global = local_to_global(j_local, 3, Br)
            y = (j_global - 1) * dy

            for i_local in 1:nx_local
                # Get global x-index
                i_global = local_to_global(i_local, 2, Br)
                x = (i_global - 1) * dx

                # Use normalized coordinates for wave patterns
                x_norm = 2π * x / G.Lx
                y_norm = 2π * y / G.Ly
                z_norm = 2π * depth / G.Lz

                # Example wave pattern with vertical decay centered at mid-depth
                Br_arr[k, i_local, j_local] = amplitude * (
                    sin(4*x_norm + z_norm) * cos(2*y_norm) * exp(-((depth - z_mid)^2)/(2*sigma_z^2)) +
                    0.3 * cos(2*x_norm) * sin(4*y_norm + 2*z_norm) * exp(-((depth - z_mid)^2)/(2*(0.6*sigma_z)^2))
                )

                Bi_arr[k, i_local, j_local] = amplitude * 0.1 * (
                    cos(4*x_norm + z_norm) * sin(2*y_norm) * exp(-((depth - z_mid)^2)/(2*sigma_z^2)) +
                    0.3 * sin(2*x_norm) * cos(4*y_norm + 2*z_norm) * exp(-((depth - z_mid)^2)/(2*(0.6*sigma_z)^2))
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
    init_surface_waves!(Bk, G::Grid, amplitude::Real, surface_depth::Real, plans; uniform=true, profile=:gaussian)

Initialize horizontally uniform surface waves with a specified vertical decay profile.

# Arguments
- `Bk`: Spectral field to populate (output)
- `G::Grid`: Grid structure
- `amplitude::Real`: Wave velocity amplitude
- `surface_depth::Real`: E-folding depth [m]
- `plans`: FFT plans for forward transform
- `uniform`: Horizontally uniform waves (default: true)
- `profile`: Vertical decay profile (:gaussian or :exponential)
"""
function init_surface_waves!(Bk, G::Grid, amplitude::Real, surface_depth::Real, plans;
                             uniform::Bool=true, profile::Symbol=:gaussian)
    surface_depth > 0 || throw(ArgumentError("surface_depth must be positive (got $surface_depth)"))

    # Initialize in real space with LOCAL dimensions (input pencil for MPI)
    B_phys = _allocate_fft_dst(Bk, plans)
    B_arr = parent(B_phys)
    T = eltype(B_arr)

    dz = G.Lz / G.nz
    for k_local in axes(B_arr, 1)
        k_global = local_to_global(k_local, 1, B_phys)
        # Depth from surface (z=0 is surface, z=-Lz is bottom).
        # Use a dz/2 shift so the top cell center corresponds to z=0.
        depth = max(zero(T), -G.z[k_global] - dz / 2)
        wave_profile = if profile == :gaussian
            exp(-(depth^2) / (surface_depth^2))
        elseif profile == :exponential
            exp(-depth / surface_depth)
        else
            throw(ArgumentError("Unknown profile=$profile. Use :gaussian or :exponential."))
        end

        if uniform
            B_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        else
            # Placeholder for future horizontal structure.
            B_arr[k_local, :, :] .= complex(T(amplitude) * wave_profile)
        end
    end

    # Transform to spectral space
    fft_forward!(Bk, B_phys, plans)
end

"""
    init_random_waves!(Bk, G::Grid, amplitude::Real; slope::Real=-2.0)

Initialize wave field with random amplitudes and phases.
"""
function init_random_waves!(Bk, G::Grid, amplitude::Real; slope::Real=-2.0)
    @info "Initializing random wave field (amplitude=$amplitude, slope=$slope)"
    
    if Bk isa PencilArray
        error("init_random_waves! does not support PencilArray. " *
              "Use parallel_initialize_fields! for MPI runs.")
    end

    # Generate random phases for real and imaginary parts
    phases_r = 2π * rand(Float64, G.nz, G.nx, G.ny)
    phases_i = 2π * rand(Float64, G.nz, G.nx, G.ny)
    
    fill!(Bk, zero(eltype(Bk)))
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        # Add some vertical structure - stronger near middle depths
        z_factor = sin(π * k / G.nz)^2
        
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:G.nx
                kx = i <= kx_max ? i - 1 : i - 1 - G.nx
                
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
                    Bk[k, i, j] = (amp_r * cis(phases_r[k, i, j])) +
                                  im * (amp_i * cis(phases_i[k, i, j]))
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
    fill!(psik, zero(eltype(psik)))
end

"""
    apply_dealiasing_mask!(field, G::Grid)

Apply 2/3 dealiasing mask to spectral field using radial cutoff.
Handles both serial (Array) and parallel (PencilArray) cases.

Uses the same radial 2/3 rule as `dealias_mask()`:
- Keep wavenumbers with |k| ≤ (2/3) × k_Nyquist = N/3
- Radial cutoff ensures isotropic dealiasing
"""
function apply_dealiasing_mask!(field, G::Grid)
    # Radial 2/3 cutoff: k_max = min(nx, ny) / 3
    kmax = floor(Int, min(G.nx, G.ny) / 3)
    kmax_sq = kmax^2

    # Get local array and its dimensions
    field_arr = parent(field)
    nz_local, nx_local, ny_local = size(field_arr)

    for k in 1:nz_local
        for j_local in 1:ny_local
            # Get global j index for wavenumber lookup
            j_global = local_to_global(j_local, 3, field)
            ky = j_global <= G.ny÷2 ? j_global-1 : j_global-1-G.ny

            for i_local in 1:nx_local
                # Get global i index for wavenumber lookup
                i_global = local_to_global(i_local, 2, field)
                kx = i_global <= G.nx÷2 ? i_global-1 : i_global-1-G.nx

                # Radial check: zero modes outside dealiasing circle
                if kx^2 + ky^2 > kmax_sq
                    field_arr[k, i_local, j_local] = zero(eltype(field_arr))
                end
            end
        end
    end
end

"""
    compute_energy_spectrum(field, G::Grid)

Compute horizontal energy spectrum E(k) from a spectral field.

In MPI mode, this computes the local contribution only (no MPI reduction).
"""
function compute_energy_spectrum(field, G::Grid)
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    k_max = min(kx_max, ky_max)
    
    spectrum = zeros(Float64, k_max)
    count = zeros(Int, k_max)

    field_arr = parent(field)
    for k in axes(field_arr, 1)
        for j_local in axes(field_arr, 3)
            j_global = local_to_global(j_local, 3, field)
            ky = j_global <= ky_max ? j_global - 1 : j_global - 1 - G.ny

            for i_local in axes(field_arr, 2)
                i_global = local_to_global(i_local, 2, field)
                kx = i_global <= kx_max ? i_global - 1 : i_global - 1 - G.nx

                k_total = round(Int, sqrt(Float64(kx^2 + ky^2)))
                if 1 <= k_total <= k_max
                    spectrum[k_total] += abs2(field_arr[k, i_local, j_local])
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
    field_r = _allocate_fft_dst(field, plans)
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
    field = zeros(ComplexF64, G.nz, G.nx, G.ny)
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        z_envelope = exp(-((k - G.nz/2)^2) / (2 * (G.nz/4)^2))
        
        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny
            
            for i in 1:G.nx
                kx = i <= kx_max ? i - 1 : i - 1 - G.nx
                
                # Gaussian envelope in wavenumber space
                k_dist2 = (kx - kx0)^2 + (ky - ky0)^2
                envelope = exp(-k_dist2 / (2 * sigma_k^2))
                
                if envelope > 1e-10
                    phase = 2π * rand()
                    field[k, i, j] = amplitude * envelope * z_envelope * cis(phase)
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

    # Get underlying arrays
    psi_arr = parent(S.psi)
    nz_local, nx_local, ny_local = size(psi_arr)

    # Compute potential vorticity q from ψ
    # q = -kh² ψ + ∂/∂z (a_ell ∂ψ/∂z) (Boussinesq)
    if hasfield(typeof(S), :q)
        compute_q_from_psi!(S.q, S.psi, G, params, a_ell, dz)
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
    compute_q_from_psi!(q, psi, G, params, a_ell, dz)

Compute QG potential vorticity from streamfunction (PDF Eq. A21, Appendix).

The PV-streamfunction relationship is (Boussinesq formulation):
    q = ∇²ψ + L(ψ)  where L = ∂/∂z (f²/N² ∂/∂z)

In spectral space with finite differences in z (PDF Eq. 32):
    q[k] = -kh² ψ[k] + (1/dz²) [a[k+1](ψ[k+1] - ψ[k]) - a[k](ψ[k] - ψ[k-1])]

where a[k] = f₀²/N²[k] is the elliptic coefficient at interface k.

Boundary conditions: ∂ψ/∂z = 0 (Neumann BC, PDF Eq. A14).
"""
function compute_q_from_psi!(q, psi, G::Grid, params, a_ell, dz)
    nz = G.nz
    dz2 = dz^2

    q_arr = parent(q)
    psi_arr = parent(psi)
    nz_local, nx_local, ny_local = size(psi_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"

    for j_local in 1:ny_local, i_local in 1:nx_local
        # Get global wavenumber indices
        i_global = local_to_global(i_local, 2, psi)
        j_global = local_to_global(j_local, 3, psi)

        kx_val = G.kx[min(i_global, length(G.kx))]
        ky_val = G.ky[min(j_global, length(G.ky))]
        kh2 = kx_val^2 + ky_val^2

        # Interior points (k = 2, ..., nz-1)
        for k in 2:nz-1
            vert_term = a_ell[k+1] * psi_arr[k+1, i_local, j_local] -
                       (a_ell[k+1] + a_ell[k]) * psi_arr[k, i_local, j_local] +
                       a_ell[k] * psi_arr[k-1, i_local, j_local]

            q_arr[k, i_local, j_local] = -kh2 * psi_arr[k, i_local, j_local] + vert_term / dz2
        end

        # Handle boundary conditions based on nz
        if nz == 1
            # Single-layer case: No vertical derivatives, q = -kh² ψ (2D barotropic mode)
            q_arr[1, i_local, j_local] = -kh2 * psi_arr[1, i_local, j_local]
        else
            # Bottom boundary (k=1): Neumann BC ψ_z = 0 ⟹ ψ[0] = ψ[1]
            vert_term = a_ell[1] * (psi_arr[2, i_local, j_local] - psi_arr[1, i_local, j_local])
            q_arr[1, i_local, j_local] = -kh2 * psi_arr[1, i_local, j_local] + vert_term / dz2

            # Top boundary (k=nz): Neumann BC ψ_z = 0 ⟹ ψ[nz+1] = ψ[nz]
            vert_term = a_ell[nz] * (psi_arr[nz-1, i_local, j_local] - psi_arr[nz, i_local, j_local])
            q_arr[nz, i_local, j_local] = -kh2 * psi_arr[nz, i_local, j_local] + vert_term / dz2
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
    nz_local, nx_local, ny_local = size(psi_arr)

    # Allocate temporary spectral arrays for velocity derivatives
    uk_temp = similar(psi)
    vk_temp = similar(psi)
    uk_arr = parent(uk_temp)
    vk_arr = parent(vk_temp)

    for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        # Get global wavenumber indices
        i_global = local_to_global(i_local, 2, psi)
        j_global = local_to_global(j_local, 3, psi)

        kx_val = G.kx[min(i_global, length(G.kx))]
        ky_val = G.ky[min(j_global, length(G.ky))]

        # Geostrophic velocities in spectral space
        uk_arr[k, i_local, j_local] = -im * ky_val * psi_arr[k, i_local, j_local]
        vk_arr[k, i_local, j_local] =  im * kx_val * psi_arr[k, i_local, j_local]
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

    nz_local, nx_local, ny_local = size(psi_arr)

    for j_local in 1:ny_local, i_local in 1:nx_local
        # Bottom boundary: b[1] from ψ[2] - ψ[1] (or extrapolation)
        if nz_local >= 2
            b_arr[1, i_local, j_local] = (psi_arr[2, i_local, j_local] - psi_arr[1, i_local, j_local]) / dz
        else
            b_arr[1, i_local, j_local] = 0
        end

        # Interior and top points
        for k in 2:nz_local
            b_arr[k, i_local, j_local] = (psi_arr[k, i_local, j_local] - psi_arr[k-1, i_local, j_local]) / dz
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
    # Note: fft_backward! returns complex arrays, extract real part for diagnostics
    psir_complex = _allocate_fft_dst(S.psi, plans)
    fft_backward!(psir_complex, S.psi, plans)
    psir = real.(parent(psir_complex))
    psi_energy = 0.5 * sum(abs2, psir) / (G.nx * G.ny * G.nz)

    # For wave field: do full complex IFFT on S.B, then extract real part
    Br_complex = _allocate_fft_dst(S.B, plans)
    fft_backward!(Br_complex, S.B, plans)
    Br = real.(parent(Br_complex))
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
