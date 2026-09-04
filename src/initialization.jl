using Random
using LinearAlgebra
using ..QGYBJplus: RuntimeGeometry, ModelFields
using ..QGYBJplus: plan_transforms!, fft_forward!, fft_backward!
using ..QGYBJplus: local_to_global
using ..QGYBJplus: allocate_fft_backward_dst
import PencilArrays: PencilArray

const _allocate_fft_dst = allocate_fft_backward_dst

"""
    init_analytical_psi!(psik, G::RuntimeGeometry, amplitude::Real, plans)

Initialize stream function with analytical expression.
Based on the generate_fields_stag routine from Fortran code.

# Arguments
- `psik`: Spectral field to populate (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `amplitude::Real`: Amplitude of the initial field
- `plans`: FFT plans for forward transform
"""
function init_analytical_psi!(psik, G::RuntimeGeometry, amplitude::Real, plans)
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
    init_random_psi!(psik, G::RuntimeGeometry, amplitude::Real; slope::Real=-3.0)

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
function init_random_psi!(psik, G::RuntimeGeometry, amplitude::Real; slope::Real=-3.0)
    @info "Initializing random stream function (amplitude=$amplitude, slope=$slope)"

    if psik isa PencilArray
        error("init_random_psi! does not support PencilArray; " *
              "use set_mean_flow!(model; method=:random) for model fields.")
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
    init_analytical_waves!(Bk, G::RuntimeGeometry, amplitude::Real, plans)

Initialize wave field (L+A) with analytical expression.

# Arguments
- `Bk`: Spectral field to populate (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `amplitude::Real`: Amplitude of the initial field
- `plans`: FFT plans for forward transform
"""
function init_analytical_waves!(Bk, G::RuntimeGeometry, amplitude::Real, plans)
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
    init_surface_waves!(Bk, G::RuntimeGeometry, amplitude::Real, surface_depth::Real, plans; uniform=true, profile=:gaussian)

Initialize horizontally uniform surface waves with a specified vertical decay profile.

# Arguments
- `Bk`: Spectral field to populate (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `amplitude::Real`: Wave velocity amplitude
- `surface_depth::Real`: E-folding depth [m]
- `plans`: FFT plans for forward transform
- `uniform`: Horizontally uniform waves (default: true)
- `profile`: Vertical decay profile (:gaussian or :exponential)
"""
function init_surface_waves!(Bk, G::RuntimeGeometry, amplitude::Real, surface_depth::Real, plans;
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
    init_random_waves!(Bk, G::RuntimeGeometry, amplitude::Real; slope::Real=-2.0)

Initialize wave field with random amplitudes and phases.
"""
function init_random_waves!(Bk, G::RuntimeGeometry, amplitude::Real; slope::Real=-2.0)
    @info "Initializing random wave field (amplitude=$amplitude, slope=$slope)"
    
    if Bk isa PencilArray
        error("init_random_waves! does not support PencilArray; " *
              "initialize waves through the model-level set! API.")
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
    apply_dealiasing_mask!(field, G::RuntimeGeometry)

Apply 2/3 dealiasing mask to spectral field using radial cutoff.
Handles both serial (Array) and parallel (PencilArray) cases.

Uses the same radial 2/3 rule as `dealias_mask()`:
- Keep integer wavenumbers with |k| < N/3
- Radial cutoff ensures isotropic dealiasing
"""
function apply_dealiasing_mask!(field, G::RuntimeGeometry)
    # Get local array and its dimensions
    field_arr = parent(field)
    nz_local, nx_local, ny_local = size(field_arr)

    for k in 1:nz_local
        for j_local in 1:ny_local
            # Get global j index for wavenumber lookup
            j_global = local_to_global(j_local, 3, field)

            for i_local in 1:nx_local
                # Get global i index for wavenumber lookup
                i_global = local_to_global(i_local, 2, field)

                # Share the canonical cutoff, including its exclusion of an
                # exact N/3 endpoint when a dimension is divisible by three.
                if !is_dealiased(i_global, j_global, G)
                    field_arr[k, i_local, j_local] = zero(eltype(field_arr))
                end
            end
        end
    end
end

"""
    compute_energy_spectrum(field, G::RuntimeGeometry)

Compute horizontal energy spectrum E(k) from a spectral field.

In MPI mode, this computes the local contribution only (no MPI reduction).
"""
function compute_energy_spectrum(field, G::RuntimeGeometry)
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
    normalize_field_energy!(field, G::RuntimeGeometry, target_energy::Real, plans)

Normalize field to have specified total energy.
"""
function normalize_field_energy!(field, G::RuntimeGeometry, target_energy::Real, plans)
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
    create_wave_packet(G::RuntimeGeometry, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real;
                       z_center=G.Lz/2, z_width=G.Lz/4)

Create a horizontally localized wave packet in spectral space with a Gaussian
vertical envelope. `z_center` and `z_width` are depths measured positively
downward from the surface.
"""
function create_wave_packet(G::RuntimeGeometry, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real;
                            z_center::Real=G.Lz / 2,
                            z_width::Real=G.Lz / 4)
    sigma_k > 0 || throw(ArgumentError("sigma_k must be positive"))
    0 <= z_center <= G.Lz || throw(ArgumentError("z_center must lie between 0 and Lz=$(G.Lz)"))
    z_width > 0 || throw(ArgumentError("z_width must be positive"))

    field = zeros(ComplexF64, G.nz, G.nx, G.ny)
    
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    
    for k in 1:G.nz
        depth = -G.z[k]
        z_envelope = exp(-((depth - z_center)^2) / (2 * z_width^2))
        
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
    add_balanced_component!(S, G, a_ell)

Add balanced component to the flow by computing geostrophically consistent fields.

This function:
1. Computes potential vorticity q from the streamfunction ψ
2. Computes geostrophically balanced velocities u = -∂ψ/∂y, v = ∂ψ/∂x
3. Computes buoyancy b = ∂ψ/∂z (from thermal wind balance)

Based on init_psi_generic and init_q from the Fortran implementation.

# Arguments
- `S::ModelFields`: Model state with streamfunction psi
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `a_ell`: Model-owned f²/N² coefficient profile

# Example
```julia
# With constant stratification
add_balanced_component!(fields, grid, coefficients.a_ell)

# With variable stratification
N2_face = evaluate_N2.(Ref(strat_profile), grid.z_faces[2:end])
a_ell = a_ell_from_N2(N2_face, FPlane(f=1e-4))
add_balanced_component!(fields, grid, a_ell)
```
"""
function add_balanced_component!(S::ModelFields, G::RuntimeGeometry,
    a_ell::AbstractVector; workspace=nothing)
    @info "Adding balanced component to initial state"

    nz = G.nz
    dz = nz > 1 ? (G.z[2] - G.z[1]) : 1.0
    length(a_ell) == nz || throw(DimensionMismatch("a_ell must have length $nz"))

    # Get underlying arrays
    psi_arr = parent(S.psi)
    nz_local, nx_local, ny_local = size(psi_arr)

    # Compute potential vorticity q from ψ
    # q = -kh² ψ + ∂/∂z (a_ell ∂ψ/∂z)
    if hasfield(typeof(S), :q)
        compute_q_from_psi!(S.q, S.psi, G, a_ell, dz; workspace)
        @info "Computed potential vorticity q from streamfunction"
    end

    # Note: Geostrophic velocities (u, v) are NOT computed here.
    # The ModelFields struct has u, v as real-space arrays, and proper velocity computation
    # requires FFT plans and workspace. Velocities will be computed consistently by
    # compute_velocities! during the first ETD-RK2 stage.

    # Compute buoyancy from thermal wind balance
    # b = ∂ψ/∂z (in QG approximation with constant N²)
    if hasfield(typeof(S), :b)
        compute_buoyancy_from_psi!(S.b, S.psi, G, dz)
        @info "Computed buoyancy b from thermal wind balance"
    end
end

"""
    compute_q_from_psi!(q, psi, G, a_ell, dz)

Compute QG potential vorticity from streamfunction.

The Boussinesq PV-streamfunction relationship is:
    q = ∇²ψ + ∂/∂z (a_ell ∂ψ/∂z)

In spectral space with finite differences in z:
    q = -kh² ψ + (1/dz²) ∂z[a_ell ∂zψ]

with Neumann BC ∂ψ/∂z = 0 at boundaries (boundary PV sheets handled by one-sided stencil).
"""
function compute_q_from_psi!(q, psi, G::RuntimeGeometry, a_ell, dz; workspace=nothing)
    with_z_local(G, (q, psi), (:out, :in);
                 scratch=z_scratch(workspace, :q_z, :psi_z)) do q_z, psi_z
        _compute_q_from_psi_kernel!(q_z, psi_z, G, a_ell, dz)
    end
    return q
end

"""Vertical stencil for `compute_q_from_psi!`; requires z to be fully local."""
function _compute_q_from_psi_kernel!(q, psi, G::RuntimeGeometry, a_ell, dz)
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
            coeff_up = a_ell[k]
            coeff_down = a_ell[k-1]

            vert_term = coeff_up * psi_arr[k+1, i_local, j_local] -
                       (coeff_up + coeff_down) * psi_arr[k, i_local, j_local] +
                       coeff_down * psi_arr[k-1, i_local, j_local]

            q_arr[k, i_local, j_local] = -kh2 * psi_arr[k, i_local, j_local] + vert_term / dz2
        end

        # Handle boundary conditions based on nz
        if nz == 1
            # Single-layer case: No vertical derivatives, q = -kh² ψ (2D barotropic mode)
            q_arr[1, i_local, j_local] = -kh2 * psi_arr[1, i_local, j_local]
        else
            # Bottom boundary (k=1): Neumann BC ψ_z = 0 ⟹ ψ[0] = ψ[1]
            coeff_up = a_ell[1]
            vert_term = coeff_up * (psi_arr[2, i_local, j_local] - psi_arr[1, i_local, j_local])
            q_arr[1, i_local, j_local] = -kh2 * psi_arr[1, i_local, j_local] + vert_term / dz2

            # Top boundary (k=nz): Neumann BC ψ_z = 0 ⟹ ψ[nz+1] = ψ[nz]
            coeff_down = a_ell[nz-1]
            vert_term = coeff_down * (psi_arr[nz-1, i_local, j_local] - psi_arr[nz, i_local, j_local])
            q_arr[nz, i_local, j_local] = -kh2 * psi_arr[nz, i_local, j_local] + vert_term / dz2
        end
    end
end

"""
    compute_barotropic_q_from_psi!(q, psi, grid)

Set potential vorticity from a vertically uniform spectral streamfunction using
`q̂ = -kₕ² ψ̂`. This is useful for prescribed barotropic flows, such as the
Asselin et al. (2020) dipole.
"""
function compute_barotropic_q_from_psi!(q, psi, G::RuntimeGeometry)
    q_arr = parent(q)
    psi_arr = parent(psi)
    size(q_arr) == size(psi_arr) ||
        throw(DimensionMismatch("q and psi must have the same local size"))

    nz_local, nx_local, ny_local = size(psi_arr)
    @inbounds for k in 1:nz_local, j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, psi)
        j_global = local_to_global(j_local, 3, psi)
        kₕ² = G.kx[i_global]^2 + G.ky[j_global]^2
        q_arr[k, i_local, j_local] = -kₕ² * psi_arr[k, i_local, j_local]
    end

    return q
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
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans for inverse transform

# Note
For typical use, velocities are computed by `compute_velocities!` in the main
timestepping loop. This function is provided for initialization or diagnostics.
"""
function compute_geostrophic_velocities!(u, v, psi, G::RuntimeGeometry, plans)
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
function compute_buoyancy_from_psi!(b, psi, G::RuntimeGeometry, dz)
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
    check_initial_conditions(S::ModelFields, G::RuntimeGeometry, plans)

Perform basic checks on initial conditions.
"""
function check_initial_conditions(S::ModelFields, G::RuntimeGeometry, plans)
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
