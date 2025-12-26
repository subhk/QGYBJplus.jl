"""
Initial condition generators inspired by init.f90 (test1).
Provides a simple random-phase ring in k-space with optional vertical structure.
"""

"""
    init_random_psi!(S, G; initial_k=5, amp_width=2.0, linear_vert_structure=0)

Populate real-space ψ with a random-phase Gaussian ring at |k|≈initial_k, then
apply optional vertical structure:
- 0 = QG-consistent kz ~ kh (kz scales with horizontal wavenumber)
- 1 = Linear in z (amplitude varies linearly from domain center)
- 2 = Single vertical mode (kz = 2π/Lz, one wavelength in z)

Returns S with ψ set in spectral space.

Note: FFT plans are created internally for each call. For efficiency in repeated
initialization, consider using the initialization routines in initialization.jl
which accept pre-computed plans.
"""
function init_random_psi!(S::State, G::Grid; initial_k=5, amp_width=2.0, linear_vert_structure=0)
    nx, ny, nz = G.nx, G.ny, G.nz
    # Build ψ in real space then FFT to spectral
    # Use element type from State for type consistency
    T = real(eltype(S.psi))
    ψᵣ = zeros(T, nz, nx, ny)
    # Deterministic pseudo-random phases based on integer hash (no Random dependency)
    phase_for(ikₓ::Int, ikᵧ::Int) = T(2π) * ((hash((ikₓ, ikᵧ)) % 1_000_000) / 1_000_000)
    for ikₓ in -2*initial_k:2*initial_k, ikᵧ in -2*initial_k:2*initial_k
        kₕ² = ikₓ^2 + ikᵧ^2
        kₕ² == 0 && continue
        kₕ = sqrt(T(kₕ²))  # Use kₕ consistently (removed redundant κ)
        amp = exp(-(kₕ - initial_k)^2 / (2*amp_width))
        for k in 1:nz, i in 1:nx, j in 1:ny
            z = G.z[k]
            phase = phase_for(ikₓ, ikᵧ)
            # Horizontal phase: uses normalized coordinates (domain-independent)
            horiz_phase = ikₓ*(i-1)*2π/nx + ikᵧ*(j-1)*2π/ny + phase
            if linear_vert_structure == 1
                # Linear in z around center of domain
                z₀ = G.Lz / 2
                ψᵣ[k, i, j] += (z - z₀) * amp * cos(horiz_phase)
            elseif linear_vert_structure == 2
                # Single vertical mode
                kz = 2π / G.Lz
                ψᵣ[k, i, j] += amp * cos(horiz_phase + kz*z)
            else
                # QG-consistent: kz ~ kh (scaled to domain)
                kz = kₕ * 2π / G.Lz
                ψᵣ[k, i, j] += amp * cos(horiz_phase + kz*z)
            end
        end
    end
    # Forward FFT to spectral ψ
    plans = plan_transforms!(G)
    fft_forward!(S.psi, ψᵣ, plans)
    return S
end
