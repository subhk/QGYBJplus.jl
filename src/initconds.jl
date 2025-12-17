"""
Initial condition generators inspired by init.f90 (test1).
Provides a simple random-phase ring in k-space with optional vertical structure.
"""

"""
    init_random_psi!(S, G; initial_k=5, amp_width=2.0, linear_vert_structure=0)

Populate real-space ψ with a random-phase Gaussian ring at |k|≈initial_k, then
apply optional vertical structure: 0=QG-consistent kz~kh, 1=linear in z,
2=constant kz=1. Returns S with ψ set in spectral space.
"""
function init_random_psi!(S::State, G::Grid; initial_k=5, amp_width=2.0, linear_vert_structure=0)
    nx, ny, nz = G.nx, G.ny, G.nz
    # Build ψ in real space then FFT to spectral
    ψᵣ = zeros(Float64, nx, ny, nz)
    # Deterministic pseudo-random phases based on integer hash (no Random dependency)
    phase_for(ikₓ::Int, ikᵧ::Int) = 2π * ((hash((ikₓ, ikᵧ)) % 1_000_000) / 1_000_000)
    for ikₓ in -2*initial_k:2*initial_k, ikᵧ in -2*initial_k:2*initial_k
        kₕ² = ikₓ^2 + ikᵧ^2
        kₕ = sqrt(kₕ²)
        kₕ² == 0 && continue
        κ = sqrt(kₕ²)
        amp = exp( - (κ - initial_k)^2 / (2*amp_width) )
        for k in 1:nz, i in 1:nx, j in 1:ny
            z = G.z[k]
            phase = phase_for(ikₓ, ikᵧ)
            # Horizontal phase: uses normalized coordinates (domain-independent)
            horiz_phase = ikₓ*(i-1)*2π/nx + ikᵧ*(j-1)*2π/ny + phase
            if linear_vert_structure == 1
                # Linear in z around center of domain
                z₀ = G.Lz / 2
                ψᵣ[i,j,k] += (z - z₀) * amp * cos(horiz_phase)
            elseif linear_vert_structure == 2
                # Single vertical mode
                kz = 2π / G.Lz
                ψᵣ[i,j,k] += amp * cos(horiz_phase + kz*z)
            else
                # QG-consistent: kz ~ kh (scaled to domain)
                kz = kₕ * 2π / G.Lz
                ψᵣ[i,j,k] += amp * cos(horiz_phase + kz*z)
            end
        end
    end
    # Forward FFT to spectral ψ
    plans = plan_transforms!(G)
    fft_forward!(S.psi, ψᵣ, plans)
    return S
end
