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
function init_random_psi!(S::State, G::Grid; initial_k=5, amp_width=2.0, linear_vert_structure=0, par::QGParams=default_params())
    nx, ny, nz = G.nx, G.ny, G.nz
    # Build ψ in real space then FFT to spectral
    using Random
    ψr = zeros(Float64, nx, ny, nz)
    φ = [2π*rand() for _ in 1:(4*initial_k+1), __ in 1:(4*initial_k+1)]
    for ikx in -2*initial_k:2*initial_k, iky in -2*initial_k:2*initial_k
        kh2 = ikx^2 + iky^2
        kh = sqrt(kh2)
        kh2 == 0 && continue
        kk = sqrt(kh2)
        amp = exp( - (kk - initial_k)^2 / (2*amp_width) )
        for k in 1:nz, i in 1:nx, j in 1:ny
            z = G.z[k]
            phase = φ[ikx+2*initial_k+1, iky+2*initial_k+1]
            if linear_vert_structure == 1
                # linear in z around z0 = π (center of domain)
                z0 = π
                ψr[i,j,k] += (z - z0) * amp * cos(ikx*(i-1)*2π/nx + iky*(j-1)*2π/ny + phase)
            elseif linear_vert_structure == 2
                kz = 1.0
                ψr[i,j,k] += amp * cos(ikx*(i-1)*2π/nx + iky*(j-1)*2π/ny + kz*z + phase)
            else
                kz = kh  # Normalized (Bu = 1.0)
                ψr[i,j,k] += amp * cos(ikx*(i-1)*2π/nx + iky*(j-1)*2π/ny + kz*z + phase)
            end
        end
    end
    # Forward FFT to spectral ψ
    plans = plan_transforms!(G)
    fft_forward!(S.psi, ψr, plans)
    return S
end

