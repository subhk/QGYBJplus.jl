#=
================================================================================
                    diagnostics.jl - Energy and Field Diagnostics
================================================================================

This file provides diagnostic routines for analyzing QG-YBJ+ simulations,
including energy computations, the omega equation RHS, and field slicing.

ENERGY DIAGNOSTICS:
-------------------
Energy is a key diagnostic for verifying model behavior:

1. FLOW KINETIC ENERGY:
   KE = (1/2) ∫∫∫ (u² + v²) dx dy dz

   In QG, KE is related to enstrophy and streamfunction.
   Conservation/decay of KE indicates model stability.

2. WAVE ENERGY:
   WE_B = (1/2) ∫∫∫ |B|² dx dy dz   (envelope-based)
   WE_A = (1/2) ∫∫∫ |A|² dx dy dz   (amplitude-based)

   Wave energy transfer between scales indicates cascade direction.
   Energy exchange with mean flow shows wave-mean interaction.

OMEGA EQUATION RHS:
-------------------
The omega equation RHS drives ageostrophic vertical motion:

    ∇²w + (f²/N²) ∂²w/∂z² = (2f/N²) J(ψ_z, ∇²ψ)

The RHS 2J(ψ_z, ∇²ψ) represents:
- Jacobian of vertical shear (thermal wind) and vorticity
- Physically: differential advection creating divergence
- Strong near fronts and eddy boundaries

FIELD SLICING:
--------------
Utility functions for extracting 2D slices from 3D spectral fields:
- slice_horizontal: x-y plane at fixed z (good for surface fields)
- slice_vertical_xz: x-z plane at fixed y (good for vertical structure)

FORTRAN CORRESPONDENCE:
-----------------------
- omega_eqn_rhs! → omega_eqn_rhs in diagnostics.f90
- wave_energy → energy diagnostics in diagnostics.f90
- flow_kinetic_energy → ke_flow in diagnostics.f90

================================================================================
=#

module Diagnostics

using ..QGYBJplus: RuntimeGeometry
using ..QGYBJplus: plan_transforms!, fft_forward!, fft_backward!
using ..QGYBJplus: local_to_global, z_is_local
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: allocate_z_pencil
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
using ..QGYBJplus: with_z_local, z_scratch
using ..QGYBJplus: with_scratch, scratch_like, scratch_physical, scratch_phys_like
import PencilArrays: PencilArray

# Reference to parent module for physics functions
const PARENT = Base.parentmodule(@__MODULE__)

# Alias for internal use
const _allocate_fft_dst = allocate_fft_backward_dst

#=
================================================================================
                    OMEGA EQUATION RHS COMPUTATION
================================================================================
The RHS of the omega equation drives the ageostrophic vertical velocity.
================================================================================
=#

"""
    omega_eqn_rhs!(rhs, psi, G, plans; Lmask=nothing, workspace=nothing)

Compute the RHS forcing for the QG omega equation.

# Physical Background
The QG omega equation relates vertical velocity w to the horizontal flow:

    N² ∇²w + f² ∂²w/∂z² = 2f J(ψ_z, ∇²ψ)

or equivalently (dividing by N²):

    ∇²w + (f²/N²) ∂²w/∂z² = (2f/N²) J(ψ_z, ∇²ψ)

This function computes 2 J(ψ_z, ∇²ψ). The solver then applies the (f/N²) scaling.

# Physical Interpretation
The Jacobian J(ψ_z, ∇²ψ) represents:
- ψ_z: Vertical shear of streamfunction (related to thermal wind/buoyancy)
- ∇²ψ: Relative vorticity ζ
- J: Cross-gradient interaction

Strong RHS forcing occurs where:
- Fronts (large ψ_z) interact with vorticity gradients
- Eddies tilt isopycnals through differential advection

# Numerical Method
1. **Vertical derivative**: ψ_z via forward finite difference
   ```
   ψ_z[k] = (ψ[k+1] - ψ[k]) / dz,  ψ_z[nz] = 0 (Neumann)
   ```

2. **Spectral derivatives**:
   - ∂ψ_z/∂x = i kₓ ψ_z
   - ∂ψ_z/∂y = i kᵧ ψ_z
   - ∂(∇²ψ)/∂x = -i kₓ kh² ψ_avg
   - ∂(∇²ψ)/∂y = -i kᵧ kh² ψ_avg
   where ψ_avg = (ψ[k+1] + ψ[k])/2 for staggered-grid consistency

3. **Jacobian in physical space**:
   ```
   J(ψ_z, ∇²ψ) = (∂ψ_z/∂x)(∂∇²ψ/∂y) - (∂ψ_z/∂y)(∂∇²ψ/∂x)
   ```

4. **Transform back**: FFT to get spectral RHS

# Arguments
- `rhs::Array{Complex,3}`: Output RHS array (modified in-place)
- `psi::Array{Complex,3}`: Spectral streamfunction
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `Lmask`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
Modified rhs array with the omega equation forcing.

# Fortran Correspondence
Matches `omega_eqn_rhs` computation in the Fortran implementation.
"""
function omega_eqn_rhs!(rhs, psi, G::RuntimeGeometry, plans; Lmask=nothing, workspace=nothing)
    with_scratch(workspace) do
        _omega_eqn_rhs!(rhs, psi, G, plans, Lmask, workspace)
    end
    return rhs
end

# Direct computation when z is fully local (serial or 1D decomposition)
# One implementation for every decomposition. Only the vertical difference and
# vertical average couple neighbouring levels, so those run under
# `with_z_local`; everything after is pointwise in z.
function _omega_eqn_rhs!(rhs, psi, G::RuntimeGeometry, plans, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    ψzₖ = scratch_like(workspace, psi)     # ∂ψ/∂z, one-sided at the top
    ψavgₖ = scratch_like(workspace, psi)   # ψ averaged onto the same half level

    with_z_local(G, (ψzₖ, ψavgₖ, psi), (:out, :out, :in);
                 scratch=z_scratch(workspace, :q_z, :A_z, :psi_z)) do ψz, ψavg, ψ
        ψz_arr = parent(ψz)
        ψavg_arr = parent(ψavg)
        ψ_arr = parent(ψ)
        nz_z, nx_z, ny_z = size(ψ_arr)
        @assert nz_z == nz "Vertical dimension must be fully local for omega RHS"
        @inbounds for k in 1:nz_z, j in 1:ny_z, i in 1:nx_z
            if k == nz
                ψz_arr[k, i, j] = 0                     # Neumann top
                ψavg_arr[k, i, j] = ψ_arr[k, i, j]
            else
                ψz_arr[k, i, j] = (ψ_arr[k+1, i, j] - ψ_arr[k, i, j]) / Δz
                ψavg_arr[k, i, j] = 0.5*(ψ_arr[k+1, i, j] + ψ_arr[k, i, j])
            end
        end
    end

    ψzₖ_arr = parent(ψzₖ)
    ψavgₖ_arr = parent(ψavgₖ)
    nz_local, nx_local, ny_local = size(ψzₖ_arr)

    # Build needed spectral derivatives
    bxₖ = scratch_like(workspace, psi); byₖ = scratch_like(workspace, psi)
    xxₖ = scratch_like(workspace, psi); xyₖ = scratch_like(workspace, psi)
    bxₖ_arr = parent(bxₖ); byₖ_arr = parent(byₖ)
    xxₖ_arr = parent(xxₖ); xyₖ_arr = parent(xyₖ)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, psi)
        j_global = local_to_global(j, 3, psi)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        bxₖ_arr[k, i, j] = im*kₓ*ψzₖ_arr[k, i, j]
        byₖ_arr[k, i, j] = im*kᵧ*ψzₖ_arr[k, i, j]
        xxₖ_arr[k, i, j] = -im*kₓ*kₕ²*ψavgₖ_arr[k, i, j]
        xyₖ_arr[k, i, j] = -im*kᵧ*kₕ²*ψavgₖ_arr[k, i, j]
    end

    # To real space - use helper for correct pencil allocation
    bxᵣ = scratch_physical(workspace, bxₖ, plans)
    byᵣ = scratch_physical(workspace, byₖ, plans)
    xxᵣ = scratch_physical(workspace, xxₖ, plans)
    xyᵣ = scratch_physical(workspace, xyₖ, plans)
    fft_backward!(bxᵣ, bxₖ, plans)
    fft_backward!(byᵣ, byₖ, plans)
    fft_backward!(xxᵣ, xxₖ, plans)
    fft_backward!(xyᵣ, xyₖ, plans)

    bxᵣ_arr = parent(bxᵣ); byᵣ_arr = parent(byᵣ)
    xxᵣ_arr = parent(xxᵣ); xyᵣ_arr = parent(xyᵣ)

    # Real-space RHS
    rhsᵣ = scratch_phys_like(workspace, bxᵣ)
    rhsᵣ_arr = parent(rhsᵣ)
    # The physical FFT input pencil can have different local x/y extents from
    # the spectral output pencil even when z is fully local.
    nz_phys, nx_phys, ny_phys = size(rhsᵣ_arr)
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        rhsᵣ_arr[k, i, j] = 2.0 * ( real(bxᵣ_arr[k, i, j])*real(xyᵣ_arr[k, i, j]) - real(byᵣ_arr[k, i, j])*real(xxᵣ_arr[k, i, j]) )
    end

    # Back to spectral. fft_backward! uses a normalized IFFT, so the
    # pseudo-spectral product is already correctly scaled; only dealias.
    fft_forward!(rhs, rhsᵣ, plans)

    rhs_arr = parent(rhs)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, rhs)
        j_global = local_to_global(j, 3, rhs)
        if !L[i_global, j_global]
            rhs_arr[k, i, j] = 0  # Dealias
        end
    end
    return rhs
end

#=
================================================================================
                    ENERGY DIAGNOSTICS
================================================================================
Energy measures for monitoring simulation health and physics.
================================================================================
=#

"""
    flow_kinetic_energy(u, v) -> KE

Compute domain-integrated kinetic energy of the geostrophic flow (simple version).

# Physical Background
The kinetic energy of the balanced flow:

    KE = (1/2) ∫∫∫ (u² + v²) dx dy dz

This is a key diagnostic for:
- Model stability (unbounded growth indicates numerical issues)
- Energy conservation/dissipation rate
- Turbulent cascade analysis

# Returns
Total kinetic energy (domain sum, not mean) in nondimensional units.

# Note
- This is NOT normalized by volume. For energy density, divide by nx×ny×nz.
- In MPI mode, this returns LOCAL energy. Use mpi_reduce_sum for global total.
- For energy with spectral dealiasing,
  use `flow_kinetic_energy_spectral` instead.
"""
function flow_kinetic_energy(u, v)
    # Works with any array (regular or PencilArray)
    # Uses real() to handle both real and complex arrays correctly
    # (for real arrays, real() is a no-op; for complex, extracts real part)
    u_arr = parent(u)
    v_arr = parent(v)
    KE = 0.0
    @inbounds for i in eachindex(u_arr)
        KE += 0.5 * (real(u_arr[i])^2 + real(v_arr[i])^2)
    end
    return KE
end

"""
    flow_kinetic_energy_spectral(uk, vk, G; Lmask=nothing) -> KE

Compute Boussinesq kinetic energy in spectral space with dealiasing.

# Physical Background (matches Fortran diag_zentrum/energy_linear)
The kinetic energy is computed as:

    KE(z) = Σₖ L(kₓ,kᵧ) × (|uₖ|² + |vₖ|²) - 0.5 × (|u₀₀|² + |v₀₀|²)

The dealiasing correction subtracts half the kh=0 mode because:
- With 2/3 dealiasing: Σₖ (1/2)|u|² = Σₖ L|u|² - 0.5|u(0,0)|²

The total KE averages over the vertical levels:

    KE_total = (1/nz) Σᵢ KE(zᵢ)

# Algorithm
1. Loop over all spectral modes (kₓ, kᵧ, z) with dealiasing mask L
2. Accumulate |u|² + |v|² at each level
3. Apply dealiasing correction: subtract half the kh=0 mode
4. Average over vertical levels (divide by nz)

# Arguments
- `uk, vk`: Spectral velocity fields (complex)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `Lmask`: Optional dealiasing mask (default: all modes included)

# Returns
Total kinetic energy per unit reference mass, normalized by nz.

# Fortran Correspondence
Matches the kinetic energy computation in `diag_zentrum` (diagnostics.f90:127-161)
and `energy_linear` (diagnostics.f90:3024-3107).

# Note
In MPI mode, returns LOCAL energy. Use mpi_reduce_sum for global total.
"""
function flow_kinetic_energy_spectral(uk, vk, G::RuntimeGeometry; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx, ny) : Lmask

    # Get local dimensions
    uk_arr = parent(uk)
    vk_arr = parent(vk)
    nz_local, nx_local, ny_local = size(uk_arr)

    KE_total = 0.0

    @inbounds for k in 1:nz_local
        ke_k = 0.0

        # Sum over horizontal wavenumbers with dealiasing
        for j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, uk)
            j_global = local_to_global(j, 3, uk)

            if L[i_global, j_global]
                # KE contribution: |u|² + |v|²
                ke_k += abs2(uk_arr[k, i, j]) + abs2(vk_arr[k, i, j])
            end
        end

        # Dealiasing correction: subtract half the kh=0 mode
        # The kh=0 mode is at global index (1,1)
        if local_to_global(1, 2, uk) == 1 && local_to_global(1, 3, uk) == 1
            # This process owns the (1,1) mode
            ke_k -= 0.5 * (abs2(uk_arr[k, 1, 1]) + abs2(vk_arr[k, 1, 1]))
        end

        KE_total += ke_k
    end

    # Normalize by nz (vertical integration)
    KE = KE_total / nz

    return KE
end

"""
    flow_potential_energy_spectral(bk, G, par; Lmask=nothing) -> PE

Compute Boussinesq potential energy in spectral space with dealiasing.

# Physical Background
The potential energy from buoyancy variance:

    PE(z) = Σₖ L(kₓ,kᵧ) × a_ell × |bₖ|² - 0.5 × correction

where a_ell = f²/N² is the elliptic coefficient.

For QG: b = ψ_z, so PE represents available potential energy from isopycnal tilting.

# Arguments
- `bk`: Spectral buoyancy field (complex)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `par`: Coefficient provider for Coriolis and stratification
- `Lmask`: Optional dealiasing mask

# Returns
Total potential energy per unit reference mass, normalized by nz.

# Fortran Correspondence
Matches the potential energy computation in `diag_zentrum` (ps term).
"""
function flow_potential_energy_spectral(bk, G::RuntimeGeometry, par; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx, ny) : Lmask

    # Get z-dependent elliptic coefficient a(z) = f²/N²(z)
    # This handles both constant_N and skewed_gaussian stratification correctly
    a_ell = if isdefined(PARENT, :a_ell_ut) && par !== nothing
        PARENT.a_ell_ut(par, G)
    else
        fill(par.f₀^2 / par.N², nz)  # Fallback to constant
    end

    # Get local dimensions
    bk_arr = parent(bk)
    nz_local, nx_local, ny_local = size(bk_arr)

    PE_total = 0.0

    @inbounds for k in 1:nz_local
        # Use global z-index for correct profile lookup in 2D decomposition
        k_global = local_to_global(k, 1, bk)
        a_ell_k = k_global <= length(a_ell) ? a_ell[k_global] : a_ell[end]

        pe_k = 0.0

        for j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, bk)
            j_global = local_to_global(j, 3, bk)

            if L[i_global, j_global]
                pe_k += a_ell_k * abs2(bk_arr[k, i, j])
            end
        end

        # Dealiasing correction
        if local_to_global(1, 2, bk) == 1 && local_to_global(1, 3, bk) == 1
            pe_k -= 0.5 * a_ell_k * abs2(bk_arr[k, 1, 1])
        end

        PE_total += pe_k
    end

    # Normalize by nz
    PE = PE_total / nz

    return PE
end

"""
    wave_energy_vavg(B, G, plans) -> WE_ave::Array{Float64,2}

Compute vertically-averaged wave energy density in physical space.

# Physical Background
The wave energy density based on envelope B:

    WE(x,y,z) = (1/2) |B|²

This function returns the vertical average:

    WE_avg(x,y) = (1/nz) Σₖ WE(x,y,k)

# Use Cases
- Visualize horizontal wave energy distribution
- Track wave energy concentration in eddies
- Monitor wave-mean flow interaction regions

# Algorithm
1. Transform B to physical space
2. Compute 0.5|B|² at each point
3. Average over vertical levels

# Returns
2D array (nx_local, ny_local) of vertically-averaged wave energy density.

# Note
In MPI mode with 2D decomposition, this returns LOCAL data only.
For full domain visualization, gather data to root first.
"""
function wave_energy_vavg(B, G::RuntimeGeometry, plans)
    nz = G.nz

    # Get local dimensions
    B_arr = parent(B)
    nz_local, nx_local, ny_local = size(B_arr)

    # Invert full complex field to physical space
    Br = _allocate_fft_dst(B, plans)
    fft_backward!(Br, B, plans)
    Br_arr = parent(Br)

    # Accumulate 0.5|B|^2 and average over nz
    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(Br_arr)
    WE = zeros(Float64, nx_phys, ny_phys)
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        WE[i,j] += 0.5 * abs2(Br_arr[k, i, j])
    end
    WE ./= nz
    return WE
end

#=
================================================================================
                    FIELD SLICING UTILITIES
================================================================================
Extract 2D slices from 3D spectral fields for visualization.
================================================================================
=#

"""
    slice_horizontal(field, G, plans; k::Int) -> Array{Float64,2}

Extract a horizontal (x-y) slice from a spectral 3D field.

# Description
Transforms a spectral field to physical space and extracts the horizontal
slice at LOCAL vertical index k.

# Use Cases
- Top-level plots (k=nz for closest to surface)
- Deep field structure (k=1 for closest to bottom)
- Vertical structure analysis at specific depths

# Arguments
- `field::Array{Complex,3}`: Spectral field (nz, nx, ny)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `k::Int`: LOCAL vertical index for slice (1 ≤ k ≤ nz_local)

# Returns
2D real array (nx_local, ny_local) with values at local z[k].

# Note
In MPI mode with 2D decomposition, k is a LOCAL index.
For full domain slices, gather data to root first.
"""
function slice_horizontal(field, G::RuntimeGeometry, plans; k::Int)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Inverse FFT entire field to get real slice
    Xr = _allocate_fft_dst(field, plans)
    fft_backward!(Xr, field, plans)
    Xr_arr = parent(Xr)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(Xr_arr)

    @assert 1 <= k <= nz_phys "k=$k must be within local range 1:$nz_phys"

    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    sl = Array{Float64}(undef, nx_phys, ny_phys)
    @inbounds for j in 1:ny_phys, i in 1:nx_phys
        sl[i,j] = real(Xr_arr[k, i, j])
    end
    return sl
end

"""
    slice_vertical_xz(field, G, plans; j::Int) -> Array{Float64,2}

Extract a vertical (x-z) slice from a spectral 3D field at fixed y.

# Description
Transforms a spectral field to physical space and extracts the x-z
slice at LOCAL y-index j.

# Use Cases
- Vertical wave structure visualization
- Eddy vertical extent analysis
- Thermocline/pycnocline interaction studies

# Arguments
- `field::Array{Complex,3}`: Spectral field (nz, nx, ny)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `j::Int`: LOCAL Y-index for slice (1 ≤ j ≤ ny_local)

# Returns
2D real array (nx_local, nz_local) with values at local y[j].

# Note
In MPI mode with 2D decomposition, j is a LOCAL index.
For full domain slices, gather data to root first.
"""
function slice_vertical_xz(field, G::RuntimeGeometry, plans; j::Int)
    nx, ny, nz = G.nx, G.ny, G.nz

    Xr = _allocate_fft_dst(field, plans)
    fft_backward!(Xr, field, plans)
    Xr_arr = parent(Xr)

    # Use physical array dimensions (may differ from spectral in 2D decomposition)
    nz_phys, nx_phys, ny_phys = size(Xr_arr)

    @assert 1 <= j <= ny_phys "j=$j must be within local range 1:$ny_phys"

    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    sl = Array{Float64}(undef, nx_phys, nz_phys)
    @inbounds for k in 1:nz_phys, i in 1:nx_phys
        sl[i,k] = real(Xr_arr[k, i, j])
    end
    return sl
end

"""
    wave_energy(B, A) -> (E_B, E_A)

Compute domain-integrated wave energy from both B and A fields (simple version).

# Physical Background
Two measures of wave energy in the model:

1. **Envelope energy** E_B = Σ |B|²
   - Based on the evolved wave envelope
   - Directly available from prognostic variable

2. **Amplitude energy** E_A = Σ |A|²
   - Based on the recovered wave amplitude
   - More physically meaningful for wave energy flux

# Use Cases
- Monitor total wave energy conservation/dissipation
- Compare E_B and E_A to verify B→A recovery
- Track energy exchange with mean flow

# Arguments
- `B::Array{Complex,3}`: Wave envelope (spectral or physical)
- `A::Array{Complex,3}`: Wave amplitude (spectral or physical)

# Returns
Tuple (E_B, E_A) of domain-summed squared magnitudes.

# Note
- These are domain SUMS, not means. For energy density, divide by grid volume.
- In MPI mode, this returns LOCAL energy. Use mpi_reduce_sum for global total.
- For wave energies with spectral dealiasing,
"""
function wave_energy(B, A)
    # Works with any array (regular or PencilArray)
    B_arr = parent(B)
    A_arr = parent(A)
    EB = 0.0; EA = 0.0
    @inbounds for x in B_arr; EB += abs2(x); end
    @inbounds for x in A_arr; EA += abs2(x); end
    return EB, EA
end


#=
================================================================================
                    GLOBAL ENERGY DIAGNOSTICS (MPI-AWARE)
================================================================================
`flow_kinetic_energy` and `wave_energy` accumulate over the local portion of a
distributed field. These wrappers add the MPI reduction, and are what the
model-level `flow_kinetic_energy(::QGYBJModel)` / `wave_energy(::QGYBJModel)`
entry points in core/io.jl call.
================================================================================
=#

"""
    flow_kinetic_energy_global(u, v, mpi_config=nothing) -> KE

Domain-integrated balanced-flow kinetic energy summed across all MPI ranks.
Pass `mpi_config=nothing` for a serial field, where the local sum is already
the global one.
"""
function flow_kinetic_energy_global(u, v, mpi_config=nothing)
    KE_local = flow_kinetic_energy(u, v)
    mpi_config === nothing && return KE_local
    return PARENT.mpi_reduce_sum(KE_local, mpi_config)
end

"""
    wave_energy_global(B, A, mpi_config=nothing) -> (E_B, E_A)

Wave envelope and amplitude energies summed across all MPI ranks. Pass
`mpi_config=nothing` for a serial field.
"""
function wave_energy_global(B, A, mpi_config=nothing)
    EB_local, EA_local = wave_energy(B, A)
    mpi_config === nothing && return EB_local, EA_local
    return PARENT.mpi_reduce_sum(EB_local, mpi_config),
           PARENT.mpi_reduce_sum(EA_local, mpi_config)
end

end # module

# Export basic diagnostics
using .Diagnostics: omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg
using .Diagnostics: slice_horizontal, slice_vertical_xz

# Export spectral energy diagnostics (Fortran-compatible)
using .Diagnostics: flow_kinetic_energy_spectral, flow_potential_energy_spectral

# Export MPI-aware global energy functions
using .Diagnostics: flow_kinetic_energy_global, wave_energy_global
