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

using ..QGYBJplus: Grid
using ..QGYBJplus: plan_transforms!, fft_forward!, fft_backward!
using ..QGYBJplus: local_to_global, z_is_local
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: allocate_z_pencil
using ..QGYBJplus: allocate_fft_backward_dst  # Centralized FFT allocation helper
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
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `Lmask`: Optional dealiasing mask
- `workspace`: Optional pre-allocated workspace for 2D decomposition

# Returns
Modified rhs array with the omega equation forcing.

# Fortran Correspondence
Matches `omega_eqn_rhs` computation in the Fortran implementation.
"""
function omega_eqn_rhs!(rhs, psi, G::Grid, plans; Lmask=nothing, workspace=nothing)
    # Check if we need 2D decomposition with transposes
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(psi, G)

    if need_transpose
        _omega_eqn_rhs_2d!(rhs, psi, G, plans, Lmask, workspace)
    else
        _omega_eqn_rhs_direct!(rhs, psi, G, plans, Lmask)
    end
    return rhs
end

# Direct computation when z is fully local (serial or 1D decomposition)
function _omega_eqn_rhs_direct!(rhs, psi, G::Grid, plans, Lmask)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    # Get local dimensions
    ψ_arr = parent(psi)
    nz_local, nx_local, ny_local = size(ψ_arr)

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local for omega RHS"

    # ψ_z in spectral space (simple finite difference)
    ψzₖ = similar(psi)
    ψzₖ_arr = parent(ψzₖ)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        if k == nz
            ψzₖ_arr[k, i, j] = 0  # Neumann top
        else
            ψzₖ_arr[k, i, j] = (ψ_arr[k+1, i, j] - ψ_arr[k, i, j]) / Δz
        end
    end

    # Build needed spectral derivatives
    bxₖ = similar(psi); byₖ = similar(psi)
    xxₖ = similar(psi); xyₖ = similar(psi)
    bxₖ_arr = parent(bxₖ); byₖ_arr = parent(byₖ)
    xxₖ_arr = parent(xxₖ); xyₖ_arr = parent(xyₖ)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, psi)
        j_global = local_to_global(j, 3, psi)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = kₓ^2 + kᵧ^2

        bxₖ_arr[k, i, j] = im*kₓ*ψzₖ_arr[k, i, j]
        byₖ_arr[k, i, j] = im*kᵧ*ψzₖ_arr[k, i, j]
        # average psi between k and k+1; at top use psi at top
        ψavg = k < nz ? 0.5*(ψ_arr[k+1, i, j] + ψ_arr[k, i, j]) : ψ_arr[k, i, j]
        xxₖ_arr[k, i, j] = -im*kₓ*kₕ²*ψavg
        xyₖ_arr[k, i, j] = -im*kᵧ*kₕ²*ψavg
    end

    # To real space - use helper for correct pencil allocation
    bxᵣ = _allocate_fft_dst(bxₖ, plans); byᵣ = _allocate_fft_dst(byₖ, plans)
    xxᵣ = _allocate_fft_dst(xxₖ, plans); xyᵣ = _allocate_fft_dst(xyₖ, plans)
    fft_backward!(bxᵣ, bxₖ, plans)
    fft_backward!(byᵣ, byₖ, plans)
    fft_backward!(xxᵣ, xxₖ, plans)
    fft_backward!(xyᵣ, xyₖ, plans)

    bxᵣ_arr = parent(bxᵣ); byᵣ_arr = parent(byᵣ)
    xxᵣ_arr = parent(xxᵣ); xyᵣ_arr = parent(xyᵣ)

    # Real-space RHS
    rhsᵣ = similar(bxᵣ)
    rhsᵣ_arr = parent(rhsᵣ)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        rhsᵣ_arr[k, i, j] = 2.0 * ( real(bxᵣ_arr[k, i, j])*real(xyᵣ_arr[k, i, j]) - real(byᵣ_arr[k, i, j])*real(xxᵣ_arr[k, i, j]) )
    end

    # Back to spectral
    fft_forward!(rhs, rhsᵣ, plans)

    # Apply dealiasing mask only (no normalization needed)
    # fft_backward! uses normalized IFFT, so pseudo-spectral products are correctly scaled.
    # Previous code incorrectly divided by nx*ny, weakening omega forcing.
    rhs_arr = parent(rhs)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, rhs)
        j_global = local_to_global(j, 3, rhs)
        if !L[i_global, j_global]
            rhs_arr[k, i, j] = 0  # Dealias
        end
    end
end

# 2D decomposition version with transposes
function _omega_eqn_rhs_2d!(rhs, psi, G::Grid, plans, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    # Allocate z-pencil workspace
    ψ_z = workspace !== nothing && hasfield(typeof(workspace), :psi_z) ? workspace.psi_z : allocate_z_pencil(G, ComplexF64)
    ψz_z = allocate_z_pencil(G, ComplexF64)
    ψavg_z = allocate_z_pencil(G, ComplexF64)

    # Transpose psi to z-pencil for vertical operations
    transpose_to_z_pencil!(ψ_z, psi, G)

    # Compute ψ_z and ψ_avg in z-pencil configuration (z now fully local)
    ψ_z_arr = parent(ψ_z)
    ψz_z_arr = parent(ψz_z)
    ψavg_z_arr = parent(ψavg_z)

    nz_local, nx_local_z, ny_local_z = size(ψ_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    @inbounds for k in 1:nz, j in 1:ny_local_z, i in 1:nx_local_z
        if k == nz
            ψz_z_arr[k, i, j] = 0  # Neumann top
            ψavg_z_arr[k, i, j] = ψ_z_arr[k, i, j]
        else
            ψz_z_arr[k, i, j] = (ψ_z_arr[k+1, i, j] - ψ_z_arr[k, i, j]) / Δz
            ψavg_z_arr[k, i, j] = 0.5*(ψ_z_arr[k+1, i, j] + ψ_z_arr[k, i, j])
        end
    end

    # Transpose back to xy-pencil for horizontal operations
    ψzₖ = similar(psi)
    ψavgₖ = similar(psi)
    transpose_to_xy_pencil!(ψzₖ, ψz_z, G)
    transpose_to_xy_pencil!(ψavgₖ, ψavg_z, G)

    # Get local dimensions for xy-pencil
    ψzₖ_arr = parent(ψzₖ)
    ψavgₖ_arr = parent(ψavgₖ)
    nz_local_xy, nx_local, ny_local = size(ψzₖ_arr)

    # Build needed spectral derivatives in xy-pencil
    bxₖ = similar(psi); byₖ = similar(psi)
    xxₖ = similar(psi); xyₖ = similar(psi)
    bxₖ_arr = parent(bxₖ); byₖ_arr = parent(byₖ)
    xxₖ_arr = parent(xxₖ); xyₖ_arr = parent(xyₖ)

    @inbounds for k in 1:nz_local_xy, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, psi)
        j_global = local_to_global(j, 3, psi)
        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        # Compute kₕ² from global kx, ky arrays (works in both serial and parallel)
        kₕ² = kₓ^2 + kᵧ^2

        bxₖ_arr[k, i, j] = im*kₓ*ψzₖ_arr[k, i, j]
        byₖ_arr[k, i, j] = im*kᵧ*ψzₖ_arr[k, i, j]
        xxₖ_arr[k, i, j] = -im*kₓ*kₕ²*ψavgₖ_arr[k, i, j]
        xyₖ_arr[k, i, j] = -im*kᵧ*kₕ²*ψavgₖ_arr[k, i, j]
    end

    # To real space (FFTs in xy-pencil) - use helper for correct pencil allocation
    bxᵣ = _allocate_fft_dst(bxₖ, plans); byᵣ = _allocate_fft_dst(byₖ, plans)
    xxᵣ = _allocate_fft_dst(xxₖ, plans); xyᵣ = _allocate_fft_dst(xyₖ, plans)
    fft_backward!(bxᵣ, bxₖ, plans)
    fft_backward!(byᵣ, byₖ, plans)
    fft_backward!(xxᵣ, xxₖ, plans)
    fft_backward!(xyᵣ, xyₖ, plans)

    bxᵣ_arr = parent(bxᵣ); byᵣ_arr = parent(byᵣ)
    xxᵣ_arr = parent(xxᵣ); xyᵣ_arr = parent(xyᵣ)

    # Real-space RHS - use physical array dimensions (may differ from spectral in 2D decomposition)
    rhsᵣ = similar(bxᵣ)
    rhsᵣ_arr = parent(rhsᵣ)
    nz_phys, nx_phys, ny_phys = size(bxᵣ_arr)
    @inbounds for k in 1:nz_phys, j in 1:ny_phys, i in 1:nx_phys
        rhsᵣ_arr[k, i, j] = 2.0 * ( real(bxᵣ_arr[k, i, j])*real(xyᵣ_arr[k, i, j]) - real(byᵣ_arr[k, i, j])*real(xxᵣ_arr[k, i, j]) )
    end

    # Back to spectral
    fft_forward!(rhs, rhsᵣ, plans)

    # Apply dealiasing mask only (no normalization needed)
    # fft_backward! uses normalized IFFT, so pseudo-spectral products are correctly scaled.
    # Previous code incorrectly divided by nx*ny, weakening omega forcing.
    rhs_arr = parent(rhs)
    @inbounds for k in 1:nz_local_xy, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, rhs)
        j_global = local_to_global(j, 3, rhs)
        if !L[i_global, j_global]
            rhs_arr[k, i, j] = 0  # Dealias
        end
    end
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
- For physically accurate energy with dealiasing and density weighting,
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
    flow_kinetic_energy_spectral(uk, vk, G, par; Lmask=nothing) -> KE

Compute kinetic energy in spectral space with dealiasing and density weighting.

# Physical Background (matches Fortran diag_zentrum/energy_linear)
The kinetic energy is computed as:

    KE(z) = Σₖ L(kₓ,kᵧ) × (|uₖ|² + |vₖ|²) - 0.5 × (|u₀₀|² + |v₀₀|²)

The dealiasing correction subtracts half the kh=0 mode because:
- With 2/3 dealiasing: Σₖ (1/2)|u|² = Σₖ L|u|² - 0.5|u(0,0)|²

The total KE integrates over z with density weighting:

    KE_total = (1/nz) Σᵢ ρₛ(zᵢ) × KE(zᵢ)

# Algorithm
1. Loop over all spectral modes (kₓ, kᵧ, z) with dealiasing mask L
2. Accumulate |u|² + |v|² at each level
3. Apply dealiasing correction: subtract half the kh=0 mode
4. Weight by density ρₛ(z) and integrate (divide by nz)

# Arguments
- `uk, vk`: Spectral velocity fields (complex)
- `G::Grid`: Grid structure
- `par`: QGParams (for density profiles)
- `Lmask`: Optional dealiasing mask (default: all modes included)

# Returns
Total kinetic energy, normalized by nz, with density weighting.

# Fortran Correspondence
Matches the kinetic energy computation in `diag_zentrum` (diagnostics.f90:127-161)
and `energy_linear` (diagnostics.f90:3024-3107).

# Note
In MPI mode, returns LOCAL energy. Use mpi_reduce_sum for global total.
"""
function flow_kinetic_energy_spectral(uk, vk, G::Grid, par; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx, ny) : Lmask

    # Get local dimensions
    uk_arr = parent(uk)
    vk_arr = parent(vk)
    nz_local, nx_local, ny_local = size(uk_arr)

    # Get density profile for weighting (ρₛ at staggered points)
    ρₛ = if isdefined(PARENT, :rho_st) && par !== nothing
        PARENT.rho_st(par, G)
    else
        ones(Float64, nz)
    end

    KE_total = 0.0

    @inbounds for k in 1:nz_local
        # Use global z-index for correct profile lookup in 2D decomposition
        k_global = local_to_global(k, 1, uk)
        ρₛₖ = k_global <= length(ρₛ) ? ρₛ[k_global] : 1.0

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

        # Weight by density and accumulate
        KE_total += ρₛₖ * ke_k
    end

    # Normalize by nz (vertical integration)
    KE = KE_total / nz

    return KE
end

"""
    flow_potential_energy_spectral(bk, G, par; Lmask=nothing) -> PE

Compute potential energy in spectral space with dealiasing and density weighting.

# Physical Background
The potential energy from buoyancy variance:

    PE(z) = Σₖ L(kₓ,kᵧ) × (a_ell × ρ₁/ρ₂) × |bₖ|² - 0.5 × correction

where a_ell = f²/N² is the elliptic coefficient.

For QG: b = ψ_z, so PE represents available potential energy from isopycnal tilting.

# Arguments
- `bk`: Spectral buoyancy field (complex)
- `G::Grid`: Grid structure
- `par`: QGParams (for f0, N2 and density profiles)
- `Lmask`: Optional dealiasing mask

# Returns
Total potential energy, normalized by nz, with density weighting.

# Fortran Correspondence
Matches the potential energy computation in `diag_zentrum` (ps term).
"""
function flow_potential_energy_spectral(bk, G::Grid, par; Lmask=nothing)
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

    # Get density profiles
    ρ₁ = if isdefined(PARENT, :rho_ut) && par !== nothing
        PARENT.rho_ut(par, G)
    else
        ones(Float64, nz)
    end

    ρ₂ = if isdefined(PARENT, :rho_st) && par !== nothing
        PARENT.rho_st(par, G)
    else
        ones(Float64, nz)
    end

    ρₛ = if isdefined(PARENT, :rho_st) && par !== nothing
        PARENT.rho_st(par, G)
    else
        ones(Float64, nz)
    end

    PE_total = 0.0

    @inbounds for k in 1:nz_local
        # Use global z-index for correct profile lookup in 2D decomposition
        k_global = local_to_global(k, 1, bk)
        a_ell_k = k_global <= length(a_ell) ? a_ell[k_global] : a_ell[end]
        ρ₁ₖ = k_global <= length(ρ₁) ? ρ₁[k_global] : 1.0
        ρ₂ₖ = k_global <= length(ρ₂) ? ρ₂[k_global] : 1.0
        ρₛₖ = k_global <= length(ρₛ) ? ρₛ[k_global] : 1.0

        pe_k = 0.0

        for j in 1:ny_local, i in 1:nx_local
            i_global = local_to_global(i, 2, bk)
            j_global = local_to_global(j, 3, bk)

            if L[i_global, j_global]
                # PE contribution: (a_ell(z) × ρ₁/ρ₂) × |b|²
                pe_k += (a_ell_k * ρ₁ₖ / ρ₂ₖ) * abs2(bk_arr[k, i, j])
            end
        end

        # Dealiasing correction
        if local_to_global(1, 2, bk) == 1 && local_to_global(1, 3, bk) == 1
            pe_k -= 0.5 * (a_ell_k * ρ₁ₖ / ρ₂ₖ) * abs2(bk_arr[k, 1, 1])
        end

        # Weight by density and accumulate
        PE_total += ρₛₖ * pe_k
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
function wave_energy_vavg(B, G::Grid, plans)
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
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `k::Int`: LOCAL vertical index for slice (1 ≤ k ≤ nz_local)

# Returns
2D real array (nx_local, ny_local) with values at local z[k].

# Note
In MPI mode with 2D decomposition, k is a LOCAL index.
For full domain slices, gather data to root first.
"""
function slice_horizontal(field, G::Grid, plans; k::Int)
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
- `G::Grid`: Grid structure
- `plans`: FFT plans
- `j::Int`: LOCAL Y-index for slice (1 ≤ j ≤ ny_local)

# Returns
2D real array (nx_local, nz_local) with values at local y[j].

# Note
In MPI mode with 2D decomposition, j is a LOCAL index.
For full domain slices, gather data to root first.
"""
function slice_vertical_xz(field, G::Grid, plans; j::Int)
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
- For physically accurate wave energies with dealiasing and density weighting,
  use `wave_energy_spectral` instead.
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

"""
    wave_energy_spectral(BR, BI, AR, AI, CR, CI, G, par; Lmask=nothing) -> (WKE, WPE, WCE)

Compute physically accurate wave energies in spectral space with dealiasing.

# Physical Background (matches YBJ+ paper)
Three components of wave energy:

1. **Wave Kinetic Energy (WKE)** - per YBJ+ equation (4.7):
   WKE = (1/2) × Σₖ |LAₖ|²

   where LA is computed directly using the L operator from equation (1.3):
   L = ∂_z(f²/N² × ∂_z)

   So LA = ∂_z(a(z) × C) where a(z) = f²/N² and C = ∂A/∂z.
   This is discretized as: LA[k] = (a[k]×C[k] - a[k-1]×C[k-1]) / Δz

2. **Wave Potential Energy (WPE)**:
   WPE = Σₖ (0.5/(ρ₂×a_ell)) × kh² × (|CRₖ|² + |CIₖ|²)

   where C = ∂A/∂z and a_ell = f²/N². This represents the potential energy from vertical wave structure.

3. **Wave Correction Energy (WCE)**:
   WCE = Σₖ (1/8) × (1/a_ell²) × kh⁴ × (|ARₖ|² + |AIₖ|²)

   Higher-order correction term from the YBJ+ formulation.

# Algorithm
1. Loop over all spectral modes with dealiasing mask L
2. Compute LA = ∂_z(a×C) using finite differences for each (i,j) mode
3. Accumulate |LA|², kh²|C|²/(ρ₂×a_ell), kh⁴|A|²/(8×a_ell²)
4. Apply dealiasing correction: subtract half the kh=0 mode from WKE
5. Integrate over z (sum local, divide by nz)

# Arguments
- `BR, BI`: Real and imaginary parts of wave envelope B (spectral)
- `AR, AI`: Real and imaginary parts of wave amplitude A (spectral)
- `CR, CI`: Real and imaginary parts of C = ∂A/∂z (spectral)
- `G::Grid`: Grid structure
- `par`: QGParams (for f0, N2)
- `Lmask`: Optional dealiasing mask

# Returns
Tuple (WKE, WPE, WCE) of wave energy components, normalized by nz.

# Fortran Correspondence
Matches `wave_energy` subroutine in diagnostics.f90 (lines 647-743).

# Note
In MPI mode, returns LOCAL energy. Use mpi_reduce_sum for global totals.
"""
function wave_energy_spectral(BR, BI, AR, AI, CR, CI, G::Grid, par; Lmask=nothing)
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
    BR_arr = parent(BR)
    BI_arr = parent(BI)
    AR_arr = parent(AR)
    AI_arr = parent(AI)
    CR_arr = parent(CR)
    CI_arr = parent(CI)

    nz_local, nx_local, ny_local = size(BR_arr)

    # Get density profile if available (for variable stratification)
    # r_2 corresponds to rho at staggered points for potential energy
    ρ₂ = if isdefined(PARENT, :rho_st)
        PARENT.rho_st(par, G)
    else
        ones(Float64, nz)
    end

    # Grid spacing for vertical derivative
    Δz = nz > 1 ? abs(G.z[2] - G.z[1]) : 1.0

    # Accumulate energy
    WKE_local = 0.0
    WPE_local = 0.0
    WCE_local = 0.0

    # For WKE, we need to compute LA = ∂_z(a(z) × C) using equation (1.3) of YBJ+
    # where L = ∂_z(f²/N² × ∂_z) and C = ∂A/∂z
    # Discretization: LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
    # This requires looping over (i,j) modes first, then computing LA across z levels

    @inbounds for j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 2, BR)
        j_global = local_to_global(j, 3, BR)

        if !L[i_global, j_global]
            continue
        end

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        # Compute LA = ∂_z(a × C) for this (i,j) mode across all z levels
        # Using the L operator from equation (1.3): L = ∂_z(f²/N² × ∂_z)
        for k in 1:nz_local
            k_global = local_to_global(k, 1, BR)
            a_ell_k = k_global <= length(a_ell) ? a_ell[k_global] : a_ell[end]
            ρ₂ₖ = k_global <= length(ρ₂) ? ρ₂[k_global] : 1.0

            # Compute LA = ∂_z(a × C) using finite differences
            # LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
            # C is defined at staggered points: C[k] represents derivative between k and k+1
            if nz == 1
                # Single layer: LA = 0 (no vertical structure)
                LA_r = 0.0
                LA_i = 0.0
            elseif k_global == 1
                # Bottom boundary: use one-sided difference
                # LA[1] = a[1] × C[1] / Δz (assuming a[0]×C[0] = 0 from BC)
                LA_r = a_ell_k * CR_arr[k, i, j] / Δz
                LA_i = a_ell_k * CI_arr[k, i, j] / Δz
            elseif k_global == nz
                # Top boundary: use one-sided difference
                # LA[nz] = -a[nz-1] × C[nz-1] / Δz (since C[nz] = 0 from BC)
                a_ell_km1 = (k_global - 1) <= length(a_ell) ? a_ell[k_global - 1] : a_ell[end]
                LA_r = -a_ell_km1 * CR_arr[k-1, i, j] / Δz
                LA_i = -a_ell_km1 * CI_arr[k-1, i, j] / Δz
            else
                # Interior: centered difference
                # LA[k] = (a[k] × C[k] - a[k-1] × C[k-1]) / Δz
                a_ell_km1 = (k_global - 1) <= length(a_ell) ? a_ell[k_global - 1] : a_ell[end]
                LA_r = (a_ell_k * CR_arr[k, i, j] - a_ell_km1 * CR_arr[k-1, i, j]) / Δz
                LA_i = (a_ell_k * CI_arr[k, i, j] - a_ell_km1 * CI_arr[k-1, i, j]) / Δz
            end

            # WKE per equation (4.7): (1/2)|LA|²
            WKE_local += 0.5 * (abs2(LA_r) + abs2(LA_i))

            # WPE: (0.5/(ρ₂×a_ell(z))) × kh² × (|CR|² + |CI|²)
            WPE_local += (0.5 / (ρ₂ₖ * a_ell_k)) * kₕ² * (abs2(CR_arr[k, i, j]) + abs2(CI_arr[k, i, j]))

            # WCE: (1/8) × (1/a_ell(z)²) × kh⁴ × (|AR|² + |AI|²)
            WCE_local += (1.0/8.0) * (1.0/(a_ell_k*a_ell_k)) * kₕ²*kₕ² * (abs2(AR_arr[k, i, j]) + abs2(AI_arr[k, i, j]))
        end
    end

    # Dealiasing correction for WKE: subtract half the kh=0 mode contribution
    # At kh=0, we still need to compute LA using the L operator
    if local_to_global(1, 2, BR) == 1 && local_to_global(1, 3, BR) == 1
        wke_zero = 0.0
        for k in 1:nz_local
            k_global = local_to_global(k, 1, BR)
            a_ell_k = k_global <= length(a_ell) ? a_ell[k_global] : a_ell[end]
            if nz == 1
                LA_r = 0.0
                LA_i = 0.0
            elseif k_global == 1
                LA_r = a_ell_k * CR_arr[k, 1, 1] / Δz
                LA_i = a_ell_k * CI_arr[k, 1, 1] / Δz
            elseif k_global == nz
                a_ell_km1 = (k_global - 1) <= length(a_ell) ? a_ell[k_global - 1] : a_ell[end]
                LA_r = -a_ell_km1 * CR_arr[k-1, 1, 1] / Δz
                LA_i = -a_ell_km1 * CI_arr[k-1, 1, 1] / Δz
            else
                a_ell_km1 = (k_global - 1) <= length(a_ell) ? a_ell[k_global - 1] : a_ell[end]
                LA_r = (a_ell_k * CR_arr[k, 1, 1] - a_ell_km1 * CR_arr[k-1, 1, 1]) / Δz
                LA_i = (a_ell_k * CI_arr[k, 1, 1] - a_ell_km1 * CI_arr[k-1, 1, 1]) / Δz
            end
            wke_zero += 0.5 * (abs2(LA_r) + abs2(LA_i))
        end
        WKE_local -= 0.5 * wke_zero / nz
    end

    # Normalize by nz (vertical integration)
    WKE = WKE_local / nz
    WPE = WPE_local / nz
    WCE = WCE_local / nz

    return WKE, WPE, WCE
end

#=
================================================================================
                    GLOBAL ENERGY DIAGNOSTICS (MPI-AWARE)
================================================================================
These functions compute global energy by reducing across all MPI processes.
In serial mode, they return the same result as the local versions.
================================================================================
=#

"""
    flow_kinetic_energy_global(u, v, mpi_config=nothing) -> KE

Compute GLOBAL domain-integrated kinetic energy across all MPI processes.

# Arguments
- `u, v`: Velocity arrays (local portion in MPI mode)
- `mpi_config`: MPI configuration (nothing for serial mode)

# Returns
Global kinetic energy (sum across all processes).

# Example
```julia
# Serial mode
KE = flow_kinetic_energy_global(state.u, state.v)

# MPI mode
KE = flow_kinetic_energy_global(state.u, state.v, mpi_config)
```
"""
function flow_kinetic_energy_global(u, v, mpi_config=nothing)
    # Compute local energy
    KE_local = flow_kinetic_energy(u, v)

    # Reduce across processes if MPI is active
    if mpi_config === nothing
        return KE_local
    else
        # Use the MPI reduce function from the main module
        return PARENT.mpi_reduce_sum(KE_local, mpi_config)
    end
end

"""
    wave_energy_global(B, A, mpi_config=nothing) -> (E_B, E_A)

Compute GLOBAL wave energy across all MPI processes.

# Arguments
- `B, A`: Wave envelope and amplitude arrays (local portion in MPI mode)
- `mpi_config`: MPI configuration (nothing for serial mode)

# Returns
Tuple (E_B, E_A) of global summed squared magnitudes.

# Example
```julia
# Serial mode
EB, EA = wave_energy_global(state.B, state.A)

# MPI mode
EB, EA = wave_energy_global(state.B, state.A, mpi_config)
```
"""
function wave_energy_global(B, A, mpi_config=nothing)
    # Compute local energy
    EB_local, EA_local = wave_energy(B, A)

    # Reduce across processes if MPI is active
    if mpi_config === nothing
        return EB_local, EA_local
    else
        # Use the MPI reduce function from the main module
        EB_global = PARENT.mpi_reduce_sum(EB_local, mpi_config)
        EA_global = PARENT.mpi_reduce_sum(EA_local, mpi_config)
        return EB_global, EA_global
    end
end

"""
    flow_kinetic_energy_spectral_global(uk, vk, G, par; Lmask=nothing, mpi_config=nothing) -> KE

Compute GLOBAL kinetic energy in spectral space across all MPI processes.

# Arguments
- `uk, vk`: Spectral velocity fields (local portion in MPI mode)
- `G::Grid`: Grid structure
- `par`: QGParams
- `Lmask`: Optional dealiasing mask
- `mpi_config`: MPI configuration (nothing for serial mode)

# Returns
Global kinetic energy with dealiasing and density weighting.
"""
function flow_kinetic_energy_spectral_global(uk, vk, G::Grid, par; Lmask=nothing, mpi_config=nothing)
    KE_local = flow_kinetic_energy_spectral(uk, vk, G, par; Lmask=Lmask)

    if mpi_config === nothing
        return KE_local
    else
        return PARENT.mpi_reduce_sum(KE_local, mpi_config)
    end
end

"""
    flow_potential_energy_spectral_global(bk, G, par; Lmask=nothing, mpi_config=nothing) -> PE

Compute GLOBAL potential energy in spectral space across all MPI processes.

# Arguments
- `bk`: Spectral buoyancy field (local portion in MPI mode)
- `G::Grid`: Grid structure
- `par`: QGParams
- `Lmask`: Optional dealiasing mask
- `mpi_config`: MPI configuration (nothing for serial mode)

# Returns
Global potential energy with dealiasing and density weighting.
"""
function flow_potential_energy_spectral_global(bk, G::Grid, par; Lmask=nothing, mpi_config=nothing)
    PE_local = flow_potential_energy_spectral(bk, G, par; Lmask=Lmask)

    if mpi_config === nothing
        return PE_local
    else
        return PARENT.mpi_reduce_sum(PE_local, mpi_config)
    end
end

"""
    wave_energy_spectral_global(BR, BI, AR, AI, CR, CI, G, par; Lmask=nothing, mpi_config=nothing) -> (WKE, WPE, WCE)

Compute GLOBAL wave energies in spectral space across all MPI processes.

# Arguments
- `BR, BI, AR, AI, CR, CI`: Spectral wave fields (local portions in MPI mode)
- `G::Grid`: Grid structure
- `par`: QGParams
- `Lmask`: Optional dealiasing mask
- `mpi_config`: MPI configuration (nothing for serial mode)

# Returns
Tuple (WKE, WPE, WCE) of global wave energy components.
"""
function wave_energy_spectral_global(BR, BI, AR, AI, CR, CI, G::Grid, par; Lmask=nothing, mpi_config=nothing)
    WKE_local, WPE_local, WCE_local = wave_energy_spectral(BR, BI, AR, AI, CR, CI, G, par; Lmask=Lmask)

    if mpi_config === nothing
        return WKE_local, WPE_local, WCE_local
    else
        WKE_global = PARENT.mpi_reduce_sum(WKE_local, mpi_config)
        WPE_global = PARENT.mpi_reduce_sum(WPE_local, mpi_config)
        WCE_global = PARENT.mpi_reduce_sum(WCE_local, mpi_config)
        return WKE_global, WPE_global, WCE_global
    end
end

end # module

# Export basic diagnostics
using .Diagnostics: omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg
using .Diagnostics: slice_horizontal, slice_vertical_xz

# Export spectral energy diagnostics (Fortran-compatible)
using .Diagnostics: flow_kinetic_energy_spectral, flow_potential_energy_spectral, wave_energy_spectral

# Export MPI-aware global energy functions
using .Diagnostics: flow_kinetic_energy_global, wave_energy_global
using .Diagnostics: flow_kinetic_energy_spectral_global, flow_potential_energy_spectral_global, wave_energy_spectral_global
