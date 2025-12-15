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

    ∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)

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

using ..QGYBJ: Grid
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!
using ..QGYBJ: local_to_global, get_local_dims
using ..QGYBJ: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJ: local_to_global_z, allocate_z_pencil

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

    ∇²w + (N²/f²) ∂²w/∂z² = 2 J(ψ_z, ∇²ψ)

This function computes the RHS: 2 J(ψ_z, ∇²ψ), which represents the forcing
for ageostrophic vertical motion.

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
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z)

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
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    # Get local dimensions
    psi_arr = parent(psi)
    nx_local, ny_local, nz_local = size(psi_arr)

    # Verify z is fully local
    @assert nz_local == nz "Vertical dimension must be fully local for omega RHS"

    # psi_z in spectral space (simple finite difference)
    psizk = similar(psi)
    psizk_arr = parent(psizk)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        if k == nz
            psizk_arr[i,j,k] = 0  # Neumann top
        else
            psizk_arr[i,j,k] = (psi_arr[i,j,k+1] - psi_arr[i,j,k]) / dz
        end
    end

    # Build needed spectral derivatives
    bxk = similar(psi); byk = similar(psi)
    xxk = similar(psi); xyk = similar(psi)
    bxk_arr = parent(bxk); byk_arr = parent(byk)
    xxk_arr = parent(xxk); xyk_arr = parent(xyk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)
        kx = G.kx[i_global]
        ky = G.ky[j_global]
        # Compute kh2 from global kx, ky arrays (works in both serial and parallel)
        kh2 = kx^2 + ky^2

        bxk_arr[i,j,k] = im*kx*psizk_arr[i,j,k]
        byk_arr[i,j,k] = im*ky*psizk_arr[i,j,k]
        # average psi between k and k+1; at top use psi at top
        ψavg = k < nz ? 0.5*(psi_arr[i,j,k+1] + psi_arr[i,j,k]) : psi_arr[i,j,k]
        xxk_arr[i,j,k] = -im*kx*kh2*ψavg
        xyk_arr[i,j,k] = -im*ky*kh2*ψavg
    end

    # To real space
    bxr = similar(psi); byr = similar(psi)
    xxr = similar(psi); xyr = similar(psi)
    fft_backward!(bxr, bxk, plans)
    fft_backward!(byr, byk, plans)
    fft_backward!(xxr, xxk, plans)
    fft_backward!(xyr, xyk, plans)

    bxr_arr = parent(bxr); byr_arr = parent(byr)
    xxr_arr = parent(xxr); xyr_arr = parent(xyr)

    # Real-space RHS
    rhsr = similar(psi)
    rhsr_arr = parent(rhsr)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        rhsr_arr[i,j,k] = 2.0 * ( real(bxr_arr[i,j,k])*real(xyr_arr[i,j,k]) - real(byr_arr[i,j,k])*real(xxr_arr[i,j,k]) )
    end

    # Back to spectral
    fft_forward!(rhs, rhsr, plans)

    # Normalize and dealias
    rhs_arr = parent(rhs)
    norm = nx*ny
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)
        if L[i_global, j_global]
            rhs_arr[i,j,k] /= norm
        else
            rhs_arr[i,j,k] = 0
        end
    end
end

# 2D decomposition version with transposes
function _omega_eqn_rhs_2d!(rhs, psi, G::Grid, plans, Lmask, workspace)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(Lmask) ? trues(nx,ny) : Lmask
    dz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0

    # Allocate z-pencil workspace
    psi_z = workspace !== nothing && hasfield(typeof(workspace), :psi_z) ? workspace.psi_z : allocate_z_pencil(G, ComplexF64)
    psiz_z = allocate_z_pencil(G, ComplexF64)
    psiavg_z = allocate_z_pencil(G, ComplexF64)

    # Transpose psi to z-pencil for vertical operations
    transpose_to_z_pencil!(psi_z, psi, G)

    # Compute psi_z and psi_avg in z-pencil configuration (z now fully local)
    psi_z_arr = parent(psi_z)
    psiz_z_arr = parent(psiz_z)
    psiavg_z_arr = parent(psiavg_z)

    nx_local_z, ny_local_z, nz_local = size(psi_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    @inbounds for k in 1:nz, j in 1:ny_local_z, i in 1:nx_local_z
        if k == nz
            psiz_z_arr[i,j,k] = 0  # Neumann top
            psiavg_z_arr[i,j,k] = psi_z_arr[i,j,k]
        else
            psiz_z_arr[i,j,k] = (psi_z_arr[i,j,k+1] - psi_z_arr[i,j,k]) / dz
            psiavg_z_arr[i,j,k] = 0.5*(psi_z_arr[i,j,k+1] + psi_z_arr[i,j,k])
        end
    end

    # Transpose back to xy-pencil for horizontal operations
    psizk = similar(psi)
    psiavgk = similar(psi)
    transpose_to_xy_pencil!(psizk, psiz_z, G)
    transpose_to_xy_pencil!(psiavgk, psiavg_z, G)

    # Get local dimensions for xy-pencil
    psizk_arr = parent(psizk)
    psiavgk_arr = parent(psiavgk)
    nx_local, ny_local, nz_local_xy = size(psizk_arr)

    # Build needed spectral derivatives in xy-pencil
    bxk = similar(psi); byk = similar(psi)
    xxk = similar(psi); xyk = similar(psi)
    bxk_arr = parent(bxk); byk_arr = parent(byk)
    xxk_arr = parent(xxk); xyk_arr = parent(xyk)

    @inbounds for k in 1:nz_local_xy, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)
        kx = G.kx[i_global]
        ky = G.ky[j_global]
        # Compute kh2 from global kx, ky arrays (works in both serial and parallel)
        kh2 = kx^2 + ky^2

        bxk_arr[i,j,k] = im*kx*psizk_arr[i,j,k]
        byk_arr[i,j,k] = im*ky*psizk_arr[i,j,k]
        xxk_arr[i,j,k] = -im*kx*kh2*psiavgk_arr[i,j,k]
        xyk_arr[i,j,k] = -im*ky*kh2*psiavgk_arr[i,j,k]
    end

    # To real space (FFTs in xy-pencil)
    bxr = similar(psi); byr = similar(psi)
    xxr = similar(psi); xyr = similar(psi)
    fft_backward!(bxr, bxk, plans)
    fft_backward!(byr, byk, plans)
    fft_backward!(xxr, xxk, plans)
    fft_backward!(xyr, xyk, plans)

    bxr_arr = parent(bxr); byr_arr = parent(byr)
    xxr_arr = parent(xxr); xyr_arr = parent(xyr)

    # Real-space RHS
    rhsr = similar(psi)
    rhsr_arr = parent(rhsr)
    @inbounds for k in 1:nz_local_xy, j in 1:ny_local, i in 1:nx_local
        rhsr_arr[i,j,k] = 2.0 * ( real(bxr_arr[i,j,k])*real(xyr_arr[i,j,k]) - real(byr_arr[i,j,k])*real(xxr_arr[i,j,k]) )
    end

    # Back to spectral
    fft_forward!(rhs, rhsr, plans)

    # Normalize and dealias
    rhs_arr = parent(rhs)
    norm = nx*ny
    @inbounds for k in 1:nz_local_xy, j in 1:ny_local, i in 1:nx_local
        i_global = local_to_global(i, 1, G)
        j_global = local_to_global(j, 2, G)
        if L[i_global, j_global]
            rhs_arr[i,j,k] /= norm
        else
            rhs_arr[i,j,k] = 0
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

Compute domain-integrated kinetic energy of the geostrophic flow.

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
"""
function flow_kinetic_energy(u, v)
    # Works with any array (regular or PencilArray)
    u_arr = parent(u)
    v_arr = parent(v)
    KE = 0.0
    @inbounds for i in eachindex(u_arr)
        KE += 0.5 * (u_arr[i]^2 + v_arr[i]^2)
    end
    return KE
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
1. Separate B into real/imaginary parts
2. Transform each to physical space
3. Compute 0.5(BR² + BI²) at each point
4. Average over vertical levels

# Returns
2D array (nx_local, ny_local) of vertically-averaged wave energy density.

# Note
In MPI mode with 2D decomposition, this returns LOCAL data only.
For full domain visualization, gather data to root first.
"""
function wave_energy_vavg(B, G::Grid, plans)
    nx, ny, nz = G.nx, G.ny, G.nz

    # Get local dimensions
    B_arr = parent(B)
    nx_local, ny_local, nz_local = size(B_arr)

    # Build BRk, BIk and invert to real
    BRk = similar(B); BIk = similar(B)
    BRk_arr = parent(BRk); BIk_arr = parent(BIk)

    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        BRk_arr[i,j,k] = Complex(real(B_arr[i,j,k]), 0)
        BIk_arr[i,j,k] = Complex(imag(B_arr[i,j,k]), 0)
    end

    BRr = similar(BRk); BIr = similar(BIk)
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)

    BRr_arr = parent(BRr); BIr_arr = parent(BIr)

    # Accumulate 0.5|B|^2 and average over nz
    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    WE = zeros(Float64, nx_local, ny_local)
    @inbounds for k in 1:nz_local, j in 1:ny_local, i in 1:nx_local
        WE[i,j] += 0.5*(real(BRr_arr[i,j,k])^2 + real(BIr_arr[i,j,k])^2)
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
- Surface vorticity plots (k=nz for surface)
- Deep field structure (k=1 for bottom)
- Vertical structure analysis at specific depths

# Arguments
- `field::Array{Complex,3}`: Spectral field (nx, ny, nz)
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

    # Get local dimensions
    field_arr = parent(field)
    nx_local, ny_local, nz_local = size(field_arr)

    @assert 1 <= k <= nz_local "k=$k must be within local range 1:$nz_local"

    # Inverse FFT entire field to get real slice
    Xr = similar(field)
    fft_backward!(Xr, field, plans)
    Xr_arr = parent(Xr)

    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    sl = Array{Float64}(undef, nx_local, ny_local)
    @inbounds for j in 1:ny_local, i in 1:nx_local
        sl[i,j] = real(Xr_arr[i,j,k])
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
- `field::Array{Complex,3}`: Spectral field (nx, ny, nz)
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

    # Get local dimensions
    field_arr = parent(field)
    nx_local, ny_local, nz_local = size(field_arr)

    @assert 1 <= j <= ny_local "j=$j must be within local range 1:$ny_local"

    Xr = similar(field)
    fft_backward!(Xr, field, plans)
    Xr_arr = parent(Xr)

    # Note: fft_backward! uses normalized IFFT (FFTW.ifft / PencilFFTs ldiv!)
    # so no additional normalization is needed
    sl = Array{Float64}(undef, nx_local, nz_local)
    @inbounds for k in 1:nz_local, i in 1:nx_local
        sl[i,k] = real(Xr_arr[i,j,k])
    end
    return sl
end

"""
    wave_energy(B, A) -> (E_B, E_A)

Compute domain-integrated wave energy from both B and A fields.

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

# Reference to parent module for MPI functions
const PARENT = Base.parentmodule(@__MODULE__)

end # module

using .Diagnostics: omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg, slice_horizontal, slice_vertical_xz
using .Diagnostics: flow_kinetic_energy_global, wave_energy_global
