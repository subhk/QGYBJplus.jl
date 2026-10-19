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

    N² ∇²w + f² ∂²w/∂z² = 2∇·Q
                        = 2f [J(ψ_z, ∇²ψ) - J(ψ_x, ψ_xz) - J(ψ_y, ψ_yz)]

It combines differential vorticity advection with the Laplacian of buoyancy
advection, and is strong near fronts and eddy boundaries. The discrete form
makes w satisfy the model's own vorticity and buoyancy equations exactly.

FIELD SLICING:
--------------
Utility functions for extracting 2D slices from 3D spectral fields:
- slice_horizontal: x-y plane at fixed z (good for surface fields)
- slice_vertical_xz: x-z plane at fixed y (good for vertical structure)

The omega RHS and the domain sums run on the fields' architecture. Slices and
the per-mode spectral energy sums return host arrays and are evaluated on a
host copy of their input.

FORTRAN CORRESPONDENCE:
-----------------------
- omega_eqn_rhs! → omega_eqn_rhs in diagnostics.f90, which keeps only the
  2J(ψ_z, ∇²ψ) part of the forcing (Trenberth's approximation)
- wave_energy → energy diagnostics in diagnostics.f90
- flow_kinetic_energy → ke_flow in diagnostics.f90

================================================================================
=#

module Diagnostics

using KernelAbstractions: @kernel, @index, @Const, get_backend
using ..QGYBJplus: RuntimeGeometry
using ..QGYBJplus: fft_forward!, fft_backward!
using ..QGYBJplus: allocate_fft_backward_dst
using ..QGYBJplus: with_scratch, scratch_like, scratch_physical, scratch_phys_like
using ..QGYBJplus: launch!, kernel_mask, keep_mode, dealias!, on_host, _check_mask_size,
                   KeepAllModes, vertical_spacing

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

Forcing of the QG omega equation, divided by `f`, at the interior faces.

# Physical Background
With geostrophic velocity `(u, v) = (-ψ_y, ψ_x)` and buoyancy `b = f ψ_z`, the
QG vorticity and buoyancy equations

    ζ_t + J(ψ, ζ) = f w_z,        b_t + J(ψ, b) + N² w = 0

combine into the omega equation

    N² ∇²w + f² w_zz = 2∇·Q = f [∂_z J(ψ, ζ) - ∇²J(ψ, ψ_z)]
                            = 2f [J(ψ_z, ζ) - J(ψ_x, ψ_xz) - J(ψ_y, ψ_yz)].

This function returns `2∇·Q / f`; the solver applies the `f/N²` scaling. The
deformation terms `J(ψ_x, ψ_xz) + J(ψ_y, ψ_yz)` are part of the exact forcing
(dropping them gives Trenberth's approximation `2f J(ψ_z, ζ)`).

# Discretization
`w` lives on the faces between levels, where the model's buoyancy
`b = f (ψ[k+1] - ψ[k]) / Δz` lives. Using `ψ[k] + ψ[k+1]` over two on a face,
`J(ψ_face, b) = (f/Δz) J(ψ[k], ψ[k+1])`, and the forcing at the face above
level `k`, stored at index `k`, is

    rhs[k] = (J(ψ[k+1], ζ[k+1]) - J(ψ[k], ζ[k]) + kₕ² Ĵ(ψ[k], ψ[k+1])) / Δz

with every Jacobian projected by `Lmask`, and `rhs[nz] = 0` at the top face.
With this forcing the omega-equation `w` satisfies the model's discrete
vorticity and buoyancy equations exactly, for the PV inversion's flux-form
stencil and any N²(z).

# Arguments
- `rhs`: Output spectral forcing `(nz, nx, ny)`, modified in place
- `psi`: Spectral streamfunction
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `Lmask`: Dealiasing mask applied to ψ before and to the products after
  multiplication (`nothing` keeps every mode)
- `workspace`: Optional model workspace for scratch buffers
- `accumulate`: Add the forcing to `rhs` instead of overwriting it
"""
function omega_eqn_rhs!(rhs, psi, G::RuntimeGeometry, plans; Lmask=nothing,
                        workspace=nothing, accumulate::Bool=false)
    with_scratch(workspace) do
        _omega_eqn_rhs!(rhs, psi, G, plans, Lmask, workspace, accumulate)
    end
    return rhs
end

"""Projected `∂ₓψ`, `∂ᵧψ`, `∂ₓζ`, `∂ᵧζ` with `ζ̂ = -kₕ² ψ̂`."""
@kernel function _omega_gradient_kernel!(ψx, ψy, ζx, ζy, @Const(ψ), @Const(kx),
                                         @Const(ky), mask)
    k, i, j = @index(Global, NTuple)
    @inbounds if keep_mode(mask, i, j)
        ikₓ = im * kx[i]
        ikᵧ = im * ky[j]
        ψ̂ = ψ[k, i, j]
        ζ̂ = -(kx[i]^2 + ky[j]^2) * ψ̂
        ψx[k, i, j] = ikₓ * ψ̂
        ψy[k, i, j] = ikᵧ * ψ̂
        ζx[k, i, j] = ikₓ * ζ̂
        ζy[k, i, j] = ikᵧ * ζ̂
    else
        ψx[k, i, j] = zero(eltype(ψx))
        ψy[k, i, j] = zero(eltype(ψy))
        ζx[k, i, j] = zero(eltype(ζx))
        ζy[k, i, j] = zero(eltype(ζy))
    end
end

"""
Physical `J(ψ[k], ζ[k])` at every level, written over `ζx`, and
`J(ψ[k], ψ[k+1])` on every face, written over `ζy`. Each work item reads only
its own `ζ` entries, so the overwrite is safe.
"""
@kernel function _omega_jacobian_kernel!(ζx, ζy, @Const(ψx), @Const(ψy), nz)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        ψxₖ = real(ψx[k, i, j])
        ψyₖ = real(ψy[k, i, j])
        ζxₖ = real(ζx[k, i, j])
        ζyₖ = real(ζy[k, i, j])
        ζx[k, i, j] = ψxₖ * ζyₖ - ψyₖ * ζxₖ
        ζy[k, i, j] = k < nz ?
            ψxₖ * real(ψy[k + 1, i, j]) - ψyₖ * real(ψx[k + 1, i, j]) :
            zero(eltype(ζy))
    end
end

@kernel function _omega_forcing_kernel!(rhs, @Const(Jζ), @Const(Jψ), @Const(kx),
                                        @Const(ky), mask, Δz, nz, accumulate)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        previous = accumulate ? rhs[k, i, j] : zero(eltype(rhs))
        if k < nz && keep_mode(mask, i, j)
            kₕ² = kx[i]^2 + ky[j]^2
            rhs[k, i, j] = previous +
                (Jζ[k + 1, i, j] - Jζ[k, i, j] + kₕ² * Jψ[k, i, j]) / Δz
        else
            rhs[k, i, j] = previous
        end
    end
end

function _omega_eqn_rhs!(rhs, psi, G::RuntimeGeometry, plans, Lmask, workspace,
                         accumulate::Bool=false)
    nz = G.nz
    backend = get_backend(rhs)
    mask = kernel_mask(Lmask, rhs, KeepAllModes())
    Δz = vertical_spacing(G)
    @assert size(psi, 1) == nz "Vertical dimension must match the grid for omega RHS"

    # Four buffers, transformed in place; the two products then replace ζx, ζy.
    ψx = scratch_like(workspace, psi); ψy = scratch_like(workspace, psi)
    ζx = scratch_like(workspace, psi); ζy = scratch_like(workspace, psi)
    launch!(_omega_gradient_kernel!, backend, size(psi),
            ψx, ψy, ζx, ζy, psi, G.kx, G.ky, mask)
    fft_backward!(ψx, ψx, plans)
    fft_backward!(ψy, ψy, plans)
    fft_backward!(ζx, ζx, plans)
    fft_backward!(ζy, ζy, plans)

    launch!(_omega_jacobian_kernel!, backend, size(ζx), ζx, ζy, ψx, ψy, nz)

    # fft_backward! is normalized, so the products are already correctly scaled.
    fft_forward!(ζx, ζx, plans)
    fft_forward!(ζy, ζy, plans)
    launch!(_omega_forcing_kernel!, backend, size(rhs),
            rhs, ζx, ζy, G.kx, G.ky, mask, Δz, nz, accumulate)
    return rhs
end

#=
================================================================================
                    ENERGY DIAGNOSTICS
================================================================================
Energy measures for monitoring simulation health and physics.
================================================================================
=#

_real_square(x) = real(x)^2

"""Host Bool matrix for a diagnostic `Lmask` (`nothing` keeps every mode)."""
_host_mask(::Nothing, nx, ny) = trues(nx, ny)
_host_mask(mask::AbstractMatrix{Bool}, nx, ny) =
    (_check_mask_size(mask, nx, ny); on_host(mask))
_host_mask(rule, nx, ny) = [keep_mode(rule, i, j) for i in 1:nx, j in 1:ny]

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
- For complex arrays only the real part contributes.
- For energy with spectral dealiasing,
  use `flow_kinetic_energy_spectral` instead.
"""
function flow_kinetic_energy(u, v)
    return 0.5 * (sum(_real_square, u) + sum(_real_square, v))
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

# Arguments
- `uk, vk`: Spectral velocity fields (complex)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `Lmask`: Optional dealiasing mask (default: all modes included)

# Returns
Total kinetic energy per unit reference mass, normalized by nz.

# Fortran Correspondence
Matches the kinetic energy computation in `diag_zentrum` (diagnostics.f90:127-161)
and `energy_linear` (diagnostics.f90:3024-3107).
"""
function flow_kinetic_energy_spectral(uk, vk, G::RuntimeGeometry; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = _host_mask(Lmask, nx, ny)
    uk_arr = on_host(uk)
    vk_arr = on_host(vk)
    nz_local, nx_local, ny_local = size(uk_arr)

    KE_total = 0.0
    @inbounds for k in 1:nz_local
        ke_k = 0.0
        for j in 1:ny_local, i in 1:nx_local
            if L[i, j]
                ke_k += abs2(uk_arr[k, i, j]) + abs2(vk_arr[k, i, j])
            end
        end
        # Dealiasing correction: subtract half the kh=0 mode at index (1,1)
        ke_k -= 0.5 * (abs2(uk_arr[k, 1, 1]) + abs2(vk_arr[k, 1, 1]))
        KE_total += ke_k
    end

    # Normalize by nz (vertical integration)
    return KE_total / nz
end

_potential_energy_coefficients(a_ell::AbstractVector, nz) = on_host(a_ell)
_potential_energy_coefficients(par, nz) = fill(par.f₀^2 / par.N², nz)

"""
    flow_potential_energy_spectral(bk, G, a_ell; Lmask=nothing) -> PE

Compute Boussinesq potential energy in spectral space with dealiasing.

# Physical Background
The potential energy from buoyancy variance:

    PE(z) = Σₖ L(kₓ,kᵧ) × a_ell × |bₖ|² - 0.5 × correction

where a_ell = f²/N² is the elliptic coefficient.

For QG: b = ψ_z, so PE represents available potential energy from isopycnal tilting.

# Arguments
- `bk`: Spectral buoyancy field (complex)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `a_ell`: The `f²/N²` profile, or an object with `f₀` and `N²` properties for
  a constant coefficient
- `Lmask`: Optional dealiasing mask

# Returns
Total potential energy per unit reference mass, normalized by nz.

# Fortran Correspondence
Matches the potential energy computation in `diag_zentrum` (ps term).
"""
function flow_potential_energy_spectral(bk, G::RuntimeGeometry, a_ell; Lmask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = _host_mask(Lmask, nx, ny)
    a = _potential_energy_coefficients(a_ell, nz)
    bk_arr = on_host(bk)
    nz_local, nx_local, ny_local = size(bk_arr)

    PE_total = 0.0
    @inbounds for k in 1:nz_local
        a_ell_k = k <= length(a) ? a[k] : a[end]
        pe_k = 0.0
        for j in 1:ny_local, i in 1:nx_local
            if L[i, j]
                pe_k += a_ell_k * abs2(bk_arr[k, i, j])
            end
        end
        # Dealiasing correction
        pe_k -= 0.5 * a_ell_k * abs2(bk_arr[k, 1, 1])
        PE_total += pe_k
    end

    return PE_total / nz
end

@kernel function _vertical_mean_energy_kernel!(WE, @Const(B), nz)
    i, j = @index(Global, NTuple)
    total = zero(eltype(WE))
    for k in 1:nz
        @inbounds total += abs2(B[k, i, j]) / 2
    end
    @inbounds WE[i, j] = total / nz
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
Host `(nx, ny)` array of vertically-averaged wave energy density.
"""
function wave_energy_vavg(B, G::RuntimeGeometry, plans)
    Br = _allocate_fft_dst(B, plans)
    fft_backward!(Br, B, plans)
    nz, nx, ny = size(Br)
    WE = similar(Br, real(eltype(Br)), nx, ny)
    launch!(_vertical_mean_energy_kernel!, Br, (nx, ny), WE, Br, nz)
    return on_host(WE)
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
slice at vertical index k.

# Use Cases
- Top-level plots (k=nz for closest to surface)
- Deep field structure (k=1 for closest to bottom)
- Vertical structure analysis at specific depths

# Arguments
- `field::Array{Complex,3}`: Spectral field (nz, nx, ny)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `k::Int`: Vertical index for slice (1 ≤ k ≤ nz)

# Returns
Host `(nx, ny)` real array with values at z[k].
"""
function slice_horizontal(field, G::RuntimeGeometry, plans; k::Int)
    Xr = _allocate_fft_dst(field, plans)
    fft_backward!(Xr, field, plans)
    nz = size(Xr, 1)
    @assert 1 <= k <= nz "k=$k must be within range 1:$nz"
    host = on_host(Xr)
    return Float64[real(host[k, i, j]) for i in axes(host, 2), j in axes(host, 3)]
end

"""
    slice_vertical_xz(field, G, plans; j::Int) -> Array{Float64,2}

Extract a vertical (x-z) slice from a spectral 3D field at fixed y.

# Description
Transforms a spectral field to physical space and extracts the x-z
slice at y-index j.

# Use Cases
- Vertical wave structure visualization
- Eddy vertical extent analysis
- Thermocline/pycnocline interaction studies

# Arguments
- `field::Array{Complex,3}`: Spectral field (nz, nx, ny)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `j::Int`: Y-index for slice (1 ≤ j ≤ ny)

# Returns
Host `(nx, nz)` real array with values at y[j].
"""
function slice_vertical_xz(field, G::RuntimeGeometry, plans; j::Int)
    Xr = _allocate_fft_dst(field, plans)
    fft_backward!(Xr, field, plans)
    ny = size(Xr, 3)
    @assert 1 <= j <= ny "j=$j must be within range 1:$ny"
    host = on_host(Xr)
    return Float64[real(host[k, i, j]) for i in axes(host, 2), k in axes(host, 1)]
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
These are domain SUMS, not means. For energy density, divide by grid volume.
"""
wave_energy(B, A) = (sum(abs2, B), sum(abs2, A))

end # module

# Export basic diagnostics
using .Diagnostics: omega_eqn_rhs!, wave_energy, flow_kinetic_energy, wave_energy_vavg
using .Diagnostics: slice_horizontal, slice_vertical_xz

# Export spectral energy diagnostics (Fortran-compatible)
using .Diagnostics: flow_kinetic_energy_spectral, flow_potential_energy_spectral
