#=
================================================================================
                    operators.jl - Velocity Field Computation
================================================================================

This file computes diagnostic velocity fields from the model's prognostic
variables (streamfunction ψ and wave amplitude A).

PHYSICAL BACKGROUND:
--------------------
In QG-YBJ+ dynamics, velocities arise from two sources:

1. GEOSTROPHIC FLOW (from streamfunction ψ):
   - Horizontal: u = -∂ψ/∂y, v = ∂ψ/∂x (geostrophic balance)
   - Vertical: w from the QG omega equation (ageostrophic correction)

2. WAVE-INDUCED FLOW (from wave amplitude A):
   - Horizontal: Stokes drift from wave envelope gradients
   - Vertical: YBJ formulation involving ∂A_z/∂x, ∂A_z/∂y

GEOSTROPHIC VELOCITIES:
-----------------------
The horizontal geostrophic velocities follow from:

    f u = -∂p'/∂y = -f ∂ψ/∂y  →  u = -∂ψ/∂y
    f v =  ∂p'/∂x =  f ∂ψ/∂x  →  v =  ∂ψ/∂x

In spectral space:
    û(k) = -i kᵧ ψ̂(k)
    v̂(k) =  i kₓ ψ̂(k)

QG OMEGA EQUATION:
------------------
The ageostrophic vertical velocity w comes from the omega equation:

    N² ∇²w + f² ∂²w/∂z² = 2∇·Q = 2f [J(ψ_z, ∇²ψ) - J(ψ_x, ψ_xz) - J(ψ_y, ψ_yz)]

This is a 3D elliptic PDE solved via:
- Horizontal: spectral differentiation (kₓ², kᵧ²)
- Vertical: tridiagonal solver at each (kₓ, kᵧ)

w is solved on the faces between levels, where the model's buoyancy f ψ_z
lives, with w = 0 on the bottom and top faces. The discrete forcing makes it
satisfy the model's own vorticity and buoyancy equations exactly.

WHERE w IS STORED:
------------------
Both vertical velocities (omega equation and YBJ) are computed on the faces
and stored at cell centres, as the average of the two faces of each cell, so
that u, v, and w share one grid for output and particle interpolation.

Physical meaning: w arises from ageostrophic convergence/divergence required
to maintain geostrophic balance as the flow evolves.

YBJ VERTICAL VELOCITY:
----------------------
For wave-induced vertical motion, following Asselin & Young (2019) equation (2.10):

    w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.

where:
- A is the wave amplitude (recovered from B via L⁺A = B)
- A_{zs} = ∂_s(A_z) = (1/2)(∂_x - i∂_y)(A_z) is the complex horizontal derivative
- The complex coordinate s = x + iy, so ∂_s = (1/2)(∂_x - i∂_y)
- e^{-ift} is the inertial oscillation factor
- c.c. denotes complex conjugate (ensures real result)

Expanding this gives the oscillating vertical velocity:
    w = -(f²/N²) * [cos(ft)·w_cos + sin(ft)·w_sin]
where:
    w_cos = Re(∂A_z/∂x) + Im(∂A_z/∂y)
    w_sin = Im(∂A_z/∂x) - Re(∂A_z/∂y)

This represents vertical motion induced by wave envelope modulation, oscillating
at the inertial frequency f.

WAVE-INDUCED STOKES DRIFT:
--------------------------
Following Wagner & Young (2016) equations (3.17)-(3.18), the Stokes drift
from near-inertial waves is computed from the wave velocity LA (not amplitude A):

    u_S = (1/f₀) Im[(LA)* ∂(LA)/∂x] = (1/f₀) |LA|² ∂φ/∂x
    v_S = (1/f₀) Im[(LA)* ∂(LA)/∂y] = (1/f₀) |LA|² ∂φ/∂y
    w_S = (1/f₀) Im[(LA)* ∂(LA)/∂z] = (1/f₀) |LA|² ∂φ/∂z

where LA = |LA|e^{iφ} is the wave velocity amplitude and φ is the wave phase.
This drift is in the direction of wave propagation (phase gradient).
These wave corrections are important for Lagrangian particle advection.

The wave velocity LA is computed from the YBJ+ relation: LA = B + (k_h²/4)A
where B is the evolved variable and A is the wave amplitude.

SPECTRAL DIFFERENTIATION:
-------------------------
All derivatives are computed in spectral space:
    ∂f/∂x → i kₓ f̂(k)
    ∂f/∂y → i kᵧ f̂(k)

Vertical derivatives use second-order finite differences.

Every loop below is a KernelAbstractions kernel launched on the backend of the
fields it writes.

================================================================================
=#
module Operators

using KernelAbstractions: @kernel, @index, @Const, get_backend
using ..QGYBJplus: RuntimeGeometry, ModelFields, YBJ, YBJPlus
using ..QGYBJplus: fft_backward!, plan_transforms!
using ..QGYBJplus: with_scratch, scratch_like, scratch_physical
using ..QGYBJplus: invert_B_to_A!
using ..QGYBJplus: launch!, on_backend, on_host, kernel_map!, kernel_mask,
                   vertical_spacing, tridiagonal_scratch, RadialDealiasing
import ..QGYBJplus.Elliptic: tridiagonal_row
using ..QGYBJplus.Elliptic: _thomas_solve!
const PARENT = Base.parentmodule(@__MODULE__)

"""
N² profile for the velocity diagnostics: `N2_profile` itself when it is a real
vector of length `nz` (kept in its own precision, so a model's device profile
is never copied), and otherwise a constant profile of `N2_const`.
"""
function _coerce_N2_profile(N2_profile, N2_const, nz, G::RuntimeGeometry)
    T = typeof(vertical_spacing(G))   # profiles share the grid's precision
    N2_profile === nothing && return fill(T(N2_const), nz)
    if length(N2_profile) != nz
        @warn "N2_profile length ($(length(N2_profile))) != nz ($nz), using constant N²=$(N2_const)"
        return fill(T(N2_const), nz)
    end
    eltype(N2_profile) === T && return N2_profile
    return T.(real.(N2_profile))
end

#=
================================================================================
                    GEOSTROPHIC VELOCITY COMPUTATION
================================================================================
The primary diagnostic: horizontal and vertical velocities from streamfunction.
================================================================================
=#

"""
    compute_velocities!(S, G; plans=nothing, f=1, N2=1, compute_w=true, N2_profile=nothing, N2_face_profile=nothing, workspace=nothing, dealias_mask=nothing, pv_forcing=nothing)

Compute the velocity of the balanced flow from the spectral streamfunction ψ̂.
As in Xie & Vanneste (2015) and Wagner & Young (2016, eq. 3.22), ψ is the
streamfunction of the Lagrangian-mean flow, so these are Lagrangian-mean
velocities. Wave orbital velocities and the Stokes drift that relates the
Lagrangian to the Eulerian mean are added by `compute_wave_velocities!`.

# Physical Equations
Horizontal velocities from geostrophic balance:
```
u = -∂ψ/∂y  →  û(k) = -i kᵧ ψ̂(k)
v =  ∂ψ/∂x  →  v̂(k) =  i kₓ ψ̂(k)
```

Vertical velocity from the QG omega equation (see
`compute_vertical_velocity!`):
```
N² ∇²w + f² ∂²w/∂z² = 2f [J(ψ_z, ∇²ψ) - J(ψ_x, ψ_xz) - J(ψ_y, ψ_yz)] + f ∂z(D_t qʷ)
```
where the last term, from wave feedback, is supplied through `pv_forcing`.
`w` is computed on the faces between levels and stored at cell centres.

# Algorithm
1. Compute û = -i kᵧ ψ̂ and v̂ = i kₓ ψ̂ in spectral space
2. Transform to physical space via inverse FFT
3. Optionally solve omega equation or use YBJ formula for w

# Arguments
- `S::ModelFields`: ModelFields with ψ (input) and u, v, w (output)
- `G::RuntimeGeometry`: RuntimeGeometry with wavenumbers kx, ky
- `plans`: FFT plans (auto-generated if nothing)
- `f`, `N2`: Scalar Coriolis frequency and fallback buoyancy frequency squared
- `compute_w::Bool`: If true, compute vertical velocity; otherwise `w` is zeroed
- `N2_profile::Vector`: Optional cell-center N²(z) profile, the fallback when
  no face profile is given
- `N2_face_profile::Vector`: Optional upper-face N²(z) profile, used by the
  vertical velocity (w lives on the faces)
- `workspace`: Optional model workspace for scratch buffers
- `dealias_mask`: Optional 2D dealiasing mask for omega equation RHS (quadratic term).
  Should be the same mask used for other nonlinear terms (typically 2/3 rule).
- `pv_forcing`: Optional function that writes the wave-PV forcing `D_t qʷ`
  into a spectral buffer (see `compute_vertical_velocity!`)

# Returns
Modified ModelFields with updated u, v, w fields.

# Note
For the instantaneous (Eulerian) velocity including the waves, use
`compute_total_velocities!`.

# Fortran Correspondence
Matches `compute_velo` in derivatives.f90.
"""
function compute_velocities!(S::ModelFields, G::RuntimeGeometry; plans=nothing,
    f::Real=1.0, N2::Real=1.0, compute_w=true, use_ybj_w::Bool=false,
    N2_profile=nothing, N2_face_profile=nothing,
    workspace=nothing, dealias_mask=nothing, pv_forcing=nothing)

    _reject_use_ybj_w(use_ybj_w)
    return with_scratch(workspace) do
        _compute_velocities!(S, G; plans, f, N2, compute_w,
                             N2_profile, N2_face_profile,
                             workspace, dealias_mask, pv_forcing)
    end
end

"""`use_ybj_w` mixed the wave and mean vertical velocities and was removed."""
function _reject_use_ybj_w(use_ybj_w::Bool)
    use_ybj_w && throw(ArgumentError(
        "use_ybj_w was removed: the mean-flow w is the omega-equation velocity, " *
        "and the wave vertical velocity at time t is part of " *
        "compute_total_velocities!(model; t) (or compute_ybj_vertical_velocity!)"))
    return nothing
end

"""û = -i kᵧ ψ̂"""
@kernel function _geostrophic_u_kernel!(uk, @Const(ψk), @Const(ky))
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        ikᵧ = im * ky[j]
        uk[k, i, j] = -ikᵧ * ψk[k, i, j]
    end
end

"""v̂ = i kₓ ψ̂"""
@kernel function _geostrophic_v_kernel!(vk, @Const(ψk), @Const(kx))
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        ikₓ = im * kx[i]
        vk[k, i, j] = ikₓ * ψk[k, i, j]
    end
end

function _compute_velocities!(S::ModelFields, G::RuntimeGeometry; plans=nothing,
    f::Real=1.0, N2::Real=1.0, compute_w=true,
    N2_profile=nothing, N2_face_profile=nothing,
    workspace=nothing, dealias_mask=nothing, pv_forcing=nothing)

    if plans === nothing
        plans = plan_transforms!(G)
    end

    # û = -i kᵧ ψ̂ and v̂ = i kₓ ψ̂, one after the other in one buffer that is
    # transformed in place. The normalized inverse transform gives
    # physical-space values directly. The buffer is returned before w is
    # computed.
    with_scratch(workspace) do
        buffer = scratch_like(workspace, S.psi)
        launch!(_geostrophic_u_kernel!, buffer, size(buffer), buffer, S.psi, G.ky)
        fft_backward!(buffer, buffer, plans)
        kernel_map!(real, S.u, buffer)
        launch!(_geostrophic_v_kernel!, buffer, size(buffer), buffer, S.psi, G.kx)
        fft_backward!(buffer, buffer, plans)
        kernel_map!(real, S.v, buffer)
    end

    # The vertical velocity is a diagnostic: the omega equation, or zero
    # (the leading-order Lagrangian-mean w).
    if compute_w
        compute_vertical_velocity!(S, G, plans; f, N2, N2_profile,
            N2_face_profile, workspace, dealias_mask, pv_forcing)
    else
        fill!(S.w, zero(eltype(S.w)))
    end

    return S
end

#=
================================================================================
                    QG OMEGA EQUATION SOLVER
================================================================================
Computes the ageostrophic vertical velocity from the QG omega equation.
This is a 3D elliptic problem solved via tridiagonal systems.
================================================================================
=#

"""
    compute_vertical_velocity!(S, G, plans; f=1, N2=1, N2_profile=nothing, N2_face_profile=nothing, workspace=nothing, dealias_mask=nothing)

Solve the QG omega equation for ageostrophic vertical velocity.

# Physical Background
In quasi-geostrophic dynamics, the leading-order horizontal flow is non-divergent
(∇·u_g = 0). Vertical motion arises from ageostrophic corrections that maintain
thermal wind balance as the flow evolves. Eliminating the time derivatives
between the vorticity and buoyancy equations,

```
ζ_t + J(ψ, ζ) = f w_z,        b_t + J(ψ, b) + N² w = 0,        b = f ψ_z,
```

gives the omega equation

```
N² ∇²w + f² ∂²w/∂z² = 2∇·Q = 2f [J(ψ_z, ∇²ψ) - J(ψ_x, ψ_xz) - J(ψ_y, ψ_yz)]
```

The first forcing term is differential vorticity advection by the thermal
wind; the other two are the deformation part of the Q-vector forcing.

# Numerical Method
1. Compute the forcing on the faces between levels with `omega_eqn_rhs!`
2. For each horizontal wavenumber (kₓ, kᵧ), solve the tridiagonal system for
   w on the interior faces with the Thomas algorithm (O(nz) per wavenumber;
   the system is strictly diagonally dominant for kₕ > 0, so no pivoting is
   needed). N² is sampled on the faces.
3. Average the two faces of each cell to store w at cell centres, and
   transform to physical space

The discrete forcing and stencil are those implied by the model's flux-form
PV inversion, so w satisfies the model's discrete vorticity and buoyancy
equations exactly.

# Boundary Conditions
w = 0 at z = -Lz and z = 0 (rigid lid and bottom).

# Arguments
- `S::ModelFields`: ModelFields with ψ (input) and w (output, cell centres)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `f`, `N2`: Scalar Coriolis frequency and fallback buoyancy frequency squared
- `N2_profile::Vector`: Optional cell-centre N²(z), used only when
  `N2_face_profile` is not given (default: constant N² = `N2`)
- `N2_face_profile::Vector`: Optional N²(z) on the upper face of each level,
  where the omega equation is solved
- `workspace`: Optional model workspace for scratch buffers
- `pv_forcing`: Optional function `pv_forcing(D)` that writes the material
  derivative of the wave PV following the mean flow, `D = ∂ₜqʷ + J(ψ, qʷ)`,
  into the spectral array `D` at the levels. With wave feedback the PV
  equation `D_t (q_g + qʷ) = 0` adds `-D_t qʷ` to the vorticity equation, so
  the omega equation gains the forcing `f ∂z(D_t qʷ)`.
- `dealias_mask`: 2D dealiasing mask for omega equation RHS. If `nothing` (default),
  the standard radial 2/3 rule is used to avoid aliasing in the
  quadratic Jacobian term.

# Fortran Correspondence
Follows the omega-equation solver of the Fortran implementation, with the full
Q-vector forcing and w on the faces.
"""
function compute_vertical_velocity!(S::ModelFields, G::RuntimeGeometry, plans;
    f::Real=1.0, N2::Real=1.0, N2_profile=nothing, N2_face_profile=nothing,
    workspace=nothing, dealias_mask=nothing, pv_forcing=nothing)
    # The omega equation involves quadratic Jacobians that need dealiasing; by
    # default use the radial two-thirds rule of `dealias_mask`.
    if dealias_mask === nothing
        dealias_mask = RadialDealiasing(G.nx, G.ny)
    end

    # w and its equation live on the faces, so N² is needed there.
    face_profile = _coerce_N2_profile(
        N2_face_profile === nothing ? N2_profile : N2_face_profile, N2, G.nz, G)
    with_scratch(workspace) do
        _compute_vertical_velocity!(S, G, plans, f, face_profile, workspace,
                                    dealias_mask, pv_forcing)
    end
    return S
end

function _compute_vertical_velocity!(S::ModelFields, G::RuntimeGeometry, plans,
    f, N2_face_profile, workspace, dealias_mask, pv_forcing=nothing)

    # One buffer holds the forcing on the faces (dealiased like the model's
    # nonlinear terms), then w, which the column solve writes over it. With
    # wave feedback the buffer first receives D_t qʷ at the levels, which is
    # differenced onto the faces before the balanced forcing is added.
    wk = scratch_like(workspace, S.psi)
    if pv_forcing === nothing
        PARENT.Diagnostics.omega_eqn_rhs!(wk, S.psi, G, plans;
                                          Lmask=dealias_mask, workspace=workspace)
    else
        pv_forcing(wk)
        launch!(_levels_to_face_difference_kernel!, wk, (G.nx, G.ny),
                wk, vertical_spacing(G), G.nz)
        PARENT.Diagnostics.omega_eqn_rhs!(wk, S.psi, G, plans;
                                          Lmask=dealias_mask, workspace=workspace,
                                          accumulate=true)
    end
    _omega_equation_solve!(wk, wk, G, f, N2_face_profile, workspace)

    # To physical space (fft_backward! is normalized), then onto cell centres.
    fft_backward!(wk, wk, plans)
    launch!(_faces_to_centers_kernel!, S.w, size(S.w), S.w, wk, false)
    return S
end

"""
Replace values at the levels by their vertical difference on the face above
each level, `(D[k+1] - D[k]) / Δz`, and zero on the top face. One work item
per column walks upward, so each value is read before it is overwritten.
"""
@kernel function _levels_to_face_difference_kernel!(D, Δz, nz)
    i, j = @index(Global, NTuple)
    for k in 1:(nz - 1)
        @inbounds D[k, i, j] = (D[k + 1, i, j] - D[k, i, j]) / Δz
    end
    @inbounds D[nz, i, j] = zero(eltype(D))
end

"""
`w` at cell centres from its values `w_face[k]` on the face above each level,
with `w = 0` on the bottom face. `w_face[nz]` is the top face, where `w = 0`.
With `accumulate`, the centre values are added to `w`.
"""
@kernel function _faces_to_centers_kernel!(w, @Const(w_face), accumulate)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        below = k == 1 ? zero(eltype(w)) : real(w_face[k - 1, i, j])
        centre = (below + real(w_face[k, i, j])) / 2
        w[k, i, j] = accumulate ? w[k, i, j] + centre : centre
    end
end

"""
Rows of `-kₕ²w + (f²/N²) ∂²w/∂z² = (f/N²) rhs` on the interior faces, with
∂²w/∂z² ≈ (w[r+1] - 2w[r] + w[r-1])/Δz². Row `r` is the face above level `r`,
`N²[r]` is sampled on that face, and w = 0 on the bottom and top faces.
"""
struct OmegaColumn{N, R, T}
    N²::N
    rhs::R
    i::Int
    j::Int
    f::T
    f²::T
    Δz::T
    kₕ²::T
end

@inline function tridiagonal_row(s::OmegaColumn, r)
    @inbounds begin
        coefficient = (s.f² / s.N²[r]) / (s.Δz * s.Δz)
        b = (s.f / s.N²[r]) * s.rhs[r, s.i, s.j]
    end
    return coefficient, -2 * coefficient - s.kₕ², coefficient, b
end

# `wk` may be `rhsk`: the Thomas sweep reads each row's right-hand side before
# writing that row of the solution, so `rhsk` is not marked @Const.
@kernel function _omega_solve_kernel!(wk, c, rhsk, @Const(kx), @Const(ky),
                                      @Const(N²), f, f², Δz, nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    # The horizontal mean of w vanishes, and nz = 1 has no interior face.
    if kₕ² > 0 && nz > 1
        system = OmegaColumn(N², rhsk, i, j, f, f², Δz, kₕ²)
        _thomas_solve!(wk, c, i, j, 0, nz - 1, system)
        @inbounds wk[nz, i, j] = zero(eltype(wk))
    else
        for k in 1:nz
            @inbounds wk[k, i, j] = zero(eltype(wk))
        end
    end
end

"""
Solve `∇²w + (f²/N²) ∂²w/∂z² = (f/N²) rhs` for w on the faces above levels
`1:(nz-1)` of every horizontal wavenumber; `wk[nz]` (the top face) is zero.
`omega_eqn_rhs!` returns the forcing divided by f, so the rows scale it by f/N².
"""
function _omega_equation_solve!(wk, rhsk, G::RuntimeGeometry, f, N2_profile, workspace)
    nz, nx, ny = size(wk)
    @assert nz == G.nz "Vertical dimension must match the grid for the omega solve"
    backend = get_backend(wk)
    Δz = vertical_spacing(G)
    f_value = convert(typeof(Δz), f)   # the column scalars share the grid's precision
    launch!(_omega_solve_kernel!, backend, (nx, ny),
            wk, tridiagonal_scratch(workspace, wk), rhsk, G.kx, G.ky,
            on_backend(backend, N2_profile), f_value, f_value^2, Δz, nz)
    return wk
end

#=
================================================================================
                    YBJ VERTICAL VELOCITY
================================================================================
Wave-induced vertical motion from the YBJ+ formulation.
================================================================================
=#

"""
    compute_ybj_vertical_velocity!(S, G, plans; f=1, N2=1, N2_profile=nothing, N2_face_profile=nothing, workspace=nothing, skip_inversion=false, t=nothing)

Compute vertical velocity from near-inertial wave envelope using YBJ+ formulation.

# Physical Background
Near-inertial waves induce vertical motion through the modulation of their
envelope. Following Asselin & Young (2019) equation (2.10):

```
w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.
```

where:
- A is the true wave amplitude (recovered from evolved B via L⁺A = B)
- A_{zs} = ∂_s(A_z) = (1/2)(∂_x - i∂_y)(A_z) is the complex horizontal derivative
- The complex derivative uses s = x + iy, so ∂_s = (1/2)(∂_x - i∂_y)
- c.c. denotes complex conjugate

Expanding this gives:
```
w = -(f²/N²) * [cos(ft)·(Re(∂A_z/∂x) + Im(∂A_z/∂y)) + sin(ft)·(Im(∂A_z/∂x) - Re(∂A_z/∂y))]
```

A_z and N² are evaluated on the faces between levels, so the face values of
w satisfy discrete continuity, `w_z = -(u_x + v_y)`, exactly for the wave
velocity `u + iv = e^{-ift} LA`; A_z = 0 makes w vanish on the bottom and top
faces. The face values are averaged to cell centres for storage.

# Physical Interpretation
This represents vertical motion induced by:
- Horizontal gradients in the wave envelope's vertical structure
- Wave packet propagation and refraction
- Strong w occurs where wave amplitude varies both horizontally and vertically
- The velocity oscillates at inertial frequency f

# Algorithm
1. **A Recovery**: Solve L⁺A = B using invert_B_to_A!
   - L⁺ is the YBJ+ elliptic operator
   - Tridiagonal solver in z for each horizontal wavenumber

2. **Vertical Derivative**: Compute A_z = ∂A/∂z
   - Uses second-order finite differences

3. **Horizontal Gradients**: Compute ∂(A_z)/∂x, ∂(A_z)/∂y
   - Spectral differentiation: multiply by i kₓ, i kᵧ

4. **Combine**: Apply equation (2.10) with oscillating e^{-ift} factor

# Arguments
- `S::ModelFields`: ModelFields with B (input) and w (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `plans`: FFT plans
- `f`, `N2`: Scalar Coriolis frequency and fallback buoyancy frequency squared
- `N2_profile::Vector`: Optional cell-centre N²(z), used only when
  `N2_face_profile` is not given (default: constant N² = `N2`)
- `N2_face_profile::Vector`: Optional N²(z) on the upper face of each level,
  used to recover A from B and in the factor f²/N² of w
- `workspace`: Optional model workspace for scratch buffers
- `skip_inversion::Bool`: If true, skip B→A re-inversion and use existing S.A, S.C.
  Use this when A/C were already computed in the timestep with the correct stratification.
  Default: false (re-inverts for safety).
- `t::Real`: Current time for computing the oscillating velocity e^{-ift}.
  If not provided (default), computes only the cosine component (t=0 snapshot).

# Fortran Correspondence
Matches YBJ vertical velocity computation in the Fortran implementation.

# Important Note
When `skip_inversion=false` (default), this function re-inverts B→A using the provided
N2_face_profile (or the legacy `N2_profile`/constant fallback). If the timestep computed A/C with a different
stratification, this will give inconsistent results. For runs with nonuniform N²,
either:
1. Pass `skip_inversion=true` and ensure A/C are already computed correctly
2. Pass the exact same `N2_face_profile` that was used in the timestep

# References
- Asselin & Young (2019), J. Fluid Mech. 876, 428-448, equation (2.10)
"""
function compute_ybj_vertical_velocity!(S::ModelFields, G::RuntimeGeometry, plans;
    f::Real=1.0, N2::Real=1.0, N2_profile=nothing, N2_face_profile=nothing,
    workspace=nothing, skip_inversion=false, t=nothing)
    # If skip_inversion=false and no N2 profile is provided, B→A is re-inverted
    # with constant N², which is inconsistent with variable stratification.
    if !skip_inversion && N2_face_profile === nothing && N2_profile === nothing
        @warn "compute_ybj_vertical_velocity!: Re-inverting B→A with constant N² (no N2_profile provided). " *
              "If your simulation uses variable stratification, pass N2_face_profile or use skip_inversion=true " *
              "to avoid inconsistent vertical velocity." maxlog=1
    end

    face_profile = _coerce_N2_profile(
        N2_face_profile === nothing ? N2_profile : N2_face_profile,
        N2, G.nz, G)
    with_scratch(workspace) do
        _compute_ybj_vertical_velocity!(S, G, plans, f, face_profile,
            workspace, skip_inversion, t)
    end
    return S
end

@kernel function _ybj_w_spectral_kernel!(dAz_dx, dAz_dy, @Const(Az), @Const(kx),
                                         @Const(ky), nz)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        # A_z is defined for levels 1:(nz-1). The top level is zeroed rather
        # than left as scratch garbage, which would pollute the transform.
        if k >= nz
            dAz_dx[k, i, j] = zero(eltype(dAz_dx))
            dAz_dy[k, i, j] = zero(eltype(dAz_dy))
        else
            ikₓ = im * kx[i]
            ikᵧ = im * ky[j]
            dAz_dx[k, i, j] = ikₓ * Az[k, i, j]
            dAz_dy[k, i, j] = ikᵧ * Az[k, i, j]
        end
    end
end

"""
YBJ w on the face above each level, written over the physical `∂ₓA_z` in
`dAz_dx`: A_z (index `k`) and N² both live on that face. A_z = 0 on the top
face (index `nz`), and so does w.
"""
@kernel function _ybj_w_face_kernel!(dAz_dx, @Const(dAz_dy), @Const(N²_face),
                                     f², cos_ft, sin_ft, nz)
    k, i, j = @index(Global, NTuple)
    @inbounds if k == nz
        dAz_dx[k, i, j] = zero(eltype(dAz_dx))
    else
        dx = dAz_dx[k, i, j]
        dy = dAz_dy[k, i, j]
        w_cos = real(dx) + imag(dy)
        w_sin = imag(dx) - real(dy)
        dAz_dx[k, i, j] = -(f² / N²_face[k]) * (cos_ft * w_cos + sin_ft * w_sin)
    end
end

function _compute_ybj_vertical_velocity!(S::ModelFields, G::RuntimeGeometry, plans,
    f, N2_face_profile, workspace, skip_inversion, t)
    nz = G.nz
    backend = get_backend(S.w)

    # Step 1: Recover A from B = L⁺A using the centralized YBJ+ inversion,
    # with a(z) = f²/N²(z) as the elliptic coefficient.
    if skip_inversion
        # Reuse S.A and S.C from the timestep, which used the correct
        # stratification. All-zero A is legitimate (a run with no waves).
        if all(iszero, S.A)
            @warn "skip_inversion=true but S.A is all zeros - vertical velocity will be zero" maxlog=1
        end
    else
        f_sq = f^2
        a_vec = [f_sq / value for value in on_host(N2_face_profile)]
        invert_B_to_A!(S, G, a_vec; workspace)
    end

    cos_ft = t === nothing ? 1.0 : cos(f * t)
    sin_ft = t === nothing ? 0.0 : sin(f * t)
    _ybj_w!(S.w, S, G, plans, f, N2_face_profile, workspace, cos_ft, sin_ft, false)
    return S
end

"""
YBJ vertical velocity of the wave at inertial phase (cos ft, sin ft), from the
A_z in `S.C`, written to (or, with `accumulate`, added to) the real centre
field `w`.
"""
function _ybj_w!(w, S::ModelFields, G::RuntimeGeometry, plans, f, N2_face_profile,
                 workspace, cos_ft, sin_ft, accumulate::Bool)
    nz = G.nz
    backend = get_backend(w)
    with_scratch(workspace) do
        # Horizontal derivatives of A_z in two buffers, transformed in place.
        dAz_dx = scratch_like(workspace, S.C)
        dAz_dy = scratch_like(workspace, S.C)
        launch!(_ybj_w_spectral_kernel!, backend, size(S.C),
                dAz_dx, dAz_dy, S.C, G.kx, G.ky, nz)
        fft_backward!(dAz_dx, dAz_dx, plans)
        fft_backward!(dAz_dy, dAz_dy, plans)

        # w on the faces, written over ∂ₓA_z. Asselin & Young (2019) (2.10):
        #   w₀ = -(f²/N²) A_{zs} e^{-ift} + c.c.,  A_{zs} = (1/2)[∂ₓA_z - i ∂ᵧA_z]
        #   w = -(f²/N²)[cos(ft)(Re ∂ₓA_z + Im ∂ᵧA_z) + sin(ft)(Im ∂ₓA_z - Re ∂ᵧA_z)]
        T = real(eltype(dAz_dx))   # the column scalars share the grid's precision
        launch!(_ybj_w_face_kernel!, backend, size(dAz_dx),
                dAz_dx, dAz_dy, on_backend(backend, N2_face_profile),
                T(float(f)^2), T(cos_ft), T(sin_ft), nz)
        launch!(_faces_to_centers_kernel!, backend, size(w), w, dAz_dx, accumulate)
    end
    return w
end

#=
================================================================================
                    TOTAL VELOCITY FOR LAGRANGIAN ADVECTION
================================================================================
For particle tracking, we need the complete velocity field including both
geostrophic flow and wave-induced motion.
================================================================================
=#

"""
    compute_total_velocities!(S, G; t, plans=nothing, f=1, N2=1, compute_w=true, N2_profile=nothing, N2_face_profile=nothing, workspace=nothing, dealias_mask=nothing, include_wave_velocity=true, formulation=YBJPlus(), pv_forcing=nothing)

Instantaneous (Eulerian) velocity at time `t`: the velocity a fluid particle
at a fixed point sees, including the near-inertial oscillation.

# Physical Background
The QG streamfunction ψ is that of the Lagrangian-mean flow (Xie & Vanneste
2015; Wagner & Young 2016, eq. 3.22: `ū + U^S = ∇⊥ψ`). The Eulerian-mean
velocity is therefore the Lagrangian mean minus the Stokes drift, and the
waves add their orbital velocity `u + iv = e^{-ift} LA` (Asselin & Young 2019,
eq. 1.2) and vertical velocity:
```
u + iv = (u^L + iv^L) - (u^S + iv^S) + e^{-ift} LA
w      = w^L - w^S + [-(f²/N²) A_{zs} e^{-ift} + c.c.]
```
`compute_velocities!` gives the Lagrangian mean (u^L, v^L, w^L), and
`compute_wave_velocities!` the rest.

Trajectories integrated with this velocity resolve the inertial oscillation,
so their time step must be small compared with 1/f; their drift averaged over
an inertial period is the Lagrangian mean. For phase-averaged trajectories use
the Lagrangian-mean velocity of `compute_velocities!` instead.

# Arguments
- `S::ModelFields`: ModelFields with ψ, A, B, C (input) and u, v, w (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `t`: Time for the inertial phase `e^{-ift}` (required)
- `plans`: FFT plans
- `f`, `N2`: Scalar Coriolis frequency and fallback buoyancy frequency squared
- `compute_w::Bool`: If true, compute the vertical velocity
- `N2_profile`, `N2_face_profile`: Optional cell-centre and upper-face N²(z)
- `workspace`: Optional model workspace for scratch buffers
- `dealias_mask`: Optional 2D dealiasing mask for the omega equation
- `include_wave_velocity::Bool`: If false, omit the orbital velocity, leaving
  the Eulerian-mean velocity
- `formulation`: `YBJPlus()` (default) or `YBJ()`, which fixes how LA follows from B
- `pv_forcing`: Wave-PV forcing of the mean-flow w in coupled runs (see
  `compute_vertical_velocity!`)
"""
function compute_total_velocities!(S::ModelFields, G::RuntimeGeometry; t=nothing,
    plans=nothing, f::Real=1.0, N2::Real=1.0, compute_w=true, use_ybj_w::Bool=false,
    N2_profile=nothing, N2_face_profile=nothing,
    workspace=nothing, dealias_mask=nothing,
    include_wave_velocity=true, formulation=YBJPlus(), pv_forcing=nothing)
    _reject_use_ybj_w(use_ybj_w)
    _require_wave_time(t, include_wave_velocity)
    compute_velocities!(S, G; plans, f, N2, compute_w, N2_profile,
        N2_face_profile, workspace, dealias_mask, pv_forcing)
    compute_wave_velocities!(S, G; t, plans, f, N2, compute_w,
        include_wave_velocity, N2_profile, N2_face_profile, formulation, workspace)
    return S
end

"""The wave orbital velocity depends on its phase, so it needs the time."""
function _require_wave_time(t, include_wave_velocity)
    include_wave_velocity && t === nothing && throw(ArgumentError(
        "the wave velocity e^{-ift} LA depends on time: pass t (the model time " *
        "of the fields), or include_wave_velocity=false for the Eulerian mean"))
    return nothing
end

#=
================================================================================
                    WAVE-INDUCED STOKES DRIFT
================================================================================
Near-inertial waves induce a Lagrangian drift in the direction of wave
propagation. This is the Stokes drift correction.
================================================================================
=#

"""
    compute_wave_velocities!(S, G; t, plans=nothing, f=1, N2=1, compute_w=true, include_wave_velocity=true, N2_profile=nothing, N2_face_profile=nothing, formulation=YBJPlus(), workspace=nothing)

Add the wave part of the instantaneous velocity at time `t` to the
Lagrangian-mean velocity in `S` (from `compute_velocities!`): subtract the
Stokes drift, which turns the Lagrangian mean into the Eulerian mean, and add
the wave orbital velocity.

# Physical Background
1. **Wave velocity** (Asselin & Young 2019, eq. 1.2): `u + iv = e^{-ift} LA`,
   with `L = ∂_z (f²/N²) ∂_z`, and the vertical velocity
   `w = -(f²/N²) A_{zs} e^{-ift} + c.c.` (eq. 2.10). For `YBJPlus()`,
   `B = L⁺A`, so `LA = B + (kₕ²/4)A` in spectral space; for `YBJ()`, `B = LA`.

2. **Stokes drift** (Wagner & Young 2016, eqs. 3.16-3.20). With the buoyancy
   action `M = (f²/N²) A_z`, so that `M_z = LA`, and `∂_s = (∂ₓ - i∂ᵧ)/2`:

       J₀ = M_z* ∂_{s*}M_z - (∂_{s*}M*) M_zz,     i f U^S = J₀,
       K₀ = M_z* M_{ss*} - (∂_{s*}M*) M_{sz},     i f w^S = K₀* - K₀,

   that is `u^S = Im(J₀)/f`, `v^S = -Re(J₀)/f`, `w^S = -2 Im(K₀)/f`, where
   `M_{ss*} = Δ_H M / 4`. The drift `(u^S, v^S, w^S)` is non-divergent. As ψ is
   the Lagrangian-mean streamfunction (their eqs. 3.22-3.23), the Eulerian
   mean is the Lagrangian mean minus this drift.

# Discretization
`M` lives on the faces between levels, with `A_z` (`S.C`) and N² there; it
vanishes on the bottom and top faces. `M_z = LA` and `M_{sz} = ∂_s LA` are
then exact at the cell centres, by the flux form of `L`, so no derivative of
the stratification is needed. The second term of `J₀` is formed on the faces,
where `M` and `M_zz = δ_z LA` both live, and averaged onto the centres; in
`K₀`, `M` is averaged onto the centres. Every term is therefore centred at
the level it is added to, and the drift converges at second order in Δz.

# Arguments
- `S::ModelFields`: ModelFields with A, B, C (input) and u, v, w modified (output)
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `t`: Time for the inertial phase `e^{-ift}`; required with `include_wave_velocity`
- `plans`: FFT plans
- `f`, `N2`: Coriolis frequency and fallback buoyancy frequency squared
- `compute_w::Bool`: If true (default), also correct the vertical velocity
- `include_wave_velocity::Bool`: If false, only subtract the Stokes drift,
  which leaves the Eulerian-mean velocity
- `N2_profile::Vector`: Optional cell-centre N²(z), used only when
  `N2_face_profile` is not given
- `N2_face_profile::Vector`: Optional N²(z) on the upper face of each level,
  where `M = (f²/N²) A_z` is formed
- `formulation`: `YBJPlus()` (default) or `YBJ()`, which fixes how LA follows from B
- `workspace`: Optional model workspace for scratch buffers

# References
- Asselin & Young (2019), J. Fluid Mech. 876, 428-448, equations (1.2), (2.10)
- Wagner & Young (2016), J. Fluid Mech. 802, 806-837, equations (3.16)-(3.23)
- Xie & Vanneste (2015), J. Fluid Mech. 774, 143-169
"""
function compute_wave_velocities!(S::ModelFields, G::RuntimeGeometry; t=nothing,
    plans=nothing, f::Real=1.0, N2::Real=1.0, compute_w=true,
    include_wave_velocity=true, N2_profile=nothing, N2_face_profile=nothing,
    formulation=YBJPlus(), workspace=nothing)

    _require_wave_time(t, include_wave_velocity)
    return with_scratch(workspace) do
        _compute_wave_velocities!(S, G; t, plans, f, N2, compute_w,
                                  include_wave_velocity, N2_profile,
                                  N2_face_profile, formulation, workspace)
    end
end

"""Weight of `kₕ²A` in `LA = B + weight kₕ²A`: `B = L⁺A` for YBJ⁺, `B = LA` for YBJ."""
_wave_velocity_weight(::YBJ) = 0.0
_wave_velocity_weight(formulation) = 0.25

"""
Spectral LA, ∂_s LA, and ∂_{s*} LA at the centres, and ∂_s M and Δ_H M / 4 on
the face above each level, with `M = (f²/N²) A_z` zero on the top face.
"""
@kernel function _wave_spectral_kernel!(LAₖ, dLA_dsₖ, dLA_ds_conjₖ, dM_dsₖ, M_ss_conjₖ,
                                        @Const(Bₖ), @Const(Aₖ), @Const(Aₖ_z),
                                        @Const(N²_face), @Const(kx), @Const(ky),
                                        f², weight, nz)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        kₓ = kx[i]
        kᵧ = ky[j]
        kₕ² = kₓ^2 + kᵧ^2
        ds = (im * kₓ + kᵧ) / 2            # ∂_s = (∂ₓ - i∂ᵧ)/2
        ds_conj = (im * kₓ - kᵧ) / 2       # ∂_{s*} = (∂ₓ + i∂ᵧ)/2

        LA = Bₖ[k, i, j] + weight * kₕ² * Aₖ[k, i, j]
        LAₖ[k, i, j] = LA
        dLA_dsₖ[k, i, j] = ds * LA
        dLA_ds_conjₖ[k, i, j] = ds_conj * LA

        M = k < nz ? (f² / N²_face[k]) * Aₖ_z[k, i, j] : zero(eltype(Aₖ_z))
        dM_dsₖ[k, i, j] = ds * M
        M_ss_conjₖ[k, i, j] = -kₕ² * M / 4   # ∂_s ∂_{s*} = Δ_H / 4
    end
end

# Without `vertical_excursions`, only the drift of horizontal orbits, the first
# term of J₀, is subtracted: the Eulerian mean seen by particles that keep
# their depth, whose orbits make up only that part of the Stokes drift.
@kernel function _stokes_drift_kernel!(u, v, w, @Const(LA), @Const(dLA_ds),
                                       @Const(dLA_ds_conj), @Const(dM_ds),
                                       @Const(M_ss_conj), inv_f, inv_Δz, nz,
                                       include_wave_velocity, compute_w,
                                       cos_ft, sin_ft, vertical_excursions)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        LAₖ = LA[k, i, j]

        # Wave orbital velocity, YBJ⁺ eq. (1.2): u + iv = e^{-ift} LA
        if include_wave_velocity
            u[k, i, j] += cos_ft * real(LAₖ) + sin_ft * imag(LAₖ)
            v[k, i, j] += cos_ft * imag(LAₖ) - sin_ft * real(LAₖ)
        end

        # ∂_s M on the faces below and above level k; M = 0 on the bottom face
        # and, through dM_ds[nz] = 0, on the top face.
        zero_M = zero(eltype(dM_ds))
        dM_ds_below = k > 1 ? dM_ds[k - 1, i, j] : zero_M
        dM_ds_above = dM_ds[k, i, j]

        # J₀ = M_z* ∂_{s*}M_z - (∂_{s*}M*) M_zz with M_z = LA. The second term
        # is formed on the faces, where M_zz = δ_z LA lives, then averaged.
        J₀ = if vertical_excursions
            T_below = k > 1 ?
                conj(dM_ds_below) * (LAₖ - LA[k - 1, i, j]) * inv_Δz : zero_M
            T_above = k < nz ?
                conj(dM_ds_above) * (LA[k + 1, i, j] - LAₖ) * inv_Δz : zero_M
            conj(LAₖ) * dLA_ds_conj[k, i, j] - (T_below + T_above) / 2
        else
            conj(LAₖ) * dLA_ds_conj[k, i, j]
        end

        # Wagner & Young (2016) eq. (3.18): i f U^S = J₀. The Stokes drift is
        # subtracted: Eulerian mean = Lagrangian mean - Stokes drift.
        u[k, i, j] -= inv_f * imag(J₀)
        v[k, i, j] += inv_f * real(J₀)

        # Eqs. (3.19)-(3.20): i f w^S = K₀* - K₀ with
        # K₀ = M_z* M_{ss*} - (∂_{s*}M*) M_{sz}, M_z = LA, M_{sz} = ∂_s LA,
        # and M averaged onto the centre.
        if compute_w
            M_ss_conj_below = k > 1 ? M_ss_conj[k - 1, i, j] : zero_M
            M_ss_conj_centre = (M_ss_conj_below + M_ss_conj[k, i, j]) / 2
            dM_ds_centre = (dM_ds_below + dM_ds_above) / 2
            K₀ = conj(LAₖ) * M_ss_conj_centre - conj(dM_ds_centre) * dLA_ds[k, i, j]
            w[k, i, j] += 2 * inv_f * imag(K₀)
        end
    end
end

function _compute_wave_velocities!(S::ModelFields, G::RuntimeGeometry; t=nothing,
    plans=nothing, f::Real=1.0, N2::Real=1.0, compute_w=true,
    include_wave_velocity=true, N2_profile=nothing, N2_face_profile=nothing,
    formulation=YBJPlus(), workspace=nothing, vertical_excursions::Bool=true)
    nz = G.nz
    backend = get_backend(S.u)
    if plans === nothing
        plans = plan_transforms!(G)
    end

    # M = (f²/N²) A_z is formed on the faces, where A_z and N² live.
    face_profile = _coerce_N2_profile(
        N2_face_profile === nothing ? N2_profile : N2_face_profile, N2, nz, G)

    cos_ft = t === nothing ? 1.0 : cos(f * t)
    sin_ft = t === nothing ? 0.0 : sin(f * t)
    T = real(eltype(S.A))   # the kernel scalars share the grid's precision
    with_scratch(workspace) do
        LAₖ = scratch_like(workspace, S.A)
        dLA_dsₖ = scratch_like(workspace, S.A)
        dLA_ds_conjₖ = scratch_like(workspace, S.A)
        dM_dsₖ = scratch_like(workspace, S.A)
        M_ss_conjₖ = scratch_like(workspace, S.A)
        launch!(_wave_spectral_kernel!, backend, size(LAₖ),
                LAₖ, dLA_dsₖ, dLA_ds_conjₖ, dM_dsₖ, M_ss_conjₖ, S.B, S.A, S.C,
                on_backend(backend, face_profile), G.kx, G.ky, T(float(f)^2),
                T(_wave_velocity_weight(formulation)), nz)

        # The drift is quadratic in the waves, so it is formed in physical
        # space; each buffer is transformed in place.
        for buffer in (LAₖ, dLA_dsₖ, dLA_ds_conjₖ, dM_dsₖ, M_ss_conjₖ)
            fft_backward!(buffer, buffer, plans)
        end

        launch!(_stokes_drift_kernel!, backend, size(S.u),
                S.u, S.v, S.w, LAₖ, dLA_dsₖ, dLA_ds_conjₖ, dM_dsₖ, M_ss_conjₖ,
                T(inv(float(f))), 1 / vertical_spacing(G), nz,
                Bool(include_wave_velocity), Bool(compute_w), T(cos_ft), T(sin_ft),
                vertical_excursions)
    end

    # The wave vertical velocity at the same phase, once the drift buffers
    # are returned.
    if include_wave_velocity && compute_w
        _ybj_w!(S.w, S, G, plans, f, face_profile, workspace, cos_ft, sin_ft, true)
    end
    return S
end

"""Spectral wave velocity `LA = B + weight kₕ² A` (weight 1/4 for YBJ⁺, 0 for YBJ)."""
@kernel function _wave_velocity_kernel!(LA, @Const(B), @Const(A), @Const(kx),
                                        @Const(ky), weight)
    k, i, j = @index(Global, NTuple)
    @inbounds LA[k, i, j] = B[k, i, j] + weight * (kx[i]^2 + ky[j]^2) * A[k, i, j]
end

"""
    wave_velocity_amplitude!(LA_r, LA_i, S, G; plans=nothing, formulation=YBJPlus(), workspace=nothing)

Fill `LA_r` and `LA_i` with the real and imaginary parts of the wave velocity
amplitude LA at the cell centres, so that at time `t` the horizontal wave
velocity is `u + iv = e^{-ift} LA`, that is, with `c = cos(ft)` and
`s = sin(ft)`,

    u = c LA_r + s LA_i,    v = c LA_i - s LA_r.

The destinations may have another precision than the model, and live on the
model's architecture. A (`S.A`) must be current.
"""
function wave_velocity_amplitude!(LA_r, LA_i, S::ModelFields, G::RuntimeGeometry;
                                  plans=nothing, formulation=YBJPlus(),
                                  workspace=nothing)
    if plans === nothing
        plans = plan_transforms!(G)
    end
    with_scratch(workspace) do
        buffer = scratch_like(workspace, S.B)
        launch!(_wave_velocity_kernel!, buffer, size(buffer), buffer, S.B, S.A,
                G.kx, G.ky, real(eltype(buffer))(_wave_velocity_weight(formulation)))
        fft_backward!(buffer, buffer, plans)
        LA_r .= real.(buffer)
        LA_i .= imag.(buffer)
    end
    return LA_r, LA_i
end

end # module

using .Operators: compute_velocities!, compute_vertical_velocity!, compute_ybj_vertical_velocity!, compute_total_velocities!, compute_wave_velocities!
