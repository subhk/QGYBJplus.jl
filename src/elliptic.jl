#=
================================================================================
                    elliptic.jl - Vertical Elliptic Solvers
================================================================================

This file contains the tridiagonal solvers for the vertical elliptic problems
that arise in the QG-YBJ+ model. These are critical for:

1. STREAMFUNCTION INVERSION (q → ψ):
   Given the QG potential vorticity q, solve for the streamfunction ψ.

2. WAVE AMPLITUDE RECOVERY (L⁺A → A):
   Given the YBJ+ evolved field L⁺A, recover the true wave amplitude A.

3. GENERAL HELMHOLTZ PROBLEMS:
   For omega equation, buoyancy inversions, etc.

MATHEMATICAL BACKGROUND:
------------------------
The QG PV inversion relates q and ψ through (PDF Eq. A13):

    q = ∇²ψ + L(ψ)    where   L = ∂/∂z(a(z) ∂/∂z)

with a(z) = f²/N²(z). Expanding via the product rule:

    L(ψ) = ∂/∂z(a ∂ψ/∂z) = a(z) ∂²ψ/∂z² + a'(z) ∂ψ/∂z

where a'(z) = ∂a/∂z arises from variable stratification N²(z).

In spectral space (∇² → -kₕ²), the equation becomes:

    ∂/∂z(a(z) ∂ψ/∂z) - kₕ² ψ = q

IMPORTANT: The staggered finite-difference discretization (PDF Eq. 32):

    [S_i(ψ_{i+1} - ψ_i) - S_{i-1}(ψ_i - ψ_{i-1})] / Δz² - kₕ² ψ_i = q_i

where S_i = (f/(N(z_i)Δz))² at interface i, **automatically captures both**
the a∂²ψ/∂z² and a'∂ψ/∂z terms through the use of different S values at
different interfaces. No explicit first-derivative term is needed.

For constant N²: S_i = S_{i-1}, and the a'∂ψ/∂z term vanishes.
For variable N²: S_i ≠ S_{i-1}, and the product rule is implicitly handled.

This is solved independently for each (kₓ, kᵧ) mode using a tridiagonal solver.

BOUNDARY CONDITIONS:
-------------------
The standard boundary conditions are:
    ψ_z = 0 at z = -Lz (bottom)
    ψ_z = 0 at z = 0 (top)

These are Neumann conditions corresponding to no buoyancy flux through
the boundaries.

NUMERICAL METHOD:
-----------------
Second-order finite differences in z lead to a tridiagonal system:

    dl[k] * x[k-1] + d[k] * x[k] + du[k] * x[k+1] = rhs[k]

Solved using the Thomas algorithm (Gaussian elimination for tridiagonal).

2D DECOMPOSITION SUPPORT:
-------------------------
For 2D parallel decomposition, vertical operations require z to be local.
Data comes in xy-pencil format and must be transposed to z-pencil format
before the solve, then transposed back afterward.

FORTRAN CORRESPONDENCE:
----------------------
- invert_q_to_psi! ↔ psi_solver (elliptic.f90)
- invert_L⁺A_to_A!   ↔ A_solver_ybj_plus (elliptic.f90)
- invert_helmholtz! ↔ helmholtzdouble (elliptic.f90)

================================================================================
=#

module Elliptic

using ..QGYBJplus: Grid, State, local_to_global, z_is_local, is_parallel_array
using ..QGYBJplus: transpose_to_z_pencil!, transpose_to_xy_pencil!
using ..QGYBJplus: local_to_global_z, allocate_z_pencil
const PARENT = Base.parentmodule(@__MODULE__)

#=
WORKSPACE STRUCTURE:
-------------------
Functions in this module accept an optional `workspace` parameter to avoid
repeated allocations in time-stepping loops. The workspace should be a
NamedTuple or struct with the following fields (not all are needed for
every function):

For `invert_q_to_psi!`:
  - `q_z`: z-pencil array for q (ComplexF64)
  - `psi_z`: z-pencil array for ψ (ComplexF64)

For `invert_helmholtz!`:
  - `work_z`: z-pencil work array (ComplexF64)
  - `psi_z`: z-pencil array for solution (ComplexF64)

For `invert_L⁺A_to_A!`:
  - `L⁺A_z`: z-pencil array for L⁺A (ComplexF64)
  - `A_z`: z-pencil array for A (ComplexF64)
  - `C_z`: z-pencil array for C (A_z derivative) (ComplexF64)

Example:
```julia
workspace = (
    q_z = allocate_z_pencil(G, ComplexF64),
    psi_z = allocate_z_pencil(G, ComplexF64),
    work_z = allocate_z_pencil(G, ComplexF64),
    L⁺A_z = allocate_z_pencil(G, ComplexF64),
    A_z = allocate_z_pencil(G, ComplexF64),
    C_z = allocate_z_pencil(G, ComplexF64),
)
```

If workspace is `nothing`, temporary arrays are allocated internally.
=#

#=
================================================================================
                    STREAMFUNCTION INVERSION: q → ψ
================================================================================
This is the core elliptic inversion that relates QGPV to streamfunction.

PHYSICS (PDF Eq. A13):
    q = ∇²ψ + L(ψ)    where   L = ∂/∂z(a(z) ∂/∂z)

with a(z) = f²/N²(z). Expanding via the product rule:

    L(ψ) = a(z) ∂²ψ/∂z² + a'(z) ∂ψ/∂z

The staggered discretization (PDF Eq. 32) for interior points:

    [S_{k+1}(ψ[k+1] - ψ[k]) - S_k(ψ[k] - ψ[k-1])] / dz² - kₕ² ψ[k] = q[k]

where S_k = (f/(N(z_k)Δz))² at interface k. Using different S values at
different interfaces automatically captures the a'∂ψ/∂z term.

Neumann BCs (ψ_z = 0) modify the boundary stencils.
================================================================================
=#

"""
    invert_q_to_psi!(S, G; a, par=nothing, workspace=nothing)

Invert spectral QGPV `q(kx,ky,z)` to obtain streamfunction `ψ(kx,ky,z)`.

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve the vertical ODE:

    ∂/∂z(a(z) ∂ψ/∂z) - kₕ² ψ = q

which expands via the product rule to: a(z)∂²ψ/∂z² + a'(z)∂ψ/∂z - kₕ²ψ = q.
The staggered discretization captures a'∂ψ/∂z automatically.

Neumann boundary conditions ψ_z = 0 at top and bottom.

# Arguments
- `S::State`: State struct containing `q` (input) and `psi` (output)
- `G::Grid`: Grid struct with wavenumbers and vertical coordinates
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = f²/N²(z), length nz
- `par`: Optional QGParams for density weighting (defaults to unity weights)
- `workspace`: Optional z-pencil workspace arrays for 2D decomposition

# Implementation Details
For 2D decomposition:
1. Transpose q from xy-pencil to z-pencil (z becomes local)
2. Perform tridiagonal solve on z-pencil data
3. Transpose ψ from z-pencil back to xy-pencil

The discrete system is tridiagonal with structure (for interior row `k`):
- Upper diagonal: `du[k] = a[k+1]` (interface above cell `k`)
- Diagonal: `d[k] = -(a[k+1] + a[k]) - kₕ² dz²`
- Lower diagonal: `dl[k] = a[k]` (interface below cell `k`)

where `a[k] = f₀²/N²[k]` is evaluated at the interface below cell `k`.

# Fortran Correspondence
This matches `psi_solver` in elliptic.f90.

# Mean Mode (kₕ=0) Handling
For the horizontal mean mode (kₓ=kᵧ=0), the equation reduces to:
    ∂/∂z(a(z) ∂ψ/∂z) = q

With Neumann boundary conditions (∂ψ/∂z=0 at both boundaries), this operator
is **singular**: the homogeneous equation has the constant function as its
null space. Consequently:
1. A solution exists only if ∫q dz = 0 (compatibility condition)
2. The solution is determined only up to an arbitrary constant

This implementation sets ψ=0 for kₕ=0 because:
- For periodic domains, the mean streamfunction doesn't affect velocities
  (u = -∂ψ/∂y, v = ∂ψ/∂x, both zero for constant ψ)
- Standard spectral QG codes typically ignore the barotropic mean
- Initial conditions and forcing are assumed to have zero horizontal mean

If your application requires tracking vertically-varying barotropic modes,
you would need to solve the singular ODE with an additional constraint
(e.g., ∫ψ dz = 0) to uniquely determine the solution.

# Example
```julia
a_vec = a_ell_ut(params, G)  # Compute a_ell = f²/N²
invert_q_to_psi!(state, grid; a=a_vec)
```
"""
function invert_q_to_psi!(S::State, G::Grid; a::AbstractVector, par=nothing, workspace=nothing)
    nz = G.nz
    @assert length(a) == nz "a must have length nz=$nz"

    # Check if we need to do transpose (2D decomposition)
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(S.q, G)

    if need_transpose
        # 2D decomposition: transpose to z-pencil, solve, transpose back
        _invert_q_to_psi_2d!(S, G, a, par, workspace)
    else
        # Serial or 1D decomposition: direct solve (z already local)
        _invert_q_to_psi_direct!(S, G, a, par)
    end

    return S
end

"""
Direct solve for serial mode or 1D decomposition (z fully local).
"""
function _invert_q_to_psi_direct!(S::State, G::Grid, a::AbstractVector, par)
    nz = G.nz

    # Get underlying arrays (works for both Array and PencilArray)
    ψ_arr = parent(S.psi)   # Output: streamfunction
    q_arr = parent(S.q)     # Input: QGPV

    # Get local dimensions
    nz_local, nx_local, ny_local = size(ψ_arr)

    # Verify z is fully local (required for vertical tridiagonal solve)
    @assert nz_local == nz "Vertical dimension must be fully local (nz_local=$nz_local, nz=$nz)"

    # Tridiagonal matrix diagonals (reused for each wavenumber)
    dₗ = zeros(eltype(a), nz)   # Lower diagonal
    d  = zeros(eltype(a), nz)   # Main diagonal
    dᵤ = zeros(eltype(a), nz)   # Upper diagonal

    # Vertical grid spacing
    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δz² = Δz^2

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhs  = zeros(eltype(a), nz)
    rhsᵢ = zeros(eltype(a), nz)
    solᵣ = zeros(eltype(a), nz)
    solᵢ = zeros(eltype(a), nz)

    # Loop over all LOCAL horizontal wavenumbers (using local indices)
    for j_local in 1:ny_local, i_local in 1:nx_local
        # Get global indices for wavenumber lookup
        i_global = local_to_global(i_local, 2, S.q)
        j_global = local_to_global(j_local, 3, S.q)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2   # Horizontal wavenumber squared

        # Special case: kₕ² = 0 (horizontal mean mode)
        # The operator ∂/∂z(a ∂/∂z) with Neumann BCs is singular (constant null space).
        # We set ψ=0 because: (1) mean ψ doesn't affect velocities in periodic domains,
        # (2) the ODE solution is only determined up to an arbitrary constant.
        # CAUTION: This discards any horizontally uniform PV without enforcing the
        # compatibility condition ∫q dz = 0. If the mean mode has significant magnitude,
        # total PV may be misrepresented in diagnostics.
        if kₕ² == 0
            # Check if mean mode has significant energy (warn once per run)
            q_mean_mag = maximum(abs, @view q_arr[:, i_local, j_local])
            if q_mean_mag > 1e-10  # Threshold for "significant"
                @warn "invert_q_to_psi!: Non-zero horizontal mean in q (max |q(k=0)|=$(q_mean_mag)). " *
                      "This barotropic component is discarded (ψ=0 for kₕ²=0) and will not " *
                      "contribute to the flow field. This is physically correct for computing " *
                      "velocities but may affect PV conservation diagnostics." maxlog=1
            end
            @inbounds for k in 1:nz
                ψ_arr[k, i_local, j_local] = 0
            end
            continue
        end

        # Special case: nz == 1 (single-layer / 2D mode)
        # With only one vertical level and Neumann BCs, vertical derivative terms vanish.
        # The equation reduces to: -kₕ² ψ = q  →  ψ = -q/kₕ²
        if nz == 1
            @inbounds ψ_arr[1, i_local, j_local] = -q_arr[1, i_local, j_local] / kₕ²
            continue
        end

        # Build tridiagonal matrix for this (kₓ, kᵧ)
        fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

        # Bottom boundary (k=1): Neumann condition ψ_z = 0
        d[1]  = -(a[1] + kₕ²*Δz²)
        dᵤ[1] = a[1]

        # Interior points (k = 2, ..., nz-1)
        @inbounds for k in 2:nz-1
            dₗ[k] = a[k]
            d[k]  = -(a[k+1] + a[k] + kₕ²*Δz²)
            dᵤ[k] = a[k+1]
        end

        # Top boundary (k=nz): Neumann condition ψ_z = 0
        dₗ[nz] = a[nz]
        d[nz]  = -(a[nz] + kₕ²*Δz²)

        # Solve for real and imaginary parts separately
        @inbounds for k in 1:nz
            rhs[k] = Δz² * real(q_arr[k, i_local, j_local])
        end
        thomas_solve!(solᵣ, dₗ, d, dᵤ, rhs)

        @inbounds for k in 1:nz
            rhsᵢ[k] = Δz² * imag(q_arr[k, i_local, j_local])
        end
        thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

        # Combine into complex solution
        @inbounds for k in 1:nz
            ψ_arr[k, i_local, j_local] = solᵣ[k] + im*solᵢ[k]
        end
    end
end

"""
2D decomposition: transpose to z-pencil, solve, transpose back.
"""
function _invert_q_to_psi_2d!(S::State, G::Grid, a::AbstractVector, par, workspace)
    nz = G.nz

    # Allocate z-pencil workspace if not provided
    q_z = workspace !== nothing ? workspace.q_z : allocate_z_pencil(G, ComplexF64)
    ψ_z = workspace !== nothing ? workspace.psi_z : allocate_z_pencil(G, ComplexF64)

    # Transpose q from xy-pencil to z-pencil
    transpose_to_z_pencil!(q_z, S.q, G)

    # Get underlying arrays in z-pencil format
    q_z_arr = parent(q_z)
    ψ_z_arr = parent(ψ_z)

    # Get local dimensions in z-pencil (z is now fully local)
    nz_local, nx_local, ny_local = size(q_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    # Tridiagonal matrix diagonals
    dₗ = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    dᵤ = zeros(eltype(a), nz)

    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δz² = Δz^2

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhs  = zeros(eltype(a), nz)
    rhsᵢ = zeros(eltype(a), nz)
    solᵣ = zeros(eltype(a), nz)
    solᵢ = zeros(eltype(a), nz)

    # Loop over LOCAL wavenumbers in z-pencil configuration
    for j_local in 1:ny_local, i_local in 1:nx_local
        # Get global indices for wavenumber lookup (z-pencil ranges)
        i_global = local_to_global_z(i_local, 2, G)
        j_global = local_to_global_z(j_local, 3, G)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        # Special case: kₕ² = 0 (horizontal mean mode)
        # CAUTION: This discards any horizontally uniform PV without enforcing the
        # compatibility condition ∫q dz = 0.
        if kₕ² == 0
            # Check if mean mode has significant energy (warn once per run)
            q_mean_mag = maximum(abs, @view q_z_arr[:, i_local, j_local])
            if q_mean_mag > 1e-10  # Threshold for "significant"
                @warn "invert_q_to_psi!: Non-zero horizontal mean in q (max |q(k=0)|=$(q_mean_mag)). " *
                      "This barotropic component is discarded (ψ=0 for kₕ²=0) and will not " *
                      "contribute to the flow field. This is physically correct for computing " *
                      "velocities but may affect PV conservation diagnostics." maxlog=1
            end
            @inbounds for k in 1:nz
                ψ_z_arr[k, i_local, j_local] = 0
            end
            continue
        end

        # Special case: nz == 1 (single-layer / 2D mode)
        if nz == 1
            @inbounds ψ_z_arr[1, i_local, j_local] = -q_z_arr[1, i_local, j_local] / kₕ²
            continue
        end

        fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

        # Bottom boundary (k=1): Neumann condition ψ_z = 0
        d[1]  = -(a[1] + kₕ²*Δz²)
        dᵤ[1] = a[1]

        # Interior points (k = 2, ..., nz-1)
        @inbounds for k in 2:nz-1
            dₗ[k] = a[k]
            d[k]  = -(a[k+1] + a[k] + kₕ²*Δz²)
            dᵤ[k] = a[k+1]
        end

        # Top boundary (k=nz): Neumann condition ψ_z = 0
        dₗ[nz] = a[nz]
        d[nz]  = -(a[nz] + kₕ²*Δz²)

        # Solve for real and imaginary parts
        @inbounds for k in 1:nz
            rhs[k] = Δz² * real(q_z_arr[k, i_local, j_local])
        end
        thomas_solve!(solᵣ, dₗ, d, dᵤ, rhs)

        @inbounds for k in 1:nz
            rhsᵢ[k] = Δz² * imag(q_z_arr[k, i_local, j_local])
        end
        thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

        @inbounds for k in 1:nz
            ψ_z_arr[k, i_local, j_local] = solᵣ[k] + im*solᵢ[k]
        end
    end

    # Transpose ψ from z-pencil back to xy-pencil
    transpose_to_xy_pencil!(S.psi, ψ_z, G)
end

#=
================================================================================
                    GENERAL HELMHOLTZ SOLVER
================================================================================
This solves general elliptic problems of the form:

    a(z) ∂²φ/∂z² + b(z) ∂φ/∂z - α kₕ² φ = f

with optional boundary condition terms. Used for omega equation, etc.
================================================================================
=#

"""
    invert_helmholtz!(dstk, rhs, G, par; a, b=zeros, scale_kh2=1.0, bot_bc=nothing, top_bc=nothing, workspace=nothing)

General vertical Helmholtz inversion for each horizontal wavenumber.

# Mathematical Problem
Solve the ODE:

    a(z) ∂²φ/∂z² + b(z) ∂φ/∂z - scale_kh2 × kₕ² φ = rhs

with Neumann boundary conditions (∂φ/∂z specified at boundaries).

# Discretization (matches Fortran helmholtzdouble)
Uses a centered stencil where same a[k], b[k] apply to all diagonals at point k:

- Bottom (k=1):  d = -a[1] - 0.5b[1]Δz - αkₕ²Δz², du = a[1] + 0.5b[1]Δz
- Interior:      d = -2a[k] - αkₕ²Δz², du = a[k] + 0.5b[k]Δz, dl = a[k] - 0.5b[k]Δz
- Top (k=nz):    d = -a[nz] + 0.5b[nz]Δz - αkₕ²Δz², dl = a[nz] - 0.5b[nz]Δz

Boundary flux terms are added to RHS:
- Bottom: rhs[1] += (a[1] - 0.5b[1]Δz) × Δz × bot_bc
- Top:    rhs[nz] -= (a[nz] + 0.5b[nz]Δz) × Δz × top_bc

# Arguments
- `dstk`: Output array (nz, nx, ny) for solution φ
- `rhs`: Right-hand side array (nz, nx, ny)
- `G::Grid`: Grid struct
- `par`: QGParams (currently unused, kept for API consistency)
- `a::AbstractVector`: Second derivative coefficient a(z), length nz
- `b::AbstractVector`: First derivative coefficient b(z), length nz (default zeros)
- `scale_kh2::Real`: Multiplier α for kₕ² term (default 1.0)
- `bot_bc`, `top_bc`: Optional boundary flux arrays (nx, ny) for non-zero Neumann BCs
- `workspace`: Optional z-pencil workspace for 2D decomposition

# Fortran Correspondence
This matches `helmholtzdouble` in elliptic.f90 exactly.

# Note
For 2D decomposition, boundary conditions are not yet supported and will trigger a warning.
"""
function invert_helmholtz!(dstk, rhs, G::Grid, par;
                           a::AbstractVector,
                           b::AbstractVector=zeros(eltype(a), length(a)),
                           scale_kh2::Real=1.0,
                           bot_bc=nothing,
                           top_bc=nothing,
                           workspace=nothing)
    nz = G.nz

    # Check if we need 2D decomposition transpose
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(rhs, G)

    if need_transpose
        _invert_helmholtz_2d!(dstk, rhs, G, par, a, b, scale_kh2, bot_bc, top_bc, workspace)
    else
        _invert_helmholtz_direct!(dstk, rhs, G, par, a, b, scale_kh2, bot_bc, top_bc)
    end

    return dstk
end

@inline function _bc_value(bc, i_local, j_local, i_global, j_global)
    if bc === nothing
        return nothing
    end
    if is_parallel_array(bc)
        return parent(bc)[i_local, j_local]
    end
    return bc[i_global, j_global]
end

"""
Direct Helmholtz solve for serial or 1D decomposition.

Matches Fortran `helmholtzdouble` discretization exactly:
- Uses centered stencil with same a[k], b[k] for all diagonals at point k
- No density weighting (coefficients used directly)
- Interior: d[k] = -2a[k] - kh²Δz²
- Boundary conditions incorporated via RHS modifications
"""
function _invert_helmholtz_direct!(dstk, rhs, G::Grid, par, a, b, scale_kh2, bot_bc, top_bc)
    nz = G.nz

    dst_arr = parent(dstk)
    rhs_arr = parent(rhs)

    nz_local, nx_local, ny_local = size(dst_arr)

    @assert nz_local == nz "Vertical dimension must be fully local"
    @assert length(a) == nz "a must have length nz=$nz"
    @assert length(b) == nz "b must have length nz=$nz"
    if bot_bc !== nothing && is_parallel_array(bot_bc)
        @assert size(parent(bot_bc)) == (nx_local, ny_local) "bot_bc local size must match (nx_local, ny_local)"
    end
    if top_bc !== nothing && is_parallel_array(top_bc)
        @assert size(parent(top_bc)) == (nx_local, ny_local) "top_bc local size must match (nx_local, ny_local)"
    end

    if nz == 1
        if bot_bc !== nothing || top_bc !== nothing
            @warn "Helmholtz solve with nz=1 ignores boundary conditions" maxlog=1
        end

        tol = sqrt(eps(real(one(eltype(rhs_arr)))))
        singular_warned = false
        for j_local in 1:ny_local, i_local in 1:nx_local
            i_global = local_to_global(i_local, 2, rhs)
            j_global = local_to_global(j_local, 3, rhs)

            kₓ = G.kx[i_global]
            kᵧ = G.ky[j_global]
            kₕ² = kₓ^2 + kᵧ^2
            denom = scale_kh2 * kₕ²

            rhs_val = rhs_arr[1, i_local, j_local]
            if abs(denom) < tol
                if !singular_warned && abs(rhs_val) > tol
                    @warn "Helmholtz solve with nz=1 has kₕ²≈0 and nonzero RHS; setting φ=0 for the mean mode." maxlog=1
                    singular_warned = true
                end
                dst_arr[1, i_local, j_local] = 0
            else
                dst_arr[1, i_local, j_local] = -rhs_val / denom
            end
        end
        return dstk
    end

    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δz² = Δz^2

    dₗ = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    dᵤ = zeros(eltype(a), nz)

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhsᵣ = zeros(eltype(a), nz)
    rhsᵢ = zeros(eltype(a), nz)
    solᵣ = zeros(eltype(a), nz)
    solᵢ = zeros(eltype(a), nz)

    for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, rhs)
        j_global = local_to_global(j_local, 3, rhs)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

        #= Build tridiagonal matrix matching Fortran helmholtzdouble exactly
           Key: uses same a[k], b[k] for all diagonals at each point k =#

        # Bottom boundary (k=1): Neumann condition
        # Fortran: d(1) = -a_helm(1) - 0.5*b_helm(1)*dz - kh2*dz*dz
        #          du(1) = a_helm(1) + 0.5*b_helm(1)*dz
        d[1]  = -a[1] - 0.5*b[1]*Δz - scale_kh2*kₕ²*Δz²
        dᵤ[1] =  a[1] + 0.5*b[1]*Δz

        # Interior points (k = 2, ..., nz-1)
        # Fortran: d(iz) = -2*a_helm(iz) - kh2*dz*dz
        #          du(iz) = a_helm(iz) + 0.5*b_helm(iz)*dz
        #          dl(iz-1) = a_helm(iz) - 0.5*b_helm(iz)*dz
        @inbounds for k in 2:nz-1
            dₗ[k] = a[k] - 0.5*b[k]*Δz
            d[k]  = -2*a[k] - scale_kh2*kₕ²*Δz²
            dᵤ[k] =  a[k] + 0.5*b[k]*Δz
        end

        # Top boundary (k=nz): Neumann condition
        # Fortran: d(n3) = -a_helm(n3) + 0.5*b_helm(n3)*dz - kh2*dz*dz
        #          dl(n3-1) = a_helm(n3) - 0.5*b_helm(n3)*dz
        dₗ[nz] = a[nz] - 0.5*b[nz]*Δz
        d[nz]  = -a[nz] + 0.5*b[nz]*Δz - scale_kh2*kₕ²*Δz²

        # Build RHS (reusing pre-allocated arrays)
        @inbounds for k in 1:nz
            rhsᵣ[k] = Δz² * real(rhs_arr[k, i_local, j_local])
            rhsᵢ[k] = Δz² * imag(rhs_arr[k, i_local, j_local])
        end

        # Add boundary condition contributions to RHS
        # Fortran: br(1) = br(1) + (a_helm(1) - 0.5*b_helm(1)*dz)*DBLE(b_bot)*dz
        if bot_bc !== nothing
            bc_val = _bc_value(bot_bc, i_local, j_local, i_global, j_global)
            rhsᵣ[1] += (a[1] - 0.5*b[1]*Δz) * Δz * real(bc_val)
            rhsᵢ[1] += (a[1] - 0.5*b[1]*Δz) * Δz * imag(bc_val)
        end
        # Fortran: br(n3) = br(n3) - (a_helm(n3) + 0.5*b_helm(n3)*dz)*DBLE(b_top)*dz
        if top_bc !== nothing
            bc_val = _bc_value(top_bc, i_local, j_local, i_global, j_global)
            rhsᵣ[nz] -= (a[nz] + 0.5*b[nz]*Δz) * Δz * real(bc_val)
            rhsᵢ[nz] -= (a[nz] + 0.5*b[nz]*Δz) * Δz * imag(bc_val)
        end

        # Solve tridiagonal systems for real and imaginary parts
        thomas_solve!(solᵣ, dₗ, d, dᵤ, rhsᵣ)
        thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

        @inbounds for k in 1:nz
            dst_arr[k, i_local, j_local] = solᵣ[k] + im*solᵢ[k]
        end
    end
end

"""
2D decomposition Helmholtz solve with transposes.

Matches Fortran `helmholtzdouble` discretization exactly:
- Uses centered stencil with same a[k], b[k] for all diagonals at point k
- No density weighting (coefficients used directly)
- Interior: d[k] = -2a[k] - kh²Δz²

"""
function _invert_helmholtz_2d!(dstk, rhs, G::Grid, par, a, b, scale_kh2, bot_bc, top_bc, workspace)
    nz = G.nz

    @assert length(a) == nz "a must have length nz=$nz"
    @assert length(b) == nz "b must have length nz=$nz"

    # Use z-pencil workspace arrays to avoid repeated allocations
    # work_z is used for input (rhs), psi_z is used for output (dst)
    rhs_z = workspace !== nothing ? workspace.work_z : allocate_z_pencil(G, ComplexF64)
    dst_z = workspace !== nothing ? workspace.psi_z : allocate_z_pencil(G, ComplexF64)

    # Transpose to z-pencil
    transpose_to_z_pencil!(rhs_z, rhs, G)

    rhs_z_arr = parent(rhs_z)
    dst_z_arr = parent(dst_z)

    nz_local, nx_local, ny_local = size(rhs_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    if nz == 1
        if bot_bc !== nothing || top_bc !== nothing
            @warn "Helmholtz solve with nz=1 ignores boundary conditions" maxlog=1
        end
        tol = sqrt(eps(real(one(eltype(rhs_z_arr)))))
        singular_warned = false

        for j_local in 1:ny_local, i_local in 1:nx_local
            i_global = local_to_global_z(i_local, 2, G)
            j_global = local_to_global_z(j_local, 3, G)

            kₓ = G.kx[i_global]
            kᵧ = G.ky[j_global]
            kₕ² = kₓ^2 + kᵧ^2
            denom = scale_kh2 * kₕ²

            rhs_val = rhs_z_arr[1, i_local, j_local]
            if abs(denom) < tol
                if !singular_warned && abs(rhs_val) > tol
                    @warn "Helmholtz solve with nz=1 has kₕ²≈0 and nonzero RHS; setting φ=0 for the mean mode." maxlog=1
                    singular_warned = true
                end
                dst_z_arr[1, i_local, j_local] = 0
            else
                dst_z_arr[1, i_local, j_local] = -rhs_val / denom
            end
        end

        transpose_to_xy_pencil!(dstk, dst_z, G)
        return dstk
    end

    if (bot_bc !== nothing && is_parallel_array(bot_bc)) || (top_bc !== nothing && is_parallel_array(top_bc))
        @warn "invert_helmholtz!: boundary conditions are not supported for 2D decomposition " *
              "when bc arrays are distributed. Ignoring bot_bc/top_bc." maxlog=1
        bot_bc = nothing
        top_bc = nothing
    end

    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δz² = Δz^2

    dₗ = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    dᵤ = zeros(eltype(a), nz)

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhsᵣ = zeros(eltype(a), nz)
    rhsᵢ = zeros(eltype(a), nz)
    solᵣ = zeros(eltype(a), nz)
    solᵢ = zeros(eltype(a), nz)

    for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global_z(i_local, 2, G)
        j_global = local_to_global_z(j_local, 3, G)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

        #= Build tridiagonal matrix matching Fortran helmholtzdouble exactly
           Key: uses same a[k], b[k] for all diagonals at each point k =#

        # Bottom boundary (k=1): Neumann condition
        # Fortran: d(1) = -a_helm(1) - 0.5*b_helm(1)*dz - kh2*dz*dz
        #          du(1) = a_helm(1) + 0.5*b_helm(1)*dz
        d[1]  = -a[1] - 0.5*b[1]*Δz - scale_kh2*kₕ²*Δz²
        dᵤ[1] =  a[1] + 0.5*b[1]*Δz

        # Interior points (k = 2, ..., nz-1)
        # Fortran: d(iz) = -2*a_helm(iz) - kh2*dz*dz
        #          du(iz) = a_helm(iz) + 0.5*b_helm(iz)*dz
        #          dl(iz-1) = a_helm(iz) - 0.5*b_helm(iz)*dz
        @inbounds for k in 2:nz-1
            dₗ[k] = a[k] - 0.5*b[k]*Δz
            d[k]  = -2*a[k] - scale_kh2*kₕ²*Δz²
            dᵤ[k] =  a[k] + 0.5*b[k]*Δz
        end

        # Top boundary (k=nz): Neumann condition
        # Fortran: d(n3) = -a_helm(n3) + 0.5*b_helm(n3)*dz - kh2*dz*dz
        #          dl(n3-1) = a_helm(n3) - 0.5*b_helm(n3)*dz
        dₗ[nz] = a[nz] - 0.5*b[nz]*Δz
        d[nz]  = -a[nz] + 0.5*b[nz]*Δz - scale_kh2*kₕ²*Δz²

        # Build RHS (reusing pre-allocated arrays)
        @inbounds for k in 1:nz
            rhsᵣ[k] = Δz² * real(rhs_z_arr[k, i_local, j_local])
            rhsᵢ[k] = Δz² * imag(rhs_z_arr[k, i_local, j_local])
        end

        # Add boundary condition contributions to RHS
        if bot_bc !== nothing
            bc_val = _bc_value(bot_bc, i_local, j_local, i_global, j_global)
            rhsᵣ[1] += (a[1] - 0.5*b[1]*Δz) * Δz * real(bc_val)
            rhsᵢ[1] += (a[1] - 0.5*b[1]*Δz) * Δz * imag(bc_val)
        end
        if top_bc !== nothing
            bc_val = _bc_value(top_bc, i_local, j_local, i_global, j_global)
            rhsᵣ[nz] -= (a[nz] + 0.5*b[nz]*Δz) * Δz * real(bc_val)
            rhsᵢ[nz] -= (a[nz] + 0.5*b[nz]*Δz) * Δz * imag(bc_val)
        end

        # Solve tridiagonal systems for real and imaginary parts
        thomas_solve!(solᵣ, dₗ, d, dᵤ, rhsᵣ)
        thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

        @inbounds for k in 1:nz
            dst_z_arr[k, i_local, j_local] = solᵣ[k] + im*solᵢ[k]
        end
    end

    # Transpose back to xy-pencil
    transpose_to_xy_pencil!(dstk, dst_z, G)
end

#=
================================================================================
                    YBJ+ WAVE INVERSION: L⁺A → A
================================================================================
In the YBJ+ formulation, the prognostic variable is L⁺A, where L⁺ is the
YBJ+ elliptic operator. After time stepping L⁺A, we need to recover A for
computing wave-related quantities (including the wave velocity amplitude LA
for GLM particle advection).

OPERATOR DEFINITIONS (from PDF):
--------------------------------
    L  (YBJ operator):   L  = ∂/∂z(f²/N² ∂/∂z)              [eq. (4)]
    L⁺ (YBJ+ operator):  L⁺ = L - k_h²/4                     [spectral space]

Or equivalently in physical space:
    L⁺ = L + (1/4)∇²     where ∇² → -k_h² in spectral space

KEY RELATION: L = L⁺ + k_h²/4

WAVE VELOCITY AMPLITUDE:
------------------------
The instantaneous wave velocity uses the YBJ operator L (NOT L⁺):
    u + iv = (LA) × e^{-ift}                                 [eq. (3)]

Since L = L⁺ + k_h²/4:
    LA = (L⁺ + k_h²/4)A = L⁺A + (k_h²/4)A

This LA is used for GLM particle advection wave displacement.

ELLIPTIC EQUATION:
------------------
With a(z) = f²/N²(z), the L⁺ operator expands via the product rule:

    L⁺A = ∂/∂z(a ∂A/∂z) - (kₕ²/4) A
        = a(z) ∂²A/∂z² + a'(z) ∂A/∂z - (kₕ²/4) A

where a'(z) = ∂a/∂z arises from variable stratification N²(z).

IMPORTANT: The staggered finite-difference discretization (PDF Eq. 35):

    [S_i(A_{i+1} - A_i) - S_{i-1}(A_i - A_{i-1})] / Δz² - (kₕ²/4) A_i = (L⁺A)_i

where S_i = (f/(N(z_i)Δz))² at interface i, **automatically captures both**
the a∂²A/∂z² and a'∂A/∂z terms. No explicit first-derivative term is needed.

We also compute C = ∂A/∂z for use in wave feedback and vertical velocity.
================================================================================
=#

"""
    invert_L⁺A_to_A!(S, G, par, a; workspace=nothing)

YBJ+ wave amplitude recovery: solve for A given L⁺A (the prognostic variable).

# Operator Definitions (from PDF)
    L  (YBJ):   L  = ∂/∂z(f²/N² ∂/∂z)                        [eq. (4)]
    L⁺ (YBJ+):  L⁺ = L - k_h²/4                               [spectral space]

Key relation: L = L⁺ + k_h²/4, so LA = L⁺A + (k_h²/4)A

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve the L⁺ elliptic equation:

    L⁺A = ∂/∂z(a(z) ∂A/∂z) - (kₕ²/4) A = (L⁺A)_input

which expands via the product rule to:

    a(z) ∂²A/∂z² + a'(z) ∂A/∂z - (kₕ²/4) A = (L⁺A)_input

with Neumann boundary conditions A_z = 0 at top and bottom.

The staggered discretization using S_i = (f/(N(z_i)Δz))² at different interfaces
automatically captures the a'(z)∂A/∂z term from variable stratification.

# Arguments
- `S::State`: State containing `L⁺A` (input), `A` and `C` (output)
- `G::Grid`: Grid struct
- `par`: QGParams (for f0, N2 parameters)
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = f²/N²(z)
- `workspace`: Optional z-pencil workspace for 2D decomposition

# Output Fields
- `S.A`: Recovered wave amplitude A
- `S.C`: Vertical derivative C = ∂A/∂z (for wave velocity computation)

# Wave Velocity Amplitude
After this function, LA can be computed as: LA = L⁺A + (k_h²/4)A
This is used for GLM particle advection (see compute_wave_displacement!).

# Mean Mode (kₕ=0) Handling
For the horizontal mean mode (kₓ=kᵧ=0), the equation reduces to:
    LA = ∂/∂z(a(z) ∂A/∂z) = L⁺A    (since L⁺ = L when k_h=0)

With Neumann boundary conditions (∂A/∂z=0 at both boundaries), this operator
is **singular** - the constant function is in its null space. To select a unique
solution, we:
1. Fix a gauge (A[1]=0) to obtain a particular solution.
2. Remove the vertical mean of A (adds a constant null-space mode).

This yields a well-defined, mean-zero A for kₕ=0 while preserving the original
equation.

# Fortran Correspondence
This matches `A_solver_ybj_plus` in elliptic.f90 (PDF Eq. 33-35).
"""
function invert_L⁺A_to_A!(S::State, G::Grid, par, a::AbstractVector; workspace=nothing)
    nz = G.nz

    # Check if we need 2D decomposition transpose
    need_transpose = G.decomp !== nothing && hasfield(typeof(G.decomp), :pencil_z) && !z_is_local(S.L⁺A, G)

    if need_transpose
        _invert_L⁺A_to_A_2d!(S, G, par, a, workspace)
    else
        _invert_L⁺A_to_A_direct!(S, G, par, a)
    end

    return S
end

"""
Direct B→A solve for serial or 1D decomposition.
"""
function _invert_L⁺A_to_A_direct!(S::State, G::Grid, par, a::AbstractVector)
    nz = G.nz

    A_arr = parent(S.A)
    L⁺A_arr = parent(S.L⁺A)
    C_arr = parent(S.C)

    nz_local, nx_local, ny_local = size(A_arr)
    @assert nz_local == nz "Vertical dimension must be fully local"

    dₗ = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    dᵤ = zeros(eltype(a), nz)

    Δz = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δz² = Δz^2
    # NOTE: The RHS should just be B, not a*B. The a(z) profile is already
    # incorporated into the LHS operator matrix. Removed incorrect a_ell_coeff scaling.

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhsᵣ = zeros(eltype(a), nz)
    rhsᵢ = zeros(eltype(a), nz)
    solᵣ = zeros(eltype(a), nz)
    solᵢ = zeros(eltype(a), nz)

    for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global(i_local, 2, S.L⁺A)
        j_global = local_to_global(j_local, 3, S.L⁺A)

        kₓ = G.kx[i_global]
        kᵧ = G.ky[j_global]
        kₕ² = kₓ^2 + kᵧ^2

        # Special case: kₕ² = 0 (horizontal mean mode)
        # The operator a(z)∂²/∂z² with Neumann BCs is singular (constant null space).
        # We resolve this by fixing one point (A[1]=0), solving the tridiagonal
        # system, then removing the vertical mean (adds a constant null-space mode).
        if kₕ² == 0
            if nz == 1
                @inbounds begin
                    A_arr[1, i_local, j_local] = 0
                    C_arr[1, i_local, j_local] = 0
                end
                continue
            end

            fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

            # Row 1 (bottom): Neumann BC at z=-Lz
            d[1]  = -a[1]
            dᵤ[1] = a[1]

            # Interior rows
            @inbounds for k in 2:nz-1
                dₗ[k] = a[k]
                d[k]  = -(a[k+1] + a[k])
                dᵤ[k] = a[k+1]
            end

            # Row nz (top): Neumann BC at z=0
            dₗ[nz] = a[nz]
            d[nz]  = -a[nz]

            @inbounds for k in 1:nz
                rhsᵣ[k] = Δz² * real(L⁺A_arr[k, i_local, j_local])
                rhsᵢ[k] = Δz² * imag(L⁺A_arr[k, i_local, j_local])
            end

            # Fix gauge: A[1] = 0 for a nonsingular solve
            d[1] = 1
            dᵤ[1] = 0
            rhsᵣ[1] = 0
            rhsᵢ[1] = 0

            thomas_solve!(solᵣ, dₗ, d, dᵤ, rhsᵣ)
            thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

            # Remove vertical mean to select a unique null-space representative
            mean_val = zero(Complex{eltype(a)})
            @inbounds for k in 1:nz
                mean_val += solᵣ[k] + im*solᵢ[k]
            end
            mean_val /= nz

            @inbounds for k in 1:nz
                A_arr[k, i_local, j_local] = (solᵣ[k] + im*solᵢ[k]) - mean_val
            end

            @inbounds for k in 1:nz-1
                C_arr[k, i_local, j_local] = (A_arr[k+1, i_local, j_local] - A_arr[k, i_local, j_local]) / Δz
            end
            C_arr[nz, i_local, j_local] = 0
            continue
        end

        # Special case: nz == 1 (single-layer / 2D mode)
        # With only one vertical level, vertical derivative terms vanish.
        # The YBJ+ equation reduces to: -(kₕ²/4) A = B  →  A = -4B/kₕ²
        # C = ∂A/∂z = 0 (no vertical structure)
        if nz == 1
            @inbounds A_arr[1, i_local, j_local] = -4 * L⁺A_arr[1, i_local, j_local] / kₕ²
            @inbounds C_arr[1, i_local, j_local] = 0
            continue
        end

        fill!(dₗ, 0); fill!(d, 0); fill!(dᵤ, 0)

        # Row 1 (bottom): Neumann BC at z=-Lz
        d[1]  = -(a[1] + (kₕ²*Δz²)/4)
        dᵤ[1] = a[1]

        # Interior rows
        @inbounds for k in 2:nz-1
            dₗ[k] = a[k]
            d[k]  = -(a[k+1] + a[k] + (kₕ²*Δz²)/4)
            dᵤ[k] = a[k+1]
        end

        # Row nz (top): Neumann BC at z=0
        dₗ[nz] = a[nz]
        d[nz]  = -(a[nz] + (kₕ²*Δz²)/4)

        # Build RHS
        @inbounds for k in 1:nz
            # RHS is just Δz² * B (no a_ell_coeff - that was incorrect)
            rhsᵣ[k] = Δz² * real(L⁺A_arr[k, i_local, j_local])
            rhsᵢ[k] = Δz² * imag(L⁺A_arr[k, i_local, j_local])
        end

        thomas_solve!(solᵣ, dₗ, d, dᵤ, rhsᵣ)
        thomas_solve!(solᵢ, dₗ, d, dᵤ, rhsᵢ)

        @inbounds for k in 1:nz
            A_arr[k, i_local, j_local] = solᵣ[k] + im*solᵢ[k]
        end

        @inbounds for k in 1:nz-1
            C_arr[k, i_local, j_local] = (A_arr[k+1, i_local, j_local] - A_arr[k, i_local, j_local])/Δz
        end
        C_arr[nz, i_local, j_local] = 0
    end
end

"""
2D decomposition B→A solve with transposes.
"""
function _invert_L⁺A_to_A_2d!(S::State, G::Grid, par, a::AbstractVector, workspace)
    nz = G.nz

    # Allocate z-pencil workspace
    L⁺A_z = workspace !== nothing ? workspace.L⁺A_z : allocate_z_pencil(G, ComplexF64)
    A_z = workspace !== nothing ? workspace.A_z : allocate_z_pencil(G, ComplexF64)
    C_z = workspace !== nothing ? workspace.C_z : allocate_z_pencil(G, ComplexF64)

    # Transpose B to z-pencil
    transpose_to_z_pencil!(L⁺A_z, S.L⁺A, G)

    L⁺A_z_arr = parent(L⁺A_z)
    A_z_arr = parent(A_z)
    C_z_arr = parent(C_z)

    nz_local, nx_local, ny_local = size(L⁺A_z_arr)
    @assert nz_local == nz "After transpose, z must be fully local"

    dl = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    du = zeros(eltype(a), nz)

    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2

    # Pre-allocate work arrays outside loop to reduce GC pressure
    rhs_r = zeros(eltype(a), nz)
    rhs_i = zeros(eltype(a), nz)
    solr  = zeros(eltype(a), nz)
    soli  = zeros(eltype(a), nz)

    for j_local in 1:ny_local, i_local in 1:nx_local
        i_global = local_to_global_z(i_local, 2, G)
        j_global = local_to_global_z(j_local, 3, G)

        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2

        # Special case: kₕ² = 0 (horizontal mean mode)
        if kh2 == 0
            if nz == 1
                @inbounds begin
                    A_z_arr[1, i_local, j_local] = 0
                    C_z_arr[1, i_local, j_local] = 0
                end
                continue
            end

            fill!(dl, 0); fill!(d, 0); fill!(du, 0)

            # Row 1 (bottom): Neumann BC at z=-Lz
            d[1]  = -a[1]
            du[1] = a[1]

            # Interior rows
            @inbounds for k in 2:nz-1
                dl[k] = a[k]
                d[k]  = -(a[k+1] + a[k])
                du[k] = a[k+1]
            end

            # Row nz (top): Neumann BC at z=0
            dl[nz] = a[nz]
            d[nz]  = -a[nz]

            @inbounds for k in 1:nz
                rhs_r[k] = Δ2 * real(L⁺A_z_arr[k, i_local, j_local])
                rhs_i[k] = Δ2 * imag(L⁺A_z_arr[k, i_local, j_local])
            end

            # Fix gauge: A[1] = 0 for a nonsingular solve
            d[1] = 1
            du[1] = 0
            rhs_r[1] = 0
            rhs_i[1] = 0

            thomas_solve!(solr, dl, d, du, rhs_r)
            thomas_solve!(soli, dl, d, du, rhs_i)

            mean_val = zero(Complex{eltype(a)})
            @inbounds for k in 1:nz
                mean_val += solr[k] + im*soli[k]
            end
            mean_val /= nz

            @inbounds for k in 1:nz
                A_z_arr[k, i_local, j_local] = (solr[k] + im*soli[k]) - mean_val
            end

            @inbounds for k in 1:nz-1
                C_z_arr[k, i_local, j_local] = (A_z_arr[k+1, i_local, j_local] - A_z_arr[k, i_local, j_local]) / Δ
            end
            C_z_arr[nz, i_local, j_local] = 0
            continue
        end

        # Special case: nz == 1 (single-layer / 2D mode)
        if nz == 1
            @inbounds A_z_arr[1, i_local, j_local] = -4 * L⁺A_z_arr[1, i_local, j_local] / kh2
            @inbounds C_z_arr[1, i_local, j_local] = 0
            continue
        end

        fill!(dl, 0); fill!(d, 0); fill!(du, 0)

        # Row 1 (bottom): Neumann BC at z=-Lz
        d[1]  = -(a[1] + (kh2*Δ2)/4)
        du[1] = a[1]

        # Interior rows
        @inbounds for k in 2:nz-1
            dl[k] = a[k]
            d[k]  = -(a[k+1] + a[k] + (kh2*Δ2)/4)
            du[k] = a[k+1]
        end

        # Row nz (top): Neumann BC at z=0
        dl[nz] = a[nz]
        d[nz]  = -(a[nz] + (kh2*Δ2)/4)

        # Build RHS
        # RHS is just Δ² * B (no a_coeff - that was incorrect)
        # The a(z) profile is already in the LHS operator matrix
        @inbounds for k in 1:nz
            rhs_r[k] = Δ2 * real(L⁺A_z_arr[k, i_local, j_local])
            rhs_i[k] = Δ2 * imag(L⁺A_z_arr[k, i_local, j_local])
        end

        thomas_solve!(solr, dl, d, du, rhs_r)
        thomas_solve!(soli, dl, d, du, rhs_i)

        @inbounds for k in 1:nz
            A_z_arr[k, i_local, j_local] = solr[k] + im*soli[k]
        end

        @inbounds for k in 1:nz-1
            C_z_arr[k, i_local, j_local] = (A_z_arr[k+1, i_local, j_local] - A_z_arr[k, i_local, j_local])/Δ
        end
        C_z_arr[nz, i_local, j_local] = 0
    end

    # Transpose A and C back to xy-pencil
    transpose_to_xy_pencil!(S.A, A_z, G)
    transpose_to_xy_pencil!(S.C, C_z, G)
end

#=
================================================================================
                        THOMAS ALGORITHM
================================================================================
The Thomas algorithm is Gaussian elimination specialized for tridiagonal
systems. It has O(n) complexity vs O(n³) for general systems.

For the system:
    dl[k] * x[k-1] + d[k] * x[k] + du[k] * x[k+1] = b[k]

Steps:
1. Forward sweep: eliminate lower diagonal
2. Back substitution: solve for x from bottom to top
================================================================================
=#

"""
    thomas_solve!(x, dl, d, du, b)

In-place Thomas algorithm for solving tridiagonal systems Ax = b.

# Arguments
- `x`: Solution vector (modified in place)
- `dl`: Lower diagonal (length n, dl[1] unused)
- `d`: Main diagonal (length n)
- `du`: Upper diagonal (length n, du[n] unused)
- `b`: Right-hand side (length n)

# Complexity
O(n) operations (vs O(n³) for general LU decomposition)
"""
function thomas_solve!(x, dₗ, d, dᵤ, b)
    n = length(d)

    # Make working copies
    c = copy(dᵤ)
    d̃ = copy(d)
    x .= b

    # Singularity tolerance
    tol = sqrt(eps(eltype(d)))

    # Forward sweep
    if abs(d̃[1]) < tol
        error("Singular matrix in Thomas solver: d[1] ≈ 0 (|d[1]| = $(abs(d̃[1]))). " *
              "This may indicate ill-conditioned elliptic problem from N²≈0 or degenerate wavenumber.")
    end
    c[1] /= d̃[1]
    x[1] /= d̃[1]
    @inbounds for i in 2:n
        denom = d̃[i] - dₗ[i]*c[i-1]
        if abs(denom) < tol
            error("Singular matrix in Thomas solver at i=$i: |denom| = $(abs(denom)). " *
                  "This may indicate ill-conditioned system from N²≈0 or unstable stratification.")
        end
        c[i] /= denom
        x[i] = (x[i] - dₗ[i]*x[i-1]) / denom
    end

    # Back substitution
    @inbounds for i in n-1:-1:1
        x[i] -= c[i]*x[i+1]
    end

    return x
end

end # module

# Export elliptic solvers to main QGYBJplus module
using .Elliptic: invert_q_to_psi!, invert_helmholtz!, invert_L⁺A_to_A!
