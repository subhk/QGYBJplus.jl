#=
================================================================================
                    elliptic.jl - Vertical Elliptic Solvers
================================================================================

This file contains the tridiagonal solvers for the vertical elliptic problems
that arise in the QG-YBJ+ model. These are critical for:

1. STREAMFUNCTION INVERSION (q → ψ):
   Given the QG potential vorticity q, solve for the streamfunction ψ.

2. WAVE AMPLITUDE RECOVERY (B → A):
   Given the YBJ+ evolved field B = L⁺A, recover the true wave amplitude A.

3. GENERAL HELMHOLTZ PROBLEMS:
   For omega equation, buoyancy inversions, etc.

MATHEMATICAL BACKGROUND:
------------------------
The Boussinesq QG PV inversion relates q and ψ through:

    q = ∇ₕ²ψ + ∂/∂z[(f²/N²) ∂ψ/∂z]

For a general Helmholtz problem, the corresponding vertical operator can be
written in expanded form as:

    a(z) ∂²ψ/∂z² + b(z) ∂ψ/∂z - kₕ² ψ = q

where:
    a(z) = f²/N²(z) (elliptic coefficient)
    b(z) = ∂a/∂z when the flux-form operator is expanded
    kₕ² = kₓ² + kᵧ² (horizontal wavenumber squared)

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
Every horizontal mode is an independent column, so each solve is one
KernelAbstractions kernel with one work item per (kₓ, kᵧ) column. The matrix
is real, so the complex right-hand side is solved in one sweep; that is
operation-for-operation the separate real and imaginary solves.

The modified superdiagonal of every sweep is kept in a real `(nz, nx, ny)`
scratch array (`tridiagonal_scratch`), borrowed from the model workspace.

FORTRAN CORRESPONDENCE:
----------------------
- invert_q_to_psi! ↔ psi_solver (elliptic.f90)
- invert_B_to_A!   ↔ A_solver_ybj_plus (elliptic.f90)
- invert_helmholtz! ↔ helmholtzdouble (elliptic.f90)

================================================================================
=#

module Elliptic

using KernelAbstractions: @kernel, @index, @Const, get_backend
using ..QGYBJplus: RuntimeGeometry, ModelFields, launch!, on_backend,
                   vertical_spacing, tridiagonal_scratch, first_value

#=
================================================================================
                    COLUMN THOMAS SWEEP
================================================================================
A tridiagonal system is described by a small isbits object answering
`tridiagonal_row(system, r) -> (dl, d, du, b)` for rows r = 1..n. Row r is
stored at vertical level r + offset of the column (i, j).
================================================================================
=#

"""
Solve one column of a tridiagonal system with the Thomas algorithm, writing
the solution to `x[offset .+ (1:n), i, j]` and using `c` as scratch.
"""
@inline function _thomas_solve!(x, c, i, j, offset, n, system)
    _, d, du, b = tridiagonal_row(system, 1)
    @inbounds c[1 + offset, i, j] = du / d
    @inbounds x[1 + offset, i, j] = b / d
    for r in 2:n
        dl, d, du, b = tridiagonal_row(system, r)
        @inbounds begin
            denominator = d - dl * c[r - 1 + offset, i, j]
            c[r + offset, i, j] = du / denominator
            x[r + offset, i, j] = (b - dl * x[r - 1 + offset, i, j]) / denominator
        end
    end
    for r in (n - 1):-1:1
        @inbounds x[r + offset, i, j] -= c[r + offset, i, j] * x[r + 1 + offset, i, j]
    end
    return nothing
end

"""
`Δz² q = a[k] (ψ[k+1] - ψ[k]) - a[k-1] (ψ[k] - ψ[k-1]) - kₕ²Δz² ψ[k]`, with the
Neumann flux removed at the bottom and top faces. `shift = kₕ²Δz²` (YBJ⁺ uses
`kₕ²Δz²/4`), which must be positive: the kₕ = 0 columns are integrated
directly.
"""
struct VerticalFluxColumn{A, R, T}
    a::A
    rhs::R
    i::Int
    j::Int
    nz::Int
    Δz²::T
    shift::T
end

@inline function tridiagonal_row(s::VerticalFluxColumn, k)
    T = typeof(s.shift)
    nz = s.nz
    @inbounds if k == 1
        a_up = T(s.a[1])
        dl, d, du = zero(T), -(a_up + s.shift), a_up
    elseif k == nz
        a_down = T(s.a[nz - 1])
        dl, d, du = a_down, -(a_down + s.shift), zero(T)
    else
        a_up = T(s.a[k])
        a_down = T(s.a[k - 1])
        dl, d, du = a_down, -(a_up + a_down + s.shift), a_up
    end
    @inbounds b = s.Δz² * s.rhs[k, s.i, s.j]
    return dl, d, du, b
end

"""
Centered `helmholtzdouble` stencil with optional Neumann boundary fluxes.
`shift = αkₕ²Δz²`. With `gauge = true`, row 1 is replaced by `φ[1] = 0`.
"""
struct HelmholtzColumn{A, B, R, BB, TB, T}
    a::A
    b::B
    rhs::R
    bot_bc::BB
    top_bc::TB
    i::Int
    j::Int
    nz::Int
    Δz::T
    Δz²::T
    shift::T
    gauge::Bool
end

@inline function tridiagonal_row(s::HelmholtzColumn, k)
    a, b, Δz, nz = s.a, s.b, s.Δz, s.nz
    @inbounds rhs = s.Δz² * s.rhs[k, s.i, s.j]
    @inbounds if k == 1
        dl = zero(s.shift)
        d = -a[1] - b[1] * Δz / 2 - s.shift
        du = a[1] + b[1] * Δz / 2
        if s.bot_bc !== nothing
            rhs += (a[1] - b[1] * Δz / 2) * Δz * s.bot_bc[s.i, s.j]
        end
        s.gauge && return zero(s.shift), one(s.shift), zero(s.shift), zero(rhs)
    elseif k == nz
        dl = a[nz] - b[nz] * Δz / 2
        d = -a[nz] + b[nz] * Δz / 2 - s.shift
        du = zero(s.shift)
        if s.top_bc !== nothing
            rhs -= (a[nz] + b[nz] * Δz / 2) * Δz * s.top_bc[s.i, s.j]
        end
    else
        dl = a[k] - b[k] * Δz / 2
        d = -2 * a[k] - s.shift
        du = a[k] + b[k] * Δz / 2
    end
    return dl, d, du, rhs
end

#=
================================================================================
                    STREAMFUNCTION INVERSION: q → ψ
================================================================================
This is the core elliptic inversion that relates QGPV to streamfunction.

PHYSICS:
    q = ∇²ψ + ∂/∂z[(f²/N²) ∂ψ/∂z]
      = ∇²ψ + ∂/∂z[a_ell ∂ψ/∂z]

where a_ell = f²/N² is the "elliptic coefficient" that varies with
stratification.

The discrete equation for interior points is:
    (a[k]/dz²)(ψ[k+1] - 2ψ[k] + ψ[k-1]) - kₕ² ψ[k] = q[k]

with Neumann BCs (ψ_z = 0) modifying the boundary stencils.
================================================================================
=#

"""
    invert_q_to_psi!(S, G; a, workspace=nothing)

Invert spectral QGPV `q(kx,ky,z)` to obtain streamfunction `ψ(kx,ky,z)`.

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve the vertical ODE:

    ∂/∂z[a(z) ∂ψ/∂z] - kₕ² ψ = q

with Neumann boundary conditions ψ_z = 0 at top and bottom.

# Arguments
- `S::ModelFields`: ModelFields struct containing `q` (input) and `psi` (output)
- `G::RuntimeGeometry`: RuntimeGeometry struct with wavenumbers and vertical coordinates
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = f²/N²(z), length
  nz, sampled at upper cell faces `grid.z_faces[2:end]`
- `workspace`: Optional model workspace supplying the Thomas scratch

# Implementation Details
The discrete system is tridiagonal with structure:
- Diagonal: d[k] = -(a[k] + a[k-1]) - kₕ² dz²
- Upper diagonal: du[k] = a[k]
- Lower diagonal: dl[k] = a[k-1]

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

This implementation retains the compatible, vertically varying part of the
horizontal-mean PV and fixes the null-space gauge with `sum(ψ) = 0`. If the
depth mean of q is nonzero, that incompatible constant component is projected
out (with a warning); no Neumann solution exists for it.

# Example
```julia
a_vec = a_ell_from_N2(N2_face_profile, FPlane(f=1e-4))
invert_q_to_psi!(state, grid; a=a_vec)
```
"""
function invert_q_to_psi!(S::ModelFields, G::RuntimeGeometry; a::AbstractVector,
    workspace=nothing)
    nz = G.nz
    @assert length(a) == nz "a must have length nz=$nz"
    _invert_q_to_psi!(S.psi, S.q, G, a, workspace)
    return S
end

"""
Warn when the horizontal-mean PV column has a depth mean beyond round-off.
Transforms and products leave a depth mean of order eps times the largest PV
coefficient, so the mean is measured against the whole field.
"""
function _check_mean_pv_compatibility(q)
    nz = size(q, 1)
    # Column (kₓ, kᵧ) = (0, 0) occupies the first nz linear indices.
    column = Vector{eltype(q)}(undef, nz)
    copyto!(column, 1, q, 1, nz)
    q̄ = sum(column) / nz
    compatibility_tolerance = 100 * eps(real(eltype(q))) * maximum(abs, q)
    if abs(q̄) > compatibility_tolerance
        @warn "invert_q_to_psi!: the horizontal-mean q column violates " *
              "the Neumann compatibility condition (depth mean=$(q̄)); " *
              "projecting out that vertically constant component." maxlog=1
    end
    return nothing
end

# The compatibility check reads from the device, so it runs for the public
# entry points; the time stepper skips it, as the dynamics conserve the depth
# mean of the horizontal-mean PV up to round-off.
function _invert_q_to_psi!(psi, q, G::RuntimeGeometry, a, workspace;
                           check_compatibility::Bool=true)
    nz, nx, ny = size(psi)
    nz == G.nz || throw(DimensionMismatch("ψ has $nz levels; the grid has $(G.nz)"))
    length(a) == nz || throw(DimensionMismatch("a must have length nz=$nz"))
    backend = get_backend(psi)
    check_compatibility && _check_mean_pv_compatibility(q)
    Δz = vertical_spacing(G)
    launch!(_invert_q_to_psi_kernel!, backend, (nx, ny),
            psi, tridiagonal_scratch(workspace, psi), q, G.kx, G.ky,
            on_backend(backend, a), Δz, Δz^2, nz)
    return psi
end

@kernel function _invert_q_to_psi_kernel!(ψ, c, @Const(q), @Const(kx),
                                          @Const(ky), @Const(a), Δz, Δz², nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2

    if kₕ² == 0
        # The constant part of ψ is a null mode, but a vertically varying,
        # mean-zero q column has a valid solution and must not be discarded.
        # Project only the incompatible depth mean of q, integrate the
        # discrete vertical flux, and impose the gauge sum(ψ)=0.
        q̄ = zero(eltype(q))
        for k in 1:nz
            @inbounds q̄ += q[k, i, j]
        end
        q̄ /= nz

        @inbounds ψ[1, i, j] = zero(eltype(ψ))
        flux = zero(eltype(q))
        for k in 1:(nz - 1)
            @inbounds begin
                flux += q[k, i, j] - q̄
                ψ[k + 1, i, j] = ψ[k, i, j] + Δz² * flux / a[k]
            end
        end

        ψ̄ = zero(eltype(ψ))
        for k in 1:nz
            @inbounds ψ̄ += ψ[k, i, j]
        end
        ψ̄ /= nz
        for k in 1:nz
            @inbounds ψ[k, i, j] -= ψ̄
        end
    elseif nz == 1
        # With one level and Neumann BCs the vertical terms vanish: ψ = -q/kₕ².
        @inbounds ψ[1, i, j] = -q[1, i, j] / kₕ²
    else
        system = VerticalFluxColumn(a, q, i, j, nz, Δz², kₕ² * Δz²)
        _thomas_solve!(ψ, c, i, j, 0, nz, system)
    end
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
    invert_helmholtz!(dstk, rhs, G; a, b=zeros, scale_kh2=1.0, bot_bc=nothing, top_bc=nothing, workspace=nothing)

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
- `G::RuntimeGeometry`: RuntimeGeometry struct
- `a::AbstractVector`: Second derivative coefficient a(z), length nz
- `b::AbstractVector`: First derivative coefficient b(z), length nz (default zeros)
- `scale_kh2::Real`: Multiplier α for kₕ² term (default 1.0)
- `bot_bc`, `top_bc`: Optional boundary flux arrays (nx, ny) for non-zero Neumann BCs
- `workspace`: Optional model workspace supplying the Thomas scratch

# Singular columns
Where `scale_kh2 × kₕ² = 0` — the horizontal-mean column, and every column when
`scale_kh2 = 0` — the Neumann problem is singular: constants are in its null
space, and a solution exists only for a compatible right-hand side. Such a
column is solved with the bottom-row equation replaced by `φ[1] = 0` (that row
is redundant when the right-hand side is compatible), and its vertical mean is
then removed. With `nz = 1` a singular column is set to zero, with a warning
when its right-hand side is nonzero.

# Fortran Correspondence
This matches `helmholtzdouble` in elliptic.f90 exactly.
"""
function invert_helmholtz!(dstk, rhs, G::RuntimeGeometry;
                           a::AbstractVector,
                           b::AbstractVector=zeros(eltype(a), length(a)),
                           scale_kh2::Real=1.0,
                           bot_bc=nothing,
                           top_bc=nothing,
                           workspace=nothing)
    nz = G.nz
    nz_local, nx, ny = size(dstk)
    @assert nz_local == nz "Vertical dimension must match the grid"
    @assert length(a) == nz "a must have length nz=$nz"
    @assert length(b) == nz "b must have length nz=$nz"
    for (bc, name) in ((bot_bc, "bot_bc"), (top_bc, "top_bc"))
        bc === nothing || size(bc) == (nx, ny) || throw(DimensionMismatch(
            "$name must be an (nx, ny) = ($nx, $ny) array (got size $(size(bc)))"))
    end

    backend = get_backend(dstk)
    # The column scalars share the grid's precision.
    α = convert(typeof(vertical_spacing(G)), scale_kh2)
    if nz == 1
        if bot_bc !== nothing || top_bc !== nothing
            @warn "Helmholtz solve with nz=1 ignores boundary conditions" maxlog=1
        end
        # A singular column (α kₕ² = 0) cannot be inverted: its right-hand
        # side is discarded, so say so when it is nonzero. That is the
        # horizontal-mean column, or every column when α = 0.
        scale = maximum(abs, rhs)
        discarded = iszero(α) ? scale > 0 :
                    abs(first_value(rhs)) > sqrt(eps(real(eltype(rhs)))) * scale
        if discarded
            @warn "Helmholtz solve with nz=1 has scale_kh2*kₕ² = 0 and a nonzero " *
                  "right-hand side; setting φ=0 for those modes." maxlog=1
        end
        launch!(_helmholtz_single_level_kernel!, backend, (nx, ny),
                dstk, rhs, G.kx, G.ky, α)
        return dstk
    end

    Δz = vertical_spacing(G)
    in_precision(profile) = eltype(profile) === typeof(Δz) ? profile : typeof(Δz).(profile)
    launch!(_helmholtz_kernel!, backend, (nx, ny),
            dstk, tridiagonal_scratch(workspace, dstk), rhs, G.kx, G.ky,
            on_backend(backend, in_precision(a)), on_backend(backend, in_precision(b)),
            on_backend(backend, bot_bc), on_backend(backend, top_bc),
            Δz, Δz^2, α, nz)
    return dstk
end

# `rhs` may alias `φ` in the Helmholtz kernels: every column reads its own
# right-hand side entries before overwriting them, so it is not marked @Const.
@kernel function _helmholtz_single_level_kernel!(φ, rhs, @Const(kx),
                                                 @Const(ky), α)
    i, j = @index(Global, NTuple)
    @inbounds begin
        denominator = α * (kx[i]^2 + ky[j]^2)
        φ[1, i, j] = iszero(denominator) ? zero(eltype(φ)) :
                     -rhs[1, i, j] / denominator
    end
end

@kernel function _helmholtz_kernel!(φ, c, rhs, @Const(kx), @Const(ky),
                                    @Const(a), @Const(b), bot_bc, top_bc,
                                    Δz, Δz², α, nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2
    singular = iszero(α * kₕ²)
    system = HelmholtzColumn(a, b, rhs, bot_bc, top_bc, i, j, nz,
                             Δz, Δz², α * kₕ² * Δz², singular)
    _thomas_solve!(φ, c, i, j, 0, nz, system)
    if singular
        # Fix the constant null mode with a zero vertical mean.
        mean_value = zero(eltype(φ))
        for k in 1:nz
            @inbounds mean_value += φ[k, i, j]
        end
        mean_value /= nz
        for k in 1:nz
            @inbounds φ[k, i, j] -= mean_value
        end
    end
end

#=
================================================================================
                    YBJ+ WAVE INVERSION: B → A
================================================================================
In the YBJ+ formulation, the prognostic variable is B = L⁺A, where L⁺ is an
elliptic operator. After time stepping B, we need to recover A for computing
wave-related quantities.

The operator L⁺ is:
    L⁺A = ∂/∂z[a(z) ∂A/∂z] - (kₕ²/4) A

So inverting gives us A from B. We also compute C = A_z for use in wave
feedback and vertical velocity calculations.
================================================================================
=#

"""
    invert_B_to_A!(S, G, a; workspace=nothing)

YBJ+ wave amplitude recovery: solve for A given B = L⁺A.

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve:

    ∂/∂z[a(z) ∂A/∂z] - (kₕ²/4) A = B

with Neumann boundary conditions A_z = 0 at top and bottom.

# Arguments
- `S::ModelFields`: ModelFields containing `B` (input), `A` and `C` (output)
- `G::RuntimeGeometry`: RuntimeGeometry struct
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = f²/N²(z), sampled
  at upper cell faces `grid.z_faces[2:end]`
- `workspace`: Optional model workspace supplying the Thomas scratch

# Output Fields
- `S.A`: Recovered wave amplitude A
- `S.C`: Vertical derivative C = ∂A/∂z (for wave velocity computation)

# Mean Mode (kₕ=0) Handling
For the horizontal mean mode (kₓ=kᵧ=0), the equation reduces to:
    ∂/∂z[a(z) ∂A/∂z] = B

With Neumann boundary conditions (∂A/∂z=0 at both boundaries), this operator
is **singular**: the constant function is in its null space, and only a B
with zero depth sum is in its range. The solve therefore:
1. Projects out the depth mean of B, which no A can produce (the model also
   removes it from B itself).
2. Integrates the discrete vertical flux upward from the bottom.
3. Removes the vertical mean of A, the null-space gauge.

This yields the mean-zero A whose `L A` is the depth-varying part of B.

# Fortran Correspondence
This matches `A_solver_ybj_plus` in elliptic.f90.
"""
function invert_B_to_A!(S::ModelFields, G::RuntimeGeometry, a::AbstractVector;
    workspace=nothing)
    nz = G.nz
    @assert length(a) == nz "a must have length nz=$nz"
    _invert_B_to_A!(S.A, S.C, S.B, G, a, workspace)
    return S
end

"""
Recover the wave amplitude `A` and its vertical derivative `C = A_z` from the
evolved field `B = L⁺A`.
"""
function _invert_B_to_A!(A, C, B, G::RuntimeGeometry, a::AbstractVector, workspace)
    nz, nx, ny = size(A)
    nz == G.nz || throw(DimensionMismatch("A has $nz levels; the grid has $(G.nz)"))
    backend = get_backend(A)
    Δz = vertical_spacing(G)
    launch!(_invert_B_to_A_kernel!, backend, (nx, ny),
            A, C, tridiagonal_scratch(workspace, A), B, G.kx, G.ky,
            on_backend(backend, a), Δz, Δz^2, nz)
    return A, C
end

@kernel function _invert_B_to_A_kernel!(A, C, c, @Const(B), @Const(kx),
                                        @Const(ky), @Const(a), Δz, Δz², nz)
    i, j = @index(Global, NTuple)
    @inbounds kₕ² = kx[i]^2 + ky[j]^2

    if nz == 1
        # With one level the vertical terms vanish: -(kₕ²/4) A = B, and C = 0.
        # The mean mode keeps the A = 0 gauge.
        @inbounds A[1, i, j] = kₕ² == 0 ? zero(eltype(A)) : -4 * B[1, i, j] / kₕ²
        @inbounds C[1, i, j] = zero(eltype(C))
    else
        if kₕ² == 0
            # ∂z(a ∂z) with Neumann BCs is singular: its range has zero depth
            # sum. Project out the depth mean of B, integrate the discrete
            # vertical flux, and fix the gauge with sum(A) = 0. (Pinning A[1]
            # and dropping a row would instead put the whole depth mean of B
            # into the bottom cell.)
            B̄ = zero(eltype(B))
            for k in 1:nz
                @inbounds B̄ += B[k, i, j]
            end
            B̄ /= nz
            @inbounds A[1, i, j] = zero(eltype(A))
            flux = zero(eltype(B))
            for k in 1:(nz - 1)
                @inbounds begin
                    flux += B[k, i, j] - B̄
                    A[k + 1, i, j] = A[k, i, j] + Δz² * flux / a[k]
                end
            end
            mean_value = zero(eltype(A))
            for k in 1:nz
                @inbounds mean_value += A[k, i, j]
            end
            mean_value /= nz
            for k in 1:nz
                @inbounds A[k, i, j] -= mean_value
            end
        else
            system = VerticalFluxColumn(a, B, i, j, nz, Δz², (kₕ² * Δz²) / 4)
            _thomas_solve!(A, c, i, j, 0, nz, system)
        end

        for k in 1:(nz - 1)
            @inbounds C[k, i, j] = (A[k + 1, i, j] - A[k, i, j]) / Δz
        end
        @inbounds C[nz, i, j] = zero(eltype(C))
    end
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

The kernels above run this per column on the device; `thomas_solve!` below is
the same algorithm for a single host vector system.
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
thomas_solve!(x, dₗ, d, dᵤ, b) = thomas_solve!(x, dₗ, d, dᵤ, b, similar(dᵤ))

"""
    thomas_solve!(x, dₗ, d, dᵤ, b, c)

Thomas algorithm with caller-supplied scratch. `c` is overwritten with the
modified superdiagonal; it must be a distinct vector of the same length as `dᵤ`.
"""
function thomas_solve!(x, dₗ, d, dᵤ, b, c)
    n = length(d)
    length(c) == length(dᵤ) ||
        throw(DimensionMismatch("Thomas scratch must match the superdiagonal"))

    copyto!(c, dᵤ)
    x .= b

    # Singularity tolerance
    tol = sqrt(eps(eltype(d)))

    # Forward sweep
    if abs(d[1]) < tol
        error("Singular matrix in Thomas solver: d[1] ≈ 0 (|d[1]| = $(abs(d[1]))). " *
              "This may indicate ill-conditioned elliptic problem from N²≈0 or degenerate wavenumber.")
    end
    c[1] /= d[1]
    x[1] /= d[1]
    @inbounds for i in 2:n
        denom = d[i] - dₗ[i]*c[i-1]
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
using .Elliptic: invert_q_to_psi!, invert_helmholtz!, invert_B_to_A!
