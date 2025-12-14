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
The QG PV inversion relates q and ψ through:

    q = ∇²ψ + (f²/N²) ∂²ψ/∂z²

Rearranging for the vertical operator (in spectral space):

    a(z) ∂²ψ/∂z² + b(z) ∂ψ/∂z - kₕ² ψ = q

where:
    a(z) = f²/N²(z) = Bu/N² (in nondimensional units)
    b(z) = coefficient from variable N² (often zero for constant density)
    kₕ² = kₓ² + kᵧ² (horizontal wavenumber squared)

This is solved independently for each (kₓ, kᵧ) mode using a tridiagonal solver.

BOUNDARY CONDITIONS:
-------------------
The standard boundary conditions are:
    ψ_z = 0 at z = 0 (bottom)
    ψ_z = 0 at z = H (top)

These are Neumann conditions corresponding to no buoyancy flux through
the boundaries.

NUMERICAL METHOD:
-----------------
Second-order finite differences in z lead to a tridiagonal system:

    dl[k] * x[k-1] + d[k] * x[k] + du[k] * x[k+1] = rhs[k]

Solved using the Thomas algorithm (Gaussian elimination for tridiagonal).

FORTRAN CORRESPONDENCE:
----------------------
- invert_q_to_psi! ↔ psi_solver (elliptic.f90)
- invert_B_to_A!   ↔ A_solver_ybj_plus (elliptic.f90)
- invert_helmholtz! ↔ helmholtzdouble (elliptic.f90)

================================================================================
=#

module Elliptic

using ..QGYBJ: Grid, State, get_kx, get_ky, get_local_dims, local_to_global
const PARENT = Base.parentmodule(@__MODULE__)

#=
================================================================================
                    STREAMFUNCTION INVERSION: q → ψ
================================================================================
This is the core elliptic inversion that relates QGPV to streamfunction.

PHYSICS:
    q = ∇²ψ + (1/a_ell) ∂²ψ/∂z²

where a_ell = Bu/N² is the "elliptic coefficient" that varies with
stratification.

The discrete equation for interior points is:
    (a[k]/dz²)(ψ[k+1] - 2ψ[k] + ψ[k-1]) - kₕ² ψ[k] = q[k]

with Neumann BCs (ψ_z = 0) modifying the boundary stencils.
================================================================================
=#

"""
    invert_q_to_psi!(S, G; a, par=nothing)

Invert spectral QGPV `q(kx,ky,z)` to obtain streamfunction `ψ(kx,ky,z)`.

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve the vertical ODE:

    a(z) ∂²ψ/∂z² - kₕ² ψ = q

with Neumann boundary conditions ψ_z = 0 at top and bottom.

# Arguments
- `S::State`: State struct containing `q` (input) and `psi` (output)
- `G::Grid`: Grid struct with wavenumbers and vertical coordinates
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = Bu/N²(z), length nz
- `par`: Optional QGParams for density weighting (defaults to unity weights)

# Implementation Details
The discrete system is tridiagonal with structure:
- Diagonal: d[k] = -(a[k] + a[k-1])/r_st[k] - kₕ² dz²
- Upper diagonal: du[k] = a[k]/r_st[k]
- Lower diagonal: dl[k] = a[k-1]/r_st[k]

where r_ut, r_st are density weights (unity for Boussinesq).

# Fortran Correspondence
This matches `psi_solver` in elliptic.f90.

# Example
```julia
a_vec = a_ell_ut(params, G)  # Compute a_ell = Bu/N²
invert_q_to_psi!(state, grid; a=a_vec)
```
"""
function invert_q_to_psi!(S::State, G::Grid; a::AbstractVector, par=nothing)
    nz = G.nz
    @assert length(a) == nz "a must have length nz=$nz"

    # Get underlying arrays (works for both Array and PencilArray)
    ψ_arr = parent(S.psi)   # Output: streamfunction
    q_arr = parent(S.q)     # Input: QGPV

    # Get local dimensions
    # With 1D decomposition in y: x and z are fully local, y is distributed
    nx_local, ny_local, nz_local = size(ψ_arr)

    # Verify z is fully local (required for vertical tridiagonal solve)
    @assert nz_local == nz "Vertical dimension must be fully local (nz_local=$nz_local, nz=$nz)"

    # Tridiagonal matrix diagonals (reused for each wavenumber)
    dl = zeros(eltype(a), nz)   # Lower diagonal
    d  = zeros(eltype(a), nz)   # Main diagonal
    du = zeros(eltype(a), nz)   # Upper diagonal

    # Vertical grid spacing
    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2

    #= Density weights for variable-density formulation
    In the Fortran code, these are rho_ut and rho_st arrays.
    For Boussinesq (constant density), they are all ones. =#
    r_ut = if par === nothing
        ones(eltype(a), nz)
    else
        isdefined(PARENT, :rho_ut) ? PARENT.rho_ut(par, G) : ones(eltype(a), nz)
    end
    r_st = if par === nothing
        ones(eltype(a), nz)
    else
        isdefined(PARENT, :rho_st) ? PARENT.rho_st(par, G) : ones(eltype(a), nz)
    end

    # Loop over all LOCAL horizontal wavenumbers (using local indices)
    for j_local in 1:ny_local, i_local in 1:nx_local
        # Get global indices for wavenumber lookup
        i_global = local_to_global(i_local, 1, G)
        j_global = local_to_global(j_local, 2, G)

        kx_val = G.kx[i_global]
        ky_val = G.ky[j_global]
        kh2 = kx_val^2 + ky_val^2   # Horizontal wavenumber squared: kx² + ky²

        # Special case: kh² = 0 (horizontal mean mode)
        # The equation becomes singular; set ψ = 0 (arbitrary constant)
        if kh2 == 0
            @inbounds for k in 1:nz
                ψ_arr[i_local, j_local, k] = 0
            end
            continue
        end

        # Build tridiagonal matrix for this (kx, ky)
        fill!(dl, 0); fill!(d, 0); fill!(du, 0)

        #= Bottom boundary (k=1): Neumann condition ψ_z = 0
        One-sided difference: (ψ[2] - ψ[1])/dz = 0
        This modifies the stencil to only use ψ[1] and ψ[2] =#
        d[1]  = -( (r_ut[1]*a[1]) / r_st[1] + kh2*Δ2 )
        du[1] =   (r_ut[1]*a[1]) / r_st[1]

        # Interior points (k = 2, ..., nz-1): standard central differences
        @inbounds for k in 2:nz-1
            dl[k] = (r_ut[k-1]*a[k-1]) / r_st[k]
            d[k]  = -( ((r_ut[k]*a[k] + r_ut[k-1]*a[k-1]) / r_st[k]) + kh2*Δ2 )
            du[k] = (r_ut[k]*a[k]) / r_st[k]
        end

        #= Top boundary (k=nz): Neumann condition ψ_z = 0
        One-sided difference from below =#
        dl[nz] = (r_ut[nz-1]*a[nz-1]) / r_st[nz]
        d[nz]  = -( (r_ut[nz-1]*a[nz-1]) / r_st[nz] + kh2*Δ2 )

        #= Solve for real and imaginary parts separately
        (LAPACK's tridiagonal solver works on real arrays) =#

        # Real part - z is fully local
        rhs = zeros(eltype(a), nz)
        @inbounds for k in 1:nz
            rhs[k] = Δ2 * real(q_arr[i_local, j_local, k])
        end
        solr = copy(rhs)
        thomas_solve!(solr, dl, d, du, rhs)

        # Imaginary part
        rhs_i = zeros(eltype(a), nz)
        @inbounds for k in 1:nz
            rhs_i[k] = Δ2 * imag(q_arr[i_local, j_local, k])
        end
        soli = copy(rhs_i)
        thomas_solve!(soli, dl, d, du, rhs_i)

        # Combine into complex solution
        @inbounds for k in 1:nz
            ψ_arr[i_local, j_local, k] = solr[k] + im*soli[k]
        end
    end

    return S
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
    invert_helmholtz!(dstk, rhs, G, par; a, b=zeros, scale_kh2=1.0, bot_bc=nothing, top_bc=nothing)

General vertical Helmholtz inversion for each horizontal wavenumber.

# Mathematical Problem
Solve the ODE:

    a(z) ∂²φ/∂z² + b(z) ∂φ/∂z - scale_kh2 × kₕ² φ = rhs

# Arguments
- `dstk`: Output array (nx, ny, nz) for solution φ
- `rhs`: Right-hand side array (nx, ny, nz)
- `G::Grid`: Grid struct
- `par`: QGParams for density profiles
- `a::AbstractVector`: Second derivative coefficient (length nz)
- `b::AbstractVector`: First derivative coefficient (length nz), default zeros
- `scale_kh2::Real`: Multiplier for kₕ² term (default 1.0)
- `bot_bc`, `top_bc`: Optional boundary flux arrays (nx, ny)

# Fortran Correspondence
This matches `helmholtzdouble` in elliptic.f90.

# Example
```julia
# Solve: ∂²w/∂z² - (N²/f²)kₕ² w = rhs
a_vec = ones(nz) ./ params.Bu  # a = f²/N² = 1/(Bu*N²)
invert_helmholtz!(w_k, rhs_k, grid, params; a=a_vec)
```
"""
function invert_helmholtz!(dstk, rhs, G::Grid, par;
                           a::AbstractVector,
                           b::AbstractVector=zeros(eltype(a), length(a)),
                           scale_kh2::Real=1.0,
                           bot_bc=nothing,
                           top_bc=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    @assert size(dstk) == (nx, ny, nz) "dstk must have size ($nx, $ny, $nz)"
    @assert size(rhs)  == (nx, ny, nz) "rhs must have size ($nx, $ny, $nz)"
    @assert length(a) == nz "a must have length nz=$nz"
    @assert length(b) == nz "b must have length nz=$nz"

    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2

    # Density-like weights
    r_ut = rho_ut(par, G)
    r_st = rho_st(par, G)

    dl = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    du = zeros(eltype(a), nz)

    for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]

        # Build tridiagonals including first derivative b-terms
        fill!(dl, 0); fill!(d, 0); fill!(du, 0)

        # Bottom level (k=1)
        α1 = r_ut[1]/r_st[1]
        d[1]  = -( α1*a[1] + 0.5*α1*b[1]*Δ + scale_kh2*kh2*Δ2 )
        du[1] =   α1*a[1] + 0.5*α1*b[1]*Δ

        # Interior levels
        @inbounds for k in 2:nz-1
            αk   = r_ut[k]/r_st[k]
            αkm1 = r_ut[k-1]/r_st[k]
            dl[k] = αkm1*a[k-1] - 0.5*αkm1*b[k-1]*Δ
            d[k]  = -( 2*αk*a[k] + scale_kh2*kh2*Δ2 )
            du[k] =  αk*a[k] + 0.5*αk*b[k]*Δ
        end

        # Top level (k=nz)
        αn = r_ut[nz-1]/r_st[nz]
        dl[nz] = αn*a[nz-1] - 0.5*αn*b[nz-1]*Δ
        d[nz]  = -( αn*a[nz-1] - 0.5*αn*b[nz-1]*Δ + scale_kh2*kh2*Δ2 )

        # Prepare RHS (scale by dz²)
        rhsR = similar(view(rhs, i, j, :), eltype(a))
        rhsI = similar(rhsR)
        @inbounds for k in 1:nz
            rhsR[k] = Δ2 * real(rhs[i,j,k])
            rhsI[k] = Δ2 * imag(rhs[i,j,k])
        end

        #= Boundary flux adjustments (for non-zero flux BCs)
        These add contributions from bottom/top buoyancy terms =#
        if bot_bc !== nothing
            rhsR[1] += (α1*(a[1] - 0.5*b[1]*Δ)) * Δ * real(bot_bc[i,j])
            rhsI[1] += (α1*(a[1] - 0.5*b[1]*Δ)) * Δ * imag(bot_bc[i,j])
        end
        if top_bc !== nothing
            rhsR[nz] -= (αn*(a[nz-1] + 0.5*b[nz-1]*Δ)) * Δ * real(top_bc[i,j])
            rhsI[nz] -= (αn*(a[nz-1] + 0.5*b[nz-1]*Δ)) * Δ * imag(top_bc[i,j])
        end

        # Solve real and imaginary parts
        solR = copy(rhsR)
        solI = copy(rhsI)
        thomas_solve!(solR, dl, d, du, rhsR)
        thomas_solve!(solI, dl, d, du, rhsI)

        @inbounds for k in 1:nz
            dstk[i,j,k] = solR[k] + im*solI[k]
        end
    end

    return dstk
end

#=
================================================================================
                    YBJ+ WAVE INVERSION: B → A
================================================================================
In the YBJ+ formulation, the prognostic variable is B = L⁺A, where L⁺ is an
elliptic operator. After time stepping B, we need to recover A for computing
wave-related quantities.

The operator L⁺ is:
    L⁺A = a(z) ∂²A/∂z² - (kₕ²/4) A

So inverting gives us A from B. We also compute C = A_z for use in wave
feedback and vertical velocity calculations.
================================================================================
=#

"""
    invert_B_to_A!(S, G, par, a)

YBJ+ wave amplitude recovery: solve for A given B = L⁺A.

# Mathematical Problem
For each horizontal wavenumber (kₓ, kᵧ), solve:

    a(z) ∂²A/∂z² - (kₕ²/4) A = B

with Neumann boundary conditions A_z = 0 at top and bottom.

# Arguments
- `S::State`: State containing `B` (input), `A` and `C` (output)
- `G::Grid`: Grid struct
- `par`: QGParams (for Burger number Bu)
- `a::AbstractVector`: Elliptic coefficient a_ell(z) = Bu/N²(z)

# Output Fields
- `S.A`: Recovered wave amplitude A
- `S.C`: Vertical derivative C = ∂A/∂z (for wave velocity computation)

# Notes
- The kₕ²/4 factor is specific to YBJ+ (vs kₕ² in normal YBJ)
- RHS is multiplied by Bu as in Fortran: rhs = dz² × Bu × B
- Top boundary C value is set to zero (A_z = 0)

# Fortran Correspondence
This matches `A_solver_ybj_plus` in elliptic.f90.

# Example
```julia
a_vec = a_ell_ut(params, grid)
invert_B_to_A!(state, grid, params, a_vec)
# Now state.A contains wave amplitude, state.C contains A_z
```
"""
function invert_B_to_A!(S::State, G::Grid, par, a::AbstractVector)
    nx, ny, nz = G.nx, G.ny, G.nz

    A = S.A   # Output: wave amplitude
    B = S.B   # Input: evolved YBJ+ field
    C = S.C   # Output: A_z (vertical derivative)

    # Tridiagonal diagonals
    dl = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    du = zeros(eltype(a), nz)

    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2

    # Density weights
    r_ut = isdefined(PARENT, :rho_ut) ? PARENT.rho_ut(par, G) : ones(eltype(a), nz)
    r_st = isdefined(PARENT, :rho_st) ? PARENT.rho_st(par, G) : ones(eltype(a), nz)

    for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]

        # Handle kh² = 0 mode
        if kh2 == 0
            @inbounds A[i,j,:] .= 0
            @inbounds C[i,j,:] .= 0
            continue
        end

        fill!(dl, 0); fill!(d, 0); fill!(du, 0)

        #= YBJ+ operator: L⁺A = (1/N²)∂²A/∂z² - (kₕ²/4)A
        Note the kₕ²/4 factor (vs kₕ² in psi inversion) =#

        # Bottom (k=1)
        d[1]  = -( (r_ut[1]*a[1]) / r_st[1] + (kh2*Δ2)/4 )
        du[1] =   (r_ut[1]*a[1]) / r_st[1]

        # Interior
        @inbounds for k in 2:nz-1
            dl[k] = (r_ut[k-1]*a[k-1]) / r_st[k]
            d[k]  = -( ((r_ut[k]*a[k] + r_ut[k-1]*a[k-1]) / r_st[k]) + (kh2*Δ2)/4 )
            du[k] = (r_ut[k]*a[k]) / r_st[k]
        end

        # Top (k=nz)
        dl[nz] = (r_ut[nz-1]*a[nz-1]) / r_st[nz]
        d[nz]  = -( (r_ut[nz-1]*a[nz-1]) / r_st[nz] + (kh2*Δ2)/4 )

        #= RHS = dz² × Bu × B
        The Bu factor comes from the nondimensionalization =#
        rhs_r = similar(dl)
        rhs_i = similar(dl)
        @inbounds for k in 1:nz
            rhs_r[k] = Δ2 * par.Bu * real(B[i,j,k])
            rhs_i[k] = Δ2 * par.Bu * imag(B[i,j,k])
        end

        # Solve for A
        solr = copy(rhs_r)
        soli = copy(rhs_i)
        thomas_solve!(solr, dl, d, du, rhs_r)
        thomas_solve!(soli, dl, d, du, rhs_i)

        @inbounds for k in 1:nz
            A[i,j,k] = solr[k] + im*soli[k]
        end

        #= Compute C = A_z using forward differences
        This is used for wave vertical velocity computation =#
        @inbounds for k in 1:nz-1
            C[i,j,k] = (A[i,j,k+1] - A[i,j,k])/Δ
        end
        # Top boundary: A_z = 0
        C[i,j,nz] = 0
    end

    return S
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

# Algorithm
The Thomas algorithm is a specialized form of Gaussian elimination:

1. **Forward sweep** (k = 1 to n):
   - Eliminate the lower diagonal entry
   - Modify the diagonal and RHS

2. **Back substitution** (k = n-1 to 1):
   - Solve for x[k] using x[k+1]

# Complexity
O(n) operations (vs O(n³) for general LU decomposition)

# Note
This modifies internal copies of du and d, leaving the inputs unchanged
(except for x which receives the solution).
"""
function thomas_solve!(x, dl, d, du, b)
    n = length(d)

    # Make working copies (algorithm modifies these)
    c = copy(du)    # Modified upper diagonal
    bb = copy(d)    # Modified main diagonal
    x .= b          # Solution starts as RHS

    #= Forward sweep: eliminate lower diagonal
    Transform the system into upper bidiagonal form =#
    c[1] /= bb[1]
    x[1] /= bb[1]
    @inbounds for i in 2:n
        denom = bb[i] - dl[i]*c[i-1]
        c[i] /= denom
        x[i] = (x[i] - dl[i]*x[i-1]) / denom
    end

    #= Back substitution: solve from bottom to top =#
    @inbounds for i in n-1:-1:1
        x[i] -= c[i]*x[i+1]
    end

    return x
end

end # module

# Export elliptic solvers to main QGYBJ module
using .Elliptic: invert_q_to_psi!, invert_helmholtz!, invert_B_to_A!
