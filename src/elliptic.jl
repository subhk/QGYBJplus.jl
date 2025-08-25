"""
Elliptic inversion routines, adapted from ellliptic.f90:
Solve for ψ in
    a(z) ∂²ψ/∂z² + b(z) ∂ψ/∂z - k_h² ψ = q,
for each horizontal wavenumber (kx,ky). We apply a tridiagonal solver along z
for each (kx,ky) independently.
"""
module Elliptic

using ..QGYBJ: Grid, State
using ..QGYBJ: a_ell_ut

"""
    invert_q_to_psi!(S, G; a)

Invert spectral PV `q(kx,ky,z)` to obtain `psi(kx,ky,z)` using the
coefficients from the Fortran scheme with Neumann ψ_z=0 at top/bottom:
discrete system with diagonals
  iz=1:   d = -(a[1]   + kh2*dz^2),   du = a[1]
  1<iz<n: d = -(a[iz]+a[iz-1] + kh2*dz^2), du=a[iz], dl=a[iz-1]
  iz=n:   d = -(a[n-1] + kh2*dz^2),   dl = a[n-1]
and RHS = dz^2 * q.
"""
function invert_q_to_psi!(S::State, G::Grid; a::AbstractVector)
    nx, ny, nz = G.nx, G.ny, G.nz
    @assert length(a) == nz
    ψ = S.psi
    q = S.q

    dl = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    du = zeros(eltype(a), nz)

    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2

    for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        if kh2 == 0
            @inbounds ψ[i,j,:] .= 0
            continue
        end
        fill!(dl, 0); fill!(d, 0); fill!(du, 0)
        # Build tri-diagonal
        d[1]  = -(a[1] + kh2*Δ2)
        du[1] =  a[1]
        @inbounds for k in 2:nz-1
            dl[k] = a[k-1]
            d[k]  = -(a[k] + a[k-1] + kh2*Δ2)
            du[k] = a[k]
        end
        dl[nz] = a[nz-1]
        d[nz]  = -(a[nz-1] + kh2*Δ2)

        # RHS = Δ^2 * q
        rhs = similar(view(q, i, j, :), eltype(a))
        @inbounds for k in 1:nz
            rhs[k] = Δ2 * real(q[i,j,k])
        end
        solr = copy(rhs)
        thomas_solve!(solr, dl, d, du, rhs)

        rhs_i = similar(solr)
        @inbounds for k in 1:nz
            rhs_i[k] = Δ2 * imag(q[i,j,k])
        end
        soli = copy(rhs_i)
        thomas_solve!(soli, dl, d, du, rhs_i)

        @inbounds for k in 1:nz
            ψ[i,j,k] = solr[k] + im*soli[k]
        end
    end
    return S
end

"""
    invert_B_to_A!(S, G, par, a)

YBJ+ inversion: for each (kx,ky), solve along z the system for A with
Neumann A_z=0 top/bottom and diagonal terms including (kh^2/4) like the
Fortran A_solver_ybj_plus. Also returns C = A_z with top value 0.
"""
function invert_B_to_A!(S::State, G::Grid, par, a::AbstractVector)
    nx, ny, nz = G.nx, G.ny, G.nz
    A = S.A
    B = S.B
    C = S.C
    dl = zeros(eltype(a), nz)
    d  = zeros(eltype(a), nz)
    du = zeros(eltype(a), nz)
    Δ = nz > 1 ? (G.z[2]-G.z[1]) : 1.0
    Δ2 = Δ^2
    for j in 1:ny, i in 1:nx
        kh2 = G.kh2[i,j]
        if kh2 == 0
            @inbounds A[i,j,:] .= 0
            @inbounds C[i,j,:] .= 0
            continue
        end
        fill!(dl, 0); fill!(d, 0); fill!(du, 0)
        d[1]  = -(a[1] + (kh2*Δ2)/4)
        du[1] =  a[1]
        @inbounds for k in 2:nz-1
            dl[k] = a[k-1]
            d[k]  = -(a[k] + a[k-1] + (kh2*Δ2)/4)
            du[k] = a[k]
        end
        dl[nz] = a[nz-1]
        d[nz]  = -(a[nz-1] + (kh2*Δ2)/4)
        # RHS = Δ2 * B (Bu = 1.0)
        rhs_r = similar(dl)
        rhs_i = similar(dl)
        @inbounds for k in 1:nz
            rhs_r[k] = Δ2 * real(B[i,j,k])
            rhs_i[k] = Δ2 * imag(B[i,j,k])
        end
        solr = copy(rhs_r)
        soli = copy(rhs_i)
        thomas_solve!(solr, dl, d, du, rhs_r)
        thomas_solve!(soli, dl, d, du, rhs_i)
        @inbounds for k in 1:nz
            A[i,j,k] = solr[k] + im*soli[k]
        end
        # C = A_z, set top C=0 and interior forward diff
        @inbounds for k in 1:nz-1
            C[i,j,k] = (A[i,j,k+1] - A[i,j,k])/Δ
        end
        C[i,j,nz] = 0
    end
    return S
end

"""
    thomas_solve!(x, dl, d, du, b)

In-place Thomas algorithm for tridiagonal systems with diagonals `dl,d,du`.
Accepts vector views for `x` and `b`.
"""
function thomas_solve!(x, dl, d, du, b)
    n = length(d)
    c = copy(du)
    bb = copy(d)
    x .= b
    # Forward sweep
    c[1] /= bb[1]
    x[1] /= bb[1]
    @inbounds for i in 2:n
        denom = bb[i] - dl[i]*c[i-1]
        c[i] /= denom
        x[i] = (x[i] - dl[i]*x[i-1]) / denom
    end
    # Back substitution
    @inbounds for i in n-1:-1:1
        x[i] -= c[i]*x[i+1]
    end
    return x
end

end # module

using .Elliptic: invert_q_to_psi!
