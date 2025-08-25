"""
Time stepping for QG–YBJ (projection + leapfrog). Implements:
- Nonlinear advection J(psi, q) and J(psi, B)
- YBJ+ inversion B -> A and C=A_z
- Hyperdiffusion via integrating factors (horizontal)
- Robert–Asselin filter
- 2/3 dealiasing mask on updates
Refraction and feedback terms can be extended to match the full Fortran model.
"""

"""
    first_projection_step!(S, G, par, plans; a)

Projection step (Forward Euler) example: given q^n, invert to ψ^n, compute
diagnostics. Extend to compute B and apply LA->A inversion similarly to Fortran.
"""
function first_projection_step!(S::State, G::Grid, par::QGParams, plans; a, dealias_mask=nothing)
    # Nonlinear terms
    L = isnothing(dealias_mask) ? trues(G.nx,G.ny) : dealias_mask
    nqk  = similar(S.q)
    nBRk = similar(S.B)
    nBIk = similar(S.B)
    rBRk = similar(S.B)
    rBIk = similar(S.B)
    dqk  = similar(S.B)

    # Compute diagnostics first
    invert_q_to_psi!(S, G; a)
    compute_velocities!(S, G; plans, params=par)

    # J terms
    convol_waqg!(nqk, nBRk, nBIk, S.u, S.v, S.q, BRk, BIk, G, plans; Lmask=L)
    # Refraction B*zeta
    refraction_waqg!(rBRk, rBIk, BRk, BIk, S.psi, G, plans; Lmask=L)
    # Vertical diffusion
    dissipation_q_nv!(dqk, S.q, par, G)

    # Special cases (follow test1 switches)
    if par.inviscid; dqk .= 0; end
    if par.linear
        nqk .= 0; nBRk .= 0; nBIk .= 0
    end
    if par.no_dispersion
        S.A .= 0; S.C .= 0
    end
    if par.passive_scalar
        S.A .= 0; S.C .= 0; rBRk .= 0; rBIk .= 0
    end
    if par.fixed_flow; nqk .= 0; end  # No mean flow advection if flow is fixed

    # Store old fields
    qok  = copy(S.q)
    BRok = similar(S.B); BIok = similar(S.B)
    @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
        BRok[i,j,k] = Complex(real(S.B[i,j,k]), 0)
        BIok[i,j,k] = Complex(imag(S.B[i,j,k]), 0)
    end

    # Forward Euler with integrating factors (test1 form)
    @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
        if L[i,j]
            kx = G.kx[i]; ky = G.ky[j]; kh2 = G.kh2[i,j]
            If = int_factor(kx, ky, par; waves=false)
            Ifw = int_factor(kx, ky, par; waves=true)
            
            # Update q only if mean flow is not fixed
            if par.fixed_flow
                # Keep q unchanged - no evolution of mean flow
                S.q[i,j,k] = qok[i,j,k]
            else
                S.q[i,j,k] = ( qok[i,j,k] - par.dt*nqk[i,j,k] + par.dt*dqk[i,j,k] ) * exp(-If)
            end
            
            # Always update wave field (B)
            BRnew = ( BRok[i,j,k] - par.dt*nBRk[i,j,k] - par.dt*0.5*kh2*Complex(imag(S.A[i,j,k]),0) + par.dt*0.5*rBIk[i,j,k] ) * exp(-Ifw)
            BInew = ( BIok[i,j,k] - par.dt*nBIk[i,j,k] + par.dt*0.5*kh2*Complex(real(S.A[i,j,k]),0) - par.dt*0.5*rBRk[i,j,k] ) * exp(-Ifw)
            S.B[i,j,k] = Complex(real(BRnew), 0) + im*Complex(real(BInew), 0)
        else
            S.q[i,j,k] = 0
            S.B[i,j,k] = 0
        end
    end

    # Feedback q* = q - qw (wave feedback on mean flow)
    wave_feedback_enabled = !par.no_feedback && !par.no_wave_feedback
    if wave_feedback_enabled
        qwk = similar(S.q)
        # Rebuild BRk/BIk from updated S.B for qw
        @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
            BRk[i,j,k] = Complex(real(S.B[i,j,k]), 0)
            BIk[i,j,k] = Complex(imag(S.B[i,j,k]), 0)
        end
        compute_qw!(qwk, BRk, BIk, par, G, plans; Lmask=L)
        @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
            if L[i,j]
                S.q[i,j,k] -= qwk[i,j,k]
            else
                S.q[i,j,k] = 0
            end
        end
    end

    # Recover psi, A (YBJ+ or normal), velocities
    # Only update psi if mean flow is not fixed
    if !par.fixed_flow
        invert_q_to_psi!(S, G; a)
    end
    if par.ybj_plus
        invert_B_to_A!(S, G, par, a)
    else
        # Normal YBJ path: optionally remove mean B, compute sigma and A
        # Build BRk/BIk from updated B
        BRk2 = similar(S.B); BIk2 = similar(S.B)
        @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
            BRk2[i,j,k] = Complex(real(S.B[i,j,k]), 0)
            BIk2[i,j,k] = Complex(imag(S.B[i,j,k]), 0)
        end
        sumB!(S.B, G; Lmask=L)
        sigma = compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask=L)
        compute_A!(S.A, S.C, BRk2, BIk2, sigma, par, G; Lmask=L)
    end
    compute_velocities!(S, G; plans, params=par)
    return S
end

"""
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a)

Skeleton for leapfrog advance; fill with actual tendencies from Fortran model.
Here just carries ψ diagnostic refresh.
"""
function leapfrog_step!(Snp1::State, Sn::State, Snm1::State,
                        G::Grid, par::QGParams, plans; a, dealias_mask=nothing)
    nx, ny, nz = G.nx, G.ny, G.nz
    L = isnothing(dealias_mask) ? trues(nx,ny) : dealias_mask
    # Ensure ψ and velocities for Sn are updated (only if mean flow evolves)
    if !par.fixed_flow
        invert_q_to_psi!(Sn, G; a)
    end
    compute_velocities!(Sn, G; plans, params=par)
    # Nonlinear terms
    nqk  = similar(Sn.q)
    nBRk = similar(Sn.B)
    nBIk = similar(Sn.B)
    rBRk = similar(Sn.B)
    rBIk = similar(Sn.B)
    dqk  = similar(Sn.B)
    # Build BRk, BIk for Sn
    BRk = similar(Sn.B); BIk = similar(Sn.B)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        BRk[i,j,k] = Complex(real(Sn.B[i,j,k]), 0)
        BIk[i,j,k] = Complex(imag(Sn.B[i,j,k]), 0)
    end
    convol_waqg!(nqk, nBRk, nBIk, Sn.u, Sn.v, Sn.q, BRk, BIk, G, plans; Lmask=L)
    refraction_waqg!(rBRk, rBIk, BRk, BIk, Sn.psi, G, plans; Lmask=L)
    dissipation_q_nv!(dqk, Snm1.q, par, G)
    
    # Special cases
    if par.inviscid; dqk .= 0; end
    if par.linear; nqk .= 0; nBRk .= 0; nBIk .= 0; end
    if par.no_dispersion; Sn.A .= 0; end
    if par.passive_scalar; Sn.A .= 0; rBRk .= 0; rBIk .= 0; end
    if par.fixed_flow; nqk .= 0; end  # No mean flow advection if flow is fixed
    # Leapfrog with integrating factors and full YBJ+ terms
    qtemp = similar(Sn.q)
    BRtemp = similar(Sn.B); BItemp = similar(Sn.B)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            kx = G.kx[i]; ky = G.ky[j]; kh2 = G.kh2[i,j]
            If  = int_factor(kx, ky, par; waves=false)
            Ifw = int_factor(kx, ky, par; waves=true)
            
            # Update q only if mean flow is not fixed
            if par.fixed_flow
                qtemp[i,j,k] = Sn.q[i,j,k]  # Keep current q unchanged
            else
                qtemp[i,j,k] = Snm1.q[i,j,k]*exp(-2If) - 2*par.dt*nqk[i,j,k]*exp(-If) + 2*par.dt*dqk[i,j,k]*exp(-2If)
            end
            
            # Always update wave field (B)
            BRtemp[i,j,k] = Complex(real(Snm1.B[i,j,k]),0)*exp(-2Ifw) - 2*par.dt*( nBRk[i,j,k] + 0.5*kh2*Complex(imag(Sn.A[i,j,k]),0) - 0.5*rBIk[i,j,k] )*exp(-Ifw)
            BItemp[i,j,k] = Complex(imag(Snm1.B[i,j,k]),0)*exp(-2Ifw) - 2*par.dt*( nBIk[i,j,k] - 0.5*kh2*Complex(real(Sn.A[i,j,k]),0) + 0.5*rBRk[i,j,k] )*exp(-Ifw)
        else
            qtemp[i,j,k] = 0; BRtemp[i,j,k] = 0; BItemp[i,j,k] = 0
        end
    end
    # Robert–Asselin filter
    γ = par.gamma
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if L[i,j]
            Snm1.q[i,j,k] = Sn.q[i,j,k] + γ*( Snm1.q[i,j,k] - 2Sn.q[i,j,k] + qtemp[i,j,k] )
            Snm1.B[i,j,k] = Sn.B[i,j,k] + γ*( Snm1.B[i,j,k] - 2Sn.B[i,j,k] + (Complex(real(BRtemp[i,j,k]),0) + im*Complex(real(BItemp[i,j,k]),0)) )
        else
            Snm1.q[i,j,k] = 0; Snm1.B[i,j,k] = 0
        end
    end
    # Accept
    Snp1.q .= qtemp
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        Snp1.B[i,j,k] = Complex(real(BRtemp[i,j,k]),0) + im*Complex(real(BItemp[i,j,k]),0)
    end
    # Feedback q* = q - qw then ψ, A, velocities (wave feedback on mean flow)
    wave_feedback_enabled = !par.no_feedback && !par.no_wave_feedback
    if wave_feedback_enabled
        qwk = similar(Snp1.q)
        # Rebuild BRk/BIk from Snp1.B for qw
        BRk2 = similar(Snp1.B); BIk2 = similar(Snp1.B)
        @inbounds for kk in 1:nz, jj in 1:ny, ii in 1:nx
            BRk2[ii,jj,kk] = Complex(real(Snp1.B[ii,jj,kk]),0)
            BIk2[ii,jj,kk] = Complex(imag(Snp1.B[ii,jj,kk]),0)
        end
        compute_qw!(qwk, BRk2, BIk2, par, G, plans; Lmask=L)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            if L[i,j]
                Snp1.q[i,j,k] -= qwk[i,j,k]
            else
                Snp1.q[i,j,k] = 0
            end
        end
    end
    
    # Only update psi if mean flow is not fixed
    if !par.fixed_flow
        invert_q_to_psi!(Snp1, G; a)
    end
    if par.ybj_plus
        invert_B_to_A!(Snp1, G, par, a)
    else
        # Normal YBJ path for Snp1
        BRk3 = similar(Snp1.B); BIk3 = similar(Snp1.B)
        @inbounds for kk in 1:nz, jj in 1:ny, ii in 1:nx
            BRk3[ii,jj,kk] = Complex(real(Snp1.B[ii,jj,kk]), 0)
            BIk3[ii,jj,kk] = Complex(imag(Snp1.B[ii,jj,kk]), 0)
        end
        sumB!(Snp1.B, G; Lmask=L)
        sigma2 = compute_sigma(par, G, nBRk, nBIk, rBRk, rBIk; Lmask=L)
        compute_A!(Snp1.A, Snp1.C, BRk3, BIk3, sigma2, par, G; Lmask=L)
    end
    compute_velocities!(Snp1, G; plans, params=par)
    return Snp1
end
