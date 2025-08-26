"""
Physics helpers: stratification profiles and derived vertical operator
coefficients matching the Fortran test1 setup.
"""

"""
    a_ell_ut(par, G) -> Vector

Compute `a_ell_ut(z) = 1.0 / N^2(z)` on unstaggered levels, using
`par.stratification` with constants from parameters_test1.
"""
function a_ell_ut(par::QGParams, G::Grid)
    nz = G.nz
    a = similar(G.z)
    if par.stratification === :constant_N
        @inbounds for k in 1:nz
            a[k] = 1.0 / 1.0  # Normalized
        end
    elseif par.stratification === :skewed_gaussian
        N02 = par.N02_sg; N12 = par.N12_sg; σ = par.sigma_sg; z0 = par.z0_sg; α = par.alpha_sg
        @inbounds for k in 1:nz
            z = G.z[k]
            N2 = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
            a[k] = 1.0 / N2  # Normalized
        end
    else
        error("Unsupported stratification: $(par.stratification)")
    end
    return a
end

"""
    rho_ut(par, G) -> Vector

Unstaggered-to-staggered density-like weight used in the Fortran vertical
operators. For now returns ones (normalized), serving as a placeholder to
enable density-weighted tridiagonals that mirror the Fortran structure.
"""
function rho_ut(par::QGParams, G::Grid)
    # If provided by params, use it
    if par.rho_ut_profile !== nothing
        @assert length(par.rho_ut_profile) == G.nz
        return copy(par.rho_ut_profile)
    end
    # Default: unity weights (normalized)
    w = similar(G.z)
    @inbounds fill!(w, 1.0)
    return w
end

"""
    rho_st(par, G) -> Vector

Staggered-grid density-like weight for vertical operators. Currently returns
ones (normalized). This matches the simplified nondimensionalization used in
the Julia port and can be refined to mirror Fortran parameters.
"""
function rho_st(par::QGParams, G::Grid)
    # If provided by params, use it
    if par.rho_st_profile !== nothing
        @assert length(par.rho_st_profile) == G.nz
        return copy(par.rho_st_profile)
    end
    # Default: unity weights (normalized)
    w = similar(G.z)
    @inbounds fill!(w, 1.0)
    return w
end

"""
    b_ell_ut(par, G) -> Vector

Vertical first-derivative coefficient b(z). Not used in the current ψ/A solvers
but provided for completeness (e.g., alternative Helmholtz problems).
Defaults to zeros unless supplied via `par.b_ell_profile`.
"""
function b_ell_ut(par::QGParams, G::Grid)
    if par.b_ell_profile !== nothing
        @assert length(par.b_ell_profile) == G.nz
        return copy(par.b_ell_profile)
    end
    b = zeros(eltype(G.z), G.nz)
    return b
end

"""
    N2_ut(par, G) -> Vector

Unstaggered Brunt–Väisälä frequency squared N^2(z) matching the chosen
stratification (used by the normal YBJ integration method).
"""
function N2_ut(par::QGParams, G::Grid)
    nz = G.nz
    N2 = similar(G.z)
    if par.stratification === :constant_N
        @inbounds fill!(N2, 1.0)
    elseif par.stratification === :skewed_gaussian
        N02 = par.N02_sg; N12 = par.N12_sg; σ = par.sigma_sg; z0 = par.z0_sg; α = par.alpha_sg
        @inbounds for k in 1:nz
            z = G.z[k]
            N2[k] = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
        end
    else
        error("Unsupported stratification: $(par.stratification)")
    end
    return N2
end

"""
    dealias_mask(G) -> Matrix{Bool}

2/3-rule horizontal dealiasing mask `L(i,j)` with radial cutoff, modeled after
the Fortran practice. True indicates mode is kept; false indicates it is
truncated.
"""
function dealias_mask(G::Grid)
    nx, ny = G.nx, G.ny
    keep = falses(nx, ny)
    # Radial 2/3 cutoff in index space
    kmax = floor(Int, min(nx, ny) / 3)
    for j in 1:ny, i in 1:nx
        ix = i-1; ix = ix <= nx÷2 ? ix : ix - nx
        jy = j-1; jy = jy <= ny÷2 ? jy : jy - ny
        r = sqrt(ix^2 + jy^2)
        keep[i,j] = (r <= kmax)
    end
    return keep
end
