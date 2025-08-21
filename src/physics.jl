"""
Physics helpers: stratification profiles and derived vertical operator
coefficients matching the Fortran test1 setup.
"""

"""
    a_ell_ut(par, G) -> Vector

Compute `a_ell_ut(z) = Bu / N^2(z)` on unstaggered levels, using
`par.stratification` with constants from parameters_test1.
"""
function a_ell_ut(par::QGParams, G::Grid)
    nz = G.nz
    a = similar(G.z)
    if par.stratification === :constant_N
        @inbounds for k in 1:nz
            a[k] = par.Bu / 1.0
        end
    elseif par.stratification === :skewed_gaussian
        N02 = par.N02_sg; N12 = par.N12_sg; σ = par.sigma_sg; z0 = par.z0_sg; α = par.alpha_sg
        @inbounds for k in 1:nz
            z = G.z[k]
            N2 = N12*exp(-((z - z0)^2)/(σ^2))*(1 + erf(α*(z - z0)/(σ*sqrt(2.0)))) + N02
            a[k] = par.Bu / N2
        end
    else
        error("Unsupported stratification: $(par.stratification)")
    end
    return a
end

"""
    dealias_mask(G) -> Matrix{Bool}

2/3-rule horizontal dealiasing mask `L(i,j)` modeled after Fortran `init_arrays`.
True indicates mode is kept; false indicates it is truncated.
"""
function dealias_mask(G::Grid)
    nx, ny = G.nx, G.ny
    keep = trues(nx, ny)
    # Following idea: keep modes with sqrt(kx^2 + ky^2) <= floor(n/3)
    kmaxx = nx ÷ 3
    kmaxy = ny ÷ 3
    # Compare in index space via integer bands around zero frequency
    for j in 1:ny, i in 1:nx
        # map frequency index to centered integer index
        ix = i-1; ix = ix <= nx÷2 ? ix : ix - nx
        jy = j-1; jy = jy <= ny÷2 ? jy : jy - ny
        keep[i,j] = (abs(ix) <= kmaxx) && (abs(jy) <= kmaxy)
    end
    # Remove the special ky = -N/2 duplicate if even: handled by FFT indexing; keep as is
    return keep
end

