"""
NetCDF I/O stubs for psi and L+A using NCDatasets.jl when available.
"""

module IO

using ..QGYBJ: Grid, State
using ..QGYBJ: plan_transforms!, fft_forward!, fft_backward!

function _require_netcdf()
    try
        import NCDatasets
        return NCDatasets
    catch
        error("NCDatasets.jl not available. Add it to your environment to use NetCDF I/O.")
    end
end

"""
    ncdump_psi(S, G, plans; path="psi.out.nc")

Write real-space psi to NetCDF with dims (x,y,z).
"""
function ncdump_psi(S::State, G::Grid, plans; path="psi.out.nc")
    NCD = _require_netcdf()
    psir = similar(S.psi)
    fft_backward!(psir, S.psi, plans)
    nx, ny, nz = G.nx, G.ny, G.nz
    ds = NCD.Dataset(path, "c")
    try
        ds.dim["x"] = nx; ds.dim["y"] = ny; ds.dim["z"] = nz
        v = ds["psi"] = NCD.defVar(ds, "psi", Float64, ("x","y","z"))
        # Normalize IFFT by nx*ny
        norm = nx*ny
        buf = Array{Float64}(undef, nx, ny, nz)
        @inbounds for k in 1:nz
            buf[:,:,k] .= real.(psir[:,:,k]) ./ norm
        end
        v[:] = buf
    finally
        close(ds)
    end
    return path
end

"""
    ncdump_la(S, G, plans; path="la.out.nc")

Write real-space L+A: variables BR and BI as separate arrays.
"""
function ncdump_la(S::State, G::Grid, plans; path="la.out.nc")
    NCD = _require_netcdf()
    BRr = similar(S.B); BIr = similar(S.B)
    # Build spectral BRk, BIk and inverse FFT
    BRk = similar(S.B); BIk = similar(S.B)
    @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
        BRk[i,j,k] = Complex(real(S.B[i,j,k]),0)
        BIk[i,j,k] = Complex(imag(S.B[i,j,k]),0)
    end
    fft_backward!(BRr, BRk, plans)
    fft_backward!(BIr, BIk, plans)
    nx, ny, nz = G.nx, G.ny, G.nz
    ds = NCD.Dataset(path, "c")
    try
        ds.dim["x"] = nx; ds.dim["y"] = ny; ds.dim["z"] = nz
        vR = ds["BR"] = NCD.defVar(ds, "BR", Float64, ("x","y","z"))
        vI = ds["BI"] = NCD.defVar(ds, "BI", Float64, ("x","y","z"))
        norm = nx*ny
        bufR = Array{Float64}(undef, nx, ny, nz)
        bufI = Array{Float64}(undef, nx, ny, nz)
        @inbounds for k in 1:nz
            bufR[:,:,k] .= real.(BRr[:,:,k]) ./ norm
            bufI[:,:,k] .= real.(BIr[:,:,k]) ./ norm
        end
        vR[:] = bufR; vI[:] = bufI
    finally
        close(ds)
    end
    return path
end

"""
    ncread_psi!(S, G, plans; path)

Read real-space psi from NetCDF and set spectral S.psi via forward FFT.
"""
function ncread_psi!(S::State, G::Grid, plans; path)
    NCD = _require_netcdf()
    ds = NCD.Dataset(path, "r")
    try
        psir = Array(ds["psi"])
        fft_forward!(S.psi, psir, plans)
    finally
        close(ds)
    end
    return S
end

"""
    ncread_la!(S, G, plans; path)

Read real-space BR and BI and set spectral S.B accordingly.
"""
function ncread_la!(S::State, G::Grid, plans; path)
    NCD = _require_netcdf()
    ds = NCD.Dataset(path, "r")
    try
        BRr = Array(ds["BR"])
        BIr = Array(ds["BI"])
        BRk = similar(S.B); BIk = similar(S.B)
        fft_forward!(BRk, BRr, plans)
        fft_forward!(BIk, BIr, plans)
        @inbounds for k in 1:G.nz, j in 1:G.ny, i in 1:G.nx
            S.B[i,j,k] = Complex(real(BRk[i,j,k]),0) + im*Complex(real(BIk[i,j,k]),0)
        end
    finally
        close(ds)
    end
    return S
end

end # module

using .IO: ncdump_psi, ncdump_la, ncread_psi!, ncread_la!

