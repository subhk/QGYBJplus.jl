"""
    Grid

Numerical grid and spectral metadata. Supports both serial arrays and
PencilArrays-backed distributed storage. Vertical grid is stored explicitly
to enable second-order finite differences like in the Fortran reference.
"""
Base.@kwdef mutable struct Grid{T, AT}
    # Sizes and spacings
    nx::Int
    ny::Int
    nz::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    z::Vector{T}           # unstaggered vertical levels (size nz)
    dz::Vector{T}          # layer thicknesses (size nz-1), mid-point scheme

    # Wavenumbers and spectral helpers
    kx::Vector{T}
    ky::Vector{T}
    kh2::AT                # kh^2 on spectral grid (nx, ny)

    # Decomposition (if using PencilArrays)
    decomp::Any
end

"""
    init_grid(par::QGParams)

Initialize grid and wavenumber arrays. Uses PencilArrays decomposition if available.
"""
function init_grid(par::QGParams)
    T = Float64
    nx, ny, nz = par.nx, par.ny, par.nz
    dx = par.Lx / nx
    dy = par.Ly / ny
    # Match Fortran nondimensional vertical domain L3 = 2π
    z = T.(collect(range(0, 2π; length=nz)))
    dz = diff(z)

    # Wavenumbers (2π periodic domain)
    kx = T.([i <= nx÷2 ? (2π/par.Lx)*(i-1) : (2π/par.Lx)*(i-1-nx) for i in 1:nx])
    ky = T.([j <= ny÷2 ? (2π/par.Ly)*(j-1) : (2π/par.Ly)*(j-1-ny) for j in 1:ny])

    # kh^2 grid
    kh2 = Array{T}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        kh2[i,j] = kx[i]^2 + ky[j]^2
    end

    decomp = nothing

    return Grid{T, typeof(kh2)}(nx, ny, nz, par.Lx, par.Ly, dx, dy, z, dz, kx, ky, kh2, decomp)
end

"""
    compute_wavenumbers!(grid)

Recompute `kx`, `ky`, `kh2` if grid sizes/domains changed.
"""
function compute_wavenumbers!(G::Grid)
    nx, ny = G.nx, G.ny
    G.kx .= [i <= nx÷2 ? (2π/G.Lx)*(i-1) : (2π/G.Lx)*(i-1-nx) for i in 1:nx]
    G.ky .= [j <= ny÷2 ? (2π/G.Ly)*(j-1) : (2π/G.Ly)*(j-1-ny) for j in 1:ny]
    @inbounds for j in 1:ny, i in 1:nx
        G.kh2[i,j] = G.kx[i]^2 + G.ky[j]^2
    end
    return G
end

"""
    init_pencil_decomposition!(G)

Attempt to initialize a PencilArrays pencil decomposition using MPI if
available. Safe to call even without MPI; leaves `G.decomp` as `nothing`.
"""
function init_pencil_decomposition!(G::Grid)
    try
        @eval import MPI
        @eval import PencilArrays
        comm = MPI.COMM_WORLD
        G.decomp = PencilArrays.PencilDecomposition((G.nx, G.ny, G.nz), comm)
    catch
        # stay in serial mode
    end
    return G
end

"""
    State

Holds prognostic and diagnostic fields. Real space fields are real-valued
arrays `(nx, ny, nz)`. Spectral fields are complex arrays `(nx, ny, nz)`.
If using PencilArrays, these are `PencilArray`s; otherwise they are `Array`s.
"""
Base.@kwdef mutable struct State{T, RT<:AbstractArray{T,3}, CT<:AbstractArray{Complex{T},3}}
    # Prognostic fields
    q::CT           # potential vorticity in spectral space
    psi::CT         # QG streamfunction in spectral space
    A::CT           # YBJ wave amplitude A (spectral)
    B::CT           # YBJ L+A combined field (spectral)
    C::CT           # A_z diagnostic (spectral)

    # Diagnostics (optional; can be allocated lazily)
    u::RT           # u in real space
    v::RT           # v in real space
    w::RT           # w in real space (if computed)
end

"""
    allocate_field(T, G; complex=false)

Allocate a 3D array of size `(nx, ny, nz)`, optionally complex, using
PencilArrays when available.
"""
function allocate_field(::Type{T}, G::Grid; complex::Bool=false) where {T}
    sz = (G.nx, G.ny, G.nz)
    if G.decomp === nothing
        return complex ? Array{Complex{T}}(undef, sz) : Array{T}(undef, sz)
    else
        if complex
            return PencilArrays.PencilArray{Complex{T}}(G.decomp, sz)
        else
            return PencilArrays.PencilArray{T}(G.decomp, sz)
        end
    end
end

"""
    init_state(G::Grid; T=Float64)

Allocate a default `State` with zeroed fields.
"""
function init_state(G::Grid; T=Float64)
    q   = allocate_field(T, G; complex=true);    fill!(q, 0)
    psi = allocate_field(T, G; complex=true);    fill!(psi, 0)
    A   = allocate_field(T, G; complex=true);    fill!(A, 0)
    B   = allocate_field(T, G; complex=true);    fill!(B, 0)
    C   = allocate_field(T, G; complex=true);    fill!(C, 0)
    u   = allocate_field(T, G; complex=false);   fill!(u, 0)
    v   = allocate_field(T, G; complex=false);   fill!(v, 0)
    w   = allocate_field(T, G; complex=false);   fill!(w, 0)
    return State{T, typeof(u), typeof(q)}(q, psi, A, B, C, u, v, w)
end
