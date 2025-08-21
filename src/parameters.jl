"""
    QGParams

Container for physical and numerical parameters of the QG–YBJ model.
This is a simplified starting point; extend as needed while porting.
"""
Base.@kwdef struct QGParams{T}
    # Domain sizes
    nx::Int
    ny::Int
    nz::Int
    Lx::T
    Ly::T

    # Time stepping
    dt::T
    nt::Int

    # Physical parameters (nondimensionalize as appropriate)
    Ro::T              # Rossby number
    Fr::T              # Froude number
    f0::T              # Coriolis parameter

    # Viscosity/hyperviscosity
    nu_h::T            # horizontal viscosity
    nu_v::T            # vertical viscosity

    # Flags
    linear_vert_structure::Int  # mapping from Fortran param

    # Derived and model choices
    Bu::T                        # Burger-like parameter (Fr^2/Ro^2)
    stratification::Symbol       # :constant_N or :skewed_gaussian

    # Skewed Gaussian params (test1 values)
    N02_sg::T
    N12_sg::T
    sigma_sg::T
    z0_sg::T
    alpha_sg::T
end

"""
    default_params(; kwargs...)

Construct a reasonable default parameter set for experimentation.
"""
function default_params(; nx=64, ny=64, nz=64, Lx=2π, Ly=2π,
                           dt=1e-3, nt=10_000, Ro=0.1, Fr=0.1, f0=1.0,
                           nu_h=0.0, nu_v=0.0, linear_vert_structure=0,
                           stratification::Symbol=:constant_N)
    T = Float64
    Bu = (Fr^2)/(Ro^2)
    # Test1 skewed Gaussian defaults (nondimensional, L3 = 2π domain)
    N02_sg = T(0.537713935783168)
    N12_sg = T(2.684198470106461)
    sigma_sg = T(0.648457170048730)
    z0_sg = T(6.121537923499139)
    alpha_sg = T(-5.338431587899242)
    return QGParams{T}(; nx, ny, nz, Lx, Ly, dt, nt, Ro, Fr, f0, nu_h, nu_v,
                         linear_vert_structure, Bu, stratification,
                         N02_sg, N12_sg, sigma_sg, z0_sg, alpha_sg)
end
