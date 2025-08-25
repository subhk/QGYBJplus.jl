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

    # Physical parameters
    f0::T              # Coriolis parameter

    # Viscosity/hyperviscosity
    nu_h::T            # horizontal viscosity
    nu_v::T            # vertical viscosity

    # Flags
    linear_vert_structure::Int  # mapping from Fortran param

    # Derived and model choices
    stratification::Symbol       # :constant_N or :skewed_gaussian
    # Wave/flow magnitude ratio squared (Uw/U)^2 for qw feedback
    W2F::T
    # Robert–Asselin filter parameter
    gamma::T

    # Horizontal hyperdiffusion parameters (two-operator form)
    nuh1::T; nuh2::T; ilap1::Int; ilap2::Int
    nuh1w::T; nuh2w::T; ilap1w::Int; ilap2w::Int
    # Vertical diffusion of q (regular viscosity)
    nuz::T

    # Switches
    inviscid::Bool
    linear::Bool
    no_dispersion::Bool
    passive_scalar::Bool
    ybj_plus::Bool
    no_feedback::Bool
    
    # Wave-mean flow interaction controls
    fixed_flow::Bool              # true: mean flow doesn't evolve (psi constant in time)
    no_wave_feedback::Bool        # true: waves don't feedback on mean flow (qw = 0)

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
                           dt=1e-3, nt=10_000, f0=1.0,
                           nu_h=0.0, nu_v=0.0, linear_vert_structure=0,
                           stratification::Symbol=:constant_N)
    T = Float64
    W2F = T( (2.5e-5/0.25)^2 )  # test1 default (Uw_scale/U_scale)^2
    gamma = T(1e-3)
    # Map test1 hyperdiffusion defaults (coarse analogue)
    nuh1 = T(0.01)
    nuh2 = T(10.0)
    ilap1 = 2; ilap2 = 6
    nuh1w = T(0.0)
    nuh2w = T(10.0)
    ilap1w = 2; ilap2w = 6
    nuz = T(0.0)
    inviscid=false; linear=false; no_dispersion=false; passive_scalar=false
    ybj_plus=true; no_feedback=true; fixed_flow=false; no_wave_feedback=true
    # Test1 skewed Gaussian defaults (nondimensional, L3 = 2π domain)
    N02_sg = T(0.537713935783168)
    N12_sg = T(2.684198470106461)
    sigma_sg = T(0.648457170048730)
    z0_sg = T(6.121537923499139)
    alpha_sg = T(-5.338431587899242)
    return QGParams{T}(; nx, ny, nz, Lx, Ly, dt, nt, f0, nu_h, nu_v,
                         linear_vert_structure, stratification, W2F, gamma,
                         nuh1, nuh2, ilap1, ilap2, nuh1w, nuh2w, ilap1w, ilap2w,
                         nuz, inviscid, linear, no_dispersion, passive_scalar,
                         ybj_plus, no_feedback, fixed_flow, no_wave_feedback,
                         N02_sg, N12_sg, sigma_sg, z0_sg, alpha_sg)
end
