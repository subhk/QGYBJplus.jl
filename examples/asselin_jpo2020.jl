#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example - MPI Parallel Version
================================================================================

MPI-parallel version of the barotropic dipole simulation from:

    Asselin, O., L. N. Thomas, W. R. Young, and L. Rainville (2020)
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439-3454

USAGE:
------
    mpirun -n 4 julia --project examples/asselin_jpo2020.jl
    mpirun -n 16 julia --project examples/asselin_jpo2020.jl

The model uses a second-order exponential Runge-Kutta time stepper. Equations
and parameters are kept in dimensional form.

================================================================================
=#

using QGYBJplus
using Printf

# ============================================================================
#                       SIMULATION PARAMETERS
# ============================================================================

nx = 256
ny = 256
nz = 128

# Physical parameters from Asselin et al. (2020)
f₀ = 1.24e-4           # Coriolis parameter [s⁻¹] (mid-latitude)
N² = 1.0e-5            # Buoyancy frequency squared [s⁻²]

# Domain size [m] (Asselin et al. 2020)
# Grid is in cardinal (X,Y) coordinates with 70 km periodic domain
# Dipole formula uses rotated (x,y) coords: x=(X-Y)/√2, y=(X+Y)/√2
Lx = 70.0e3            # 70 km horizontal domain in (X,Y)
Ly = 70.0e3            # 70 km horizontal domain in (X,Y)
Lz = 3.0e3             # H = 3 km depth, surface at z = 0

# Time stepping
n_inertial_periods = 15.0
T_inertial = 2π / f₀   # Inertial period = 2π/f [s] ≈ 14 hours
dt = 2.0                # [s]
nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave parameters
u0_wave = 0.10         # Wave velocity amplitude [m/s] (u0 = 10 cm/s)
surface_layer_depth = 30.0  # Surface layer depth [m] (s = 30 m)

# Flow parameters
U0_flow = 0.335        # Flow velocity scale [m/s] (U = 33.5 cm/s)
k_dipole = sqrt(2) * π / Lx  # κ = √2π/(70 km) per Asselin et al. (2020)
psi0 = U0_flow / k_dipole  # Streamfunction amplitude [m²/s]
vorticity_gradient = 2 * k_dipole^2 * U0_flow  # γ = 2κ²U ≈ 2.7e-9 m⁻¹ s⁻¹
rossby_rms = k_dipole * U0_flow / f₀           # κU/f ≈ 0.17

# Output settings
output_dir = "output_asselin"
save_interval_IP = 5.0  # Paper figures use 5, 10, and 15 inertial periods
diag_interval_IP = 0.5  # Print diagnostics every 0.5 inertial periods

"""
    asselin_dipole_streamfunction(X, Y, z)

Dimensional streamfunction from Asselin et al. (2020), Eq. (2).

The code grid uses cardinal coordinates `(X, Y)`. The paper's dipole formula is
written in coordinates rotated by 45 degrees:

    x = (X - Y) / √2,   y = (X + Y) / √2.
"""
function asselin_dipole_streamfunction(X, Y, z)
    x = (X - Y) / sqrt(2)
    y = (X + Y) / sqrt(2)
    return psi0 * sin(k_dipole * x) * cos(k_dipole * y)
end

# The paper specifies weak horizontal wave hyperdiffusion but not a coefficient.
# This value keeps the damping confined to the grid scale.
νₕ₁ʷ_wave = 1.0e5  # [m⁴/s]

grid = RectilinearGrid(size = (nx, ny, nz),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0))

model = QGYBJModel(grid = grid,
                   coriolis = FPlane(f = f₀),
                   stratification = ConstantStratification(N² = N²),
                   closure = HorizontalHyperdiffusivity(waves = νₕ₁ʷ_wave,
                                                         wave_laplacian_order = 2),
                   flow = :fixed,
                   feedback = :none,
                   ybj_plus = true,
                   parallel_io = false,
                   verbose = false)

set!(model;
     ψ = asselin_dipole_streamfunction,
     pv_method = :barotropic,
     waves = SurfaceWave(amplitude = u0_wave,
                         scale = surface_layer_depth,
                         profile = :gaussian))

simulation = Simulation(model;
                        Δt = dt,
                        stop_time = n_inertial_periods * inertial_period(model),
                        output = NetCDFOutput(path = output_dir,
                                              schedule = TimeInterval(save_interval_IP * inertial_period(model)),
                                              fields = (:ψ, :waves)),
                        diagnostics = IterationInterval(max(1, round(Int, diag_interval_IP * inertial_period(model) / dt))),
                        verbose = true)

if is_root(simulation)
    println("="^70)
    println("Asselin et al. (2020) Dipole")
    println("="^70)
    @printf("Resolution: %d × %d × %d, Duration: %.1f IP\n", nx, ny, nz, n_inertial_periods)
    @printf("Domain: %.1f km × %.1f km × %.1f km\n", Lx/1e3, Ly/1e3, Lz/1e3)
    @printf("Timestepper: exponential RK2, dt = %.1f s\n", dt)
    @printf("Dipole checks: γ = %.3e m⁻¹ s⁻¹, κU/f = %.3f\n", vorticity_gradient, rossby_rms)
    println("Output directory: $output_dir")
end

run!(simulation)
finalize_simulation!(simulation)
