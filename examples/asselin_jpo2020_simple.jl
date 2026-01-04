#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example - SIMPLIFIED VERSION
================================================================================

MPI-parallel simulation of the barotropic dipole from Asselin et al. (2020):
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439-3454

This example demonstrates the simplified high-level API that hides MPI
complexity from users.

USAGE:
------
    mpirun -n 4 julia --project examples/asselin_jpo2020_simple.jl
    mpirun -n 16 julia --project examples/asselin_jpo2020_simple.jl

================================================================================
=#

using QGYBJplus

# ============================================================================
#                       SIMULATION PARAMETERS
# ============================================================================

# Physical parameters from Asselin et al. (2020)
const f₀ = 1.24e-4           # Coriolis parameter [s⁻¹]
const N² = 1e-5              # Buoyancy frequency squared [s⁻²]

# Domain size [m]
const Lx = 70e3              # 70 km horizontal domain
const Ly = 70e3
const Lz = 2000.0            # 2 km depth

# Time stepping
const n_inertial_periods = 15
const T_inertial = 2π / f₀   # Inertial period ≈ 14 hours
const dt = 20.0              # Time step [s] (IMEX allows larger dt)
const nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave and flow parameters
const u0_wave = 0.10         # Wave velocity amplitude [m/s]
const surface_depth = 30.0   # Surface layer depth [m]
const U0_flow = 0.335        # Flow velocity scale [m/s]
const k_dipole = sqrt(2) * π / Lx  # Dipole wavenumber

# ============================================================================
#                       MAIN SIMULATION
# ============================================================================

# Initialize simulation (handles all MPI setup automatically!)
# centered=true gives domain x,y ∈ [-35km, +35km) matching Fig. 2 of paper
sim = initialize_simulation(
    nx = 256, ny = 256, nz = 128,
    Lx = Lx, Ly = Ly, Lz = Lz,
    centered = true,  # Center domain at origin: x,y ∈ [-Lx/2, Lx/2)
    f₀ = f₀, N² = N²,
    dt = dt, nt = nt,
    ybj_plus = true,
    fixed_flow = true,
    no_wave_feedback = true,
    νₕ₁ʷ = 1.0e5,  # Biharmonic hyperdiffusion for waves
    ilap1w = 2,    # 4th order (∇⁴)
    γ = 0.001      # Robert-Asselin filter coefficient
)

# Set initial conditions with simple high-level functions
dipole = (x, y, z) -> begin
    x_rot = (x - y) / sqrt(2)
    y_rot = (x + y) / sqrt(2)
    (U0_flow / k_dipole) * sin(k_dipole * x_rot) * cos(k_dipole * y_rot)
end
set_mean_flow!(sim; psi_func=dipole)
set_surface_waves!(sim; amplitude=u0_wave, surface_depth=surface_depth)

# Run simulation
run!(sim;
    output_dir = "output_asselin_simple",
    timestepper = :imex_cn,
    save_interval = T_inertial,  # Save every inertial period
    diagnostics_interval = 10
)

# Clean up
if is_root(sim)
    println("\nSimulation complete!")
end
finalize_simulation!(sim)
