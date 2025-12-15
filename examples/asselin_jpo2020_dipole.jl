#=
================================================================================
    Asselin et al. (2020) JPO Dipole Example
================================================================================

This example reproduces the barotropic dipole simulation from:

    Asselin, O., L. N. Thomas, W. R. Young, and L. Rainville (2020)
    "Refraction and Straining of Near-Inertial Waves by Barotropic Eddies"
    Journal of Physical Oceanography, 50, 3439-3454
    DOI: 10.1175/JPO-D-20-0109.1

PHYSICAL SETUP:
---------------
- Location: NISKINe region (~58.5°N, 500 km south of Iceland)
- Domain: 70 km × 70 km horizontally, 3 km depth
- Flow: Steady barotropic vortex dipole
- Waves: Surface-confined near-inertial oscillations (k=0 initially)
- Stratification: Uniform N² = 10⁻⁵ s⁻²

KEY PHYSICS:
------------
The paper examines how near-inertial waves are modified by mesoscale eddies:
1. ζ-refraction: Waves acquire horizontal scale from vorticity gradients
2. Straining: Differential advection stretches wavevector (cancelled by k-advection!)
3. Dispersion: Low modes escape to anticyclone, high modes remain near surface

The key result is that for steady barotropic flows with initially uniform waves,
straining is INEFFECTIVE because k-advection exactly cancels it.

DIPOLE STREAMFUNCTION:
----------------------
    ψ = U/κ × sin(κx) × cos(κy)
    ζ = -2κ²ψ = -γ/κ × sin(κx) × cos(κy)

where γ = 2κ²U is the maximum vorticity gradient.

WAVE INITIAL CONDITION:
-----------------------
    u(t=0) = u₀ × exp(-z²/σ²),  v(t=0) = 0

A horizontally uniform, surface-confined near-inertial oscillation.

================================================================================
=#

using QGYBJ
using Printf

# ============================================================================
#                       PHYSICAL PARAMETERS (from paper)
# ============================================================================

# Domain dimensions (dimensional)
const L_domain = 70e3      # Horizontal domain size [m]
const H_depth = 3000.0     # Ocean depth [m]

# Coriolis parameter at 58.5°N
const f_dim = 1.24e-4      # [s⁻¹]

# Stratification (uniform)
const N2_dim = 1e-5        # [s⁻²]
const N_dim = sqrt(N2_dim) # [s⁻¹]

# Dipole flow parameters
const U_jet = 0.335        # Max jet velocity [m/s] (33.5 cm/s)
const gamma_max = 2.7e-9   # Max vorticity gradient [m⁻¹ s⁻¹]

# Compute dipole wavenumber: γ = 2κ²U → κ = √(γ/(2U))
const kappa_dim = sqrt(gamma_max / (2 * U_jet))  # ≈ √2π/(70 km)

# Wave initial condition
const u0_wave = 0.10       # Initial wave velocity [m/s] (10 cm/s)
const sigma_wave = 30.0    # Surface layer depth [m]

# ============================================================================
#                       NONDIMENSIONALIZATION
# ============================================================================
#
# Following the QGYBJ.jl convention:
#   - Horizontal length scale: L = L_domain / (2π)
#   - Vertical length scale: chosen so domain is [0, 2π]
#   - Velocity scale: U = U_jet
#   - Time scale: T = L/U
#
# Nondimensional numbers:
#   Ro = U/(f×L)           Rossby number
#   Bu = (N×H_nd)²/(f×L)²  Burger number (H_nd = nondim vertical scale)
#   W2F = (u0/U)²          Wave-to-flow velocity ratio squared

const L_scale = L_domain / (2π)        # Horizontal scale ≈ 11.14 km
const H_scale = H_depth / (2π)         # Vertical scale ≈ 477 m
const U_scale = U_jet                   # Velocity scale
const T_scale = L_scale / U_scale       # Time scale ≈ 33,200 s ≈ 9.2 hours

# Nondimensional parameters
const Ro = U_scale / (f_dim * L_scale)
const Bu = (N_dim * H_scale / (f_dim * L_scale))^2
const W2F = (u0_wave / U_scale)^2

# Nondimensional wave parameters
const u0_nd = u0_wave / U_scale         # ≈ 0.30
const sigma_nd = sigma_wave / H_scale   # Surface layer in nondim units

# Inertial period in nondimensional time
const T_inertial_dim = 2π / f_dim       # ≈ 50,700 s ≈ 14.1 hours
const T_inertial_nd = T_inertial_dim / T_scale  # In nondim time units

println("="^70)
println("Asselin et al. (2020) Dipole Simulation - Parameter Summary")
println("="^70)
println("\nDimensional Parameters:")
@printf("  Domain:        %.0f km × %.0f km × %.1f km\n", L_domain/1e3, L_domain/1e3, H_depth/1e3)
@printf("  Coriolis f:    %.2e s⁻¹\n", f_dim)
@printf("  Stratification N²: %.1e s⁻²  (N = %.2e s⁻¹)\n", N2_dim, N_dim)
@printf("  Jet velocity U:    %.1f cm/s\n", U_jet*100)
@printf("  Wave velocity u₀:  %.1f cm/s\n", u0_wave*100)
@printf("  Surface layer σ:   %.0f m\n", sigma_wave)
@printf("  Inertial period:   %.1f hours\n", T_inertial_dim/3600)

println("\nNondimensional Numbers:")
@printf("  Rossby number Ro:  %.3f\n", Ro)
@printf("  Burger number Bu:  %.2f\n", Bu)
@printf("  Wave/Flow W2F:     %.4f\n", W2F)
@printf("  Nondim inertial period: %.2f\n", T_inertial_nd)

# ============================================================================
#                       MODEL PARAMETERS
# ============================================================================

# Grid resolution (paper doesn't specify, use reasonable values)
const nx = 128
const ny = 128
const nz = 64

# Time stepping
const n_inertial_periods = 15  # Run for 15 inertial periods (as in paper)
const dt = 0.001               # Time step (nondimensional)
const nt = round(Int, n_inertial_periods * T_inertial_nd / dt)

println("\nSimulation Setup:")
@printf("  Resolution:    %d × %d × %d\n", nx, ny, nz)
@printf("  Time step dt:  %.4f (nondim)\n", dt)
@printf("  Total steps:   %d\n", nt)
@printf("  Duration:      %.1f inertial periods\n", n_inertial_periods)

# Create parameter struct
par = QGYBJ.QGParams{Float64}(
    # Domain
    nx = nx, ny = ny, nz = nz,
    Lx = 2π, Ly = 2π,

    # Time stepping
    dt = dt,
    nt = nt,

    # Physical parameters
    f0 = 1.0,  # Nondimensional Coriolis

    # Nondimensional numbers (from paper)
    Ro = Ro,
    Bu = Bu,
    W2F = W2F,
    gamma = 1e-3,  # Robert-Asselin filter

    # Hyperdiffusion (for numerical stability)
    nuh1 = 1e-4, ilap1 = 2,    # Biharmonic
    nuh2 = 1e-2, ilap2 = 6,    # Hyper-6
    nuh1w = 0.0, ilap1w = 2,   # Wave hyperdiffusion
    nuh2w = 1e-2, ilap2w = 6,
    nuz = 0.0,                  # No vertical diffusion

    # Legacy viscosity (unused)
    nu_h = 0.0, nu_v = 0.0,

    # Physics switches
    linear_vert_structure = 0,
    stratification = :constant_N,  # Uniform N² as in paper
    inviscid = false,
    linear = false,                # Include nonlinear advection
    no_dispersion = false,         # Include wave dispersion
    passive_scalar = false,        # Waves are dynamically active
    ybj_plus = true,               # Use YBJ+ formulation (as in paper!)
    no_feedback = true,            # No wave feedback on mean flow
    fixed_flow = true,             # STEADY flow (key assumption!)
    no_wave_feedback = true,

    # Stratification parameters (not used for constant_N)
    N02_sg = 1.0, N12_sg = 0.0, sigma_sg = 1.0, z0_sg = π, alpha_sg = 0.0
)

# ============================================================================
#                       INITIALIZE GRID AND STATE
# ============================================================================

println("\nInitializing grid and state...")

G = QGYBJ.init_grid(par)
S = QGYBJ.init_state(G)

# Vertical coordinate (0 to 2π, surface at z=2π)
z = G.z
dz = G.nz > 1 ? z[2] - z[1] : 1.0

println("  Grid spacing: dx = $(G.dx), dz = $dz")
println("  Vertical levels: z ∈ [$(minimum(z)), $(maximum(z))]")

# ============================================================================
#                       SET UP DIPOLE STREAMFUNCTION
# ============================================================================
#
# From paper Eq. (2): ψ = U/κ × sin(κx) × cos(κy)
#
# In nondimensional coordinates with Lx = Ly = 2π:
#   x_dim = x × L_scale
#   κ_nd = κ_dim × L_scale
#
# The dipole has:
#   - Anticyclone centered at (x,y) = (π/2, 0)  [negative vorticity]
#   - Cyclone centered at (x,y) = (-π/2, 0)     [positive vorticity]
#   - Jet between them along y = 0

println("\nSetting up dipole streamfunction...")

# Nondimensional dipole wavenumber
# From paper: κ = √2π/(70 km), so κ×L_scale = √2π/(70 km) × (70 km/2π) = 1/√2
const kappa_nd = kappa_dim * L_scale  # ≈ 1/√2 ≈ 0.707

# Actually, for a 2π periodic domain, we want integer wavenumber
# Use k = 1 (fundamental mode) which gives similar structure
const k_dipole = 1.0

# Amplitude: U/κ in nondim units = 1.0/k_dipole (since U_nd = 1)
const psi_amp = 1.0 / k_dipole

# Initialize ψ in real space, then FFT
psi_real = zeros(Float64, nx, ny, nz)

for k in 1:nz, j in 1:ny, i in 1:nx
    x = (i-1) * G.dx
    y = (j-1) * G.dy

    # Dipole streamfunction: ψ = (U/κ) sin(κx) cos(κy)
    # Shift x by π/2 to center anticyclone at origin
    psi_real[i,j,k] = psi_amp * sin(k_dipole * (x - π/2)) * cos(k_dipole * y)
end

# FFT to spectral space
plans = QGYBJ.plan_transforms!(G)
QGYBJ.fft_forward!(S.psi, complex.(psi_real), plans)

# Compute q = ∇²ψ (in spectral space, q = -kh² × ψ)
for k in 1:nz, j in 1:ny, i in 1:nx
    kh2 = G.kx[i]^2 + G.ky[j]^2
    S.q[i,j,k] = -kh2 * S.psi[i,j,k]
end

# Compute max vorticity for verification
zeta_real = zeros(Float64, nx, ny, nz)
zeta_spec = similar(S.q)
for k in 1:nz, j in 1:ny, i in 1:nx
    kh2 = G.kx[i]^2 + G.ky[j]^2
    zeta_spec[i,j,k] = -kh2 * S.psi[i,j,k]
end
QGYBJ.fft_backward!(complex.(zeta_real), zeta_spec, plans)
zeta_max = maximum(abs.(real.(zeta_real)))

@printf("  Dipole wavenumber κ = %.3f\n", k_dipole)
@printf("  Max vorticity |ζ|/f = %.3f (paper: ~0.34)\n", zeta_max)

# ============================================================================
#                       SET UP WAVE INITIAL CONDITION
# ============================================================================
#
# From paper Eq. (4): u(t=0) = u₀ exp(-z²/σ²), v(t=0) = 0
#
# In YBJ+, we work with B = L⁺A where LA = (u + iv) e^{ift}
# At t=0: LA = u₀ exp(-z²/σ²) + i×0 = u₀ exp(-z²/σ²)
#
# For horizontally uniform initial condition, only the (kx,ky)=(0,0) mode is nonzero.

println("\nSetting up wave initial condition...")

# Surface-confined wave profile
# In nondim units, surface is at z = 2π, so profile peaks there
z_surface = 2π  # Surface location
sigma_z = sigma_nd * (2π)  # Convert to z units (scale by domain height)

# Actually, the paper uses z=0 at surface, z<0 below
# In QGYBJ.jl, z goes from 0 (bottom) to 2π (top/surface)
# So the wave profile should peak at z = 2π

# For simplicity, use exponential decay from surface:
# exp(-(z_surface - z)²/σ²)

# Initialize B with surface-confined structure
# Only the kh=0 mode (i=1, j=1) is nonzero for horizontally uniform IC
for k in 1:nz
    z_k = z[k]
    # Distance from surface (z = 2π)
    depth = z_surface - z_k

    # Gaussian profile peaking at surface
    # Use σ = 30m / H_scale ≈ 0.063 in nondim z ∈ [0, 2π]
    sigma_z_nd = sigma_wave / H_scale  # In units where domain is [0, 2π]
    sigma_z_scaled = sigma_z_nd * (2π / (2π))  # Already in [0, 2π]

    # Actually let's be more careful:
    # Physical depth from surface: d_phys = H - z_phys = H×(1 - z_nd/(2π))
    # If z_nd = 2π (top), d_phys = 0 (surface)
    # If z_nd = 0 (bottom), d_phys = H (full depth)

    # Profile: exp(-d²/σ²) where d = distance from surface
    # d_nd = (2π - z_nd) × (H/(2π)) = physical depth
    # Or in nondim: d_nd = 2π - z_nd

    d_nd = 2π - z_k  # Nondim distance from surface
    d_phys = d_nd * H_scale  # Physical distance [m]

    wave_profile = exp(-(d_phys^2) / (sigma_wave^2))

    # Set only the (0,0) mode
    S.B[1, 1, k] = u0_nd * wave_profile * (nx * ny)  # Factor for FFT normalization
end

# Verify wave energy distribution
wave_profile_check = [real(S.B[1,1,k]) / (nx*ny) for k in 1:nz]
@printf("  Wave amplitude at surface: %.3f (target: %.3f)\n",
        maximum(wave_profile_check), u0_nd)
@printf("  Wave amplitude at depth 100m: %.4f\n",
        wave_profile_check[end-round(Int, 100/H_scale * nz / (2π))])

# ============================================================================
#                       DIAGNOSTIC SETUP
# ============================================================================

# Compute elliptic coefficient a = Bu/N² = Bu (for constant N²=1)
a_ell = QGYBJ.a_ell_ut(par, G)

# Dealiasing mask
L_mask = QGYBJ.dealias_mask(G)

# Compute velocities from ψ
QGYBJ.compute_velocities!(S, G; plans=plans, params=par)

# Verify flow setup
u_max = maximum(abs.(S.u))
v_max = maximum(abs.(S.v))
@printf("\nFlow verification:\n")
@printf("  Max |u| = %.3f (nondim)\n", u_max)
@printf("  Max |v| = %.3f (nondim)\n", v_max)

# ============================================================================
#                       TIME INTEGRATION
# ============================================================================

println("\n" * "="^70)
println("Starting time integration...")
println("="^70)

# Output interval (in inertial periods)
output_interval_IP = 1.0
output_interval_steps = round(Int, output_interval_IP * T_inertial_nd / dt)

# Storage for diagnostics
times = Float64[]
wave_energies_B = Float64[]
wave_energies_A = Float64[]

# Initial diagnostics
EB, EA = QGYBJ.wave_energy(S.B, S.A)
push!(times, 0.0)
push!(wave_energies_B, EB)
push!(wave_energies_A, EA)

@printf("\nt = 0.0 IP: E_B = %.4e, E_A = %.4e\n", EB, EA)

# Run projection step (Forward Euler for first step)
println("\nRunning projection step...")
QGYBJ.first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask)

# Create states for leapfrog
Sn = deepcopy(S)     # State at time n
Snm1 = deepcopy(S)   # State at time n-1 (will store filtered values)
Snp1 = deepcopy(S)   # State at time n+1

# Main time loop
for step in 1:nt
    # Leapfrog step
    QGYBJ.leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a_ell, dealias_mask=L_mask)

    # Rotate states: Snm1 now has filtered n values, Snp1 has n+1 values
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1

    # Output diagnostics
    if step % output_interval_steps == 0
        t_IP = step * dt / T_inertial_nd
        EB, EA = QGYBJ.wave_energy(Sn.B, Sn.A)

        push!(times, t_IP)
        push!(wave_energies_B, EB)
        push!(wave_energies_A, EA)

        @printf("t = %.1f IP: E_B = %.4e, E_A = %.4e\n", t_IP, EB, EA)
    end
end

println("\n" * "="^70)
println("Simulation complete!")
println("="^70)

# ============================================================================
#                       FINAL ANALYSIS
# ============================================================================

# Copy final state
S_final = Sn

# Compute wave energy at different depths
println("\nWave Energy Distribution (Final State):")
println("-"^50)

# Surface layer (top 10% of domain)
k_surface = round(Int, 0.9 * nz):nz
B_surface = S_final.B[:,:,k_surface]
EB_surface = sum(abs2.(B_surface))

# Interior (10-50% depth)
k_interior = round(Int, 0.5 * nz):round(Int, 0.9 * nz)
B_interior = S_final.B[:,:,k_interior]
EB_interior = sum(abs2.(B_interior))

# Deep (below 50%)
k_deep = 1:round(Int, 0.5 * nz)
B_deep = S_final.B[:,:,k_deep]
EB_deep = sum(abs2.(B_deep))

EB_total = EB_surface + EB_interior + EB_deep

@printf("  Surface layer (top 10%%):    %.1f%%\n", 100 * EB_surface / EB_total)
@printf("  Interior (10-50%% depth):    %.1f%%\n", 100 * EB_interior / EB_total)
@printf("  Deep (below 50%%):           %.1f%%\n", 100 * EB_deep / EB_total)

println("\nKey Results (compare with Asselin et al. 2020):")
println("-"^50)
println("1. Wave energy concentrates in anticyclone (check visually)")
println("2. Shear bands form with linear k, m growth")
println("3. Band slope is time-independent: dz/dx = -(3fγ|z|/2N²)^(1/3)")
println("4. Straining is ineffective in steady barotropic flow")

# Save final state info
println("\nFinal state saved in: S_final")
println("Grid saved in: G")
println("Parameters saved in: par")

# Return key objects for further analysis
(S_final, G, par, times, wave_energies_B, wave_energies_A)
