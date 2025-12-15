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
- Steady barotropic vortex dipole
- Surface-confined near-inertial waves (k=0 initially)
- Uniform stratification N² = 1, f = 1

KEY PHYSICS:
------------
1. Waves acquire horizontal structure from vorticity gradients (ζ-refraction)
2. Wave energy concentrates in anticyclone (ζ < 0)
3. For steady barotropic flow, straining is INEFFECTIVE

================================================================================
=#

using QGYBJ
using Printf

# ============================================================================
#                       SIMULATION PARAMETERS
# ============================================================================

# Grid resolution
const nx = 128
const ny = 128
const nz = 64

# Simulation duration
const n_inertial_periods = 15
const T_inertial = 2π  # Inertial period = 2π/f = 2π when f=1
const dt = 0.001
const nt = round(Int, n_inertial_periods * T_inertial / dt)

# Wave amplitude relative to flow
const u0_wave = 0.3  # Wave/flow velocity ratio

# Surface layer depth (fraction of domain)
const sigma_z = 0.01 * 2π  # 1% of domain depth

println("="^70)
println("Asselin et al. (2020) Dipole Simulation")
println("="^70)
@printf("\nResolution: %d × %d × %d\n", nx, ny, nz)
@printf("Duration: %.1f inertial periods (%d steps)\n", n_inertial_periods, nt)
@printf("Wave amplitude: %.2f (relative to flow)\n", u0_wave)

# Create parameters - f0=1, N2=1 by default
par = QGYBJ.default_params(
    nx = nx, ny = ny, nz = nz,
    dt = dt, nt = nt,
    W2F = u0_wave^2,       # Wave-to-flow ratio squared
    ybj_plus = true,       # YBJ+ formulation
    fixed_flow = true,     # Steady flow
    no_wave_feedback = true
)

# ============================================================================
#                       INITIALIZE GRID AND STATE
# ============================================================================

println("\nInitializing grid and state...")
G = QGYBJ.init_grid(par)
S = QGYBJ.init_state(G)
plans = QGYBJ.plan_transforms!(G)

# ============================================================================
#                       SET UP DIPOLE STREAMFUNCTION
# ============================================================================
#
# ψ = (U/κ) sin(κx) cos(κy)  with U=1, κ=1

println("Setting up dipole streamfunction...")

const k_dipole = 1.0
const psi_amp = 1.0 / k_dipole

psi_real = zeros(Float64, nx, ny, nz)
for k in 1:nz, j in 1:ny, i in 1:nx
    x = (i-1) * G.dx
    y = (j-1) * G.dy
    psi_real[i,j,k] = psi_amp * sin(k_dipole * (x - π/2)) * cos(k_dipole * y)
end

QGYBJ.fft_forward!(S.psi, complex.(psi_real), plans)

# Compute q = ∇²ψ = -kh² × ψ
for k in 1:nz, j in 1:ny, i in 1:nx
    kh2 = G.kx[i]^2 + G.ky[j]^2
    S.q[i,j,k] = -kh2 * S.psi[i,j,k]
end

# ============================================================================
#                       SET UP WAVE INITIAL CONDITION
# ============================================================================
#
# Surface-confined: B(z) = u₀ exp(-(z_surface - z)² / σ²)
# Only (kx,ky)=(0,0) mode is nonzero for horizontally uniform IC

println("Setting up wave initial condition...")

z_surface = 2π
for k in 1:nz
    depth = z_surface - G.z[k]
    wave_profile = exp(-(depth^2) / (sigma_z^2))
    S.B[1, 1, k] = u0_wave * wave_profile * (nx * ny)  # FFT normalization
end

@printf("  Surface amplitude: %.3f\n", real(S.B[1,1,nz]) / (nx*ny))

# ============================================================================
#                       TIME INTEGRATION
# ============================================================================

a_ell = QGYBJ.a_ell_ut(par, G)
L_mask = QGYBJ.dealias_mask(G)
QGYBJ.compute_velocities!(S, G; plans=plans, params=par)

println("\n" * "="^70)
println("Starting time integration...")
println("="^70)

output_interval = round(Int, T_inertial / dt)

# Initial energy
EB = sum(abs2.(S.B)) / (nx * ny * nz)
@printf("\nt = 0.0 IP: E_B = %.4e\n", EB)

# First step
QGYBJ.first_projection_step!(S, G, par, plans; a=a_ell, dealias_mask=L_mask)

Sn = deepcopy(S)
Snm1 = deepcopy(S)
Snp1 = deepcopy(S)

# Main loop
for step in 1:nt
    QGYBJ.leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a_ell, dealias_mask=L_mask)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1

    if step % output_interval == 0
        t_IP = step * dt / T_inertial
        EB = sum(abs2.(Sn.B)) / (nx * ny * nz)
        @printf("t = %.1f IP: E_B = %.4e\n", t_IP, EB)
    end
end

println("\n" * "="^70)
println("Simulation complete!")
println("="^70)

# Return for analysis
(Sn, G, par)
