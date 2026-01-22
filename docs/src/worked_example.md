# [Worked Example](@id worked_example)

```@meta
CurrentModule = QGYBJplus
```

This comprehensive tutorial builds a complete QG-YBJ+ simulation from scratch. We'll explain every step so you understand what's happening and why.

**Time required**: ~30 minutes

**What you'll build**: A simulation of near-inertial waves interacting with ocean eddies, including:
- Dipole vortex (anticyclone + cyclone pair)
- Surface-trapped wave initial condition
- Wave refraction and energy diagnostics
- NetCDF output for visualization

---

## Overview

We'll simulate a 500km × 500km × 4km ocean domain with:
- A dipole vortex (cyclone + anticyclone pair)
- Surface-trapped near-inertial waves
- 5 inertial periods of evolution

The waves will refract toward the anticyclone, demonstrating the key wave-trapping phenomenon.

---

## Step 1: Load Packages and Set Parameters

```julia
using QGYBJplus
using Printf

# ============================================================================
#                       PHYSICAL PARAMETERS
# ============================================================================

# Domain size (realistic ocean values)
const Lx = 500e3    # 500 km horizontal extent
const Ly = 500e3    # 500 km horizontal extent
const Lz = 4000.0   # 4 km depth

# Grid resolution
const nx = 128      # Grid points in x
const ny = 128      # Grid points in y
const nz = 64       # Grid points in z (vertical levels)

# Physical constants
const f₀ = 1e-4     # Coriolis parameter [s⁻¹] (mid-latitude value)
const N² = 1e-5     # Buoyancy frequency squared [s⁻²]

# Time stepping
const T_inertial = 2π / f₀           # Inertial period ≈ 17.5 hours
const dt = 10.0                       # Time step [s] - use IMEX for larger dt
const n_inertial_periods = 5          # Simulation duration
const nt = round(Int, n_inertial_periods * T_inertial / dt)

println("Simulation setup:")
println("  Domain: $(Lx/1e3) km × $(Ly/1e3) km × $(Lz/1e3) km")
println("  Grid: $nx × $ny × $nz")
println("  Inertial period: $(T_inertial/3600) hours")
println("  Time steps: $nt ($(n_inertial_periods) inertial periods)")
```

**What's happening?**
- `f₀ = 1e-4 s⁻¹` is the Coriolis parameter at ~45° latitude
- `N² = 1e-5 s⁻²` gives N ≈ 0.003 s⁻¹, typical for the ocean interior
- The inertial period is `2π/f₀ ≈ 17.5 hours`

---

## Step 2: Create Parameters and Initialize

```julia
# ============================================================================
#                       CREATE MODEL PARAMETERS
# ============================================================================

# Note: Lx, Ly, Lz are REQUIRED - there are no defaults
par = default_params(
    # Grid dimensions
    nx = nx, ny = ny, nz = nz,

    # Domain size [meters]
    Lx = Lx, Ly = Ly, Lz = Lz,

    # Physics
    f₀ = f₀,
    N² = N²,
    ybj_plus = true,       # Use YBJ+ formulation (includes k² correction)

    # Time stepping
    dt = dt,
    nt = nt,

    # Coupling options
    no_wave_feedback = true,  # Disable wave feedback (qʷ) for clarity
    fixed_flow = false,        # Let mean flow evolve

    # Dissipation (hyperdiffusion)
    νₕ₁ = 1e8,    # Biharmonic viscosity [m⁴/s]
    ilap1 = 2,    # 4th order (∇⁴)
)

println("\nParameters created:")
println("  f₀ = $(par.f₀)")
println("  N² = $(par.N²)")
println("  dt = $(par.dt) s")
println("  ybj_plus = $(par.ybj_plus)")
```

**Key parameters explained:**

| Parameter | Value | Meaning |
|:----------|:------|:--------|
| `ybj_plus=true` | Use full YBJ+ | Includes `k²/4` correction in L⁺ operator |
| `no_wave_feedback=true` | One-way coupling | Waves don't affect mean flow (simpler) |
| `νₕ₁, ilap1` | Biharmonic | Scale-selective dissipation at grid scale |

---

## Step 3: Setup Grid, State, and FFT Plans

```julia
# ============================================================================
#                       INITIALIZE SIMULATION
# ============================================================================

# setup_model returns: Grid, State, FFT plans, elliptic coefficient
G, S, plans, a_ell = setup_model(par)

# Get dealiasing mask (2/3 rule)
L = dealias_mask(G)

println("\nGrid initialized:")
println("  x range: [$(G.x[1]/1e3), $(G.x[end]/1e3)] km")
println("  z range: [$(G.z[1]/1e3), $(G.z[end]/1e3)] km")
println("  dx = $(G.dx/1e3) km, dz = $(G.dz[1]) m")
```

**What did `setup_model` create?**

- **G (Grid)**: x, y, z coordinates; kx, ky wavenumbers; kh² for each mode; dx, dy, dz spacings
- **S (State)**: q, B (prognostic, spectral); psi, A, C (diagnostic, spectral); u, v, w (velocities, real space)
- **plans**: FFTW plans for efficient transforms
- **a_ell**: Coefficient array for elliptic inversions: `a(z) = f₀²/N²(z)`

---

## Step 4: Set Up Initial Conditions

### 4a: Create a Dipole Vortex

```julia
# ============================================================================
#                       DIPOLE INITIAL CONDITION
# ============================================================================

# Dipole parameters
U0 = 0.3           # Maximum velocity [m/s]
R0 = 50e3          # Vortex radius [m]
separation = 150e3 # Distance between vortex centers [m]

# Dipole centers (relative to domain center)
x_center = Lx / 2
y_center = Ly / 2
x_pos = x_center + separation / 2   # Anticyclone (positive ψ)
x_neg = x_center - separation / 2   # Cyclone (negative ψ)

# Build streamfunction in physical space
psi_phys = zeros(nz, nx, ny)

for k in 1:nz, j in 1:ny, i in 1:nx
    x = G.x[i]
    y = G.y[j]

    # Distance from each vortex center
    r_pos = sqrt((x - x_pos)^2 + (y - y_center)^2)
    r_neg = sqrt((x - x_neg)^2 + (y - y_center)^2)

    # Gaussian vortices
    psi_anticyclone = +U0 * R0 * exp(-(r_pos/R0)^2)  # Positive ψ = anticyclone
    psi_cyclone     = -U0 * R0 * exp(-(r_neg/R0)^2)  # Negative ψ = cyclone

    psi_phys[k, i, j] = psi_anticyclone + psi_cyclone
end

# Transform to spectral space
psi_spectral = zeros(ComplexF64, nz, nx, ny)
for k in 1:nz
    psi_spectral[k, :, :] = fft(psi_phys[k, :, :])
end
S.psi .= psi_spectral

# Compute q from ψ: q = ∇²ψ + (f²/N²)∂²ψ/∂z²
compute_q_from_psi!(S, G, par)

println("\nDipole vortex created:")
println("  Max velocity: $U0 m/s")
println("  Vortex radius: $(R0/1e3) km")
println("  Separation: $(separation/1e3) km")
```

**Physical meaning:**
- **Anticyclone** (positive ψ): Clockwise rotation, **traps waves**
- **Cyclone** (negative ψ): Counter-clockwise rotation, **expels waves**

### 4b: Create Surface Wave Initial Condition

```julia
# ============================================================================
#                       WAVE INITIAL CONDITION
# ============================================================================

# Horizontally uniform, surface-trapped wave
A0 = 0.1           # Wave amplitude [m/s]
decay_depth = 100.0  # e-folding depth [m]

# Build wave amplitude in physical space
A_phys = zeros(ComplexF64, nz, nx, ny)

for k in 1:nz
    z = G.z[k]                              # z runs from -Lz to 0
    depth = -z                              # Depth from surface (positive)
    vertical_structure = exp(-depth / decay_depth)
    A_phys[k, :, :] .= A0 * vertical_structure  # Uniform horizontally
end

# Transform to spectral space
A_spectral = zeros(ComplexF64, nz, nx, ny)
for k in 1:nz
    A_spectral[k, :, :] = fft(A_phys[k, :, :])
end
S.A .= A_spectral

# Compute B from A using L⁺ operator
# B = L⁺(A) = ∂/∂z[(f²/N²)∂A/∂z] - (k²/4)A
# For uniform A in x,y, this simplifies significantly
# Here we just set B = A as initial approximation (refined by inversion)
S.L⁺A .= S.A

println("\nWave field created:")
println("  Amplitude: $A0 m/s")
println("  Decay depth: $decay_depth m")
println("  Vertical structure: exp(-z/$decay_depth)")
```

This mimics wind-generated near-inertial waves that are strongest near the surface and decay with depth.

---

## Step 5: Run the Time-Stepping Loop

```julia
# ============================================================================
#                       TIME INTEGRATION
# ============================================================================

# Create state arrays for leapfrog (need 3 time levels)
Snm1 = copy_state(S)  # n-1
Sn = copy_state(S)    # n
Snp1 = copy_state(S)  # n+1

# Initialize with projection step (Forward Euler)
println("\nRunning projection step...")
first_projection_step!(Sn, G, par, plans; a=a_ell, dealias_mask=L)
copy_state!(Snm1, Sn)

# Storage for diagnostics
times = Float64[]
wave_energies = Float64[]
flow_energies = Float64[]

# Diagnostic output interval (every 0.5 inertial periods)
diag_interval = round(Int, 0.5 * T_inertial / dt)

println("Starting main time loop...")
println("="^60)

for step in 1:nt
    # Leapfrog step with Robert-Asselin filter
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans;
                   a=a_ell, dealias_mask=L)

    # Rotate time levels (efficient pointer swap)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1

    # Periodic diagnostics
    if step % diag_interval == 0
        current_time = step * dt
        time_IP = current_time / T_inertial  # Time in inertial periods

        # Compute energies
        KE = flow_kinetic_energy(Sn.u, Sn.v)
        # Wave kinetic energy per YBJ+ equation (4.7): WKE = (1/2)|LA|²
        WKE, WPE, WCE = compute_detailed_wave_energy(Sn, G, par)

        # Store for later analysis
        push!(times, time_IP)
        push!(wave_energies, WKE)
        push!(flow_energies, KE)

        @printf("  t = %.2f IP | Flow KE = %.4e | Wave KE = %.4e\n",
                time_IP, KE, WKE)
    end
end

println("="^60)
println("Simulation complete!")
```

**What happens each time step?**

1. **Invert** q → ψ and B → A
2. **Compute** u, v from ψ
3. **Evaluate** tendencies: J(ψ,q), J(ψ,B), refraction ζ·B, dispersion k²A
4. **Update**: φ^{n+1} = φ^{n-1} + 2·dt·F^n
5. **Rotate** time levels: Snm1 ← Sn ← Snp1

---

## Step 6: Analyze Results

```julia
# ============================================================================
#                       ANALYZE RESULTS
# ============================================================================

# Get final fields
final_psi = Sn.psi
final_B = Sn.L⁺A
final_A = Sn.A

# Compute final vorticity ζ = ∇²ψ
# In spectral space: ζ̂ = -k²·ψ̂
zeta_spectral = zeros(ComplexF64, nz, nx, ny)
for k in 1:nz, j in 1:ny, i in 1:nx
    kh2 = G.kx[i]^2 + G.ky[j]^2
    zeta_spectral[k, i, j] = -kh2 * final_psi[k, i, j]
end

# Extract horizontal slice at mid-depth
k_mid = nz ÷ 2
psi_slice = slice_horizontal(Sn.psi, G, plans; k=k_mid)
A_slice = slice_horizontal(Sn.A, G, plans; k=k_mid)

println("\nFinal state analysis:")
println("  ψ at z=$(G.z[k_mid]) m:")
println("    min = $(minimum(real.(psi_slice)))")
println("    max = $(maximum(real.(psi_slice)))")
println("  |A| at z=$(G.z[k_mid]) m:")
println("    min = $(minimum(abs.(A_slice)))")
println("    max = $(maximum(abs.(A_slice)))")

# Check wave concentration
# Waves should be stronger in the anticyclone (positive ψ region)
println("\nWave refraction check:")
println("  If waves concentrated in anticyclone, max|A| should be")
println("  near max(ψ) location")
```

---

## Step 7: Save Output

```julia
# ============================================================================
#                       SAVE OUTPUT
# ============================================================================

using NCDatasets

output_file = "dipole_waves.nc"

NCDataset(output_file, "c") do ds
    # Define dimensions
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)
    defDim(ds, "z", nz)
    defDim(ds, "time", length(times))

    # Coordinate variables
    x_var = defVar(ds, "x", Float64, ("x",))
    y_var = defVar(ds, "y", Float64, ("y",))
    z_var = defVar(ds, "z", Float64, ("z",))
    t_var = defVar(ds, "time", Float64, ("time",))

    x_var[:] = G.x
    y_var[:] = G.y
    z_var[:] = G.z
    t_var[:] = times

    # Final state (mid-depth slice)
    psi_var = defVar(ds, "psi_final", Float64, ("x", "y"))
    A_var = defVar(ds, "A_amplitude_final", Float64, ("x", "y"))

    psi_var[:, :] = real.(psi_slice)'
    A_var[:, :] = abs.(A_slice)'

    # Time series
    KE_var = defVar(ds, "flow_KE", Float64, ("time",))
    WE_var = defVar(ds, "wave_energy", Float64, ("time",))

    KE_var[:] = flow_energies
    WE_var[:] = wave_energies

    # Attributes
    ds.attrib["description"] = "QG-YBJ+ dipole-wave simulation"
    ds.attrib["inertial_period_hours"] = T_inertial / 3600
end

println("\nOutput saved to: $output_file")
```

---

## Complete Script

Here's the full script you can copy and run:

```julia
# dipole_waves.jl - Complete QG-YBJ+ simulation example
#
# Run with: julia --project dipole_waves.jl

using QGYBJplus
using Printf

# === Parameters ===
const Lx, Ly, Lz = 500e3, 500e3, 4000.0
const nx, ny, nz = 128, 128, 64
const f₀, N² = 1e-4, 1e-5
const T_inertial = 2π / f₀
const dt = 10.0
const nt = round(Int, 5 * T_inertial / dt)

# === Create model ===
par = default_params(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz,
                     f₀=f₀, N²=N², dt=dt, nt=nt,
                     ybj_plus=true, no_wave_feedback=true)
G, S, plans, a_ell = setup_model(par)
L = dealias_mask(G)

# === Initial conditions ===
# (Add dipole and wave initialization here - see sections above)

# === Time stepping ===
Snm1, Sn, Snp1 = copy_state(S), copy_state(S), copy_state(S)
first_projection_step!(Sn, G, par, plans; a=a_ell, dealias_mask=L)
copy_state!(Snm1, Sn)

for step in 1:nt
    leapfrog_step!(Snp1, Sn, Snm1, G, par, plans; a=a_ell, dealias_mask=L)
    Snm1, Sn, Snp1 = Sn, Snp1, Snm1
end

println("Done! Final KE = ", flow_kinetic_energy(Sn.u, Sn.v))
```

---

## What's Next?

Now that you've built a complete simulation:

1. **[Configuration Guide](@ref configuration)** - Customize all parameters
2. **[Stratification](@ref stratification)** - Use realistic N²(z) profiles
3. **[MPI Parallelization](@ref parallel)** - Scale to larger domains
4. **[Particle Advection](@ref particles)** - Track Lagrangian trajectories
5. **[Physics Overview](@ref physics-overview)** - Understand the equations deeply

---

## Common Modifications

### Use IMEX Time Stepping (10x faster)

```julia
# Replace leapfrog with IMEX-CN
imex_ws = init_imex_workspace(Sn, G)
first_imex_step!(Sn, G, par, plans, imex_ws; a=a_ell, dealias_mask=L)

for step in 1:nt
    imex_cn_step!(Snp1, Sn, G, par, plans, imex_ws; a=a_ell, dealias_mask=L)
    parent(Sn.L⁺A) .= parent(Snp1.L⁺A)
    parent(Sn.q) .= parent(Snp1.q)
end
```

### Add Particle Tracking

```julia
# Create particles at mid-depth
pconfig = particles_in_box(-2000.0; x_max=Lx, y_max=Ly, nx=20, ny=20)
tracker = ParticleTracker(pconfig, G)
initialize_particles!(tracker, pconfig)

# In time loop, add:
advect_particles!(tracker, Sn, G, dt, current_time)

# After simulation:
write_particle_trajectories("trajectories.nc", tracker)
```

### Use Realistic Stratification

```julia
par = default_params(
    Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz,
    stratification = :skewed_gaussian,  # Realistic pycnocline
    # Other parameters...
)
G, S, plans, a_ell, N2_profile = setup_model_with_profile(par)
```

### Use an Analytical N(z) Profile

```julia
a = 0.01    # s^-1
b = -2.0e-6 # s^-1 m^-1
@inline N(z) = a + b * z  # z is negative below the surface

domain = create_domain_config(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz)
strat = create_stratification_config(type=:analytical, N_func=N)
sim = setup_simulation(domain, strat)
```
