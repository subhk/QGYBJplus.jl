# [Quick Start Tutorial](@id quickstart)

```@meta
CurrentModule = QGYBJplus
```

Run your first QGYBJ+.jl simulation in 5 minutes.

---

## Minimal Example

```@raw html
<div class="quickstart-card">
```

```julia
using QGYBJplus

# Configure (Lx, Ly, Lz are REQUIRED)
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,  # Domain size [m]
    nx=64, ny=64, nz=32,             # Grid points
    dt=0.001, total_time=1.0,        # Time stepping
    output_interval=100
)

# Run
result = run_simple_simulation(config)

# Check results
println("Kinetic Energy: ", flow_kinetic_energy(result.state.u, result.state.v))
```

```@raw html
</div>
```

---

## Step-by-Step Breakdown

### Step 1: Create Configuration

```julia
config = create_simple_config(
    Lx = 500e3,        # Domain length x [m] (REQUIRED)
    Ly = 500e3,        # Domain length y [m] (REQUIRED)
    Lz = 4000.0,       # Domain depth [m] (REQUIRED)
    nx = 64, ny = 64, nz = 32,  # Grid dimensions
    dt = 0.001,        # Time step
    total_time = 1.0,  # Total simulation time
)
```

!!! warning "Domain size is required"
    There are no default values for `Lx`, `Ly`, `Lz`. Omitting them causes a `MethodError`.

### Step 2: Run Simulation

```julia
result = run_simple_simulation(config)
```

This returns a `Simulation` object containing:
- `result.state` — Final state with all fields
- `result.grid` — Grid information
- `result.params` — Simulation parameters

### Step 3: Access Results

```julia
state = result.state

# Spectral fields (complex, in Fourier space)
state.psi    # Streamfunction
state.L⁺A    # Wave envelope (L⁺A where L⁺ = L - k_h²/4)
state.A      # Wave amplitude (diagnosed from L⁺A)
state.C      # Vertical derivative of A

# Physical fields (real, in physical space)
state.u      # Zonal velocity
state.v      # Meridional velocity
state.w      # Vertical velocity
```

### Step 4: Compute Diagnostics

```julia
# Mean flow kinetic energy
KE = flow_kinetic_energy(state.u, state.v)

# Wave energy components per YBJ+ equation (4.7)
WKE, WPE, WCE = compute_detailed_wave_energy(state, result.grid, result.params)

# Vertically-averaged wave kinetic energy (uses LA = L⁺A + k_h²/4 * A)
WE = wave_energy_vavg(state.L⁺A, state.A, grid, plans)
```

---

## Common Configuration Options

```@raw html
<div class="feature-grid">
<div class="feature-card">
    <h3>Physics Options</h3>
    <p>Control the physical model behavior</p>
</div>
<div class="feature-card">
    <h3>Stratification</h3>
    <p>Choose ocean density profile</p>
</div>
</div>
```

```julia
config = create_simple_config(
    Lx=500e3, Ly=500e3, Lz=4000.0,
    nx=64, ny=64, nz=32,

    # Physics options
    ybj_plus = true,          # YBJ+ formulation (default)
    linear = false,           # Set true to disable nonlinear terms
    inviscid = true,          # Set true for no dissipation
    no_wave_feedback = true,  # Set true for one-way coupling (eddies → waves only)

    # Stratification
    stratification_type = :constant_N,  # or :skewed_gaussian
)
```

| Option | Default | Description |
|:-------|:--------|:------------|
| `ybj_plus` | `true` | Use YBJ+ formulation (recommended) |
| `linear` | `false` | Disable nonlinear advection terms |
| `inviscid` | `false` | Disable all dissipation |
| `no_wave_feedback` | `false` | Disable wave feedback on mean flow |
| `stratification_type` | `:constant_N` | Ocean density profile type |

---

## Output Files

By default, simulations save to `./output_simple/`:

```
output_simple/
├── state0001.nc          # Field snapshots
├── state0002.nc
└── diagnostic/           # Energy time series
    ├── wave_KE.nc
    ├── mean_flow_KE.nc
    └── total_energy.nc
```

---

## What's Next?

```@raw html
<div class="learning-path">
<div class="path-step">
    <div class="step-number">→</div>
    <div class="step-content">
        <strong><a href="../worked_example/">Worked Example</a></strong> — Detailed step-by-step walkthrough
    </div>
</div>
<div class="path-step">
    <div class="step-number">→</div>
    <div class="step-content">
        <strong><a href="../guide/configuration/">Configuration Guide</a></strong> — All available parameters
    </div>
</div>
<div class="path-step">
    <div class="step-number">→</div>
    <div class="step-content">
        <strong><a href="../advanced/parallel/">MPI Parallelization</a></strong> — Run large-scale simulations
    </div>
</div>
</div>
```
