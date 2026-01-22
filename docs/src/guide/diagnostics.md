# [Diagnostics](@id diagnostics)

```@meta
CurrentModule = QGYBJplus
```

This page describes diagnostic quantities and analysis tools in QGYBJ+.jl.

## Energy Diagnostics

### Flow Kinetic Energy

The kinetic energy of the balanced flow:

```math
KE = \frac{1}{2}\int (u^2 + v^2) \, dV
```

```julia
KE = flow_kinetic_energy(state.u, state.v)
```

### Flow Potential Energy (Spectral)

Available potential energy computed in spectral space with proper dealiasing:

```math
PE = \frac{1}{2}\int \frac{f_0^2}{N^2}\left(\frac{\partial\psi}{\partial z}\right)^2 dV
```

```julia
# Compute buoyancy field first (b = ∂ψ/∂z)
bk = similar(state.psi)
# ... compute vertical derivative ...
PE = flow_potential_energy_spectral(bk, grid, params)
```

### Wave Energy

The wave kinetic energy is computed per YBJ+ equation (4.7):

```math
\text{WKE} = \frac{1}{2}\int |LA|^2 \, dV
```

where ``LA`` is computed directly using the L operator from equation (1.3):

```math
L = \partial_z \left( \frac{f^2}{N^2} \partial_z \right)
```

So ``LA = \partial_z(a(z) \times C)`` where ``a(z) = f^2/N^2`` and ``C = \partial A/\partial z``.

```julia
# Detailed wave energy components
WKE, WPE, WCE = compute_detailed_wave_energy(state, grid, params)

# Simple wave kinetic energy
WE = compute_wave_energy(state, grid, plans; params=params)

# Vertically-averaged wave kinetic energy using LA = L⁺A + (k_h²/4)A
WE = wave_energy_vavg(state.L⁺A, state.A, grid, plans)
```

!!! note "Physical interpretation"
    WKE uses ``|LA|^2`` where LA is computed as ``LA = L^+A + (k_h^2/4)A`` in spectral space.
    This uses the original YBJ ``L`` operator (not ``L^+``) for physical consistency.

## Energy Diagnostics Output Files

QGYBJ+.jl automatically saves energy diagnostics to separate files in a dedicated `diagnostic/` folder, following the structure used in the Fortran QG_YBJp code.

### Output Folder Structure

```
output_dir/
├── state0001.nc              # Field snapshots
├── state0002.nc
├── diagnostics_0001.nc       # Legacy combined diagnostics
└── diagnostic/               # Separate energy files
    ├── wave_KE.nc            # Wave kinetic energy time series
    ├── wave_PE.nc            # Wave potential energy time series
    ├── wave_CE.nc            # Wave correction energy (YBJ+)
    ├── mean_flow_KE.nc       # Mean flow kinetic energy
    ├── mean_flow_PE.nc       # Mean flow potential energy
    └── total_energy.nc       # Summary file with all energies
```

### Wave Kinetic Energy (WKE)

The wave kinetic energy is computed per YBJ+ equation (4.7):

```math
\text{WKE} = \frac{1}{2} \sum_{k_x, k_y, z} |LA|^2 - \text{(dealiasing correction)}
```

where ``LA = L^+A + (k_h^2/4)A`` in spectral space. This relationship comes from:
- ``L^+A = LA - \frac{k_h^2}{4}A`` (definition of L⁺ operator)
- In spectral space: ``\Delta \rightarrow -k_h^2``, so ``L^+A = LA - (k_h^2/4)A``
- Therefore: ``LA = L^+A + (k_h^2/4)A``

**Physical interpretation**: WKE represents the kinetic energy contained in the near-inertial wave field, computed from the wave velocity amplitude ``LA`` using the original YBJ ``L`` operator (not the evolved envelope ``L^+A``). This ensures consistency with the energy budget in the YBJ+ formulation.

### Wave Potential Energy (WPE)

The wave potential energy captures vertical wave structure through ``C = \partial A/\partial z``:

```math
\text{WPE} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{k_h^2}{2 a_{ell}} \left( |C_R|^2 + |C_I|^2 \right)
```

where:
- ``a_{ell} = f_0^2 / N^2`` is the elliptic coefficient
- ``C = \partial A / \partial z`` is the vertical derivative of wave amplitude
- ``k_h^2 = k_x^2 + k_y^2`` is the horizontal wavenumber squared

**Physical interpretation**: WPE represents the potential energy from wave-induced isopycnal displacements. It scales with ``N^2 \eta^2`` where ``\eta`` is the vertical displacement.

### Wave Correction Energy (WCE)

The YBJ+ formulation includes a higher-order correction term:

```math
\text{WCE} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{k_h^4}{8 a_{ell}^2} \left( |A_R|^2 + |A_I|^2 \right)
```

**Physical interpretation**: WCE is a higher-order correction from the YBJ+ equation that accounts for horizontal wave dispersion. It becomes important for short horizontal wavelengths.

### Mean Flow Kinetic Energy

The balanced flow kinetic energy is computed from the geostrophic velocities:

```math
\text{KE}_{flow} = \frac{1}{2} \sum_{k_x, k_y, z} \left( |u_k|^2 + |v_k|^2 \right) - \frac{1}{2} |u(k_h=0)|^2
```

where the velocities are derived from the streamfunction:
```math
u = -\frac{\partial \psi}{\partial y} = -ik_y \hat{\psi}, \quad v = \frac{\partial \psi}{\partial x} = ik_x \hat{\psi}
```

This gives:
```math
|u|^2 + |v|^2 = k_h^2 |\hat{\psi}|^2
```

**Physical interpretation**: ``\text{KE}_{flow}`` represents the kinetic energy of the large-scale quasi-geostrophic eddies and jets.

### Mean Flow Potential Energy

The available potential energy from buoyancy:

```math
\text{PE}_{flow} = \frac{1}{2} \sum_{k_x, k_y, z} \frac{f_0^2}{N^2} |b_k|^2
```

where buoyancy ``b`` is related to the streamfunction via thermal wind balance:
```math
b = \frac{\partial \psi}{\partial z}
```

**Physical interpretation**: ``\text{PE}_{flow}`` represents the energy stored in tilted isopycnals (density surfaces). It can be released through baroclinic instability.

### Total Energy Conservation

In the inviscid limit, the total energy is conserved:

```math
E_{total} = \underbrace{\text{KE}_{flow} + \text{PE}_{flow}}_{\text{Mean flow}} + \underbrace{\text{WKE} + \text{WPE} + \text{WCE}}_{\text{Waves}} = \text{const}
```

Energy exchange between waves and mean flow occurs via:
- **Refraction**: Waves gain/lose energy from vorticity gradients
- **Wave feedback** ``q^w``: Waves modify the effective PV

### Using the EnergyDiagnosticsManager

The energy diagnostics manager is automatically created during simulation setup:

```julia
# Energy diagnostics are computed and saved automatically during simulation
sim = setup_simulation(config)
run_simulation!(sim)

# After simulation, files are in:
# output_dir/diagnostic/wave_KE.nc
# output_dir/diagnostic/wave_PE.nc
# etc.
```

For manual control:

```julia
using QGYBJplus

# Create manager manually
energy_manager = EnergyDiagnosticsManager(
    "output_dir";
    output_interval=1.0  # Time between outputs
)

# Record energies at each diagnostic time
record_energies!(
    energy_manager,
    current_time,
    wave_KE, wave_PE, wave_CE,
    mean_flow_KE, mean_flow_PE
)

# Write all files at end
write_all_energy_files!(energy_manager)
```

### Reading Energy Output Files

```julia
using NCDatasets

# Read wave KE time series
ds = NCDataset("output_dir/diagnostic/wave_KE.nc", "r")
time = ds["time"][:]
wave_KE = ds["wave_KE"][:]
close(ds)

# Read total energy summary
ds = NCDataset("output_dir/diagnostic/total_energy.nc", "r")
time = ds["time"][:]
total_wave = ds["total_wave_energy"][:]
total_flow = ds["total_flow_energy"][:]
total = ds["total_energy"][:]
close(ds)

# Plot energy evolution
using Plots
plot(time, total, label="Total", linewidth=2)
plot!(time, total_flow, label="Mean flow", linestyle=:dash)
plot!(time, total_wave, label="Waves", linestyle=:dot)
xlabel!("Time")
ylabel!("Energy")
```

### Energy Budget Verification

Check energy conservation:

```julia
# After simulation
ds = NCDataset("output_dir/diagnostic/total_energy.nc", "r")
E = ds["total_energy"][:]
close(ds)

# Relative change
dE_rel = (E[end] - E[1]) / E[1]
println("Relative energy change: $(dE_rel)")

if abs(dE_rel) < 1e-6
    println("Energy well conserved")
else
    println("Check time step or dissipation settings")
end
```

## MPI-Aware Energy Functions

For parallel simulations, use the global reduction versions:

```julia
# Physical-space energy with MPI reduction
KE = flow_kinetic_energy_global(state.u, state.v, mpi_config)
WE_B, WE_A = wave_energy_global(state.L⁺A, state.A, mpi_config)

# Spectral energy with MPI reduction
KE_spectral = flow_kinetic_energy_spectral_global(uk, vk, grid, params; mpi_config=mpi_config)
PE_spectral = flow_potential_energy_spectral_global(bk, grid, params; mpi_config=mpi_config)
WKE, WPE, WCE = wave_energy_spectral_global(BR, BI, AR, AI, CR, CI, grid, params; mpi_config=mpi_config)
```

## Wave Diagnostics

### Wave Amplitude

```julia
# |A|² field
A2 = abs2.(state.A)

# Maximum amplitude
A_max = maximum(sqrt.(A2))

# Volume-averaged amplitude
using Statistics
A_rms = sqrt(mean(A2))
```

### Wave Velocities

Compute wave-induced velocities (in-place):

```julia
# Updates state.u, state.v with wave velocities
compute_wave_velocities!(state, grid; plans=plans, params=params, compute_w=true)
```

## Omega Equation

The omega equation computes ageostrophic vertical velocity:

```math
\nabla^2 w + \frac{f_0^2}{N^2}\frac{\partial^2 w}{\partial z^2} = \frac{2f_0}{N^2}J\left(\frac{\partial\psi}{\partial z}, \nabla^2\psi\right)
```

Use `omega_eqn_rhs!` to compute the right-hand side:

```julia
# Compute omega equation RHS
omega_eqn_rhs!(rhs, state.psi, grid, params, plans)
```

## Time Series Analysis

### Recording Diagnostics Manually

```julia
# Initialize storage arrays
time_series = Float64[]
KE_series = Float64[]
WKE_series = Float64[]

for step = 1:nsteps
    # ... time stepping code ...

    # Record diagnostics
    push!(time_series, step * dt)
    push!(KE_series, flow_kinetic_energy(state.u, state.v))
    WKE, WPE, WCE = compute_detailed_wave_energy(state, grid, params)
    push!(WKE_series, WKE)
end
```

### Energy Conservation Check

```julia
# Total energy should be conserved (inviscid)
E_total = KE_series .+ WKE_series

# Check conservation
dE = (E_total[end] - E_total[1]) / E_total[1]
println("Relative energy change: $dE")

if abs(dE) > 1e-6
    @warn "Energy not well conserved!"
end
```

## Diagnostic Output Example

```julia
function compute_diagnostics(state, grid, params)
    diag = Dict{String, Any}()

    # Energy
    diag["KE"] = flow_kinetic_energy(state.u, state.v)
    diag["WKE"], diag["WPE"], diag["WCE"] = compute_detailed_wave_energy(state, grid, params)
    diag["WE_B"], diag["WE_A"] = wave_energy(state.L⁺A, state.A)

    # Extrema
    diag["A_max"] = maximum(abs.(state.A))
    diag["psi_max"] = maximum(abs.(state.psi))

    return diag
end

function print_diagnostics(diag, step, time)
    println("=" ^ 60)
    println("Step: $step, Time: $(round(time, digits=4))")
    println("-" ^ 60)
    println("  KE = $(round(diag["KE"], sigdigits=6))")
    println("  WKE = $(round(diag["WKE"], sigdigits=6))")
    println("  |A|_max = $(round(diag["A_max"], sigdigits=4))")
    println("=" ^ 60)
end
```

## API Reference

See the [Physics API Reference](../api/physics.md) for complete documentation of diagnostic functions:
- `flow_kinetic_energy` - Mean flow kinetic energy
- `flow_kinetic_energy_spectral` - Spectral kinetic energy with dealiasing
- `flow_potential_energy_spectral` - Spectral potential energy
- `wave_energy` - Basic wave energy from B and A fields
- `wave_energy_spectral` - Spectral wave energy components
- `compute_detailed_wave_energy` - Detailed WKE, WPE, WCE computation
- `compute_wave_energy` - Simple wave energy computation
- `flow_kinetic_energy_global` / `wave_energy_global` - MPI-aware global reductions
- `omega_eqn_rhs!` - Omega equation RHS computation
- `EnergyDiagnosticsManager` - Automatic energy output to separate files
