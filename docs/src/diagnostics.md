## Diagnostics

The `Diagnostics` module provides helpers for computing derived quantities and
extracting slices.

### Examples

```julia
using QGYBJ

# Create parameters with domain size (REQUIRED)
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)  # 500km × 500km × 4km
G, S, plans, a = setup_model(par)

# Invert q → ψ and compute omega‑equation RHS
invert_q_to_psi!(S, G; a, par=par)
rhs = similar(S.psi)
omega_eqn_rhs!(rhs, S.psi, G, plans)

# Wave energy (domain‑sum style)
EB, EA = wave_energy(S.B, S.A)

# Slices (back to real space internally)
sl_xy = slice_horizontal(S.psi, G, plans; k=G.nz ÷ 2)
sl_xz = slice_vertical_xz(S.psi, G, plans; j=G.ny ÷ 2)
```

