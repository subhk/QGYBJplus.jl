"""
Convenience setup helpers to bootstrap a simulation.
"""

"""
    setup_model(par::QGParams) -> (G, S, plans, a)

Initialize grid, state, FFT plans, and elliptic coefficient for a basic run.

# Arguments
- `par`: Model parameters (REQUIRED - use `default_params(Lx=..., Ly=..., Lz=...)`)

# Returns
Tuple of (Grid, State, Plans, a_ell)

# Example
```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)
G, S, plans, a = setup_model(par)
```
"""
function setup_model(par::QGParams)
    G = init_grid(par)
    S = init_state(G)
    plans = plan_transforms!(G)
    a = a_ell_ut(par, G)
    return G, S, plans, a
end
