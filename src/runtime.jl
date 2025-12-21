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

# Note on Stratification
This function uses constant N² (from par.N²) for the elliptic coefficient.
If `par.stratification != :constant_N`, use `setup_model_with_profile()` instead
to ensure consistent physics across all operators.

# Example
```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0)
G, S, plans, a = setup_model(par)
```
"""
function setup_model(par::QGParams)
    # Warn if stratification is non-constant
    if hasfield(typeof(par), :stratification) && par.stratification != :constant_N
        @warn "setup_model: Using constant N² from par.N² but par.stratification=$(par.stratification). " *
              "For non-constant stratification, use setup_model_with_profile() or the high-level " *
              "QGYBJSimulation API to ensure consistent physics in wave dispersion and vertical velocity." maxlog=1
    end

    G = init_grid(par)
    S = init_state(G)
    plans = plan_transforms!(G)
    a = a_ell_ut(par, G)
    return G, S, plans, a
end

"""
    setup_model_with_profile(par::QGParams) -> (G, S, plans, a, N2_profile)

Initialize grid, state, FFT plans, elliptic coefficient, and N² profile.

This function uses `N2_ut` to build the N² profile from `par.stratification`.
For profile-based stratification types (e.g., `:tanh_profile`, `:from_file`),
it falls back to constant N² with a warning. Use the high-level stratification
API or provide an explicit N² profile for those cases.

# Returns
Tuple of (Grid, State, Plans, a_ell, N2_profile)

# Example
```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, stratification=:skewed_gaussian)
G, S, plans, a, N2_profile = setup_model_with_profile(par)
# Use N2_profile in compute_vertical_velocity!, etc.
```
"""
function setup_model_with_profile(par::QGParams)
    G = init_grid(par)
    S = init_state(G)
    plans = plan_transforms!(G)

    # Compute N² profile based on stratification type
    # Uses the same logic as physics.N2_ut (including warnings for profile-based types).
    N2_profile = N2_ut(par, G)
    a = a_ell_from_N2(N2_profile, par)

    return G, S, plans, a, N2_profile
end

"""
