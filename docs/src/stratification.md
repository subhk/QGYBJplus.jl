## Stratification

Stratification enters via N²(z) and the vertical operator coefficient
`a_ell_ut = 1/N²`. QGYBJ.jl offers several profiles.

### Built‑in Profiles

- `:constant_N` with `N0`
- `:skewed_gaussian` (test case values from the reference Fortran)
- `:tanh_profile` (pycnocline‑like)
- `:from_file` (load N² from NetCDF)

```julia
strat = create_stratification_config(:constant_N, N0=1.0)
```

### Using the Profile

During setup, the stratification profile is computed on the grid and used to
derive `a_ell_ut` for the vertical inversions. The density‑like weights used by
the vertical solvers default to unity (Boussinesq) for parity with the Fortran
test cases, and are populated during `setup_simulation`.

You can override them if needed via parameter hooks:

```julia
par = default_params(Lx=500e3, Ly=500e3, Lz=4000.0, nx=64, ny=64, nz=32)
rho_ut = ones(par.nz); rho_st = ones(par.nz)   # or custom profiles
par = with_density_profiles(par; rho_ut=rho_ut, rho_st=rho_st)
```

