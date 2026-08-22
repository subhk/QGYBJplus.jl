# [Asselin et al. dipole](@id worked_example)

```@meta
CurrentModule = QGYBJplus
```

`examples/asselin_jpo2020.jl` implements the barotropic dipole setup from
Asselin et al. (2020) with the composition-first API.

## Run

```bash
mpiexecjl -n 4 julia --project=. examples/asselin_jpo2020.jl
```

Defaults are 256×256×128 points, a 70 km square horizontal domain, 3 km depth,
and 15 inertial periods with `Δt = 2 s`.

For a smaller run, reduce `size` and `stop_time` in the script. The setup uses
a fixed barotropic flow, YBJ⁺ waves, no wave feedback, and wave
hyperdiffusion.

## Output

Snapshots are written below `output_asselin/`. Use `examples/compute_energy.jl`
for spatial kinetic energy, or see [I/O and restart](@ref io-output) for direct
FFTW analysis with `grid.kx` and `grid.ky`.

## Reference

Asselin, O., Thomas, L. N., Young, W. R., & Rainville, L. (2020),
[“Refraction and Straining of Near-Inertial Waves by Barotropic
Eddies”](https://doi.org/10.1175/JPO-D-20-0109.1), *Journal of Physical
Oceanography*, 50, 3439–3454.
