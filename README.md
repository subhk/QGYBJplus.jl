QG-YBJ+ Model
==============

[![CI](https://github.com/subhk/QGYBJplus.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJplus.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/QGYBJplus.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/QGYBJplus.jl/dev/)

This numerical model simulates interactions between near-inertial waves and
(Lagrangian-mean) balanced flow. It combines the YBJ⁺ equation of Asselin &
Young (2019), the wave-feedback formulation of Xie & Vanneste (2015),
horizontal pseudo-spectral methods, second-order vertical differences, and
ETD-RK2 time integration.

```julia
using QGYBJplus

grid = RectilinearGrid(size=(64, 64, 32),
                       extent=(500e3, 500e3, 4000.0), centered=true)
model = QGYBJModel(grid=grid,
                   coriolis=FPlane(f=1e-4),
                   stratification=ConstantStratification(N²=1e-5),
                   flow=EvolvingFlow(),
                   feedback=WaveMeanFeedback(),
                   formulation=YBJPlus())

set!(model;
     ψ=(x, y, z) -> 1e3 * sin(2π*x/500e3) * cos(2π*y/500e3),
     waves=SurfaceWave(amplitude=0.1, scale=30.0))

simulation = Simulation(model; Δt=20.0, stop_time=86400.0,
                        output=NetCDFOutput(path="output",
                                            schedule=TimeInterval(3600.0)))
try
    run!(simulation)
finally
    finalize_simulation!(simulation)
end
```


## References

- Asselin, O., & Young, W. R. (2019). An improved model of near-inertial wave dynamics. *J. Fluid Mech.*, 876, 428–448. https://doi.org/10.1017/jfm.2019.557
- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
