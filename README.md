QG-YBJ+ Model
==============

[![CI](https://github.com/subhk/QGYBJplus.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJplus.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/QGYBJplus.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/QGYBJplus.jl/dev/)

This numerical model simulates the coupling between near-inertial waves and (Lagrangian-mean) balanced eddies. Wave dynamics follow the YBJ+ equation (Asselin & Young 2019), while potential vorticity evolution is governed by the quasigeostrophic equation, incorporating the wave feedback formulation of Xie & Vanneste (2015). The model employs pseudo-spectral methods horizontally, second-order finite differences vertically, and second-order exponential Runge–Kutta (ETD-RK2) time integration.

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

- Asselin, O., & Young, W. R. (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Phys. Oceanogr.*, 49, 1699-1717.
- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
