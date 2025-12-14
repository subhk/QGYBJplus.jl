QG-YBJ+ Model
==============

[![CI](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/QGYBJ.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/QGYBJ.jl/dev/)

This is a numerical model for the two-way interaction of near-inertial waves with (Lagrangian-mean) balanced eddies. Wave evolution is governed by the YBJ+ equation (Asselin & Young 2019). The traditional quasigeostrophic equation dictates the evolution of potential vorticity, which includes the wave feedback term of Xie & Vanneste (2015). The model is pseudo-spectral in the horizontal and uses second-order finite differences to evaluate vertical and time derivatives.

## Quick Start

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/QGYBJ.jl")

using QGYBJ

# Setup model
par = default_params(nx=64, ny=64, nz=32)
G, S, plans, a = setup_model(; par)

# Initialize and run
init_random_psi!(S, G, par, plans; a=a)
first_projection_step!(S, G, par, plans; a=a, dealias_mask=dealias_mask(par, G))
```

## References

- Asselin, O., & Young, W. R. (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Phys. Oceanogr.*, 49, 1699-1717.
- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
