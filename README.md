QG-YBJ+ Model
==============

[![CI](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/QGYBJ.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/QGYBJ.jl/dev/)

This numerical model simulates the coupling between near-inertial waves and (Lagrangian-mean) balanced eddies. Wave dynamics follow the YBJ+ equation (Asselin & Young 2019), while potential vorticity evolution is governed by the quasigeostrophic equation, incorporating the wave feedback formulation of Xie & Vanneste (2015). The model employs pseudo-spectral methods horizontally and second-order finite differencing for vertical and temporal derivatives.


## References

- Asselin, O., & Young, W. R. (2019). Penetration of wind-generated near-inertial waves into a turbulent ocean. *J. Phys. Oceanogr.*, 49, 1699-1717.
- Xie, J.-H., & Vanneste, J. (2015). A generalised-Lagrangian-mean model of the interactions between near-inertial waves and mean flow. *J. Fluid Mech.*, 774, 143-169.
