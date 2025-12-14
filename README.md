QG-YBJ+ Model
==============

[![CI](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml)
[![Docs](https://github.com/subhk/QGYBJ.jl/actions/workflows/docs.yml/badge.svg)](https://subhk.github.io/QGYBJ.jl)

This is a numerical model for the two-way interaction of near-inertial waves with (Lagrangian-mean) balanced eddies. Wave evolution is governed by the YBJ+ equation (Asselin & Young 2019). The traditional quasigeostrophic equation dictates the evolution of potential vorticity, which includes the wave feedback term of Xie & Vanneste (2015). The model is pseudo-spectral in the horizontal and uses second-order finite differences to evaluate vertical and time derivatives.

