# Oceananigans-Style Reporting Design

## Goal

Make detailed simulation progress follow the compact Oceananigans convention
while retaining QGYBJplus's useful MPI and runtime initialization reports.

## Design

`run!(simulation; progress=true)` keeps its existing cadence and global MPI
reductions. Rank zero prints one direct `@printf` line containing the padded
iteration, human-readable simulation time and timestep, maximum wave and flow
speeds, and cumulative wall time for the current `run!` call. An internal time
formatter follows Oceananigans's seconds-based unit thresholds without adding
a package dependency or public API.

The existing structured `@info` reports remain unchanged and visible on rank
zero:

- MPI process count and 2D topology
- validated grid size, topology, and decomposition dimensions
- pencil decomposition layouts
- model runtime grid size and rank count

Tests cover the exact progress labels and time formatting, confirm wall time is
present, and protect all four initialization messages. Documentation shows the
new output without adding new configuration concepts.
