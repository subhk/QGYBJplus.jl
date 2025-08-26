## Getting Started

This page walks you through installing QGYBJ.jl, running a quick example,
and understanding the core concepts (Grid, State, Params).

### Installation

- Clone the repository and activate the project
  - `git clone https://github.com/subhk/QGYBJ.jl`
  - `cd QGYBJ.jl`
  - `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`

Notes:
- FFTW.jl is required for transforms. The optional NCDatasets.jl is only needed
  for NetCDF I/O. If NCDatasets is not installed, NetCDF I/O is disabled.

### Quick Example

```julia
using QGYBJ

# Create a simple configuration and run a short simulation
config = create_simple_config(
    dt=1e-3, total_time=2.0,
)
sim = setup_simulation(config)
run_simulation!(sim)
```

This will:
- Build a grid and state
- Create transforms (FFTs)
- Initialize fields from the config
- Step forward in time and (optionally) write outputs

### Core Concepts

- `QGParams`: numerical/physical parameters (grid sizes, dt, viscosity, flags)
- `Grid`: grid geometry and spectral metadata (kx, ky, kh², z)
- `State`: prognostic and diagnostic fields in spectral/real space
  - Spectral: `q`, `psi`, `A`, `B`, `C=A_z`
  - Real: `u`, `v`, `w` (computed as needed)
- Transforms: 2D FFTs per z (FFTW in serial; PencilFFTs in parallel)

