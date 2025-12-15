# [QG Equations](@id qg-equations)

```@meta
CurrentModule = QGYBJ
```

This page details the quasi-geostrophic (QG) equations implemented in QGYBJ.jl.

## Potential Vorticity Evolution

The core QG equation is the conservation of potential vorticity:

```math
\frac{\partial q}{\partial t} + J(\psi, q) = \mathcal{D}_q + \mathcal{F}_q
```

where ``\mathcal{D}_q`` is dissipation and ``\mathcal{F}_q`` is forcing.

### Potential Vorticity Definition

```math
q = \underbrace{\nabla^2\psi}_{\text{relative vorticity}} + \underbrace{\frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right)}_{\text{stretching term}}
```

For uniform stratification (``N^2 = \text{const}``):
```math
q = \nabla^2\psi + \frac{f_0^2}{N^2}\frac{\partial^2\psi}{\partial z^2}
```

### Physical Interpretation

| Term | Physical Meaning |
|:-----|:-----------------|
| ``\nabla^2\psi`` | Relative vorticity ``\zeta`` |
| ``(f_0^2/N^2)\partial_z^2\psi`` | Vortex stretching due to vertical motion |
| ``J(\psi, q)`` | Advection of PV by geostrophic flow |

## Streamfunction Inversion

Given ``q``, we solve for ``\psi`` via the elliptic equation:

```math
\nabla^2\psi + \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right) = q
```

### Spectral Representation

In spectral space (horizontal) with vertical finite differences:

```math
-k_h^2 \hat{\psi} + \frac{\partial}{\partial z}\left(a(z)\frac{\partial\hat{\psi}}{\partial z}\right) = \hat{q}
```

where:
- ``k_h^2 = k_x^2 + k_y^2`` (horizontal wavenumber squared)
- ``a(z) = f_0^2/N^2(z)`` (stretching coefficient)

### Tridiagonal System

For each horizontal wavenumber ``(k_x, k_y)``, the vertical discretization gives:

```math
a_k \hat{\psi}_{k-1} + b_k \hat{\psi}_k + c_k \hat{\psi}_{k+1} = \hat{q}_k
```

This is solved efficiently with the Thomas algorithm in O(nz) operations.

### Boundary Conditions

- **Top and Bottom**: ``\frac{\partial\psi}{\partial z} = 0`` (no buoyancy flux)

In the code:
```julia
# Called for each time step
invert_q_to_psi!(state, grid, params, a_ell)
```

## Velocity Fields

### Geostrophic Velocities

From geostrophic balance:
```math
u = -\frac{\partial\psi}{\partial y}, \quad v = \frac{\partial\psi}{\partial x}
```

In spectral space:
```math
\hat{u} = -ik_y\hat{\psi}, \quad \hat{v} = ik_x\hat{\psi}
```

### Vertical Velocity

The QG omega equation gives the ageostrophic vertical velocity:

```math
\nabla^2 w + \frac{N^2}{f_0^2}\frac{\partial^2 w}{\partial z^2} = 2J\left(\frac{\partial\psi}{\partial z}, \nabla^2\psi\right)
```

The RHS represents frontogenesis/frontolysis forcing.

## Jacobian Operator

The Jacobian ``J(a, b)`` is computed pseudo-spectrally:

```math
J(a, b) = \frac{\partial a}{\partial x}\frac{\partial b}{\partial y} - \frac{\partial a}{\partial y}\frac{\partial b}{\partial x}
```

### Algorithm
1. Compute ``\partial a/\partial x``, ``\partial a/\partial y`` in spectral space
2. Transform to physical space
3. Multiply in physical space
4. Transform back to spectral space
5. Apply dealiasing (2/3 rule)

### Conservation Properties

The Jacobian satisfies:
- ``\int J(a, b) \, dA = 0`` (integral vanishes)
- ``J(a, a) = 0`` (anti-symmetry)

These ensure energy and enstrophy conservation in the inviscid limit.

## Dissipation

### Hyperdiffusion

The model uses scale-selective hyperdiffusion:

```math
\mathcal{D}_q = -\nu_{h1}(-\nabla^2)^{p_1} q - \nu_{h2}(-\nabla^2)^{p_2} q - \nu_z\frac{\partial^2 q}{\partial z^2}
```

where:
- ``\nu_{h1}, p_1``: Large-scale dissipation (drag)
- ``\nu_{h2}, p_2``: Small-scale dissipation (hyperviscosity)
- ``\nu_z``: Vertical diffusion

### Integrating Factor Method

To handle stiff diffusion, we use integrating factors:

```math
\tilde{q} = q \cdot e^{\nu k^{2p} \Delta t}
```

This allows larger time steps while maintaining stability.

## Wave Feedback

When waves are present, the QG equation includes a feedback term through a modified effective PV.

### The Wave Feedback Mechanism

The wave-induced PV ``q^w`` is computed from the wave envelope ``B``:

```math
q^w = Ro \cdot W2F \cdot \left[ \frac{i}{2} J(B^*, B) - \frac{1}{4} \nabla_h^2 |B|^2 \right]
```

where:
- ``Ro = U/(f_0 L)`` is the Rossby number
- ``W2F = (U_w/U)^2`` is the wave-to-flow velocity ratio squared
- ``B = B_R + i B_I`` is the complex wave envelope

### Effective PV for Inversion

The streamfunction is obtained by inverting the **effective** PV:

```math
q^* = q - q^w
```

```math
\nabla^2\psi + \frac{\partial}{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial\psi}{\partial z}\right) = q^*
```

This means the wave feedback **modifies the inversion** rather than appearing as an explicit advection term.

See [Wave-Mean Interaction](@ref wave-mean) for detailed formulas and implementation.

## Implementation

### Key Functions

The QG equation implementation uses these core functions:
- `invert_q_to_psi!` - Solve elliptic equation for streamfunction
- `jacobian_spectral!` - Compute Jacobian pseudo-spectrally
- `compute_velocities!` - Get (u, v) from streamfunction

See the [Physics API Reference](../api/physics.md) for detailed documentation.

### Code Location

- `elliptic.jl`: Streamfunction inversion (`invert_q_to_psi!`)
- `nonlinear.jl`: Jacobian computation (`jacobian_spectral!`)
- `operators.jl`: Velocity computation (`compute_velocities!`)

## References

- Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics*. Cambridge University Press.
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*. Springer.
