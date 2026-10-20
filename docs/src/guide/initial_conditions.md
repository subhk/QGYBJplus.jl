# [Initial conditions](@id initial-conditions)

```@meta
CurrentModule = QGYBJplus
```

Model-level setters update model-owned arrays and use runtime transform plans
automatically.

## Analytical mean flow

```julia
set!(
    model;
    ψ=(x, y, z) -> 1e3 * sin(2π * x / model.grid.extent[1]),
    pv_method=:qg,
)
```

Use `pv_method=:barotropic` for an imposed vertically uniform flow, as in the
Asselin dipole example. `pv_method=:none` leaves q unchanged, so it suits only
a `FixedFlow()` model, whose ψ is prescribed: an evolving flow is inverted from
q, and `set!` rejects `:none` for it. `set!` checks all of its arguments, and
reads every array and file, before it changes the model.

## Mean flow from vorticity

The mean flow can be given by its relative vorticity instead of its
streamfunction:

```julia
Lx, Ly = model.grid.extent[1], model.grid.extent[2]
set!(model; ζ=(x, y, z) -> 1e-5 * sin(2π * x / Lx) * cos(2π * y / Ly))
set!(model; ζ=FieldFile("eddies.jld2", "zeta"))
```

`ζ` (or `zeta`) takes a function of `(x, y, z)`, an array, a
[`FieldArray`](@ref), or a [`FieldFile`](@ref). The streamfunction follows by
inverting ``\nabla_h^2 \psi = \zeta`` at each level, and `pv_method` then
builds the potential vorticity from ψ as it does for a given ψ. The horizontal
mean of ζ at each level is dropped, since the circulation around a doubly
periodic domain vanishes; the model warns when that mean exceeds 1% of the
field's rms, as it does for a single eddy cut out of a larger region.

## Deterministic random mean flow

```julia
set!(
    model;
    mean_flow=RandomStreamfunction(
        amplitude=0.1,          # rms horizontal velocity, m/s
        spectral_slope=-3,
        seed=42,
    ),
)
```

`amplitude` is the root-mean-square horizontal velocity over the domain, in
the model's units, whatever the resolution. Each horizontal Fourier mode
within the two-thirds dealiasing radius has a random phase and a variance
proportional to ``|k|^{\text{spectral\_slope}}``, with ``|k|`` the mode index.
The field is the same at every level; `profile=z -> ...` gives it a vertical
structure, for example `profile=z -> exp(z / 500)` for an eddy field
decaying downward over 500 m. Each coefficient is generated from a hash of
its own indices, so a fixed seed gives the same field on every architecture,
up to rounding in the GPU math library.

## Surface-confined waves

```julia
set!(
    model;
    waves=SurfaceWave(
        amplitude=0.1,
        scale=30.0,
        profile=:gaussian,
    ),
)
```

The profile may be `:gaussian` or `:exponential`, decaying over `scale`
downward from the top of the domain, its rigid lid. It is evaluated at the
depth of each cell's upper face, so the top cell holds the full `amplitude`
however thin the layer. The wave is horizontally uniform, and its depth mean
is removed, since the wave velocity has no barotropic part (see
[YBJ⁺ wave model](@ref ybj-plus)). For a surface layer much thinner than the
ocean that changes the profile very little; for a layer comparable to the
depth, it adds a weak opposite flow below the layer. Resolve the layer with
several cells: a `scale` below the grid spacing puts the whole wave in the
top cell.

## Wave packets

```julia
set_wave_packet!(
    model;
    amplitude=0.1,           # largest |B|
    kx=4,
    ky=0,
    sigma_k=1.5,
    center=(250e3, 250e3),
)
```

The packet is a plane wave of wavenumber ``(2\pi k_x/L_x, 2\pi k_y/L_y)`` (`kx`
and `ky` are integer mode indices) under a Gaussian envelope centred at
`center`, the middle of the domain by default. Its spectral width `sigma_k`,
in mode indices, sets its horizontal width, about ``L_x / (2\pi\,\sigma_k)``.
`amplitude` is the packet's largest ``|B|``. Its Gaussian vertical envelope is
centered `z_center` below the top of the domain, with width `z_width`; they
default to half and a quarter of the depth. The packet involves no
randomness, so it is the same on every call and every architecture.

## Direct array access

Pass a global `(z, x, y)` array directly to initialize a physical field:

```julia
set!(model; ψ=psi_values, pv_method=:qg)
set!(model; B=wave_values)
```

Use [`FieldArray`](@ref) to declare spectral data or `(x, y, z)` layout:

```julia
set!(model; B=FieldArray(B_hat; space=:spectral, layout=:zxy))
```

File-backed fields use [`FieldFile`](@ref), which reads NetCDF, or JLD2 when
the name ends in `.jld2`. A NetCDF variable's dimensions are put in order by
their names when they name the axes (`x`, `y`, and `z` or `depth`, or common
alternatives such as `lon`, `lat`, and `lev`), so a field stored as
`(y, x, z)` is read correctly; otherwise, and for JLD2 arrays, `layout` gives
the order, by default the file-oriented `(x, y, z)`:

```julia
set!(model; B=FieldFile("initial.nc", "B"))
```

A single time (a dimension of length one) is dropped. Missing (fill) values,
NaN, and Inf are errors, for files and arrays alike.

Public setters rebuild dependent `q`, `A`, `C`, and velocity diagnostics.

## Fields on another grid

A physical field given on another rectilinear grid is interpolated onto the
model's grid when the coordinates of its points are known. A `FieldFile` finds
them as the vectors `x`, `y`, and `z` (or `depth`) stored beside the field:
`z` is in the grid's coordinates, positive upward, and `depth` is measured
down from the top of the grid (a `z` with no point below the top and some
above it is read as depth, as in earlier versions). A `FieldArray` takes them
as keywords, with `z` in the grid's coordinates:

```julia
set!(model; ζ=FieldFile("eddies.jld2", "zeta"))
set!(model; ψ=FieldFile("coarse_run.nc", "psi"))
set!(model; ζ=FieldArray(ζ_data; layout=:xyz, x=xs, y=ys, z=zs))
```

`group` selects the NetCDF group, or the JLD2 group, NamedTuple, or Dict,
that holds the field and its coordinates, as in
`FieldFile("input.jld2", "zeta"; group="eddies")`. A stratification profile
`N2` kept in the same place shares the field's vertical coordinate, so one
file can hold both:

```julia
profile = load_stratification_from_file("input.jld2")   # z (or depth) and N2
model = QGYBJModel(grid=grid, stratification=profile)
set!(model; ζ=FieldFile("input.jld2", "zeta"))           # x, y, z, and zeta
```

Interpolation is linear along each axis. In x and y it is periodic with the
model's domain, so the data must hold one period of a periodic field, in any
horizontal frame; a last point one period after the first is taken to repeat
it. The model warns when the data leave a gap of more than a grid spacing at
the periodic boundary, as for a region cut out of a larger domain, which also
makes the field jump across that boundary. Vertically, the values are held
beyond the data, with the same warning as for stratification profiles. Data
at the model's own points are copied unchanged. Each point takes the value at
its location rather than an average around it, so smooth data on a much finer
grid to the model's resolution first. Spectral fields must be given on the
model's grid.
