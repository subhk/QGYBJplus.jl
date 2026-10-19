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
Asselin dipole example.

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
        amplitude=0.4,
        spectral_slope=-3,
        seed=42,
    ),
)
```

Each Fourier coefficient is generated from a hash of its own indices, so a
fixed seed gives the same field on every architecture, up to rounding in the
GPU math library.

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

The profile may be `:gaussian` or `:exponential`. The wave is horizontally
uniform, and its depth mean is removed, since the wave velocity has no
barotropic part (see [YBJ⁺ wave model](@ref ybj-plus)). For a surface layer
much thinner than the ocean that changes the profile very little; for a
layer comparable to the depth, it adds a weak opposite flow below the layer.

## Wave packets

```julia
set_wave_packet!(
    model;
    amplitude=0.1,
    kx=2,
    ky=1,
    sigma_k=0.5,
)
```

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
the name ends in `.jld2`. Its default layout is the file-oriented `(x, y, z)`
ordering:

```julia
set!(model; B=FieldFile("initial.nc", "B"))
```

Public setters rebuild dependent `q`, `A`, `C`, and velocity diagnostics.

## Fields on another grid

A physical field given on another rectilinear grid is interpolated onto the
model's grid when the coordinates of its points are known. A `FieldFile` finds
them as the vectors `x`, `y`, and `z` (or `depth`) stored beside the field,
read with the vertical conventions of
[stratification files](@ref stratification); a `FieldArray` takes them as
keywords, with `z` positive upward:

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
