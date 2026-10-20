#=
================================================================================
            initialization.jl - Spectral initial-condition builders
================================================================================

Kernels that build or derive spectral fields for `set!`: the deterministic
random streamfunction, the dealiasing projection, and the PV-from-ψ stencils.
Functions of user-supplied coordinates are evaluated on the host by `set!`
and copied to the model's architecture there.
================================================================================
=#

using Random

#=
Deterministic, architecture-independent random numbers. Every Fourier
coefficient hashes its own global indices, so a given seed produces the same
uniform numbers on every architecture and with any kernel launch order. The
field built from them agrees to rounding: a GPU math library may round `log`,
`cos`, and `^` differently from the host.
=#

const _U53 = 0x1.0p-53

@inline function _mix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

# Indices are positive, so `% UInt64` is the same value as `UInt64(index)`
# without a conversion check that GPU code would have to carry.
@inline function _hash_u01(seed::UInt64, tag::UInt64, k::Int, i::Int, j::Int)
    x = seed
    x = _mix64(x ⊻ (k % UInt64))
    x = _mix64(x ⊻ (i % UInt64))
    x = _mix64(x ⊻ (j % UInt64))
    x = _mix64(x ⊻ tag)
    return Float64((x >> 11) + 1) * _U53
end

@inline function _hash_randn(seed::UInt64, tag::UInt64, k::Int, i::Int, j::Int)
    u1 = _hash_u01(seed, tag, k, i, j)
    u2 = _hash_u01(seed, tag + UInt64(1), k, i, j)
    return sqrt(-2 * log(u1)) * cos(2 * pi * u2)
end

"""
The unit-amplitude random coefficient of horizontal mode `(i, j)`: a Gaussian
amplitude of variance proportional to `|k|^slope` (`|k|` the mode index) and a
uniform phase, both hashed from the mode, Hermitian so that ψ is real. It
does not depend on the level, so the field built from it is vertically
coherent.
"""
@inline function _random_psi_coefficient(base_seed::UInt64, slope, i, j, nx, ny)
    # FFTW order, as for the grid's wavenumbers.
    kx = i <= (nx + 1) ÷ 2 ? i - 1 : i - 1 - nx
    ky = j <= (ny + 1) ÷ 2 ? j - 1 : j - 1 - ny
    (kx == 0 && ky == 0) && return zero(ComplexF64)
    # Hashed in Float64, so all architectures agree.
    variance = sqrt(Float64(kx^2 + ky^2))^slope
    i_conj = i == 1 ? 1 : nx - i + 2
    j_conj = j == 1 ? 1 : ny - j + 2
    if i == i_conj && j == j_conj
        # Self-conjugate modes must be real for a real ψ.
        return complex(sqrt(2 * variance) *
                       _hash_randn(base_seed, UInt64(0x10), 1, i, j))
    end
    # Hash the canonical member of each conjugate pair, so both members agree
    # and the field is Hermitian.
    canonical = (i < i_conj) || (i == i_conj && j <= j_conj)
    i_can = canonical ? i : i_conj
    j_can = canonical ? j : j_conj
    amplitude = sqrt(2 * variance) * _hash_randn(base_seed, UInt64(0x20), 1, i_can, j_can)
    phase = 2 * pi * _hash_u01(base_seed, UInt64(0x30), 1, i_can, j_can)
    value = amplitude * cis(phase)
    return canonical ? value : conj(value)
end

@kernel function _random_psi_kernel!(ψ, base_seed, slope, @Const(weights), mask, nx, ny)
    k, i, j = @index(Global, NTuple)
    value = keep_mode(mask, i, j) ?
            @inbounds(weights[k]) * _random_psi_coefficient(base_seed, slope, i, j, nx, ny) :
            zero(ComplexF64)
    @inbounds ψ[k, i, j] = value
end

"""
Root-mean-square horizontal velocity over the domain of the random
streamfunction with unit coefficients, level `k` scaled by `weights[k]`,
computed on the host from the same hashed coefficients as the kernel.
"""
function _random_psi_rms_velocity(G::RuntimeGeometry, base_seed, slope, weights)
    nx, ny, nz = G.nx, G.ny, G.nz
    kx, ky = on_host(G.kx), on_host(G.ky)
    mask = RadialDealiasing(nx, ny)
    total = 0.0
    for j in 1:ny, i in 1:nx
        keep_mode(mask, i, j) || continue
        total += (Float64(kx[i])^2 + Float64(ky[j])^2) *
                 abs2(_random_psi_coefficient(base_seed, slope, i, j, nx, ny))
    end
    # Parseval for the unnormalised FFT: over the domain, the mean of
    # |∇ψ|² = u² + v² is Σ kₕ²|ψ̂|² / (nx ny)² per level, averaged over levels.
    return sqrt(total * sum(abs2, weights) / nz) / (nx * ny)
end

"""
    init_random_psi!(psik, G, amplitude; slope=-3.0, seed=0, seed_offset=0, profile=nothing)

Fill `psik` with a random streamfunction whose horizontal velocity has
root-mean-square `amplitude` over the domain. Every horizontal Fourier mode
within the radial two-thirds dealiasing radius has a random phase and a
variance proportional to `|k|^slope`, with `|k|` the mode index. The field is
the same at every level, times `profile[k]` at level `k` when a profile
vector is given, and ψ is real (Hermitian coefficients). The values depend
only on the seeds and the global mode indices, so every architecture produces
the same field up to rounding.
"""
function init_random_psi!(psik, G::RuntimeGeometry, amplitude::Real;
                          slope::Real=-3.0, seed::Int=0, seed_offset::Int=0,
                          profile=nothing)
    isfinite(amplitude) && isfinite(slope) ||
        throw(ArgumentError("random streamfunction parameters must be finite"))
    amplitude >= 0 || throw(ArgumentError("amplitude must be non-negative"))
    size(psik) == (G.nz, G.nx, G.ny) || throw(DimensionMismatch(
        "psik has size $(size(psik)); expected $((G.nz, G.nx, G.ny))"))
    weights = profile === nothing ? ones(G.nz) : Float64.(collect(profile))
    length(weights) == G.nz && all(isfinite, weights) || throw(ArgumentError(
        "the profile must give a finite value at each of the $(G.nz) levels"))
    base_seed = _mix64(UInt64(seed) ⊻ UInt64(seed_offset))
    rms = _random_psi_rms_velocity(G, base_seed, Float64(slope), weights)
    scale = rms > 0 ? Float64(amplitude) / rms : 0.0
    backend = get_backend(psik)
    launch!(_random_psi_kernel!, backend, size(psik), psik, base_seed,
            Float64(slope), on_backend(backend, scale .* weights),
            RadialDealiasing(G.nx, G.ny), G.nx, G.ny)
    return psik
end

"""
    apply_dealiasing_mask!(field, G::RuntimeGeometry)

Apply the radial two-thirds dealiasing rule to a spectral field (see
[`dealias_mask`](@ref)):
- Keep integer wavenumbers with |k| < N/3
- Radial cutoff ensures isotropic dealiasing
"""
apply_dealiasing_mask!(field, G::RuntimeGeometry) =
    dealias!(field, field, RadialDealiasing(G.nx, G.ny))

"""
    compute_energy_spectrum(field, G::RuntimeGeometry)

Compute the horizontal energy spectrum E(k) of a spectral field, averaged over
the modes in each integer shell. Evaluated on a host copy of `field`.
"""
function compute_energy_spectrum(field, G::RuntimeGeometry)
    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2
    k_max = min(kx_max, ky_max)
    spectrum = zeros(Float64, k_max)
    count = zeros(Int, k_max)

    field_arr = on_host(field)
    for k in axes(field_arr, 1)
        for j in axes(field_arr, 3)
            ky = j <= ky_max ? j - 1 : j - 1 - G.ny
            for i in axes(field_arr, 2)
                kx = i <= kx_max ? i - 1 : i - 1 - G.nx
                k_total = round(Int, sqrt(Float64(kx^2 + ky^2)))
                if 1 <= k_total <= k_max
                    spectrum[k_total] += abs2(field_arr[k, i, j])
                    count[k_total] += 1
                end
            end
        end
    end

    for i in 1:k_max
        if count[i] > 0
            spectrum[i] /= count[i]
        end
    end
    return spectrum
end

"""
    create_wave_packet(G::RuntimeGeometry, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real;
                       center=nothing, z_center=G.Lz/2, z_width=G.Lz/4)

The Fourier coefficients, as a host array, of a Gaussian wave packet: a plane
wave of mode indices `(kx0, ky0)`, wavenumber `(2π kx0/Lx, 2π ky0/Ly)`, under
a horizontal envelope centred at `center = (x, y)` (the middle of the domain
by default) and a vertical one centred `z_center` below the top of the domain
(its rigid lid) with width `z_width`. The coefficients are Gaussian in mode
index around `(kx0, ky0)` with width `sigma_k`, so the packet is about
`Lx / (2π sigma_k)` wide; their phases place it at `center`. `amplitude` is
the packet's largest |B|. The packet involves no randomness.
"""
function create_wave_packet(G::RuntimeGeometry, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real;
                            center=nothing,
                            z_center::Real=G.Lz / 2,
                            z_width::Real=G.Lz / 4)
    isfinite(amplitude) || throw(ArgumentError("amplitude must be finite"))
    isfinite(sigma_k) && sigma_k > 0 || throw(ArgumentError("sigma_k must be positive"))
    0 <= z_center <= G.Lz || throw(ArgumentError("z_center must lie between 0 and Lz=$(G.Lz)"))
    z_width > 0 || throw(ArgumentError("z_width must be positive"))
    nx, ny, nz = G.nx, G.ny, G.nz
    x_center, y_center = center === nothing ?
        (G.x0 + G.Lx / 2, G.y0 + G.Ly / 2) : Float64.(center)
    isfinite(x_center) && isfinite(y_center) ||
        throw(ArgumentError("center must be finite"))

    # Horizontal pattern: x_i = x0 + (i - 1) dx carries the phase
    # 2π kx (x_i - x0)/Lx, so the phase factor below aligns every mode at
    # the centre.
    pattern = zeros(ComplexF64, nx, ny)
    for j in 1:ny, i in 1:nx
        kx = i <= (nx + 1) ÷ 2 ? i - 1 : i - 1 - nx
        ky = j <= (ny + 1) ÷ 2 ? j - 1 : j - 1 - ny
        weight = exp(-((kx - kx0)^2 + (ky - ky0)^2) / (2 * sigma_k^2))
        weight > 1e-12 || continue
        phase = 2π * (kx * (x_center - G.x0) / G.Lx + ky * (y_center - G.y0) / G.Ly)
        pattern[i, j] = weight * cis(-phase)
    end
    # Scale the largest |B| on the grid to `amplitude`; the backward
    # transform, like the model's, divides by nx ny.
    peak = maximum(abs, FFTW.ifft(pattern))
    peak > 0 || throw(ArgumentError(
        "the packet has no modes on this grid: (kx, ky) = ($kx0, $ky0) is too far " *
        "outside the resolved wavenumbers for sigma_k = $sigma_k"))
    pattern .*= amplitude / peak

    field = zeros(ComplexF64, nz, nx, ny)
    top = last(G.z_faces)
    for k in 1:nz
        depth = top - G.z[k]
        z_envelope = exp(-((depth - z_center)^2) / (2 * z_width^2))
        @views field[k, :, :] .= z_envelope .* pattern
    end
    return field
end

"""
    add_balanced_component!(S, G, a_ell; workspace=nothing)

Derive the balanced potential vorticity from the streamfunction,
`q = -kₕ²ψ + ∂/∂z (a_ell ∂ψ/∂z)` (see [`compute_q_from_psi!`](@ref)).
Velocities are diagnosed separately by `compute_velocities!`.

Based on init_psi_generic and init_q from the Fortran implementation.

# Arguments
- `S::ModelFields`: Model state with streamfunction psi
- `G::RuntimeGeometry`: RuntimeGeometry structure
- `a_ell`: Model-owned f²/N² coefficient profile

# Example
```julia
# With variable stratification
N2_face = evaluate_N2.(Ref(strat_profile), grid.z_faces[2:end])
a_ell = a_ell_from_N2(N2_face, FPlane(f=1e-4))
add_balanced_component!(fields, geometry, a_ell)
```
"""
function add_balanced_component!(S::ModelFields, G::RuntimeGeometry,
    a_ell::AbstractVector; workspace=nothing)
    @info "Adding balanced component to initial state"

    nz = G.nz
    length(a_ell) == nz || throw(DimensionMismatch("a_ell must have length $nz"))
    compute_q_from_psi!(S.q, S.psi, G, a_ell, vertical_spacing(G); workspace)
    @info "Computed potential vorticity q from streamfunction"
    return S
end

"""
    compute_q_from_psi!(q, psi, G, a_ell, dz; workspace=nothing)

Compute QG potential vorticity from streamfunction.

The Boussinesq PV-streamfunction relationship is:
    q = ∇²ψ + ∂/∂z (a_ell ∂ψ/∂z)

In spectral space with finite differences in z:
    q = -kh² ψ + (1/dz²) ∂z[a_ell ∂zψ]

with Neumann BC ∂ψ/∂z = 0 at boundaries (boundary PV sheets handled by one-sided stencil).
"""
function compute_q_from_psi!(q, psi, G::RuntimeGeometry, a_ell, dz; workspace=nothing)
    nz = size(psi, 1)
    nz == G.nz || throw(DimensionMismatch("ψ has $nz levels; the grid has $(G.nz)"))
    size(q) == size(psi) ||
        throw(DimensionMismatch("q and psi must have the same size"))
    length(a_ell) == nz || throw(DimensionMismatch("a_ell must have length $nz"))
    # The stencil reads ψ at k ± 1 after q[k] is written, so q cannot be ψ.
    q === psi && throw(ArgumentError("q and psi must be distinct arrays"))
    backend = get_backend(q)
    T = real(eltype(q))
    launch!(_q_from_psi_kernel!, backend, size(q), q, psi, G.kx, G.ky,
            on_backend(backend, in_precision(a_ell, T)), T(dz)^2, nz)
    return q
end

@kernel function _q_from_psi_kernel!(q, @Const(ψ), @Const(kx), @Const(ky),
                                     @Const(a), dz2, nz)
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        kh2 = kx[i]^2 + ky[j]^2
        if nz == 1
            # Single-layer case: no vertical derivatives, q = -kh² ψ
            q[k, i, j] = -kh2 * ψ[k, i, j]
        elseif k == 1
            # Bottom boundary: Neumann BC ψ_z = 0 ⟹ ψ[0] = ψ[1]
            vert_term = a[1] * (ψ[2, i, j] - ψ[1, i, j])
            q[k, i, j] = -kh2 * ψ[1, i, j] + vert_term / dz2
        elseif k == nz
            # Top boundary: Neumann BC ψ_z = 0 ⟹ ψ[nz+1] = ψ[nz]
            vert_term = a[nz - 1] * (ψ[nz - 1, i, j] - ψ[nz, i, j])
            q[k, i, j] = -kh2 * ψ[nz, i, j] + vert_term / dz2
        else
            coeff_up = a[k]
            coeff_down = a[k - 1]
            vert_term = coeff_up * ψ[k + 1, i, j] -
                        (coeff_up + coeff_down) * ψ[k, i, j] +
                        coeff_down * ψ[k - 1, i, j]
            q[k, i, j] = -kh2 * ψ[k, i, j] + vert_term / dz2
        end
    end
end

"""
    compute_barotropic_q_from_psi!(q, psi, grid)

Set potential vorticity from a vertically uniform spectral streamfunction using
`q̂ = -kₕ² ψ̂`. This is useful for prescribed barotropic flows, such as the
Asselin et al. (2020) dipole.
"""
function compute_barotropic_q_from_psi!(q, psi, G::RuntimeGeometry)
    size(q) == size(psi) ||
        throw(DimensionMismatch("q and psi must have the same size"))
    launch!(_barotropic_q_kernel!, q, size(q), q, psi, G.kx, G.ky)
    return q
end

@kernel function _barotropic_q_kernel!(q, @Const(ψ), @Const(kx), @Const(ky))
    k, i, j = @index(Global, NTuple)
    @inbounds q[k, i, j] = -(kx[i]^2 + ky[j]^2) * ψ[k, i, j]
end

"""
    compute_psi_from_vorticity!(field, grid)

Overwrite the spectral relative vorticity in `field` with the streamfunction,
`ψ̂ = -ζ̂/kₕ²`, setting the horizontal mean (`kₕ = 0`) of ψ to zero.
"""
function compute_psi_from_vorticity!(field, G::RuntimeGeometry)
    launch!(_psi_from_vorticity_kernel!, field, size(field), field, G.kx, G.ky)
    return field
end

@kernel function _psi_from_vorticity_kernel!(field, @Const(kx), @Const(ky))
    k, i, j = @index(Global, NTuple)
    @inbounds begin
        kₕ² = kx[i]^2 + ky[j]^2
        field[k, i, j] = kₕ² > 0 ? -field[k, i, j] / kₕ² : zero(eltype(field))
    end
end
