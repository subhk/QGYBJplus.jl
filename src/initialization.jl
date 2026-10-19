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

@kernel function _random_psi_kernel!(ψ, base_seed, amplitude, slope, nx, ny)
    k, i, j = @index(Global, NTuple)
    kx_max = nx ÷ 2
    ky_max = ny ÷ 2
    kx = i <= kx_max ? i - 1 : i - 1 - nx
    ky = j <= ky_max ? j - 1 : j - 1 - ny
    value = zero(ComplexF64)   # hashed in Float64, so all architectures agree
    if !(kx == 0 && ky == 0)
        k_total = sqrt(Float64(kx^2 + ky^2))
        energy = amplitude * k_total^slope
        i_conj = i == 1 ? 1 : nx - i + 2
        j_conj = j == 1 ? 1 : ny - j + 2
        if i == i_conj && j == j_conj
            # Self-conjugate modes must be real for a real ψ.
            value = complex(sqrt(2 * energy) *
                            _hash_randn(base_seed, UInt64(0x10), k, i, j))
        else
            # Hash the canonical member of each conjugate pair, so both
            # members agree and the field is Hermitian.
            canonical = (i < i_conj) || (i == i_conj && j <= j_conj)
            i_can = canonical ? i : i_conj
            j_can = canonical ? j : j_conj
            amp = sqrt(2 * energy) * _hash_randn(base_seed, UInt64(0x20), k, i_can, j_can)
            phase = 2 * pi * _hash_u01(base_seed, UInt64(0x30), k, i_can, j_can)
            canonical_value = amp * cis(phase)
            value = canonical ? canonical_value : conj(canonical_value)
        end
    end
    @inbounds ψ[k, i, j] = value
end

"""
    init_random_psi!(psik, G, amplitude; slope=-3.0, seed=0, seed_offset=0)

Fill `psik` with a random streamfunction spectrum `E(k) ∝ amplitude k^slope`
that has Hermitian symmetry (a real ψ in physical space), then apply the
radial two-thirds mask. The values depend only on the seeds and the global
mode indices, so every architecture produces the same field up to rounding.
"""
function init_random_psi!(psik, G::RuntimeGeometry, amplitude::Real;
                          slope::Real=-3.0, seed::Int=0, seed_offset::Int=0)
    isfinite(amplitude) && isfinite(slope) ||
        throw(ArgumentError("random streamfunction parameters must be finite"))
    amplitude >= 0 || throw(ArgumentError("amplitude must be non-negative"))
    base_seed = _mix64(UInt64(seed) ⊻ UInt64(seed_offset))
    launch!(_random_psi_kernel!, psik, size(psik), psik, base_seed,
            Float64(amplitude), Float64(slope), G.nx, G.ny)
    apply_dealiasing_mask!(psik, G)
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
                       z_center=G.Lz/2, z_width=G.Lz/4)

Create a horizontally localized wave packet in spectral space with a Gaussian
vertical envelope, as a host array. `z_center` and `z_width` are depths
measured positively downward from the top of the domain (its rigid lid).
"""
function create_wave_packet(G::RuntimeGeometry, kx0::Int, ky0::Int, sigma_k::Real, amplitude::Real;
                            z_center::Real=G.Lz / 2,
                            z_width::Real=G.Lz / 4)
    sigma_k > 0 || throw(ArgumentError("sigma_k must be positive"))
    0 <= z_center <= G.Lz || throw(ArgumentError("z_center must lie between 0 and Lz=$(G.Lz)"))
    z_width > 0 || throw(ArgumentError("z_width must be positive"))

    field = zeros(ComplexF64, G.nz, G.nx, G.ny)

    kx_max = G.nx ÷ 2
    ky_max = G.ny ÷ 2

    top = last(G.z_faces)
    for k in 1:G.nz
        depth = top - G.z[k]
        z_envelope = exp(-((depth - z_center)^2) / (2 * z_width^2))

        for j in 1:G.ny
            ky = j <= ky_max ? j-1 : j-1-G.ny

            for i in 1:G.nx
                kx = i <= kx_max ? i - 1 : i - 1 - G.nx

                # Gaussian envelope in wavenumber space
                k_dist2 = (kx - kx0)^2 + (ky - ky0)^2
                envelope = exp(-k_dist2 / (2 * sigma_k^2))

                if envelope > 1e-10
                    phase = 2π * rand()
                    field[k, i, j] = amplitude * envelope * z_envelope * cis(phase)
                end
            end
        end
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
    @assert nz == G.nz "Vertical dimension must match the grid"
    size(q) == size(psi) ||
        throw(DimensionMismatch("q and psi must have the same size"))
    backend = get_backend(q)
    launch!(_q_from_psi_kernel!, backend, size(q), q, psi, G.kx, G.ky,
            on_backend(backend, a_ell), dz^2, nz)
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
