"""
Compute `a(z) = f² / N²(z)`, with `N²` bounded below by `√eps` times its
largest value. The bound is relative so that it means the same in every unit
system: an absolute `√eps ≈ 1.5e-8` would clamp realistic abyssal N² in s⁻².
"""
function a_ell_from_N2(N2_profile::AbstractVector, f::Real)
    isempty(N2_profile) &&
        throw(ArgumentError("the stratification profile cannot be empty"))
    T = promote_type(eltype(N2_profile), typeof(float(f)))
    for (k, value) in enumerate(N2_profile)
        isfinite(value) ||
            throw(ArgumentError("N² must be finite at vertical level $k"))
        value > zero(value) ||
            throw(ArgumentError("N² must be positive at vertical level $k"))
    end
    threshold = sqrt(eps(T)) * T(maximum(N2_profile))
    warned = false
    coefficients = Vector{T}(undef, length(N2_profile))
    @inbounds for k in eachindex(N2_profile)
        value = T(N2_profile[k])
        if value < threshold && !warned
            @warn "N² is below √eps times its largest value; clamping elliptic coefficients" level=k threshold maxlog=1
            warned = true
        end
        coefficients[k] = T(f)^2 / max(value, threshold)
    end
    return coefficients
end

a_ell_from_N2(N2_profile::AbstractVector, coriolis::FPlane) =
    a_ell_from_N2(N2_profile, coriolis.f)

"""Return the two-thirds dealiasing mask for global geometry."""
function dealias_mask(geometry::Union{RectilinearGrid, RuntimeGeometry})
    nx, ny = geometry isa RectilinearGrid ? geometry.size[1:2] :
                                            (geometry.nx, geometry.ny)
    return Bool[is_dealiased(i, j, nx, ny) for i in 1:nx, j in 1:ny]
end

"""
Test whether the Fourier mode with indices `(i, j)` lies inside the two-thirds
cutoff: the ellipse `(iₓ/Kₓ)² + (i_y/K_y)² ≤ 1` in mode index, with
`Kₓ = ⌊nₓ/3⌋` and `K_y = ⌊n_y/3⌋`, which is a disk on a square grid.
"""
@inline function is_dealiased(i::Int, j::Int, nx::Int, ny::Int)
    ix = i - 1
    ix = ix <= nx ÷ 2 ? ix : ix - nx
    jy = j - 1
    jy = jy <= ny ÷ 2 ? jy : jy - ny
    kx_max, ky_max = fld(nx, 3), fld(ny, 3)
    # A dimension too short to resolve any mode keeps only its mean.
    (kx_max == 0 && ix != 0) && return false
    (ky_max == 0 && jy != 0) && return false
    # The ellipse in exact integer arithmetic, so boundary modes are never
    # decided by rounding.
    sx, sy = max(kx_max, 1)^2, max(ky_max, 1)^2
    radius² = ix^2 * sy + jy^2 * sx
    # The exact |k| = n/3 boundary is unsafe for an unpadded quadratic
    # product: two boundary modes can alias back onto the retained region.
    # Exclude it when a dimension is divisible by three, so that every
    # retained mode has |iₓ| < nₓ/3 and |i_y| < n_y/3 and products of
    # retained modes never alias onto one.
    return nx % 3 == 0 || ny % 3 == 0 ? radius² < sx * sy : radius² <= sx * sy
end

@inline is_dealiased(i::Int, j::Int, geometry::RuntimeGeometry) =
    is_dealiased(i, j, geometry.nx, geometry.ny)
@inline is_dealiased(i::Int, j::Int, geometry::RectilinearGrid) =
    is_dealiased(i, j, geometry.size[1], geometry.size[2])

"""
    compute_hyperdiff_coeff(; dx, dy, dt, order=4, efold_steps=10, kmax_fraction=2/3)

Coefficient `ν` of a hyperdiffusion term `ν kₕ^order` that damps the target
wavenumber `kmax_fraction π / min(dx, dy)` by one e-fold every `efold_steps`
steps of length `dt`.

The default, `kmax_fraction = 2/3`, targets the largest wavenumber the
two-thirds dealiasing keeps. `kmax_fraction = 1` targets the Nyquist
wavenumber instead; nonlinear runs remove it, so the smallest scale they keep
is then damped `(3/2)^order` times more weakly than intended.
"""
function compute_hyperdiff_coeff(; dx::Real, dy::Real, dt::Real,
    order::Int=4, efold_steps::Int=10, kmax_fraction::Real=2/3)

    all(isfinite, (dx, dy, dt, kmax_fraction)) ||
        throw(ArgumentError("spacing, step size, and scale fraction must be finite"))
    dx > 0 && dy > 0 && dt > 0 ||
        throw(ArgumentError("spacing and step size must be positive"))
    order > 0 && iseven(order) ||
        throw(ArgumentError("hyperdiffusion order must be positive and even"))
    efold_steps > 0 || throw(ArgumentError("efold_steps must be positive"))
    0 < kmax_fraction <= 1 ||
        throw(ArgumentError("kmax_fraction must lie in (0, 1]"))

    k_max = kmax_fraction * π / min(dx, dy)
    return inv(efold_steps * dt * k_max^order)
end

"""
Compute a coefficient and Laplacian power from domain dimensions (see
[`compute_hyperdiff_coeff`](@ref) for `kmax_fraction`).
"""
function compute_hyperdiff_params(; nx::Int, ny::Int,
    Lx::Real, Ly::Real, dt::Real, order::Int=4, efold_steps::Int=10,
    kmax_fraction::Real=2/3)

    nx > 0 && ny > 0 || throw(ArgumentError("dimensions must be positive"))
    coefficient = compute_hyperdiff_coeff(
        ; dx=Lx / nx, dy=Ly / ny, dt, order, efold_steps, kmax_fraction)
    return (ν=coefficient, ilap=order ÷ 2, order)
end

"""Return matching typed closure coefficients for flow and waves."""
function dimensional_hyperdiff_params(; nx::Int, ny::Int,
    Lx::Real, Ly::Real, dt::Real, order::Int=4, efold_steps::Int=10,
    kmax_fraction::Real=2/3)

    result = compute_hyperdiff_params(
        ; nx, ny, Lx, Ly, dt, order, efold_steps, kmax_fraction)
    return (
        νₕ₁=result.ν,
        ilap1=result.ilap,
        νₕ₂=0.0,
        ilap2=2,
        νₕ₁ʷ=result.ν,
        ilap1w=result.ilap,
        νₕ₂ʷ=0.0,
        ilap2w=2,
        order,
        efold_steps,
    )
end
