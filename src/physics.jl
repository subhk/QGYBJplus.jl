"""Vertical and horizontal coefficients used by the spectral kernels."""

"""Compute `a(z) = f² / N²(z)` with a stable lower bound on `N²`."""
function a_ell_from_N2(N2_profile::AbstractVector, f::Real)
    isempty(N2_profile) &&
        throw(ArgumentError("the stratification profile cannot be empty"))
    T = promote_type(eltype(N2_profile), typeof(float(f)))
    threshold = sqrt(eps(T))
    warned = false
    coefficients = Vector{T}(undef, length(N2_profile))
    @inbounds for k in eachindex(N2_profile)
        value = T(N2_profile[k])
        isfinite(value) ||
            throw(ArgumentError("N² must be finite at vertical level $k"))
        value > zero(T) ||
            throw(ArgumentError("N² must be positive at vertical level $k"))
        if value < threshold && !warned
            @warn "N² is close to zero; clamping elliptic coefficients" level=k threshold maxlog=1
            warned = true
        end
        coefficients[k] = T(f)^2 / max(value, threshold)
    end
    return coefficients
end

a_ell_from_N2(N2_profile::AbstractVector, coriolis::FPlane) =
    a_ell_from_N2(N2_profile, coriolis.f)

"""Return the radial two-thirds dealiasing mask for global geometry."""
function dealias_mask(geometry::Union{RectilinearGrid, RuntimeGeometry})
    nx, ny = geometry isa RectilinearGrid ? geometry.size[1:2] :
                                            (geometry.nx, geometry.ny)
    return Bool[is_dealiased(i, j, nx, ny) for i in 1:nx, j in 1:ny]
end

"""Test whether a global Fourier index lies inside the radial cutoff."""
@inline function is_dealiased(i_global::Int, j_global::Int,
                              nx::Int, ny::Int)
    kmax = fld(min(nx, ny), 3)
    ix = i_global - 1
    ix = ix <= nx ÷ 2 ? ix : ix - nx
    jy = j_global - 1
    jy = jy <= ny ÷ 2 ? jy : jy - ny
    return ix^2 + jy^2 <= kmax^2
end

@inline is_dealiased(i::Int, j::Int, geometry::RuntimeGeometry) =
    is_dealiased(i, j, geometry.nx, geometry.ny)
@inline is_dealiased(i::Int, j::Int, geometry::RectilinearGrid) =
    is_dealiased(i, j, geometry.size[1], geometry.size[2])

"""Coefficient giving one e-fold of damping at the target scale."""
function compute_hyperdiff_coeff(; dx::Real, dy::Real, dt::Real,
    order::Int=4, efold_steps::Int=10, kmax_fraction::Real=1)

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

"""Compute a coefficient and Laplacian power from domain dimensions."""
function compute_hyperdiff_params(; nx::Int, ny::Int,
    Lx::Real, Ly::Real, dt::Real, order::Int=4, efold_steps::Int=10)

    nx > 0 && ny > 0 || throw(ArgumentError("dimensions must be positive"))
    coefficient = compute_hyperdiff_coeff(
        dx=Lx / nx, dy=Ly / ny, dt, order, efold_steps)
    return (ν=coefficient, ilap=order ÷ 2, order)
end

"""Return matching typed closure coefficients for flow and waves."""
function dimensional_hyperdiff_params(; nx::Int, ny::Int,
    Lx::Real, Ly::Real, dt::Real, order::Int=4, efold_steps::Int=10)

    result = compute_hyperdiff_params(
        ; nx, ny, Lx, Ly, dt, order, efold_steps)
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
