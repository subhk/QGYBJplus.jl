"""
    @local_spectral_loop field begin ... end

Loop over a local `(z, x, y)` spectral field while exposing `k`, `i`, and `j`.
Use this for boilerplate array copies and format conversions, not for hiding
the numerical equations themselves.
"""
macro local_spectral_loop(field, body)
    field_expr = esc(field)
    body_expr = esc(body)
    field_sym = gensym(:field)
    arr_sym = gensym(:arr)
    nz_sym = gensym(:nz_local)
    nx_sym = gensym(:nx_local)
    ny_sym = gensym(:ny_local)

    return quote
        local $field_sym = $field_expr
        local $arr_sym = parent($field_sym)
        local $nz_sym, $nx_sym, $ny_sym = size($arr_sym)
        @inbounds for $(esc(:k)) in 1:$nz_sym, $(esc(:j)) in 1:$ny_sym, $(esc(:i)) in 1:$nx_sym
            $body_expr
        end
    end
end

"""
    @dealiased_spectral_loop field mask begin ... end begin ... end

Loop over a local spectral field while exposing `k`, `i`, `j`, `i_global`, and
`j_global`. The first block runs for kept modes; the second block runs for
masked modes.
"""
macro dealiased_spectral_loop(field, mask, kept_body, masked_body)
    field_expr = esc(field)
    mask_expr = esc(mask)
    kept_expr = esc(kept_body)
    masked_expr = esc(masked_body)
    field_sym = gensym(:field)
    arr_sym = gensym(:arr)
    mask_sym = gensym(:mask)
    nz_sym = gensym(:nz_local)
    nx_sym = gensym(:nx_local)
    ny_sym = gensym(:ny_local)

    return quote
        local $field_sym = $field_expr
        local $mask_sym = $mask_expr
        local $arr_sym = parent($field_sym)
        local $nz_sym, $nx_sym, $ny_sym = size($arr_sym)
        @inbounds for $(esc(:k)) in 1:$nz_sym, $(esc(:j)) in 1:$ny_sym, $(esc(:i)) in 1:$nx_sym
            $(esc(:i_global)) = local_to_global($(esc(:i)), 2, $field_sym)
            $(esc(:j_global)) = local_to_global($(esc(:j)), 3, $field_sym)
            if $mask_sym[$(esc(:i_global)), $(esc(:j_global))]
                $kept_expr
            else
                $masked_expr
            end
        end
    end
end

"""
    @dealiased_wavenumber_loop field grid mask begin ... end begin ... end

Like `@dealiased_spectral_loop`, also exposing `kₓ`, `kᵧ`, and `kₕ²` for the
current horizontal mode.
"""
macro dealiased_wavenumber_loop(field, grid, mask, kept_body, masked_body)
    field_expr = esc(field)
    grid_expr = esc(grid)
    mask_expr = esc(mask)
    kept_expr = esc(kept_body)
    masked_expr = esc(masked_body)
    field_sym = gensym(:field)
    grid_sym = gensym(:grid)
    mask_sym = gensym(:mask)
    arr_sym = gensym(:arr)
    nz_sym = gensym(:nz_local)
    nx_sym = gensym(:nx_local)
    ny_sym = gensym(:ny_local)

    return quote
        local $field_sym = $field_expr
        local $grid_sym = $grid_expr
        local $mask_sym = $mask_expr
        local $arr_sym = parent($field_sym)
        local $nz_sym, $nx_sym, $ny_sym = size($arr_sym)
        @inbounds for $(esc(:k)) in 1:$nz_sym, $(esc(:j)) in 1:$ny_sym, $(esc(:i)) in 1:$nx_sym
            $(esc(:i_global)) = local_to_global($(esc(:i)), 2, $field_sym)
            $(esc(:j_global)) = local_to_global($(esc(:j)), 3, $field_sym)
            $(esc(:kₓ)) = $grid_sym.kx[$(esc(:i_global))]
            $(esc(:kᵧ)) = $grid_sym.ky[$(esc(:j_global))]
            $(esc(:kₕ²)) = $(esc(:kₓ))^2 + $(esc(:kᵧ))^2
            if $mask_sym[$(esc(:i_global)), $(esc(:j_global))]
                $kept_expr
            else
                $masked_expr
            end
        end
    end
end
