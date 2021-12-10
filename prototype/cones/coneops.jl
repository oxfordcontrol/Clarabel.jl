# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function cones_update_scaling!(
    cones::ConeSet{T},
    s::SplitVector{T},
    z::SplitVector{T},
    λ::SplitVector{T}
) where {T}

    # update scalings by passing subview to each of
    # the appropriate cone types.
    foreach(update_scaling!,cones,s.views,z.views,λ.views)

    return nothing
end

function cones_set_identity_scaling!(
    cones::ConeSet{T}
) where {T}

    foreach(set_identity_scaling!,cones)

    return nothing
end


# The diagonal part of the KKT scaling
# matrix for each cone
function cones_get_diagonal_scaling!(
    cones::ConeSet{T},
    diagW2::SplitVector{T}
) where {T}

    foreach(get_diagonal_scaling!,cones,diagW2.views)
    return nothing
end

# x = y ∘ z
function cones_circle_op!(
    cones::ConeSet{T},
    x::SplitVector{T},
    y::SplitVector{T},
    z::SplitVector{T}
) where {T}

    foreach(circle_op!,cones,x.views,y.views,z.views)
    return nothing
end

# x = y \ z
function cones_inv_circle_op!(
    cones::ConeSet{T},
    x::SplitVector{T},
    y::SplitVector{T},
    z::SplitVector{T}
) where {T}

    foreach(inv_circle_op!,cones,x.views,y.views,z.views)
    return nothing
end

# place a vector to some nearby point in the cone
function cones_shift_to_cone!(
    cones::ConeSet{T},
    z::SplitVector{T}
) where {T}

    foreach(shift_to_cone!,cones,z.views)
    return nothing
end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface
function cones_gemv_W!(
    cones::ConeSet{T},
    is_transpose::Bool,
    x::SplitVector{T},
    y::SplitVector{T},
    α::T,
    β::T
) where {T}

    foreach((c,x,y)->gemv_W!(c,is_transpose,x,y,α,β),cones,x.views,y.views)
    return nothing
end

# computes y = αW^{-1}x + βy, or y = αW⁻ᵀx + βy, i.e.
# similar to the BLAS gemv interface
function cones_gemv_Winv!(
    cones::ConeSet{T},
    is_transpose::Bool,
    x::SplitVector{T},
    y::SplitVector{T},
    α::T,
    β::T
) where {T}

    foreach((c,x,y)->gemv_Winv!(c,is_transpose,x,y,α,β),cones,x.views,y.views)
    return nothing
end

# computes y = (W^TW){-1}x
function cones_mul_WtWinv!(
    cones::ConeSet{T},
    x::SplitVector{T},
    y::SplitVector{T}
) where {T}

    foreach(mul_WtWinv!,cones,x.views,y.views)
    return nothing
end

# computes y = (W^TW)x
function cones_mul_WtW!(
    cones::ConeSet{T},
    x::SplitVector{T},
    y::SplitVector{T}
) where {T}

    foreach(mul_WtW!,cones,x.views,y.views)
    return nothing
end

#computes y = y + αe
function cones_add_scaled_e!(
    cones::ConeSet{T},
    x::SplitVector{T},
    α::T
) where {T}

    foreach((c,x)->add_scaled_e!(c,x,α),cones,x.views)
    return nothing
end

# maximum allowed step length over all cones
function cones_step_length(
    cones::ConeSet{T},
    dz::SplitVector{T},
    ds::SplitVector{T},
     z::SplitVector{T},
     s::SplitVector{T},
     λ::SplitVector{T}
) where {T}

    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views
    λ     = λ.views

    α = minimum(map(step_length,cones,dz,ds,z,s,λ))
    return α
end
