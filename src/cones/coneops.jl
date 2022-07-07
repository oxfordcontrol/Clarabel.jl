# -----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

function _conedispatch(x, call)
    thetypes = collect(values(ConeDict))
    foldr((t, tail) -> :(if $x isa $t; $call else $tail end), thetypes, init=Expr(:block))
end

macro conedispatch(call)
    esc(_conedispatch(:cone, call))
end

#for debugging.  Replace @conedispatch with @noop
#to disable the type expansion.
macro noop(call)
    esc(call)
end

# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function cones_rectify_equilibration!(
    cones::ConeSet{T},
     δ::ConicVector{T},
     e::ConicVector{T}
) where{T}

    any_changed = false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    δ .= 1

    for (cone,δi,ei) in zip(cones,δ.views,e.views)
        @conedispatch any_changed |= rectify_equilibration!(cone,δi,ei)
    end

    return any_changed
end



function cones_update_scaling!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T},
) where {T}

    for (cone,si,zi) in zip(cones,s.views,z.views)
        @conedispatch update_scaling!(cone,si,zi)
    end

    return nothing
end

function cones_set_identity_scaling!(
    cones::ConeSet{T}
) where {T}

    for cone in cones
        @conedispatch set_identity_scaling!(cone)
    end

    return nothing
end

# The WtW block for each cone.
function cones_get_WtW_blocks!(
    cones::ConeSet{T},
    WtWblocks::Vector{Vector{T}}
) where {T}

    for (cone, block) in zip(cones,WtWblocks)
        @conedispatch get_WtW_block!(cone,block)
    end
    return nothing
end

# x = λ ∘ λ
function cones_λ_circ_λ!(
    cones::ConeSet{T},
    x::ConicVector{T}
) where {T}

    for (cone,xi) in zip(cones,x.views)
        @conedispatch λ_circ_λ!(cone,xi)
    end
    return nothing
end

# x = y ∘ z
function cones_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,xi,yi,zi) in zip(cones, x.views, y.views, z.views)
        @conedispatch circ_op!(cone,xi,yi,zi)
    end
    return nothing
end

# x = λ \ z,  where λ is scaled internal
# variable for each cone
function cones_λ_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,xi,zi) in zip(cones, x.views, z.views)
        @conedispatch λ_inv_circ_op!(cone,xi,zi)
    end
    return nothing
end

# x = y \ z
function cones_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,xi,yi,zi) in zip(cones, x.views, y.views, z.views)
        @conedispatch inv_circ_op!(cone,xi,yi,zi)
    end
    return nothing
end

# place a vector to some nearby point in the cone
function cones_shift_to_cone!(
    cones::ConeSet{T},
    z::ConicVector{T}
) where {T}

    for (cone,zi) in zip(cones, z.views)
        @conedispatch shift_to_cone!(cone,zi)
    end
    return nothing
end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface.
#Warning: x must not alias y.
function cones_gemv_W!(
    cones::ConeSet{T},
    is_transpose::Symbol,
    x::ConicVector{T},
    y::ConicVector{T},
    α::T,
    β::T
) where {T}

    for (cone,xi,yi) in zip(cones, x.views, y.views)
        @conedispatch gemv_W!(cone,is_transpose,xi,yi,α,β)
    end
    return nothing
end

# computes y = αW^{-1}x + βy, or y = αW⁻ᵀx + βy, i.e.
# similar to the BLAS gemv interface.
#Warning: x must not alias y.
function cones_gemv_Winv!(
    cones::ConeSet{T},
    is_transpose::Symbol,
    x::ConicVector{T},
    y::ConicVector{T},
    α::T,
    β::T
) where {T}

    for (cone,xi,yi) in zip(cones, x.views, y.views)
        @conedispatch gemv_Winv!(cone,is_transpose,xi,yi,α,β)
    end
    return nothing
end

#computes y = y + αe
function cones_add_scaled_e!(
    cones::ConeSet{T},
    x::ConicVector{T},
    α::T
) where {T}

    for (cone,xi) in zip(cones, x.views)
        @conedispatch add_scaled_e!(cone,xi,α)
    end
    return nothing
end

# maximum allowed step length over all cones
function cones_step_length(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
     z::ConicVector{T},
     s::ConicVector{T}
) where {T}

    huge    = floatmax(T)
    (αz,αs) = (huge, huge)

    for (cone,dzi,dsi,zi,si) in zip(cones,dz.views,ds.views,z.views,s.views)
        @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si)
        αz = min(αz, nextαz)
        αs = min(αs, nextαs)
    end

    return (αz,αs)
end
