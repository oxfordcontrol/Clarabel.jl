# -------------------------------------
# Zero Cone
# -------------------------------------

degree(K::ZeroCone{T}) where {T} = 0

function rectify_equilibration!(
    K::ZeroCone{T},
    δ::AbstractVector{T},
    e::AbstractVector{T}
) where{T}

    #allow elementwise equilibration scaling
    δ .= e
    return false
end


function update_scaling!(
    K::ZeroCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #nothing to do.
    #This cone acts like λ = 0 everywhere.
    return nothing
end

function set_identity_scaling!(
    K::ZeroCone{T}
) where {T}

    #do nothing.   "Identity" scaling will be zero for equalities
    return nothing
end


function get_WtW_block!(
    K::ZeroCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #expecting only a diagonal here, and
    #setting it to zero since this is an
    #equality condition
    WtWblock .= zero(T)

    return nothing
end

function λ_circ_λ!(
    K::ZeroCone{T},
    x::AbstractVector{T}
) where {T}

    x .= zero(T)

end

# implements x = y ∘ z for the zero cone
function circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# implements x = λ \ z for the zerocone.
# We treat λ as zero always for this cone
function λ_inv_circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# implements x = y \ z for the zero cone
function inv_circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# place vector into zero cone
function shift_to_cone!(
    K::ZeroCone{T},z::AbstractVector{T}
) where{T}

    z .= zero(T)

    return nothing
end

# implements y = αWx + βy for the zero cone
function gemv_W!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

    #treat W like zero
    y .= β.*y

    return nothing
end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #treat Winv like zero
  y .= β.*y

  return nothing
end

# implements y = y + αe for the zero cone
function add_scaled_e!(
    K::ZeroCone{T},
    x::AbstractVector{T},α::T
) where {T}

    #e = 0, do nothing
    return nothing

end


function step_length(
     K::ZeroCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T}
) where {T}

    #equality constraints allow arbitrary step length
    huge = inv(eps(T))
    return (huge,huge)
end
