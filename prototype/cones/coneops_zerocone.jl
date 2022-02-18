# -------------------------------------
# Zero Cone
# -------------------------------------

degree(K::ZeroCone{T}) where {T} = 0

function rectify_equilibration!(
    K::ZeroCone{T},
    δ::VectorView{T},
    e::VectorView{T}
) where{T}

    #allow elementwise equilibration scaling
    δ .= e
    return false
end


function update_scaling!(
    K::ZeroCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}
) where {T}

    λ   .= 0

    return nothing
end

function set_identity_scaling!(
    K::ZeroCone{T}
) where {T}

    #do nothing.   "Identity" scaling will be zero for equalities
    return nothing
end


function get_diagonal_scaling!(
    K::ZeroCone{T},
    diagW2::VectorView{T}
) where {T}

    diagW2 .= 0.

    return nothing
end

# implements x = y ∘ z for the zero cone
function circle_op!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    x .= 0

    return nothing
end

# implements x = y \ z for the zero cone
function inv_circle_op!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    x .= 0

    return nothing
end

# place vector into zero cone
function shift_to_cone!(
    K::ZeroCone{T},z::VectorView{T}
) where{T}

    z .= 0

    return nothing
end

# implements y = αWx + βy for the zero cone
function gemv_W!(
    K::ZeroCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
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
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

  #treat Winv like zero
  y .= β.*y

  return nothing
end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

  #treat inv(W^TW) like zero
  y .= 0

  return nothing
end

# implements y = W^TWx
function mul_WtW!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

  #treat (W^TW) like zero
  y .= 0

  return nothing
end

# implements y = y + αe for the zero cone
function add_scaled_e!(
    K::ZeroCone{T},
    x::VectorView{T},α::T
) where {T}

    #e = 0, do nothing
    return nothing

end


function step_length(
     K::ZeroCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}
) where {T}

    #equality constraints allow arbitrary step length
    return 1/eps(T)

end
