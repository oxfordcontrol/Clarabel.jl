## ------------------------------------
# Nonnegative Cone
# -------------------------------------

function rectify_equilibration!(
    K::NonnegativeCone{T},
    enew::VectorView{T},
    eold::VectorView{T}
) where{T}

    #allow elementwise equilibration scaling
    return false
end

function update_scaling!(
    K::NonnegativeCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}
) where {T}

    λ   .= sqrt.(s.*z)
    K.w .= sqrt.(s./z)

    return nothing
end

#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::NonnegativeCone{T}
) where {T}

    K.w .= 1

    return nothing
end

function get_diagonal_scaling!(
    K::NonnegativeCone{T},
    diagW2::VectorView{T}
) where {T}

    diagW2 .= -K.w.^2

    return nothing
end


# implements x = y ∘ z for the nn cone
function circle_op!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    x .= y.*z

    return nothing
end

# implements x = y \ z for the nn cone
function inv_circle_op!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    x .= z./y

    return nothing
end

# place vector into nn cone
function shift_to_cone!(
    K::NonnegativeCone{T},
    z::VectorView{T}
) where{T}

    α = minimum(z)
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        z .+= -α
        z .+=  1
    end

    return nothing
end


# implements y = αWx + βy for the nn cone
function gemv_W!(
    K::NonnegativeCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

    #W is diagonal so ignore transposition
    y .= α.*(K.w.*x) + β.*y

    return nothing
end

# implements y = αW^{-1}x + βy for the nn cone
function gemv_Winv!(
    K::NonnegativeCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

  #W is diagonal, so ignore transposition
  y .= α.*(x./K.w) + β.*y

  return nothing
end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

  y .= x./(K.w.^2)

  return nothing
end

# implements y = W^TW^x
function mul_WtW!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

  y .= x.*(K.w.^2)

  return nothing
end

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::NonnegativeCone,
    x::VectorView{T},α::T
) where {T}

    #e is a vector of ones, so just shift
    x .+= α

    return nothing
end


#return maximum allowable step length while remaining in the nn cone
function step_length(
    K::NonnegativeCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}
) where {T}

    f    = (dv,v)->(dv<0 ? -v/dv : 1/eps(T))
    αz   = minimum(map(f, dz, z))
    αs   = minimum(map(f, ds, s))
    α    = min(αz,αs)

    return α
end
