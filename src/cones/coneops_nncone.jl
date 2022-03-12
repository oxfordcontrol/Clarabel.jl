## ------------------------------------
# Nonnegative Cone
# -------------------------------------

function rectify_equilibration!(
    K::NonnegativeCone{T},
    δ::AbstractVector{T},
    e::AbstractVector{T}
) where{T}

    #allow elementwise equilibration scaling
    δ .= e
    return false
end

function update_scaling!(
    K::NonnegativeCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
) where {T}

    @. K.λ = sqrt(s*z)
    @. K.w = sqrt(s/z)

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
    diagW2::AbstractVector{T}
) where {T}

    @. diagW2 = -K.w^2

    return nothing
end

# returns x = λ∘λ for the nn cone
function λ_circ_λ!(
    K::NonnegativeCone{T},
    x::AbstractVector{T}
) where {T}

    @. x = K.λ^2

    return nothing
end

# implements x = y ∘ z for the nn cone
function circ_op!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @. x = y*z

    return nothing
end

# implements x = λ \ z for the nn cone, where λ
# is the internally maintained scaling variable.
function λ_inv_circ_op!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    inv_circ_op!(K, x, K.λ, z)

end

# implements x = y \ z for the nn cone
function inv_circ_op!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @. x = z/y

    return nothing
end

# place vector into nn cone
function shift_to_cone!(
    K::NonnegativeCone{T},
    z::AbstractVector{T}
) where{T}

    α = minimum(z)
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        @. z += -α
        @. z +=  one(T)
    end

    return nothing
end


# implements y = αWx + βy for the nn cone
function gemv_W!(
    K::NonnegativeCone{T},
    is_transpose::Bool,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #W is diagonal so ignore transposition
  #@. y = α*(x*K.w) + β*y
  @inbounds for i = eachindex(y)
      y[i] = α*(x[i]*K.w[i]) + β*y[i]
  end

  return nothing
end

# implements y = αW^{-1}x + βy for the nn cone
function gemv_Winv!(
    K::NonnegativeCone{T},
    is_transpose::Bool,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #W is diagonal, so ignore transposition
  #@. y = α*(x/K.w) + β.*y
  @inbounds for i = eachindex(y)
      y[i] = α*(x[i]/K.w[i]) + β*y[i]
  end

  return nothing
end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

  @. y = x/(K.w^2)

  return nothing
end

# implements y = W^TW^x
function mul_WtW!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

  @. y = x*(K.w^2)

  return nothing
end

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::NonnegativeCone,
    x::AbstractVector{T},α::T
) where {T}

    #e is a vector of ones, so just shift
    @. x += α

    return nothing
end


#return maximum allowable step length while remaining in the nn cone
function step_length(
    K::NonnegativeCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
) where {T}

    αz = 1/eps(T)
    αs = 1/eps(T)

    for i in eachindex(ds)
        αz = dz[i] < 0 ? min(αz,-z[i]/dz[i]) : αz
        αs = ds[i] < 0 ? min(αs,-s[i]/ds[i]) : αs
    end

    α = min(αz,αs)

    return α
end
