# ----------------------------------------------------
# Positive Semidefinite Cone
# ----------------------------------------------------
#PJG: Remove when complete
macro __FUNCTION__()
    return :($(esc(Expr(:isdefined, :var"#self#"))) ? $(esc(:var"#self#")) : nothing)
end

dim(K::PSDCone{T})    where {T} = K.dim     #number of elements
degree(K::PSDCone{T}) where {T} = K.n       #side dimension, M \in \mathcal{S}^{n×n}


function update_scaling!(
    K::PSDCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
    @. λ   = sqrt(s*z)
    @. K.w = sqrt(s/z)

    return nothing
end


#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::PSDCone{T}
) where {T}

    K.W .= I(K.n)

    return nothing
end

function get_diagonal_scaling!(
    K::PSDCone{T},
    diagW2::VectorView{T}
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
    @. diagW2 = -K.w^2

    return nothing
end


# implements x = y ∘ z for the SDP cone
#PJG Bottom p5, CVXOPT
function circle_op!(
    K::PSDCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #make square views
    (X,Y,Z) = map(m->reshape(m,K.n,:), (x,y,z))

    X  .= Y*Z + Z*Y
    X .*= 0.5

    return nothing
end

# implements x = y \ z for the SDP cone
# PJG, Top page 14, \S5, CVXOPT 
function inv_circle_op!(
    K::PSDCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #make square views
    (X,Y,Z) = map(m->reshape(m,K.n,:), (x,y,z))
    Γ = similar(X)
    for i = 1:K.n
        for j = 1:K.n
            Γ[i,j] = (Y[i,j] + Y[j,i])/2
        end
    end
    X .= Z./Γ

    return nothing
end

# place vector into nn cone
function shift_to_cone!(
    K::PSDCone{T},
    z::VectorView{T}
) where{T}

    print("Placeholder at :", @__FUNCTION__, "\n")
    α = minimum(z)
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        @. z += -α
        @. z +=  1.
    end

    return nothing
end


# implements y = αWx + βy for the nn cone
function gemv_W!(
    K::PSDCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

print("Placeholder at :", @__FUNCTION__, "\n")
  #W is diagonal so ignore transposition
  #@. y = α*(x*K.w) + β*y
  @inbounds for i = eachindex(y)
      y[i] = α*(x[i]*K.w[i]) + β*y[i]
  end

  return nothing
end

# implements y = αW^{-1}x + βy for the nn cone
function gemv_Winv!(
    K::PSDCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
  #W is diagonal, so ignore transposition
  #@. y = α*(x/K.w) + β.*y
  @inbounds for i = eachindex(y)
      y[i] = α*(x[i]/K.w[i]) + β*y[i]
  end

  return nothing
end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::PSDCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
  @. y = x/(K.w^2)

  return nothing
end

# implements y = W^TW^x
function mul_WtW!(
    K::PSDCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
  @. y = x*(K.w^2)

  return nothing
end

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::PSDCone,
    x::VectorView{T},α::T
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
    #e is a vector of ones, so just shift
    @. x += α

    return nothing
end


#return maximum allowable step length while remaining in the nn cone
function step_length(
    K::PSDCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}
) where {T}

    print("Placeholder at :", @__FUNCTION__, "\n")
    αz = 1/eps(T)
    αs = 1/eps(T)

    for i in eachindex(ds)
        αz = dz[i] < 0 ? min(αz,-z[i]/dz[i]) : αz
        αs = ds[i] < 0 ? min(αs,-s[i]/ds[i]) : αs
    end

    α = min(αz,αs)

    return α
end
