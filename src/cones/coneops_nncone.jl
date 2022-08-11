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
    μ::T,
    scaling_strategy::ScalingStrategy
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

function get_WtW_block!(
    K::NonnegativeCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #this block is diagonal, and we expect here
    #to receive only the diagonal elements to fill
    @. WtWblock = K.w^2

    return nothing
end

# returns x = λ∘λ for the nn cone
function affine_ds!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
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
        add_scaled_e!(K,z,-α)
        add_scaled_e!(K,z,one(T))
    end

    return nothing
end

# asymmetric initialization
function asymmetric_init!(
   K::NonnegativeCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    s .= one(T)
    z .= one(T)

   return nothing
end

# implements y = αWx + βy for the nn cone
function gemv_W!(
    K::NonnegativeCone{T},
    is_transpose::Symbol,
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
    is_transpose::Symbol,
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

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::NonnegativeCone{T},
    x::AbstractVector{T},α::T
) where {T}

    #e is a vector of ones, so just shift
    @. x += α

    return nothing
end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function combined_ds!(
    K::NonnegativeCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    tmp = dz                #alias
    dz .= step_z            #copy for safe call to gemv_W
    gemv_W!(K,:N,tmp,step_z,one(T),zero(T))         #Δz <- WΔz
    tmp .= step_s           #copy for safe call to gemv_Winv
    gemv_Winv!(K,:T,tmp,step_s,one(T),zero(T))      #Δs <- W⁻¹Δs
    circ_op!(K,tmp,step_s,step_z)                   #tmp = W⁻¹Δs ∘ WΔz
    add_scaled_e!(K,tmp,-σμ)                        #tmp = W⁻¹Δs ∘ WΔz - σμe

    return nothing
end

# compute the generalized step Wᵀ(λ \ ds)
function Wt_λ_inv_circ_ds!(
    K::NonnegativeCone{T},
    lz::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
    Wtlinvds::AbstractVector{T}
) where {T}

    tmp = lz;
    @. tmp = rz  #Don't want to modify our RHS
    λ_inv_circ_op!(K,tmp,rs)                  #tmp = λ \ ds
    gemv_W!(K,:T,tmp,Wtlinvds,one(T),zero(T)) #Wᵀ(λ \ ds) = Wᵀ(tmp)

    return nothing
end

# compute the generalized step of -WᵀWΔz
function WtW_Δz!(
    K::NonnegativeCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    gemv_W!(K,:N,lz,workz,one(T),zero(T))    #work = WΔz
    gemv_W!(K,:T,workz,ls,-one(T),zero(T))   #Δs = -WᵀWΔz

end

#return maximum allowable step length while remaining in the nn cone
function step_length(
    K::NonnegativeCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     α::T
) where {T}

    αz = floatmax(T)
    αs = floatmax(T)

    for i in eachindex(ds)
        αz = dz[i] < 0 ? prevfloat(min(αz,-z[i]/dz[i])) : αz
        αs = ds[i] < 0 ? prevfloat(min(αs,-s[i]/ds[i])) : αs
    end

    return (αz,αs)
end

function compute_centrality(
    K::NonnegativeCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    barrier = T(0)
    for i = 1:K.dim
        barrier += -log(s[i]) - log(z[i])
    end

    return barrier
end
