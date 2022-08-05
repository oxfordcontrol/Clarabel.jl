# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

#degree = 1 for SOC, since e'*e = 1
degree(K::SecondOrderCone{T}) where {T} = 1

function update_scaling!(
    K::SecondOrderCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    flag::Bool
) where {T}

    #first calculate the scaled vector w
    @views zscale = sqrt(z[1]^2 - dot(z[2:end],z[2:end]))
    @views sscale = sqrt(s[1]^2 - dot(s[2:end],s[2:end]))
    gamma  = sqrt((1 + dot(s,z)/(zscale*sscale) ) / 2)

    w = K.w

    w     .= s./(2*sscale*gamma)
    w[1]  += z[1]/(2*zscale*gamma)

    @views w[2:end] .-= z[2:end]/(2*zscale*gamma)

    #various intermediate calcs for u,v,d,η
    w0p1 = w[1] + 1
    @views w1sq = dot(w[2:end],w[2:end])
    w0sq = w[1]*w[1]
    α  = w0p1 + w1sq / w0p1
    β  = 1 + 2/w0p1 + w1sq / (w0p1*w0p1)

    #Scalar d is the upper LH corner of the diagonal
    #term in the rank-2 update form of W^TW
    K.d = w0sq/2 + w1sq/2 * (1 - (α*α)/(1+w1sq*β))

    #the leading scalar term for W^TW
    K.η = sqrt(sscale/zscale)

    #the vectors for the rank two update
    #representation of W^TW
    u0 = sqrt(w0sq + w1sq - K.d)
    u1 = α/u0
    v0 = zero(T)
    v1 = sqrt(u1*u1 - β)
    K.u[1] = u0
    @views K.u[2:end] .= u1.*K.w[2:end]
    K.v[1] = v0
    @views K.v[2:end] .= v1.*K.w[2:end]

    #λ = Wz
    gemv_W!(K,:N,z,K.λ,one(T),zero(T))

    return nothing
end

#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::SecondOrderCone{T}
) where {T}

    K.d  = one(T)
    K.u .= zero(T)
    K.v .= zero(T)
    K.η  = one(T)
    K.w .= zero(T)

    return nothing
end

function get_WtW_block!(
    K::SecondOrderCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal D block from the
    #sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The ConicVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries

    WtWblock    .= (K.η^2)
    WtWblock[1] *= K.d

    return nothing
end


# returns x = λ ∘ λ for the socone
function affine_ds!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    circ_op!(K,x,K.λ,K.λ)

    return nothing
end


# implements x = y ∘ z for socone
function circ_op!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x[1] = dot(y,z)
    y0   = y[1]
    z0   = z[1]
    for i = 2:length(x)
        x[i] = y0*z[i] + z0*y[i]
    end

    return nothing
end

# implements x = λ \ z for the socone, where λ
# is the internally maintained scaling variable.
function λ_inv_circ_op!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    inv_circ_op!(K, x, K.λ, z)

end

# implements x = y \ z for the socone
function inv_circ_op!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @views p = (y[1]^2 - dot(y[2:end],y[2:end]))
    pinv = 1/p
    @views v = dot(y[2:end],z[2:end])

    x[1]      = (y[1]*z[1] - v)*pinv
    @views x[2:end] .= pinv*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

    return nothing
end

# place vector into socone
function shift_to_cone!(
    K::SecondOrderCone{T},
    z::AbstractVector{T}
) where{T}

    z[1] = max(z[1],0)

    @views α = z[1]^2 - dot(z[2:end],z[2:end])
    if(α < eps(T))
        #done in two stages since otherwise (1.-α) = -α for
        #large α, which makes z exactly 0.0 (or worse, -0.0 )
        z[1] -=  α
        z[1] +=  one(T)
    end

    return nothing
end

# unsymmetric initialization
function unsymmetric_init!(
   K::SecondOrderCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    s .= zero(T)
    z .= zero(T)
    add_scaled_e!(K,s,one(T))
    add_scaled_e!(K,z,one(T))

   return nothing
end

# implements y = αWx + βy for the socone
function gemv_W!(
    K::SecondOrderCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #NB: symmetric, so ignore transpose

  # use the fast product method from ECOS ECC paper
  @views ζ = dot(K.w[2:end],x[2:end])
  c = x[1] + ζ/(1+K.w[1])

  y[1] = α*K.η*(K.w[1]*x[1] + ζ) + β*y[1]

  for i = 2:length(y)
    y[i] = (α*K.η)*(x[i] + c*K.w[i]) + β*y[i]
  end

  return nothing
end

# implements y = αW^{-1}x + βy for the socone
function gemv_Winv!(
    K::SecondOrderCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

    #NB: symmetric, so ignore transpose

    # use the fast inverse product method from ECOS ECC paper
    @views ζ = dot(K.w[2:end],x[2:end])
    c = -x[1] + ζ/(1+K.w[1])

    y[1] = (α/K.η)*(K.w[1]*x[1] - ζ) + β*y[1]

    for i = 2:length(y)
        y[i] = (α/K.η)*(x[i] + c*K.w[i]) + β*y[i]
    end

    return nothing
end


# implements y = y + αe for the socone
function add_scaled_e!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},α::T
) where {T}

    #e is (1,0.0..0)
    x[1] += α

    return nothing
end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function combined_ds!(
    K::SecondOrderCone{T},
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
    K::SecondOrderCone{T},
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
    K::SecondOrderCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    gemv_W!(K,:N,lz,workz,one(T),zero(T))    #work = WΔz
    gemv_W!(K,:T,workz,ls,-one(T),zero(T))   #Δs = -WᵀWΔz

end

#return maximum allowable step length while remaining in the socone
function step_length(
    K::SecondOrderCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T}
) where {T}

    αz   = _step_length_soc_component(dz,z)
    αs   = _step_length_soc_component(ds,s)

    return (αz,αs)
end

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
function _step_length_soc_component(
    y::AbstractVector{T},
    x::AbstractVector{T}
) where {T}

    # assume that x is in the SOC, and
    # find the minimum positive root of
    # the quadratic equation:
    # ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = y[1]^2 - dot(y[2:end],y[2:end])
    @views b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    @views c = x[1]^2 - dot(x[2:end],x[2:end])  #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        throw(DomainError(c, "starting point of line search not in SOC"))
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return floatmax(T)

    else
        sqrtd = sqrt(d)
        r1 = (-b + sqrtd)/(2*a)
        r2 = (-b - sqrtd)/(2*a)
        #return the minimum positive root
        r1 = r1 < 0 ? floatmax(T) : r1
        r2 = r2 < 0 ? floatmax(T) : r2
        return prevfloat(min(r1,r2))
    end

end

function compute_centrality(
    K::SecondOrderCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
    
    barrier_s = s[1]^2 - dot(s[2:end],s[2:end])
    barrier_z = z[1]^2 - dot(z[2:end],z[2:end])

    # avoid numerical issue that barrier_s <= 0 or barrier_z <= 0
    if barrier_s > 0 && barrier_z > 0
        return (- log(barrier_s) - log(barrier_z))/2
    else
        return Inf
    end
end

# check neighbourhood
function _check_neighbourhood(
    K::SecondOrderCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    η::T
) where {T}
    # # NB: need to avoid redundant memory later
    # tmp = zeros(T,length(s))

    # circ_op!(K,tmp,s,z)
    # add_scaled_e!(K,tmp,-μ)

    # if norm(tmp) < η
    #     return true
    # end

    # return false

    return true

end
