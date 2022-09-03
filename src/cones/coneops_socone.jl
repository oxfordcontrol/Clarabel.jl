# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

#degree = 1 for SOC, since e'*e = 1
degree(K::SecondOrderCone{T}) where {T} = 1

# place vector into socone
function shift_to_cone!(
    K::SecondOrderCone{T},
    z::AbstractVector{T}
) where{T}

    z[1] = max(z[1],0)

    @views α = _soc_residual(z)
    if(α < sqrt(eps(T)))
        #done in two stages since otherwise (1.-α) = -α for
        #large α, which makes z exactly 0.0 (or worse, -0.0 )
        z[1] -=  α
        z[1] +=  one(T)
    end

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    K::SecondOrderCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
) where{T}
 
     z .= zero(T)
     s .= zero(T)
     add_scaled_e!(K,z,one(T))
     add_scaled_e!(K,s,one(T))
 
    return nothing
end 

# configure cone internals to provide W = I scaling
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

function update_scaling!(
    K::SecondOrderCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    #first calculate the scaled vector w
    @views zscale = sqrt(_soc_residual(z))
    @views sscale = sqrt(_soc_residual(s))
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
    mul_W!(K,:N,K.λ,z,one(T),zero(T))

    return nothing
end

function get_Hs!(
    K::SecondOrderCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal D block from the
    #sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The ConicVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries

    Hsblock    .= (K.η^2)
    Hsblock[1] *= K.d

    return nothing
end

# compute the product y = WᵀWx
function mul_Hs!(
    K::SecondOrderCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    mul_W!(K,:N,work,x,one(T),zero(T))    #work = Wx
    mul_W!(K,:T,y,work,one(T),zero(T))         #y = c Wᵀwork = W^TWx

end

# returns x = λ ∘ λ for the socone
function affine_ds!(
    K::SecondOrderCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    circ_op!(K,ds,K.λ,K.λ)

    return nothing
end

function combined_ds_shift!(
    K::SecondOrderCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    _combined_ds_shift_symmetric!(K,shift,step_z,step_s,σμ);
end

function Δs_from_Δz_offset!(
    K::SecondOrderCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    _Δs_from_Δz_offset_symmetric!(K,out,ds,work);
end

#return maximum allowable step length while remaining in the socone
function step_length(
    K::SecondOrderCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T
) where {T}

    αz   = _step_length_soc_component(z,dz,αmax)
    αs   = _step_length_soc_component(s,ds,αmax)

    return (αz,αs)
end

function compute_barrier(
    K::SecondOrderCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    res_s = _soc_residual_shifted(s,ds,α)
    res_z = _soc_residual_shifted(z,dz,α)

    # avoid numerical issue if res_s <= 0 or res_z <= 0
    if res_s > 0 && res_z > 0
        return -logsafe(res_s*res_z)/2
    else
        return Inf
    end
end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------

# implements y = y + αe for the socone
function add_scaled_e!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},α::T
) where {T}

    #e is (1,0.0..0)
    x[1] += α

    return nothing
end

# implements y = αWx + βy for the socone
function mul_W!(
    K::SecondOrderCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
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
function mul_Winv!(
    K::SecondOrderCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
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

# implements x = λ \ z for the socone, where λ
# is the internally maintained scaling variable.
function λ_inv_circ_op!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    inv_circ_op!(K, x, K.λ, z)

end

# ---------------------------------------------
# Jordan algebra operations for symmetric cones 
# ---------------------------------------------

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

# implements x = y \ z for the socone
function inv_circ_op!(
    K::SecondOrderCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    p = _soc_residual(y)
    pinv = 1/p
    @views v = dot(y[2:end],z[2:end])

    x[1]      = (y[1]*z[1] - v)*pinv
    @views x[2:end] .= pinv*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

    return nothing
end

# ---------------------------------------------
# internal operations for second order cones 
# ---------------------------------------------

@inline function _soc_residual(z:: AbstractVector{T}) where {T} 
    @views res = z[1]*z[1] - dot(z[2:end],z[2:end])
end 

#compute the residual at z + \alpha dz 
#without storing the intermediate vector
@inline function _soc_residual_shifted(
    z::AbstractVector{T}, 
    dz::AbstractVector{T}, 
    α::T
) where {T} 
    
    sc = z[1] + α * dz[1];
        
    @views res = sc * sc -  dot_shifted(z[2:end],z[2:end],dz[2:end],dz[2:end],α)


    return res
end 

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
function _step_length_soc_component(
    x::AbstractVector{T},
    y::AbstractVector{T},
    αmax::T
) where {T}

    # assume that x is in the SOC, and find the minimum positive root
    # of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = _soc_residual(y)
    @views b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    @views c = _soc_residual(x) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        throw(DomainError(c, "starting point of line search not in SOC"))
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return αmax

    elseif a == 0
        #edge case with only one root.  This corresponds to
        #the case where the search direction is exactly on the 
        #cone boundary.   The root should be -c/b, but b can't 
        #be negative since both (x,y) are in the cone and it is 
        #self dual, so <x,y> \ge 0 necessarily.
        return αmax

    elseif c == 0
        #Edge case with one of the roots at 0.   This corresponds 
        #to the case where the initial point is exactly on the 
        #cone boundary.  The other root is -b/a.   If the search 
        #direction is in the cone, then a >= 0 and b can't be 
        #negative due to self-duality.  If a < 0, then the 
        #direction is outside the cone and b can't be positive.
        #Either way, step length is determined by whether or not 
        #the search direction is in the cone.

        return (a >= 0 ? αmax : zero(T)) 
    end 


    # if we got this far then we need to calculate a pair 
    # of real roots and choose the smallest positive one.  
    # We need to be cautious about cancellations though.  
    # See §1.4: Goldberg, ACM Computing Surveys, 1991 
    # https://dl.acm.org/doi/pdf/10.1145/103162.103163

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    #return the minimum positive root, up to αmax
    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)

end