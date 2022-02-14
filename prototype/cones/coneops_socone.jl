# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

#degree = 1 for SOC, since e'*e = 1
degree(K::SecondOrderCone{T}) where {T} = 1

function update_scaling!(
    K::SecondOrderCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}
) where {T}

    #first calculate the scaled vector w
    zscale = sqrt(z[1]^2 - dot(z[2:end],z[2:end]))
    sscale = sqrt(s[1]^2 - dot(s[2:end],s[2:end]))
    gamma  = sqrt((1 + dot(s,z)/(zscale*sscale) ) / 2)

    K.w         .= s./(2*sscale*gamma)
    K.w[1]      += z[1]/(2*zscale*gamma)
    K.w[2:end] .-= z[2:end]/(2*zscale*gamma)

    #various intermediate calcs for u,v,d,η
    w0p1 = K.w[1] + 1
    w1sq = dot(K.w[2:end],K.w[2:end])
    w0sq = K.w[1]*K.w[1]
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
    v0 = 0
    v1 = sqrt(u1*u1 - β)
    K.u[1] = u0
    K.u[2:end] .= u1.*K.w[2:end]
    K.v[1] = 0.0
    K.v[2:end] .= v1.*K.w[2:end]

    #λ = Wz
    gemv_W!(K,false,z,λ,1.,0.0)

    return nothing
end

#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::SecondOrderCone{T}
) where {T}

    K.d  = 1.0
    K.u .= 0.0
    K.v .= 0.0
    K.η  = 1.0
    K.w[1]      = 0.0
    K.w[2:end] .= 0.0

    return nothing
end

function get_diagonal_scaling!(
    K::SecondOrderCone{T},
    diagW2::VectorView{T}
) where {T}

    #NB: we are returning here the D block from the
    #sparse representation of -W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The SplitVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries

    diagW2    .= -(K.η^2)
    diagW2[1] *= K.d

    return nothing
end


# implements x = y ∘ z for the socone
function circle_op!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
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
function inv_circle_op!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    p    = (y[1]^2 - dot(y[2:end],y[2:end]))
    pinv = 1/p
    v = dot(y[2:end],z[2:end])

    x[1]      = (y[1]*z[1] - v)*pinv
    x[2:end] .= pinv*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

    return nothing
end

# place vector into socone
function shift_to_cone!(
    K::SecondOrderCone{T},
    z::VectorView{T}
) where{T}

    z[1] = max(z[1],0)

    α = z[1] - norm(z[2:end])
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0.0 (or worse, -0.0 )
        z[1] += -α
        z[1] +=  1
    end

    return nothing
end

# implements y = αWx + βy for the socone
function gemv_W!(
    K::SecondOrderCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

  # use the fast product method from ECOS ECC paper
  ζ = dot(K.w[2:end],x[2:end])
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
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

    # use the fast inverse product method from ECOS ECC paper
    ζ = dot(K.w[2:end],x[2:end])
    c = -x[1] + ζ/(1+K.w[1])

    y[1] = (α/K.η)*(K.w[1]*x[1] - ζ) + β*y[1]

    for i = 2:length(y)
        y[i] = (α/K.η)*(x[i] + c*K.w[i]) + β*y[i]
    end

    return nothing
end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    #PJG: W is symmetric, so just multiply
    #by the inverse twice.  Could be made
    #faster if needed
    gemv_Winv!(K,true,y,y,1.0,0.0)
    gemv_Winv!(K,true,x,y,1.0,0.0)

    return nothing
end

# implements y = W^TWx
function mul_WtW!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    #PJG: W is symmetric, so just multiply
    #by W twice.  Could be made
    #faster if needed
    gemv_W!(K,true,y,y,1.0,0.0)
    gemv_W!(K,true,x,y,1.0,0.0)

    return nothing
end

# implements y = y + αe for the socone
function add_scaled_e!(
    K::SecondOrderCone,
    x::VectorView{T},α::T
) where {T}

    #e is (1,0.0..0)
    x[1] += α

    return nothing
end


#return maximum allowable step length while remaining in the socone
function step_length(
    K::SecondOrderCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}
) where {T}

    αz   = _step_length_soc_component(K,dz,z)
    αs   = _step_length_soc_component(K,ds,s)
    α    = min(αz,αs)

    return α
end

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
function _step_length_soc_component(
    K::SecondOrderCone{T},
    y::VectorView{T},
    x::VectorView{T}
) where {T}

    # assume that x is in the SOC, and
    # find the minimum positive root of
    # the quadratic equation:
    # ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    a = y[1]^2 - dot(y[2:end],y[2:end])
    b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    c = x[1]^2 - dot(x[2:end],x[2:end])  #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        throw(DomainError(c, "starting point of line search not in SOC"))
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return 1/eps(T)

    else
        sqrtd = sqrt(d)
        r1 = (-b + sqrtd)/(2*a)
        r2 = (-b - sqrtd)/(2*a)
        #return the minimum positive root
        r1 = r1 < 0 ? 1/eps(T) : r1
        r2 = r2 < 0 ? 1/eps(T) : r2
        return min(r1,r2)
    end

end
