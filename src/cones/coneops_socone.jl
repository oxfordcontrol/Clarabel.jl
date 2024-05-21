# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

#degree = 1 for SOC, since e'*e = 1
degree(K::SecondOrderCone{T}) where {T} = 1
numel(K::SecondOrderCone{T}) where {T} = K.dim

function is_sparse_expandable(K::SecondOrderCone{T}) where{T}
    return !isnothing(K.sparse_data)
end

function margins(
    K::SecondOrderCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone
) where{T}

    @views α = z[1] - norm(z[2:end])
    β = max(zero(T),α)
    (α,β)
end

# place vector into socone
function scaled_unit_shift!(
    K::SecondOrderCone{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone
) where{T}

    z[1] += α

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    K::SecondOrderCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
) where{T}
 
    s .= zero(T)
    z .= zero(T)

    #Primal or Dual doesn't matter here
    #since the cone is self dual anyway
    scaled_unit_shift!(K,s,one(T),PrimalCone)
    scaled_unit_shift!(K,z,one(T),DualCone)

 
    return nothing
end 

# configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::SecondOrderCone{T}
) where {T}

    K.w .= zero(T)
    K.w[1] = one(T)
    K.η  = one(T)

    if !isnothing(K.sparse_data)
        K.sparse_data.d  = T(0.5)
        K.sparse_data.u .= zero(T)
        K.sparse_data.u[1] = sqrt(T(0.5))
        K.sparse_data.v .= zero(T)
    end 

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
    zscale = _sqrt_soc_residual(z)
    sscale = _sqrt_soc_residual(s)

    # Fail if either s or z is not an interior point
    if iszero(zscale) || iszero(sscale)
        return is_scaling_success = false
    end

    #the leading scalar term for W^TW
    K.η = sqrt(sscale/zscale)

    # construct w and normalize
    w = K.w
    w     .= s./(sscale)
    w[1]  += z[1]/(zscale)

    @inbounds for i in 2:length(w)
        w[i] -= z[i]/(zscale)
    end

    wscale = _sqrt_soc_residual(w)
    # Fail if w is not an interior point
    if iszero(wscale)
        return is_scaling_success = false
    end
    w ./= wscale

    #try to force badly scaled w to come out normalized
    @views w1sq = sumsq(w[2:end])
    w[1] = sqrt(1 + w1sq)

    #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
    γ = 0.5 * wscale
    K.λ[1] = γ 
    s1 = @view s[2:end];
    z1 = @view z[2:end];
    λ1 = @view K.λ[2:end];
    λ1 .= ((γ + z[1]/zscale)/sscale).*s1 +((γ + s[1]/sscale)/zscale).*z1 
    K.λ[2:end] *= inv(s[1]/sscale + z[1]/zscale + 2*γ)
    K.λ .*= sqrt(sscale*zscale)

    #Populate sparse expansion terms if allocated
    if is_sparse_expandable(K)

        sparse_data = K.sparse_data

        #various intermediate calcs for u,v,d
        α  = 2*w[1]

        #Scalar d is the upper LH corner of the diagonal
        #term in the rank-2 update form of W^TW
        wsq    = w[1]*w[1] + w1sq
        wsqinv = 1/wsq
        sparse_data.d    = wsqinv / 2

        #the vectors for the rank two update
        #representation of W^TW
        u0  = sqrt(wsq - sparse_data.d)
        u1 = α/u0
        v0 = zero(T)
        v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
        
        sparse_data.u[1] = u0
        @views K.sparse_data.u[2:end] .= u1.*w[2:end]
        sparse_data.v[1] = v0
        @views sparse_data.v[2:end] .= v1.*w[2:end]

    end 

    return is_scaling_success = true
end

function get_Hs!(
    K::SecondOrderCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    if is_sparse_expandable(K)
        #For sparse form, we are returning here the diagonal D block 
        #from the sparse representation of W^TW, but not the
        #extra two entries at the bottom right of the block.
        Hsblock    .= K.η^2
        Hsblock[1] *= K.sparse_data.d
    
    else 
        #for dense form, we return H = \eta^2 (2*ww^T - J), where 
        #J = diag(1,-I).  We are packing into dense triu form

        Hsblock[1] = 2*K.w[1]^2 - one(T)
        hidx = 2

        @inbounds for col in 2:K.dim
            wcol = K.w[col]
            @inbounds for row in 1:col 
                Hsblock[hidx] = 2*K.w[row]*wcol
                hidx += 1
            end 
            #go back to add the offset term from J 
            Hsblock[hidx-1] += one(T)
        end
        Hsblock .*= K.η^2
    end

    return nothing

    
end

function Hs_is_diagonal(
    K::SecondOrderCone{T}
) where{T}
    return is_sparse_expandable(K)
end

# compute the product y = WᵀWx
function mul_Hs!(
    K::SecondOrderCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    # y = = H^{-1}x = W^TWx
    # where H^{-1} = \eta^{-2} (2*ww^T - J)
    c = 2*dot(K.w,x)
    y .= x
    y[1] = -x[1]
    y .+= c*K.w
    y .*= K.η^2  
    return nothing
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
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
   
    #out = Wᵀ(λ \ ds).  Below is equivalent,
    #but appears to be a little more stable 
    
    resz = _soc_residual(z)

    @views λ1ds1  = dot(K.λ[2:end],ds[2:end])
    @views w1ds1  = dot(K.w[2:end],ds[2:end])

    out .= -z
    out[1] = +z[1]

    c = K.λ[1]*ds[1] - λ1ds1
    out .*= c/resz

    out[1]              += K.η*w1ds1
    @views out[2:end]  .+= K.η*(ds[2:end] + w1ds1/(1+K.w[1]).*K.w[2:end])

    out .*= (1/K.λ[1])

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

  @inbounds for i in 2:length(y)
      y[i] = α*K.η*(x[i] + c*K.w[i]) + β*y[i]
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

    @inbounds for i = 2:length(y)
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
    @views x[2:end] .= pinv.*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

    return nothing
end

# ---------------------------------------------
# internal operations for second order cones 
# ---------------------------------------------

@inline function _soc_residual(z:: AbstractVector{T}) where {T} 

    @views z[1]*z[1] - sumsq(z[2:end])
end 

@inline function _sqrt_soc_residual(z:: AbstractVector{T}) where {T} 
    res = _soc_residual(z)
    # set res to 0 when z is not an interior point
    res = res > 0.0 ? sqrt(res) : zero(T)
end 

#compute the residual at z + \alpha dz 
#without storing the intermediate vector
@inline function _soc_residual_shifted(
    z::AbstractVector{T}, 
    dz::AbstractVector{T}, 
    α::T
) where {T} 
    
    x0 = z[1] + α * dz[1];
    @views x1_sq = dot_shifted(z[2:end],z[2:end],dz[2:end],dz[2:end],α)
    res = x0*x0 - x1_sq
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

    @views a = _soc_residual(y) #NB: could be negative
    @views b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    @views c = max(zero(T),_soc_residual(x)) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        # This should never be reachable since c ≥ 0 above
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