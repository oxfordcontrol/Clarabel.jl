# -----------------------------------------------------
# operators on multiple cones
# -----------------------------------------------------

# Order of the cone is the same as dimension
# by default.   Order will be defined differently
# for the zero cone though (order=0 in that case)
dim(K::AbstractCone{T}) where {T} = K.dim
order(K::AbstractCone{T}) where {T} = K.dim


# The diagonal part of the KKT scaling
# matrix for each cone
function set_scaling_diagonal!(
    scalings::DefaultConeScalings,
    diagW2::SplitVector{T}) where {T}

    foreach(set_scaling_diagonal!,scalings.cones,diagW2.views)

end

# x = y ∘ z
function circle_op!(
    scalings::DefaultConeScalings,
    x::SplitVector{T},
    y::SplitVector{T},
    z::SplitVector{T}) where {T}

    foreach(circle_op!,scalings.cones,x.views,y.views,z.views)

end

# x = y \ z
function inv_circle_op!(
    scalings::DefaultConeScalings,
    x::SplitVector{T},
    y::SplitVector{T},
    z::SplitVector{T}) where {T}

    foreach(inv_circle_op!,scalings.cones,x.views,y.views,z.views)

end

# place a vector to some nearby point in the cone
function shift_to_cone!(
    scalings::DefaultConeScalings,
    z::SplitVector{T}) where {T}

    foreach(shift_to_cone!,scalings.cones,z.views)

end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface
function gemv_W!(
    scalings::DefaultConeScalings,
    is_transpose::Bool,
    x::SplitVector{T},
    y::SplitVector{T},
    α::T,
    β::T) where {T}

    cones = scalings.cones
    xv    = x.views
    yv    = y.views

    foreach((c,x,y)->gemv_W!(c,is_transpose,x,y,α,β),cones,xv,yv)
end

# computes y = αW^{-1}x + βy, or y = αW⁻ᵀx + βy, i.e.
# similar to the BLAS gemv interface
function gemv_Winv!(
    scalings::DefaultConeScalings,
    is_transpose::Bool,
    x::SplitVector{T},
    y::SplitVector{T},
    α::T,
    β::T) where {T}

    cones = scalings.cones
    xv    = x.views
    yv    = y.views

    foreach((c,x,y)->gemv_Winv!(c,is_transpose,x,y,α,β),cones,xv,yv)
end

#computes y = y + αe
function add_scaled_e!(
    scalings::DefaultConeScalings,
    x::SplitVector{T},
    α::T) where {T}

    foreach((c,x)->add_scaled_e!(c,x,α),scalings.cones,x.views)

end

# maximum allowed step length over all cones
function step_length(
    scalings::DefaultConeScalings,
    dz::SplitVector{T},
    ds::SplitVector{T},
     z::SplitVector{T},
     s::SplitVector{T},
     λ::SplitVector{T}) where {T}

    cones = scalings.cones
    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views
    λ     = λ.views

    minimum(map(step_length,cones,dz,ds,z,s,λ))

end



# -------------------------------------
# Zero Cone
# -------------------------------------

order(K::ZeroCone{T}) where {T} = 0

function UpdateScaling!(
    K::ZeroCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}) where {T}

    λ   .= 0

end

function IdentityScaling!(
    K::ZeroCone{T}) where {T}

    #do nothing.   "Identity" scaling will be zero for equalities

end


# implements x = y ∘ z for the zero cone
function circle_op!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    x .= 0

end

# implements x = y \ z for the zero cone
function inv_circle_op!(
    K::ZeroCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    x .= 0

end

# place vector into zero cone
function shift_to_cone!(
    K::ZeroCone{T},z::VectorView{T}) where{T}

    z .= 0

end

# implements y = αWx + βy for the zero cone
function gemv_W!(
    K::ZeroCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}


    #treat W like zero
    y .= β.*y

end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(
    K::ZeroCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}

  #treat Winv like zero
  y .= β.*y

end

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::ZeroCone{T},
    x::VectorView{T},α::T) where {T}

    #do nothing

end

function set_scaling_diagonal!(
    K::ZeroCone{T},
    diagW2::VectorView{T}) where {T}

    diagW2 .= 0.
end

function step_length(
     K::ZeroCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}) where {T}

    #equality constraints allow arbitrary step length
    return 1/eps(T)

end



## ------------------------------------
# Nonnegative Cone
# -------------------------------------

function UpdateScaling!(
    K::NonnegativeCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}) where {T}

    λ   .= sqrt.(s.*z)
    K.w .= sqrt.(s./z)

end

function IdentityScaling!(
    K::NonnegativeCone{T}) where {T}

    K.w .= 1

end


# implements x = y ∘ z for the nn cone
function circle_op!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    x .= y.*z
    
end

# implements x = y \ z for the nn cone
function inv_circle_op!(
    K::NonnegativeCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    x .= z./y

end

# place vector into nn cone
function shift_to_cone!(
    K::NonnegativeCone{T},
    z::VectorView{T}) where{T}

    α = minimum(z)
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        z .+= -α
        z .+=  1
    end

end

# implements y = αWx + βy for the nn cone
function gemv_W!(
    K::NonnegativeCone,
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}

  #W is diagonal, so ignore transposition
  y .= α.*(K.w.*x) + β.*y

end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(
    K::NonnegativeCone,
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}

  #W is diagonal, so ignore transposition
  y .= α.*(x./K.w) + β.*y

end

# implements y = y + αe for the nn cone
function add_scaled_e!(
    K::NonnegativeCone,
    x::VectorView{T},α::T) where {T}

    #e is a vector of ones, so just shift
    x .+= α

end

function set_scaling_diagonal!(
    K::NonnegativeCone{T},
    diagW2::VectorView{T}) where {T}

    diagW2 .= -K.w.^2

end

function step_length(
    K::NonnegativeCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}) where {T}

    f    = (dv,v)->(dv<0 ? -v/dv : 1/eps(T))
    αz   = minimum(map(f, dz, z))
    αs   = minimum(map(f, ds, s))
    α    = min(αz,αs)

end

# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

#PJG: AD thesis p17 : should this be one?  Or maybe 2 according to Sturm?
order(K::SecondOrderCone{T}) where {T} = K.dim

function UpdateScaling!(
    K::SecondOrderCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}) where {T}

    #first calculate the scaled vector w
    zscale = sqrt(z[1]^2 - dot(z[2:end],z[2:end]))
    sscale = sqrt(s[1]^2 - dot(s[2:end],s[2:end]))
    gamma  = sqrt((1 + dot(s,z)/zscale/sscale) / 2)

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
    K.v[1] = 0.
    K.v[2:end] .= v1.*K.w[2:end]

    #λ = Wz
    gemv_W!(K,false,z,λ,1.,0.)

end

function IdentityScaling!(
    K::SecondOrderCone{T}) where {T}

    K.d  = 1.
    K.u .= 0.
    K.v .= 0.
    K.η  = 1.
    K.w[1]      = 0.
    K.w[2:end] .= 0.
end


# implements x = y ∘ z for the socone
function circle_op!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    x[1] = dot(y,z)
    y0   = y[1]
    z0   = z[1]
    for i = 2:length(x)
        x[i] = y0*z[i] + z0*y[i]
    end

end

# implements x = y \ z for the socone
function inv_circle_op!(
    K::SecondOrderCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}) where {T}

    p = (y[1]^2 - dot(y[2:end],y[2:end]))
    v = dot(y[2:end],z[2:end])

    x[1]      =  (y[1]*z[1] - v)/p
    x[2:end] .=  ((v/y[1] - z[1])/p).*y[2:end] + (1/y[1]).*z[2:end]

end

# place vector into socone
function shift_to_cone!(
    K::SecondOrderCone{T},
    z::VectorView{T}) where{T}

    z[1] = max(z[1],0)

    α = z[1] - norm(z[2:end])
    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        z[1] += -α
        z[1] +=  1
    end

end

# implements y = αWx + βy for the socone
function gemv_W!(
    K::SecondOrderCone,
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}

  # use the fast product method from ECOS ECC paper
  ζ = dot(K.w[2:end],x[2:end])
  c = x[1] + ζ/(1+K.w[1])

  y[1] = α*K.η*(K.w[1]*x[1] + ζ) + β*y[1]

  for i = 2:length(y)
    y[i] = (α*K.η)*(x[i] + c*K.w[i]) + β*y[i]
  end

end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(
    K::SecondOrderCone,
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T) where {T}


    # use the fast inverse product method from ECOS ECC paper
    ζ = dot(K.w[2:end],x[2:end])
    c = -x[1] + ζ/(1+K.w[1])

    y[1] = (α/K.η)*(K.w[1]*x[1] - ζ) + β*y[1]

    for i = 2:length(y)
        y[i] = (α/K.η)*(x[i] + c*K.w[i]) + β*y[i]
    end

end

# implements y = y + αe for the socone
function add_scaled_e!(
    K::SecondOrderCone,
    x::VectorView{T},α::T) where {T}

    #e is (1,0...0)
    x[1] += α

end

function set_scaling_diagonal!(
    K::SecondOrderCone{T},
    diagW2::VectorView{T}) where {T}

    diagW2    .= -(K.η^2)
    diagW2[1] *= K.d

end

function step_length(
    K::SecondOrderCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}) where {T}

    αz   = step_length_soc_component(K,dz,z)
    αs   = step_length_soc_component(K,ds,s)
    α    = min(αz,αs)
    return α

end

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
function step_length_soc_component(
    K::SecondOrderCone{T},
    y::VectorView{T},
    x::VectorView{T}) where {T}

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
