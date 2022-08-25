# ----------------------------------------------------
# Power Cone
# ----------------------------------------------------

# degree of the cone is always 3 for PowerCone
dim(K::PowerCone{T}) where {T} = 3
degree(K::PowerCone{T}) where {T} = dim(K)
numel(K::PowerCone{T}) where {T} = dim(K)

is_symmetric(::PowerCone{T}) where {T} = false

#Power cone returns a dense WtW block
function WtW_is_diagonal(
    K::PowerCone{T}
) where{T}
    return false
end

function update_scaling!(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}
    # update both gradient and Hessian for function f*(z) at the point z
    # NB: the update order can't be switched as we reuse memory in the Hessian computation
    # Hessian update
    update_grad_HBFGS(K,s,z,scaling_strategy)

    # K.z .= z
    @inbounds for i = 1:3
        K.z[i] = z[i]
    end
end

# return μH*(z) for power cone
function get_WtW_block!(
    K::PowerCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    # _pack_triu(WtWblock,K.μH)
    _pack_triu(WtWblock,K.HBFGS)

end

# return x = y for asymmetric cones
function affine_ds!(
    K::PowerCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:3
        x[i] = y[i]
    end

end

# unit initialization for asymmetric solves
function unit_initialization!(
   K::PowerCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    α = K.α

    s[1] = one(T)*sqrt(1+α)
    s[2] = one(T)*sqrt(1+(1-α))
    s[3] = zero(T)

    #@. z = s
    @inbounds for i = 1:3
        z[i] = s[i]
    end

   return nothing
end

# compute ds in the combined step where μH(z)Δz + Δs = - ds
function combined_ds!(
    K::PowerCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    η = K.grad_work      
    higher_correction!(K,η,step_s,step_z)             #3rd order correction requires input variables.z
    @inbounds for i = 1:3
        dz[i] = K.grad[i]*σμ - η[i]
    end

    return nothing
end

# compute the generalized step ds
function Wt_λ_inv_circ_ds!(
    K::PowerCone{T},
    lz::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
    Wtlinvds::AbstractVector{T}
) where {T}

    # @. Wtlinvds = rs    #Wᵀ(λ \ ds) <- ds
    @inbounds for i = 1:3
        Wtlinvds[i] = rs[i]
    end

    return nothing
end

# compute the generalized step of -μH(z)Δz
function WtW_Δz!(
    K::PowerCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # mul!(ls,K.HBFGS,lz,-one(T),zero(T))
    H = K.HBFGS
    @inbounds for i = 1:3
        ls[i] = - H[i,1]*lz[1] - H[i,2]*lz[2] - H[i,3]*lz[3]
    end

end

#return maximum allowable step length while remaining in the Power cone
function step_length(
    K::PowerCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    backtrack = settings.linesearch_backtrack_step

    αz = _step_length_powcone(K.vec_work, dz, z, αmax, K.α, backtrack, is_dual_feasible_powcone)
    αs = _step_length_powcone(K.vec_work, ds, s, αmax, K.α, backtrack, is_primal_feasible_powcone)

    return (αz,αs)
end





###############################################
# Basic operations for Power Cones
# Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# We use the dual barrier function: f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# Evaluates the gradient of the dual Power cone ∇f*(z) at z, and stores the result at g

function compute_centrality(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    α = K.α
    barrier = zero(T)

    # Dual barrier
    barrier += -logsafe((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*logsafe(z[1]) - α*logsafe(z[2])

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    g = K.vec_work
    gradient_primal(K,s,g)     #compute g(s)
    barrier += logsafe((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*logsafe(-g[1]) + α*logsafe(-g[2]) - 3

    return barrier
end

# Returns true if s is primal feasible
function is_primal_feasible_powcone(s::AbstractVector{T},α::T) where {T}

    if (s[1] > 0 && s[2] > 0)
        res = exp(2*α*logsafe(s[1]) + 2*(1-α)*logsafe(s[2])) - s[3]*s[3]
        if res > 0
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function is_dual_feasible_powcone(z::AbstractVector{T},α::T) where {T}

    if (z[1] > 0 && z[2] > 0)
        res = exp(2*α*logsafe(z[1]/α) + 2*(1-α)*logsafe(z[2]/(1-α))) - z[3]*z[3]
        if res > 0
            return true
        end
    end

    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function gradient_primal(
    K::PowerCone{T},
    s::AbstractVector{T},
    g::AbstractVector{T}
) where {T}

    α = K.α

    # unscaled ϕ
    ϕ = (s[1])^(2*α)*(s[2])^(2-2*α)

    # obtain g3 from the Newton-Raphson method
    abs_s = abs(s[3])
    if abs_s > eps(T)
        g[3] = newton_raphson(abs_s,ϕ,α)
        if s[3] < zero(T)
            g[3] = -g[3]
        end
        g[1] = -(α*g[3]*s[3] + 1 + α)/s[1]
        g[2] = -((1-α)*g[3]*s[3] + 2 - α)/s[2]
    else
        g[3] = zero(T)
        g[1] = -(1+α)/s[1]
        g[2] = -(2-α)/s[2]
    end

end

# Newton-Raphson method
# solve an one-dimensional equation f(x) = 0
# x(k+1) = x(k) - f(x(k))/f'(x(k))
# When we initialize x0 such that 0 < x0 < x*, the Newton-Raphson method converges quadratically
function newton_raphson(
    s3::T,
    ϕ::T,
    α::T
) where {T}
    # init point x0: since our dual barrier has an additional shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x),
    # the previous selection from Hypatia is still feasible, i.e. f(x0) > 0
    x = -one(T)/s3 + 2*(s3 + sqrt(4*ϕ*ϕ/s3/s3 + 3*ϕ))/(4*ϕ - s3*s3)

    t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)    # additional shift due to the choice of dual barrier
    t1 = x*x
    t2 = x*2/s3

    f0 = 2*α*logsafe(2*α*t1 + (1+α)*t2) + 2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - logsafe(ϕ) - logsafe(t1+t2) - 2*logsafe(t2) + t0
    f1 = 2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)

    xnew = x - f0/f1

    # terminate when abs(xnew - x) <= eps(T)
    while (xnew - x) > eps(T)
        # println("x is ",x)
        x = xnew

        t1 = x*x
        t2 = x*2/s3
        f0 = 2*α*logsafe(2*α*t1 + (1+α)*t2) + 2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - logsafe(ϕ) - logsafe(t1+t2) - 2*logsafe(t2) + t0
        f1 = 2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)
        xnew = x - f0/f1
    end

    return xnew
end

# 3rd-order correction at the point z, w.r.t. directions u,v, and then save it to η
function higher_correction!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs

    H = K.H
    u = K.vec_work
    z = K.z

    # lu factorization
    getrf!(H,K.ws)
    if K.ws.info[] == 0     # lu decomposition is successful
        # @. u = ds
        @inbounds for i = 1:3
            u[i] = ds[i]
        end
        getrs!(H,K.ws,u)    # solve H*u = ds
    else
        # @. η = zero(T)
        @inbounds for i = 1:3
            η[i] = zero(T)
        end
        return nothing
    end

    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # Reuse K.H memory for computation
    Hψ = K.H
    
    η[1] = 2*α*ϕ/z[1]
    η[2] = 2*(1-α)*ϕ/z[2]
    η[3] = -2*z[3]

    # 3rd order correction: η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
    # where: 
    # Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
    #         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
    #         0                       0                          -2;]
    Hψ[1,1] = 2*α*(2*α-1)*ϕ/(z[1]*z[1])
    Hψ[1,2] = 4*α*(1-α)*ϕ/(z[1]*z[2])
    Hψ[2,1] = Hψ[1,2]
    # Hψ[1,3] = 0
    # Hψ[3,1] = Hψ[1,3]
    Hψ[2,2] = 2*(1-α)*(1-2*α)*ϕ/(z[2]*z[2])
    # Hψ[2,3] = 0
    # Hψ[3,2] = Hψ[2,3]
    Hψ[3,3] = -2.

    dotψu = dot(η,u)
    dotψv = dot(η,v)

    Hψv = K.vec_work_2
    Hψv[1] = Hψ[1,1]*v[1]+Hψ[1,2]*v[2]
    Hψv[2] = Hψ[2,1]*v[1]+Hψ[2,2]*v[2]
    Hψv[3] = -2*v[3]

    coef = (dot(u,Hψv)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
    coef2 = 4*α*(2*α-1)*(1-α)*ϕ*(u[1]/z[1] - u[2]/z[2])*(v[1]/z[1] - v[2]/z[2])/ψ
    inv_ψ2 = 1/ψ/ψ

    η[1] = coef*η[1] - 2*(1-α)*u[1]*v[1]/(z[1]*z[1]*z[1]) + coef2/z[1] + Hψv[1]*dotψu*inv_ψ2
    η[2] = coef*η[2] - 2*α*u[2]*v[2]/(z[2]*z[2]*z[2]) - coef2/z[2] + Hψv[2]*dotψu*inv_ψ2
    η[3] = coef*η[3] + Hψv[3]*dotψu*inv_ψ2

    Hψu = K.vec_work_2
    Hψu[1] = Hψ[1,1]*u[1]+Hψ[1,2]*u[2]
    Hψu[2] = Hψ[2,1]*u[1]+Hψ[2,2]*u[2]
    Hψu[3] = -2*u[3]
    @. η = η + Hψu*dotψv*inv_ψ2

    @inbounds for i = 1:3
        η[i] /= 2
    end

end


######################################
# primal-dual scaling
######################################

# Implementation sketch
# 1) only need to replace μH by W⊤W,
#   where W⊤W is the primal-dual scaling matrix generated by BFGS, i.e. W⊤W*[z,̃z] = [s,̃s]
#   ̃z = -f'(s), ̃s = - f*'(z)

# YC: PowerCone utilizes the dual scaling strategy only

function update_grad_HBFGS(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    scaling_strategy::ScalingStrategy
) where {T}
    # reuse memory
    st = K.grad
    zt = K.vec_work
    δs = K.grad_work
    tmp = K.z
    H = K.H
    HBFGS = K.HBFGS

    # Hessian computation, compute μ locally
    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # compute the gradient at z
    st[1] = -2*α*ϕ/(z[1]*ψ) - (1-α)/z[1]
    st[2] = -2*(1-α)*ϕ/(z[2]*ψ) - α/z[2]
    st[3] = 2*z[3]/ψ


    # use workspace K.vec_work temporarily
    gψ = K.vec_work
    gψ[1] = 2*α*ϕ/(z[1]*ψ)
    gψ[2] = 2*(1-α)*ϕ/(z[2]*ψ)
    gψ[3] = -2*z[3]/ψ

    H[1,1] = gψ[1]*gψ[1] - 2*α*(2*α-1)*ϕ/(z[1]*z[1]*ψ) + (1-α)/(z[1]*z[1])
    H[1,2] = gψ[1]*gψ[2] - 4*α*(1-α)*ϕ/(z[1]*z[2]*ψ)
    H[2,1] = H[1,2]
    H[2,2] = gψ[2]*gψ[2] - 2*(1-α)*(1-2*α)*ϕ/(z[2]*z[2]*ψ) + α/(z[2]*z[2])
    H[1,3] = gψ[1]*gψ[3]
    H[3,1] = H[1,3]
    H[2,3] = gψ[2]*gψ[3]
    H[3,2] = H[2,3]
    H[3,3] = gψ[3]*gψ[3] + 2/ψ

    μ = dot(z,s)/3

    # HBFGS .= μ*H
    @inbounds for i = 1:3
        @inbounds for j = 1:3
            HBFGS[i,j] = μ*H[i,j]
        end
    end

    # compute zt,st,μt locally
    # YC: note the definitions of zt,st have a sign difference compared to the Mosek's paper

    # use the dual scaling
    if scaling_strategy == Dual
        return nothing
    end
    gradient_primal(K,s,zt)
 
    
    μt = dot(zt,st)/3

    # δs = s + μ*st
    # δz = z + μ*zt
    @inbounds for i = 1:3
        δs[i] = s[i] + μ*st[i]
    end

    de1 = μ*μt-1
    de2 = dot(zt,H,zt) - 3*μt*μt

    if (de1 > eps(T) && de2 > eps(T))
        # tmp = μt*st - H*zt
        @inbounds for i = 1:3
            tmp[i] = μt*st[i] - H[i,1]*zt[1] - H[i,2]*zt[2] - H[i,3]*zt[3]
        end

        # store (s - μ*st + δs/de1) into zt
        @inbounds for i = 1:3
            zt[i] = s[i] - μ*st[i] + δs[i]/de1
        end

        # Hessian HBFGS:= μ*H + 1/(2*μ*3)*δs*(s - μ*st + δs/de1)' + 1/(2*μ*3)*(s - μ*st + δs/de1)*δs' - μ/de2*tmp*tmp'
        coef1 = 1/(2*μ*3)
        coef2 = μ/de2
        # HBFGS .+= coef1*δs*zt' + coef1*zt*δs' - coef2*tmp*tmp'
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] += coef1*δs[i]*zt[j] + coef1*zt[i]*δs[j] - coef2*tmp[i]*tmp[j]
            end
        end
    else
        return nothing
    end
end

