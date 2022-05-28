# ----------------------------------------------------
# Power Cone
# ----------------------------------------------------

# degree of the cone
dim(K::PowerCone{T}) where {T} = K.dim
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
    μ::T
) where {T}
    #update both gradient and Hessian for function f*(z) at the point z
    muHessianF(K,z,K.μH,μ)
    GradF(K,z,K.grad)
    K.z .= z

    # # Hessian update
    # update_HBFGS(K,s,z,μ)
end

# return μH*(z) for power cone
function get_WtW_block!(
    K::PowerCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    _pack_triu(WtWblock,K.μH)
    # _pack_triu(WtWblock,K.HBFGS)

end

# return x = y for unsymmetric cones
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

#  unsymmetric initialization
function unsymmetric_init!(
   K::PowerCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    α = K.α

    s[1] = one(T)*sqrt(1+α)
    s[2] = one(T)*sqrt(1+(1-α))
    s[3] = zero(T)

    @. z = s

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
    # η = similar(dz)
    # higherCorrection!(K,η,step_s,step_z)             #3rd order correction requires input variables.z
    # @. dz = η + σμ*K.grad

    # @. dz = σμ*K.grad                   #dz <- σμ*g(z)
    @inbounds for i = 1:3
        dz[i] = σμ*K.grad[i]                
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

    mul!(ls,K.μH,lz,-one(T),zero(T))

end

#return maximum allowable step length while remaining in the Power cone
function unsymmetric_step_length(
    K::PowerCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::T,
     scaling::T
) where {T}

    if isnan(α)
        error("numerical error")
    end

    # avoid abuse of α
    αExp = K.α

    αz = _step_length_power_dual(K.vecWork,dz,z,α,scaling,αExp)
    αs = _step_length_power_primal(K.vecWork,ds,s,α,scaling,αExp)

    return min(αz,αs)
end


# find the maximum step length α≥0 so that
# s + α*ds stays in the Power cone
function _step_length_power_primal(
    ws::AbstractVector{T},
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α::T,
    scaling::T,
    αExp::T
) where {T}

    # @. ws = s + α*ds
    @inbounds for i = 1:3
        ws[i] = s[i] + α*ds[i]                
    end

    while !checkPowerPrimalFeas(ws,αExp)
        α *= scaling    #backtrack line search
        # @. ws = s + α*ds
        @inbounds for i = 1:3
            ws[i] = s[i] + α*ds[i]                
        end
    end

    return α
end
# z + α*dz stays in the dual Power cone
function _step_length_power_dual(
    ws::AbstractVector{T},
    dz::AbstractVector{T},
    z::AbstractVector{T},
    α::T,
    scaling::T,
    αExp::T
) where {T}

    # @. ws = z + α*dz
    @inbounds for i = 1:3
        ws[i] = z[i] + α*dz[i]                
    end

    while !checkPowerDualFeas(ws,αExp)
        α *= scaling    #backtrack line search
        # @. ws = z + α*dz
        @inbounds for i = 1:3
            ws[i] = z[i] + α*dz[i]                
        end
    end

    return α
end



###############################################
# Basic operations for Power Cones
# Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# We use the dual barrier function: f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# Evaluates the gradient of the dual Power cone ∇f*(z) at z, and stores the result at g
function GradF(
    K::PowerCone{T},
    z::AbstractVector{T},
    g::AbstractVector{T}
) where {T}

    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    g[1] = -2*α*ϕ/(z[1]*ψ) - (1-α)/z[1]
    g[2] = -2*(1-α)*ϕ/(z[2]*ψ) - α/z[2]
    g[3] = 2*z[3]/ψ

end

# Evaluates the Hessian of the Power dual cone barrier at z and stores the upper triangular part of the matrix μH*(z)
# NB:could reduce H to an upper triangular matrix later, remove duplicate updates
function muHessianF(
    K::PowerCone{T},
    z::AbstractVector{T},
    H::AbstractMatrix{T},
    μ::T
) where {T}

    α = K.α
    gψ = K.vecWork

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]
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

    H .*= μ
end

function f_sum(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    α = K.α
    barrier = T(0)

    # Dual barrier
    barrier += -log((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*log(z[1]) - α*log(z[2])

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -3 = - ν 

    g = K.vecWork
    GradPrim(K,s,g)     #compute g(s)
    barrier += log((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*log(-g[1]) + α*log(-g[2]) - 3

    return barrier
end

# Returns true if s is primal feasible
function checkPowerPrimalFeas(s::AbstractVector{T},α::T) where {T}
    s1 = s[1]
    s2 = s[2]
    s3 = s[3]

    if (s1 > 0 && s2 > 0)
        res = exp(2*α*log(s1) + 2*(1-α)*log(s2)) - s3*s3
        if res > 0
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function checkPowerDualFeas(z::AbstractVector{T},α::T) where {T}

    if (z[1] > 0 && z[2] > 0)
        res = exp(2*α*log(z[1]/α) + 2*(1-α)*log(z[2]/(1-α))) - z[3]*z[3]
        if res > 0
            return true
        end
    end

    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function GradPrim(
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
        g[3] = NewtonRaphson(abs_s,ϕ,α)    
        if s[3] < 0
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
function NewtonRaphson(
    s3::T,
    ϕ::T,
    α::T
) where {T}
    # init point x0: since our dual barrier has an additional shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x), 
    # the previous selection from Hypatia is still feasible, i.e. f(x0) > 0 
    x = -one(T)/s3 + 2*(s3 + sqrt(4*ϕ*ϕ/s3/s3 + 3*ϕ))/(4*ϕ - s3*s3)

    t0 = - 2*α*log(α) - 2*(1-α)*log(1-α)    # additional shift due to the choice of dual barrier
    t1 = x*x
    t2 = x*2/s3

    f0 = 2*α*log(2*α*t1 + (1+α)*t2) + 2*(1-α)*log(2*(1-α)*t1 + (2-α)*t2) - log(ϕ) - log(t1+t2) - 2*log(t2) + t0  
    f1 = 2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)

    xnew = x - f0/f1 

    # terminate when abs(xnew - x) <= eps(T)
    while (xnew - x) > eps(T)
        # println("x is ",x)
        x = xnew

        t1 = x*x
        t2 = x*2/s3
        f0 = 2*α*log(2*α*t1 + (1+α)*t2) + 2*(1-α)*log(2*(1-α)*t1 + (2-α)*t2) - log(ϕ) - log(t1+t2) - 2*log(t2) + t0  
        f1 = 2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)
        xnew = x - f0/f1 
    end

    return xnew
end


######################################
# May need to be removed later
######################################

# 3rd-order correction at the point z, w.r.t. directions u,v, and then save it to η
# NB: not finished yet
function higherCorrection!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs
    #NB: need to be refined later
    μH = K.μHWork
    u = K.vecWork
    F = K.FWork
    z = K.z

    # recompute Hessian
    muHessianF(K,z,μH, one(T))

    if F === nothing
        F = lu(μH, check= false)
    else
        F = lu!(μH, check= false)
    end

    if !issuccess(F)
        increase_diag!(μH)
        F = lu!(μH)
    end

    ldiv!(u,F,ds)    #equivalent to Hinv*ds

    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # memory allocation
    gψ = K.gradWork
    Hψ = K.μHWork

    gψ[1] = 2*α*ϕ/z[1]
    gψ[2] = 2*(1-α)*ϕ/z[2]
    gψ[3] = -2*z[3]

    # Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
    #         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
    #         0                       0                          -2;]
    Hψ[1,1] = 2*α*(2*α-1)*ϕ/(z[1]*z[1])
    Hψ[1,2] = 4*α*(1-α)*ϕ/(z[1]*z[2])
    Hψ[2,1] = Hψ[1,2]
    Hψ[1,3] = 0
    Hψ[3,1] = Hψ[1,3]
    Hψ[2,2] = 2*(1-α)*(1-2*α)*ϕ/(z[2]*z[2])
    Hψ[2,3] = 0
    Hψ[3,2] = Hψ[2,3]
    Hψ[3,3] = -2

    dotψu = dot(gψ,u)
    dotψv = dot(gψ,v)

    dotψuv = 4*α*(2*α-1)*(1-α)*ϕ*[-u[1]*v[1]/(z[1]*z[1]*z[1]) + (u[2]*v[1]+u[1]*v[2])/(z[1]*z[1]*z[2]) - u[2]*v[2]/(z[1]*z[2]*z[2]); u[1]*v[1]/(z[1]*z[1]*z[2]) - (u[2]*v[1]+u[1]*v[2])/(z[1]*z[2]*z[2]) + u[2]*v[2]/(z[2]*z[2]*z[2]); 0]
    dothuv = [-2*(1-α)*u[1]*v[1]/(z[1]*z[1]*z[1]); -2*α*u[2]*v[2]/(z[2]*z[2]*z[2]); 0]
    Hψv = Hψ*v
    Hψu = Hψ*u

    η .= (dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv
    η ./= -2

end

# check neighbourhood
function _check_neighbourhood(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    η::T
) where {T}

    grad = K.gradWork
    μH = K.μHWork
    F = K.FWork
    tmp = K.vecWork

    # compute gradient and Hessian at z
    GradF(K, z, grad)
    muHessianF(K, z, μH, μ)

    if F === nothing
        F = lu(μH, check= false)
    else
        F = lu!(μH, check= false)
    end

    if !issuccess(F)
        increase_diag!(μH)
        lu!(F,μH)
    end

    # grad as a workspace for s + μ*grad
    axpby!(one(T), s, μ, grad)

    ldiv!(tmp,F,grad)
    if (dot(tmp,grad)/μ < η)
        return true
    end

    return false
end


######################################
# primal-dual scaling 
######################################

# Implementation sketch
# 1) only need to replace μH by W⊤W, 
#   where W⊤W is the primal-dual scaling matrix generated by BFGS, i.e. W⊤W*[z,̃z] = [s,̃s]
#   ̃z = -f'(s), ̃s = - f*'(z)

# NB: better to create two matrix spaces, one for the Hessian at z, H*(z), and another one for BFGS (primal-dual scaling) matrix, H-BFGS(z,s)


function update_HBFGS(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T
) where {T}

    st = rand(T,3)
    zt = rand(T,3)
    GradPrim(K,s,zt)
    zt .*= -1
    GradF(K,z,st)
    st .*= -1

    H = K.H
    HBFGS = K.HBFGS

    # should compute μ, μt globally
    μt = dot(zt,st)/3

    muHessianF(K,z,H,one(T))

    δs = s - μ*st
    δz = z - μ*zt

    tmp = H*zt - μt*st
    de1 = μ*μt-1
    de2 = dot(zt,H,zt) - 3*μt*μt
    if (de1 > eps(T) && de2 > eps(T))
        HBFGS .= μ*H + 1/(2*μ*3)*δs*(s + μ*st + δs/de1)' + 1/(2*μ*3)*(s + μ*st + δs/de1)*δs' - μ/de2*tmp*tmp'
    else
        HBFGS .= μ*H
    end

end

# compute shadow iterates for centrality check
function shadow_iterates!(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    st::AbstractVector{T},
    zt::AbstractVector{T},
) where {T}

    GradPrim(K,s,zt)
    zt .*= -1
    GradF(K,z,st)
    st .*= -1
    
end
