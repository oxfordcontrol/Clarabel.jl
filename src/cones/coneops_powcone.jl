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
end

# return μH*(z) for exponetial cone
function get_WtW_block!(
    K::PowerCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    WtWblock .= _pack_triu(WtWblock,K.μH)

end

# return x = y for unsymmetric cones
function affine_ds!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    @. x = y

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

# add gradient to x
function add_grad!(
    K::PowerCone{T},
    x::AbstractVector{T},
    α::T
) where {T}

    #e is a vector of ones, so just shift
    @. x += α*K.grad

    return nothing
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
    exp = K.α
    
    αz = _step_length_power_dual(K.vecWork,dz,z,α,scaling,exp)
    αs = _step_length_power_primal(K.vecWork,ds,s,α,scaling,exp)

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
    exp::T
) where {T}

    # NB: additional memory, may need to remove it later
    @. ws = s + α*ds

    while !checkPowerPrimalFeas(ws,exp)
        α *= scaling    #backtrack line search
        @. ws = s + α*ds
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
    exp::T
) where {T}

    # NB: additional memory, may need to remove it later
    @. ws = z + α*dz

    while !checkPowerDualFeas(ws,exp)
        α *= scaling    #backtrack line search
        @. ws = z + α*dz
    end

    return α
end



###############################################
# Basic operations for Power Cones
# Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# We use the dual barrier function: f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - log(z1) - log(z2):
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

# 3rd-order correction at the point z, w.r.t. directions u,v, and then save it to η
# NB: not finished yet
function higherCorrection!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T}, 
    v::AbstractVector{T},
    z::AbstractVector{T},
    μ::T
) where {T}

    # u for H^{-1}*Δs 
    #NB: need to be refined later
    μH = K.μHWork
    u = K.vecWork
    F = K.FWork

    # recompute Hessian
    muHessianF(K,z,μH, μ)

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
    @. u *= μ

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

    return η
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





# Hessian operator δ = μH*(z)[δz], where v contains μH*.
function compute_muHessianF(δ::AbstractVector{T}, H::AbstractMatrix{T}, δz::AbstractVector{T}) where {T}
    mul!(δ,H,δz)
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
    z1 = z[1]          
    z2 = z[2]    
    z3 = z[3]

    if (z1 > 0 && z2 > 0)
        res = exp(2*α*log(z1/α) + 2*(1-α)*log(z2/(1-α))) - z3*z3
        if res > 0
            return true
        end
    end

    return false
end

