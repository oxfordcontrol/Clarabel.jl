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
    muHessianF(K,z,K.μH,K.Hinv,μ)
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

    z[1] = one(T)*sqrt(1+α)
    z[2] = one(T)*sqrt(1+(1-α))
    z[3] = zero(T)

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
    
    αz = _step_length_power_dual(dz,z,α,scaling,exp)
    αs = _step_length_power_primal(ds,s,α,scaling,exp)

    return min(αz,αs)
end


# find the maximum step length α≥0 so that
# s + α*ds stays in the Power cone
function _step_length_power_primal(
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α::T,
    scaling::T,
    exp::T
) where {T}

    # NB: additional memory, may need to remove it later
    ws = s + α*ds

    while !checkPowerPrimalFeas(ws,exp)
        α *= scaling    #backtrack line search
        @. ws = s + α*ds
    end

    return α
end
# z + α*dz stays in the dual Power cone
function _step_length_power_dual(
    dz::AbstractVector{T},
    z::AbstractVector{T},
    α::T,
    scaling::T,
    exp::T
) where {T}

    # NB: additional memory, may need to remove it later
    ws = z + α*dz

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

    z1 = z[1]               #z1
    z2 = z[2]               #z2
    z3 = z[3]               #z3
    α = K.α

    ϕ = (z1/α)^(2*α)*(z2/(1-α))^(2-2*α)
    ψ = ϕ - z3*z3

    g[1] = -2*α*ϕ/(z1*ψ) - (1-α)/z1
    g[2] = -2*(1-α)*ϕ/(z2*ψ) - α/z2
    g[3] = 2*z3/ψ

end

# Evaluates the Hessian of the Power dual cone barrier at z and stores the upper triangular part of the matrix μH*(z)
# NB:could reduce H to an upper triangular matrix later, remove duplicate updates
function muHessianF(
    K::PowerCone{T},
    z::AbstractVector{T}, 
    H::AbstractMatrix{T}, 
    Hinv::AbstractMatrix{T}, 
    μ::T
) where {T}

    z1 = z[1]               #z1
    z2 = z[2]               #z2
    z3 = z[3]               #z3
    α = K.α

    ϕ = (z1/α)^(2*α)*(z2/(1-α))^(2-2*α)
    ψ = ϕ - z3*z3
    divgψ = [2*α*ϕ/(z1*ψ); 2*(1-α)*ϕ/(z2*ψ); -2*z3/ψ]
    gψ1 = divgψ[1]
    gψ2 = divgψ[2]
    gψ3 = divgψ[3]

    H[1,1] = gψ1*gψ1 - 2*α*(2*α-1)*ϕ/(z1*z1*ψ) + (1-α)/(z1*z1)
    H[1,2] = gψ1*gψ2 - 4*α*(1-α)*ϕ/(z1*z2*ψ)
    H[2,1] = H[1,2]
    H[2,2] = gψ2*gψ2 - 2*(1-α)*(1-2*α)*ϕ/(z2*z2*ψ) + α/(z2*z2)
    H[1,3] = gψ1*gψ3 
    H[3,1] = H[1,3]
    H[2,3] = gψ2*gψ3
    H[3,2] = H[2,3]
    H[3,3] = gψ3*gψ3 + 2/ψ

    # compute H^{-1}, 3x3 inverse is easy
    # NB: may need to be modified later rather than using inv() directly
    Hinv .= inv(H) 

    H .*= μ
end

# 3rd-order correction at the point z, w.r.t. directions u,v, and then save it to η
function higherCorrection!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T}, 
    v::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
    z1 = z[1]               #z1
    z2 = z[2]               #z2
    z3 = z[3]               #z3   
    v1 = v[1]
    v2 = v[2]
    v3 = v[3]

    # u for H^{-1}*Δs 
    u = K.Hinv*ds

    u1 = u[1]
    u2 = u[2]
    u3 = u[3]

    α = K.α

    ϕ = (z1/α)^(2*α)*(z2/(1-α))^(2-2*α)
    ψ = ϕ - z3*z3
    gψ = [2*α*ϕ/z1; 2*(1-α)*ϕ/z2; -2*z3]
    Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
            4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
            0                       0                          -2;]

    dotψu = dot(gψ,u)
    dotψv = dot(gψ,v)

    dotψuv = 4*α*(2*α-1)*(1-α)*ϕ*[-u1*v1/(z1*z1*z1) + (u2*v1+u1*v2)/(z1*z1*z2) - u2*v2/(z1*z2*z2); u1*v1/(z1*z1*z2) - (u2*v1+u1*v2)/(z1*z2*z2) + u2*v2/(z2*z2*z2); 0]
    dothuv = [-2*(1-α)*u1*v1/(z1*z1*z1); -2*α*u2*v2/(z2*z2*z2); 0]
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

    grad = zeros(T,3)
    μH = zeros(T,3,3)
    Hinv = zeros(T,3,3)

    # compute gradient and Hessian at z
    GradF(K, z, grad)
    muHessianF(K, z, μH, Hinv, μ)
    
    # grad as a workspace for s + μ*grad
    grad .*= μ
    grad .+= s

    if (norm(dot(grad,Hinv,grad)/(μ*μ)) < η^2)
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

