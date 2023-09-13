# ----------------------------------------------------
# Power Cone
# ----------------------------------------------------

# degree of the cone is always 3 for PowerCone
degree(K::PowerCone{T}) where {T} = 3
numel(K::PowerCone{T}) where {T} = 3

is_symmetric(::PowerCone{T}) where {T} = false

function margins(
    K::PowerCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone
) where{T}

    # We should never end up computing margins for this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function scaled_unit_shift!(
    K::PowerCone{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function unit_initialization!(
    K::PowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
     α = K.α
 
     s[1] = sqrt(one(T)+α)
     s[2] = sqrt(one(T)+((one(T)-α)))
     s[3] = zero(T)
 
     #@. z = s
     @inbounds for i = 1:3
         z[i] = s[i]
     end
 
    return nothing
 end

function set_identity_scaling!(
    K::PowerCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    update_dual_grad_H(K,z)

    # update the scaling matrix Hs
    update_Hs(K,s,z,μ,scaling_strategy)

    # K.z .= z
    @inbounds for i = 1:3
        K.z[i] = z[i]
    end

    return is_scaling_success = true
end

function Hs_is_diagonal(
    K::PowerCone{T}
) where{T}
    return false
end

# return μH*(z) for power cone
function get_Hs!(
    K::PowerCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    pack_triu(Hsblock,K.Hs)

end

# compute the product y = Hₛx = μH(z)x
function mul_Hs!(
    K::PowerCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # mul!(ls,K.Hs,lz,-one(T),zero(T))
    H = K.Hs
    @inbounds for i = 1:3
        y[i] =  H[i,1]*x[1] + H[i,2]*x[2] + H[i,3]*x[3]
    end

end

function affine_ds!(
    K::PowerCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:3
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::PowerCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    η = similar(K.grad); η .= zero(T)

    #3rd order correction requires input variables z
    higher_correction!(K, η, step_s,step_z)     

    @inbounds for i = 1:3
        shift[i] = K.grad[i]*σμ - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::PowerCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @inbounds for i = 1:3
        out[i] = ds[i]
    end

    return nothing
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

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    work = similar(K.grad); work .= zero(T)

    #need functions as closures to capture the power K.α
    #and use the same backtrack mechanism as the expcone
    is_prim_feasible_fcn = s -> is_primal_feasible(K,s,K.α)
    is_dual_feasible_fcn = s -> is_dual_feasible(K,s,K.α)

    αz = backtrack_search(K, dz, z, αmax, αmin, step, is_dual_feasible_fcn, work)
    αs = backtrack_search(K, ds, s, αmax, αmin, step, is_prim_feasible_fcn, work)

    return (αz,αs)
end

function compute_barrier(
    K::PowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    barrier = zero(T)

    # we want to avoid allocating a vector for the intermediate 
    # sums, so the two barrier functions are written to accept 
    # both vectors and 3-element tuples. 
    cur_z    = (z[1] + α*dz[1], z[2] + α*dz[2], z[3] + α*dz[3])
    cur_s    = (s[1] + α*ds[1], s[2] + α*ds[2], s[3] + α*ds[3])

    barrier += barrier_dual(K, cur_z)
    barrier += barrier_primal(K, cur_s)

    return barrier
end


# ----------------------------------------------
#  nonsymmetric cone operations for power cones
#
# Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# We use the dual barrier function: 
# f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# Evaluates the gradient of the dual Power cone ∇f*(z) at z, 
# and stores the result at g


@inline function barrier_dual(
    K::PowerCone{T},
    z::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Dual barrier
    α = K.α
    return -logsafe((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*logsafe(z[1]) - α*logsafe(z[2])

end

@inline function barrier_primal(
    K::PowerCone{T},
    s::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    α = K.α

    g = gradient_primal(K,s)     #compute g(s)
    return logsafe((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*logsafe(-g[1]) + α*logsafe(-g[2]) - 3
end 



# Returns true if s is primal feasible
function is_primal_feasible(
    ::PowerCone{T},
    s::AbstractVector{T},α::T
) where {T}

    if (s[1] > 0 && s[2] > 0)
        res = exp(2*α*logsafe(s[1]) + 2*(1-α)*logsafe(s[2])) - s[3]*s[3]
        if res > 0
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function is_dual_feasible(
    ::PowerCone{T},
    z::AbstractVector{T},α::T) where {T}

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
    s::Union{AbstractVector{T}, NTuple{3,T}},
) where {T}

    α = K.α;

    # unscaled ϕ
    ϕ = (s[1])^(2*α)*(s[2])^(2-2*α)
    g = similar(K.grad); g .= zero(T)


    # obtain g3 from the Newton-Raphson method
    abs_s = abs(s[3])
    if abs_s > eps(T)
        g[3] = _newton_raphson_powcone(abs_s,ϕ,α)
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
    return SVector(g)

end


# 3rd-order correction at the point z.  Output is η.

# 3rd order correction: 
# η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + 
#            dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - 
#            dotψuv/ψ + dothuv]
# where: 
# Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
#         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
#         0                       0                          -2;]
function higher_correction!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs
    H = K.H_dual
    z = K.z

    #solve H*u = ds
    cholH = similar(K.H_dual); cholH .= zero(T)
    issuccess = cholesky_3x3_explicit_factor!(cholH,H)
    if issuccess 
        u = cholesky_3x3_explicit_solve!(cholH,ds)
    else 
        return SVector(zero(T),zero(T),zero(T))
    end

    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # Reuse cholH memory for further computation
    Hψ = cholH
    
    η[1] = 2*α*ϕ/z[1]
    η[2] = 2*(1-α)*ϕ/z[2]
    η[3] = -2*z[3]

    Hψ[1,1] = 2*α*(2*α-1)*ϕ/(z[1]*z[1])
    Hψ[1,2] = 4*α*(1-α)*ϕ/(z[1]*z[2])
    Hψ[2,1] = Hψ[1,2]
    Hψ[1,3] = 0
    Hψ[3,1] = 0
    Hψ[2,2] = 2*(1-α)*(1-2*α)*ϕ/(z[2]*z[2])
    Hψ[2,3] = 0
    Hψ[3,2] = 0
    Hψ[3,3] = -2.

    dotψu = dot(η,u)
    dotψv = dot(η,v)

    Hψv = similar(K.grad); Hψv .= zero(T)
    Hψv[1] = Hψ[1,1]*v[1]+Hψ[1,2]*v[2]
    Hψv[2] = Hψ[2,1]*v[1]+Hψ[2,2]*v[2]
    Hψv[3] = -2*v[3]

    coef = (dot(u,Hψv)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
    coef2 = 4*α*(2*α-1)*(1-α)*ϕ*(u[1]/z[1] - u[2]/z[2])*(v[1]/z[1] - v[2]/z[2])/ψ
    inv_ψ2 = 1/ψ/ψ

    η[1] = coef*η[1] - 2*(1-α)*u[1]*v[1]/(z[1]*z[1]*z[1]) + 
           coef2/z[1] + Hψv[1]*dotψu*inv_ψ2

    η[2] = coef*η[2] - 2*α*u[2]*v[2]/(z[2]*z[2]*z[2]) - 
           coef2/z[2] + Hψv[2]*dotψu*inv_ψ2

    η[3] = coef*η[3] + Hψv[3]*dotψu*inv_ψ2

    # reuse vector Hψv
    Hψu = Hψv
    Hψu[1] = Hψ[1,1]*u[1]+Hψ[1,2]*u[2]
    Hψu[2] = Hψ[2,1]*u[1]+Hψ[2,2]*u[2]
    Hψu[3] = -2*u[3]

    # @. η <= (η + Hψu*dotψv*inv_ψ2)/2
    @inbounds for i = 1:3
        η[i] = (η[i] + Hψu[i]*dotψv*inv_ψ2)/2
    end
    # coercing to an SArray means that the MArray we computed 
    # locally in this function is seemingly not heap allocated 
    SArray(η)
end


# update gradient and Hessian at dual z
function update_dual_grad_H(
    K::PowerCone{T},
    z::AbstractVector{T}
) where {T}
    
    H = K.H_dual
    α = K.α

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # use K.grad as a temporary workspace
    gψ = K.grad
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

    # compute the gradient at z
    grad = K.grad
    grad[1] = -2*α*ϕ/(z[1]*ψ) - (1-α)/z[1]
    grad[2] = -2*(1-α)*ϕ/(z[2]*ψ) - α/z[2]
    grad[3] = 2*z[3]/ψ
end


# Newton-Raphson method:
# solve a one-dimensional equation f(x) = 0
# x(k+1) = x(k) - f(x(k))/f'(x(k))
# When we initialize x0 such that 0 < x0 < x*, 
# the Newton-Raphson method converges quadratically

function _newton_raphson_powcone(
    s3::T,
    ϕ::T,
    α::T
) where {T}

    # init point x0: f(x0) > 0
    x0 = -one(T)/s3 + (2*s3 + sqrt(ϕ*ϕ/s3/s3 + 3*ϕ))/(ϕ - s3*s3)

    # additional shift due to the choice of dual barrier
    t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)   

    # function for f(x) = 0
    function f0(x)
        t1 = x*x; t2 = 2*x/s3;
        2*α*logsafe(2*α*t1 + (1+α)*t2) + 
             2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - 
             logsafe(ϕ) - logsafe(t1+t2) - 
             2*logsafe(t2) + t0
    end

    # first derivative
    function f1(x)
        t1 = x*x; t2 = x*2/s3;
        2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + 
             (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)
    end
    
     return _newton_raphson_onesided(x0,f0,f1)
end
