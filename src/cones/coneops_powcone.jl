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
    # NB: the update order can't be switched as we reuse memory in the 
    # Hessian update
    _update_grad_HBFGS(K,s,z,μ,scaling_strategy)

    # K.z .= z
    @inbounds for i = 1:3
        K.z[i] = z[i]
    end
end

# return μH*(z) for power cone
function get_WtW!(
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
   z::AbstractVector{T},
   s::AbstractVector{T}
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
function combined_ds_shift!(
    K::PowerCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    η = K.grad_work      
    _higher_correction!(K,η,step_s,step_z)             #3rd order correction requires input variables.z
    @inbounds for i = 1:3
        shift[i] = K.grad[i]*σμ - η[i]
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

# compute the product y = c ⋅ μH(z)x
function mul_WtW!(
    K::PowerCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    c::T,
    workz::AbstractVector{T}
) where {T}

    # mul!(ls,K.HBFGS,lz,-one(T),zero(T))
    H = K.HBFGS
    @inbounds for i = 1:3
        y[i] =  c * (H[i,1]*x[1] + H[i,2]*x[2] + H[i,3]*x[3])
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
    αmin      = settings.min_terminate_step_length

    #need functions as closures to capture the power K.α
    #and use the same backtrack mechanism as the expcone
    is_primal_feasible_fcn = s -> _is_dual_feasible_powcone(s,K.α)
    is_dual_feasible_fcn   = s -> _is_primal_feasible_powcone(s,K.α)

    αz = _step_length_3d_cone(K.vec_work, dz, z, αmax, αmin, backtrack, is_primal_feasible_fcn)
    αs = _step_length_3d_cone(K.vec_work, ds, s, αmax, αmin, backtrack, is_dual_feasible_fcn)

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

    barrier += _barrier_dual(K, cur_z)
    barrier += _barrier_primal(K, cur_s)

    return barrier
end


# ----------------------------------------------
#  internal operations for power cones
#
# Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# We use the dual barrier function: 
# f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# Evaluates the gradient of the dual Power cone ∇f*(z) at z, 
# and stores the result at g


@inline function _barrier_dual(
    K::PowerCone{T},
    z::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Dual barrier
    α = K.α
    return -logsafe((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*logsafe(z[1]) - α*logsafe(z[2])

end

@inline function _barrier_primal(
    K::PowerCone{T},
    s::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    g = K.vec_work
    α = K.α

    _gradient_primal(K,s,g)     #compute g(s)
    return logsafe((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*logsafe(-g[1]) + α*logsafe(-g[2]) - 3
end 



# Returns true if s is primal feasible
function _is_primal_feasible_powcone(s::AbstractVector{T},α::T) where {T}

    if (s[1] > 0 && s[2] > 0)
        res = exp(2*α*logsafe(s[1]) + 2*(1-α)*logsafe(s[2])) - s[3]*s[3]
        if res > 0
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function _is_dual_feasible_powcone(z::AbstractVector{T},α::T) where {T}

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
function _gradient_primal(
    K::PowerCone{T},
    s::Union{AbstractVector{T}, NTuple{3,T}},
    g::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    α = K.α

    # unscaled ϕ
    ϕ = (s[1])^(2*α)*(s[2])^(2-2*α)

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

    # init point x0: since our dual barrier has an additional 
    # shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x),
    # the previous selection is still feasible, i.e. f(x0) > 0
    x = -one(T)/s3 + 2*(s3 + sqrt(4*ϕ*ϕ/s3/s3 + 3*ϕ))/(4*ϕ - s3*s3)

    # additional shift due to the choice of dual barrier
    t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)    
    t1 = x*x
    t2 = x*2/s3

    f0 = 2*α*logsafe(2*α*t1 + (1+α)*t2) + 
         2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - 
         logsafe(ϕ) - logsafe(t1+t2) - 2*logsafe(t2) + t0
        
    f1 = 2*α*α/(α*x + (1+α)/s3) + 
         2*(1-α)*(1-α)/((1-α)*x + (2-α)/s3) - 
         2*(x + 1/s3)/(t1 + t2)

    xnew = x - f0/f1

    # terminate when abs(xnew - x) <= eps(T)
    while (xnew - x) > eps(T)
        # println("x is ",x)
        x = xnew

        t1 = x*x
        t2 = x*2/s3

        f0 = 2*α*logsafe(2*α*t1 + (1+α)*t2) + 
             2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - 
             logsafe(ϕ) - logsafe(t1+t2) - 
             2*logsafe(t2) + t0

        f1 = 2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + 
             (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)

        xnew = x - f0/f1
    end

    return xnew
end

# 3rd-order correction at the point z 
# w.r.t. directions u,v. Writes to η
function _higher_correction!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs

    H = K.H
    u = K.vec_work
    z = K.z

    #solve H*u = ds
    issuccess = cholesky_3x3_explicit_factor!(K.cholH,H)
    if issuccess 
        cholesky_3x3_explicit_solve!(u,K.cholH,ds)
    else 
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

    # 3rd order correction: 
    # η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + 
    #            dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - 
    #            dotψuv/ψ + dothuv]
    # where: 
    # Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
    #         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
    #         0                       0                          -2;]
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

    Hψv = K.vec_work_2
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

    Hψu = K.vec_work_2
    Hψu[1] = Hψ[1,1]*u[1]+Hψ[1,2]*u[2]
    Hψu[2] = Hψ[2,1]*u[1]+Hψ[2,2]*u[2]
    Hψu[3] = -2*u[3]
    @. η = η + Hψu*dotψv*inv_ψ2

    @inbounds for i = 1:3
        η[i] /= 2
    end

end


#-------------------------------------
# primal-dual scaling
#-------------------------------------

# Implementation sketch
# 1) only need to replace μH by W^TW, where
#    W^TW is the primal-dual scaling matrix 
#    generated by BFGS, i.e. W^T W*[z,\tilde z] = [s,\tile s]
#   \tilde z = -f'(s), \tilde s = - f*'(z)

function _update_grad_HBFGS(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}
    # reuse memory
    st = K.grad
    zt = K.vec_work
    δs = K.grad_work
    tmp = K.z   #shared workspace for δz, tmp, axis_z
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

    #Use the local mu with primal dual strategy.  Otherwise 
    #we use the global one 
    if(scaling_strategy == Dual::ScalingStrategy)
        # HBFGS .= μ*H
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] = μ*H[i,j]
            end
        end
        return nothing
    end 
    dot_sz = dot(z,s)
    μ = dot_sz/3

    # compute zt,st,μt locally
    # NB: zt,st have different sign convention wrt Mosek paper
    _gradient_primal(K,s,zt)

    μt = dot(zt,st)/3

    # δs = s + μ*st
    # δz = z + μ*zt     

    @inbounds for i = 1:3
        δs[i] = s[i] + μ*st[i]
    end

    δz = tmp
    @inbounds for i = 1:3
        δz[i] = z[i] + μ*zt[i]
    end    
    dot_δsz = dot(δs,δz)

    de1 = μ*μt-1
    de2 = dot(zt,H,zt) - 3*μt*μt

    if !(abs(de1) > eps(T) && abs(de2) > eps(T))
        # HBFGS when s,z are on the central path
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] = μ*H[i,j]
            end
        end
        return nothing
    else
        # compute t
        # tmp = μt*st - H*zt
        @inbounds for i = 1:3
            tmp[i] = μt*st[i] - H[i,1]*zt[1] - 
                     H[i,2]*zt[2] - H[i,3]*zt[3]
        end

        # HBFGS as a workspace
        copyto!(HBFGS,H)
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] -= st[i]*st[j]/3 + tmp[i]*tmp[j]/de2
            end
        end

        t = μ*norm(HBFGS)  #Frobenius norm

        @assert dot_sz > 0
        @assert dot_δsz > 0
        @assert t > 0

        # generate the remaining axis
        # axis_z = cross(z,zt)
        axis_z = tmp
        axis_z[1] = z[2]*zt[3] - z[3]*zt[2]
        axis_z[2] = z[3]*zt[1] - z[1]*zt[3]
        axis_z[3] = z[1]*zt[2] - z[2]*zt[1]
        normalize!(axis_z)

        # HBFGS = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
        @inbounds for i = 1:3
            @inbounds for j = i:3
                HBFGS[i,j] = s[i]*s[j]/dot_sz + δs[i]*δs[j]/dot_δsz + t*axis_z[i]*axis_z[j]
            end
        end

        # symmetrize matrix
        HBFGS[2,1] = HBFGS[1,2]
        HBFGS[3,1] = HBFGS[1,3]
        HBFGS[3,2] = HBFGS[2,3]

        return nothing
    end
end
