# ----------------------------------------------------
# Power Cone
# ----------------------------------------------------

# degree of the cone
# PJG: hardcoded 3 unless it can be different
# MOSEK seems to allow n-dimensional, but maybe
# it's always possible to combine a 3d power cone
# with a bigger SOC or something to get the same
# behaviour
# YC: We should remove it at present and add it back
# when it is extended to the generalized power cone.
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
    flag::Bool
) where {T}
    # update both gradient and Hessian for function f*(z) at the point z
    # NB: the update order can't be switched as we reuse memory in the Hessian computation
    # Hessian update
    # update_HBFGS(K,s,z,flag)
    update_Hessian(K,s,z,μ)

    gradient_f(K,z,K.grad)
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
    σμ::T,
    scale_flag::Bool
) where {T}
    # NB: The higher-order correction is under development

    # PJG: remove dead code in comments here
    # YC: I need to test whether higherorder correction 
    # is effective for the power cone

    # η = K.grad_work      #share the same memory as gψ in higher_correction!()
    # higher_correction!(K,η,step_s,step_z)             #3rd order correction requires input variables.z
    # @inbounds for i = 1:3
    #     dz[i] = η[i] + K.grad[i]*σμ
    # end

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

    mul!(ls,K.HBFGS,lz,-one(T),zero(T))

end

#return maximum allowable step length while remaining in the Power cone
function step_length(
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

    αz = _step_length_power_dual(K.vec_work,dz,z,α,scaling,αExp)
    αs = _step_length_power_primal(K.vec_work,ds,s,α,scaling,αExp)

    return (αz,αs)
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

    while !check_power_primal_feas(ws,αExp)
        if (α < 1e-4)
            error("Power cone's step size fails in primal feasibility check!")
        end
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

    while !check_power_dual_feas(ws,αExp)
        if (α < 1e-4)
            error("Power cone's step size fails in dual feasibility check!")
        end
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
function gradient_f(
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
function compute_Hessian(
    K::PowerCone{T},
    z::AbstractVector{T},
    H::AbstractMatrix{T},
) where {T}

    α = K.α
    gψ = K.vec_work

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

end

function compute_centrality(
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

    g = K.vec_work
    gradient_primal(K,s,g)     #compute g(s)
    barrier += log((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*log(-g[1]) + α*log(-g[2]) - 3

    return barrier
end

# Returns true if s is primal feasible
function check_power_primal_feas(s::AbstractVector{T},α::T) where {T}
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
function check_power_dual_feas(z::AbstractVector{T},α::T) where {T}

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

# 3rd-order correction at the point z, w.r.t. directions u,v, and then save it to η
function higher_correction!(
    K::PowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs
    #NB: need to be refined later
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

    # memory allocation
    gψ = K.grad_work
    Hψ = Matrix{T}(undef,3,3)

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


######################################
# primal-dual scaling
######################################

# Implementation sketch
# 1) only need to replace μH by W⊤W,
#   where W⊤W is the primal-dual scaling matrix generated by BFGS, i.e. W⊤W*[z,̃z] = [s,̃s]
#   ̃z = -f'(s), ̃s = - f*'(z)

# NB: better to create two matrix spaces, one for the Hessian at z, H*(z), and another one for BFGS (primal-dual scaling) matrix, H-BFGS(z,s)

# YC: This is the temporary implementation for the dual scaling strategy
function update_Hessian(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T
) where {T}
    H = K.H
    HBFGS = K.HBFGS
    μ = dot(z,s)/3

    compute_Hessian(K,z,H)

    copyto!(HBFGS,H)
    BLAS.scal!(μ,HBFGS)
end

#PJG: There was a type error in the function below, where
#the first argument was K::ExponentalCone.   I don't understand
#how the code could possibly have worked like that, unless This
#function was never called at all.

# YC: I'm testing the primal-dual scaling with 
#     higher order correction and the function is for that purpose

function update_HBFGS(
    K::PowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    flag::Bool
) where {T}
    # reuse memory
    st = K.grad_work
    zt = K.vec_work
    δs = K.grad
    tmp = K.z
    H = K.H
    HBFGS = K.HBFGS

    # Hessian computation, compute μ locally
    compute_Hessian(K,z,H)
    μ = dot(z,s)/3
    # HBFGS .= μ*H
    @inbounds for i = 1:3
        @inbounds for j = 1:3
            HBFGS[i,j] = μ*H[i,j]
        end
    end

    # use the dual scaling
    if !flag
        return nothing
    end

    # compute zt,st,μt locally
    gradient_primal(K,s,zt)
    zt .*= -1
    gradient_f(K,z,st)
    st .*= -1
    μt = dot(zt,st)/3

    # δs = s - μ*st
    # δz = z - μ*zt
    @inbounds for i = 1:3
        δs[i] = s[i] - μ*st[i]
    end

    de1 = μ*μt-1
    de2 = dot(zt,H,zt) - 3*μt*μt

    if (de1 > eps(T) && de2 > eps(T))
        # tmp = H*zt - μt*st
        mul!(tmp,H,zt)
        @inbounds for i = 1:3
            tmp[i] -= μt*st[i]
        end

        # store (s + μ*st + δs/de1) into zt
        @inbounds for i = 1:3
            zt[i] = s[i] + μ*st[i] + δs[i]/de1
        end

        # Hessian HBFGS:= μ*H + 1/(2*μ*3)*δs*(s + μ*st + δs/de1)' + 1/(2*μ*3)*(s + μ*st + δs/de1)*δs' - μ/de2*tmp*tmp'
        coef1 = 1/(2*μ*3)
        coef2 = μ/de2
        # HBFGS .+= coef1*δs*zt' + coef1*zt*δs' - coef2*tmp*tmp'
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] += coef1*δs[i]*zt[j] + coef1*zt[i]*δs[j] - coef2*tmp[i]*tmp[j]
            end
        end
        # YC: to do but require H to be symmetric with upper triangular parts
        # syr2k!(uplo, trans, alpha, A, B, beta, C)
    else
        return nothing
    end

end
