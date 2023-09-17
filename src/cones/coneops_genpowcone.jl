# ----------------------------------------------------
# Generalized Power Cone
# ----------------------------------------------------

#dimensions of the subcomponents
dim1(K::GenPowerCone{T}) where {T} = length(K.α)
dim2(K::GenPowerCone{T}) where {T} = K.dim2

# degree of the cone is the dim of power vector + 1
dim(K::GenPowerCone{T}) where {T} = dim1(K) + dim2(K)
degree(K::GenPowerCone{T}) where {T} = dim1(K) + 1
numel(K::GenPowerCone{T}) where {T} = dim(K)

function is_sparse_expandable(::GenPowerCone{T}) where{T}
    # we do not curently have a way of representing
    # this cone in non-expanded form
    return true
end

is_symmetric(::GenPowerCone{T}) where {T} = false
allows_primal_dual_scaling(::GenPowerCone{T}) where {T} = false

function shift_to_cone!(
    K::GenPowerCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit initialization
    error("This function should never be reached.");
    # 
end

function unit_initialization!(
    K::GenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
    # init u[i] = √(1+αi), i ∈ [dim1(K)]
    @inbounds for i = 1:dim1(K)
        s[i] = sqrt(one(T)+K.α[i])
    end
    # init w = 0
    s[dim1(K)+1:end] .= zero(T)
 
     #@. z = s
     @inbounds for i = 1:dim(K)
         z[i] = s[i]
     end
 
    return nothing
 end

function set_identity_scaling!(
    K::GenPowerCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::GenPowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    update_dual_grad_H(K,z)
    K.data.μ = μ

    # K.z .= z
    @inbounds for i in eachindex(z)
        K.data.z[i] = z[i]
    end

    return is_scaling_success = true
end

function Hs_is_diagonal(
    K::GenPowerCone{T}
) where{T}
    return true
end

# return μH*(z) for generalized power cone
function get_Hs!(
    K::GenPowerCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal D = [d1; d2] block from the
    #sparse representation of W^TW, but not the
    #extra 3 entries at the bottom right of the block.
    #The ConicVector for s and z (and its views) don't
    #know anything about the 3 extra sparsifying entries
    dim1 = Clarabel.dim1(K)
    data = K.data
    
    @. Hsblock[1:dim1]     = data.μ*data.d1
    @. Hsblock[dim1+1:end] = data.μ*data.d2

end

# compute the product y = Hs*x = μH(z)x
function mul_Hs!(
    K::GenPowerCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # Hs = μ*(D + pp' -qq' -rr')

    data = K.data

    rng1 = 1:dim1(K)
    rng2 = (dim1(K)+1):dim(K)

    coef_p = dot(data.p,x)
    @views coef_q = dot(data.q,x[rng1])
    @views coef_r = dot(data.r,x[rng2])
    
    @. y[rng1] = data.d1*x[rng1] - coef_q*K.data.q
    @. y[rng2] = data.d2*x[rng2] - coef_r*K.data.r

    @. y += coef_p*data.p
    @. y *= data.μ

end

function affine_ds!(
    K::GenPowerCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:dim(K)
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::GenPowerCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}
    
    #YC: No 3rd order correction at present

    # #3rd order correction requires input variables z
    # and an allocated vector for the correction η
    # higher_correction!(K,η,step_s,step_z)     

    @inbounds for i = 1:Clarabel.dim(K)
        shift[i] = K.data.grad[i]*σμ # - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::GenPowerCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @inbounds for i = 1:dim(K)
        out[i] = ds[i]
    end

    return nothing
end

#return maximum allowable step length while remaining in the generalized power cone
function step_length(
    K::GenPowerCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    work = K.data.work

    is_prim_feasible_fcn = s -> is_primal_feasible(K,s)
    is_dual_feasible_fcn = s -> is_dual_feasible(K,s)

    αz = backtrack_search(K, dz, z, αmax, αmin, step, is_dual_feasible_fcn,work)
    αs = backtrack_search(K, ds, s, αmax, αmin, step, is_prim_feasible_fcn,work)

    return (αz,αs)
end

function compute_barrier(
    K::GenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    barrier = zero(T)
    work = K.data.work

    #primal barrier
    @inbounds for i = 1:dim(K)
        work[i] = s[i] + α*ds[i]
    end
    barrier += barrier_primal(K, work)

    #dual barrier
    @inbounds for i = 1:dim(K)
        work[i] = z[i] + α*dz[i]
    end
    barrier += barrier_dual(K, work)

    return barrier
end


# ----------------------------------------------
#  internal operations for generalized power cones
#
# Primal generalized power cone: ∏_{i ∈ [d1]}s[i]^{α[i]} ≥ ||s[d1+1:end]||, s[1:d1] ≥ 0
# Dual generalized power cone: ∏_{i ∈ [d1]}(z[i]/α[i])^{α[i]} ≥ ||z[d1+1:end]||, z[1:d1] ≥ 0
# We use the dual barrier function: 
# f*(z) = -log((∏_{i ∈ [d1]}(z[i]/α[i])^{2*α[i]} - ||z[d1+1:end]||^2) - ∑_{i ∈ [d1]} (1-α[i])*log(z[i]):
# Evaluates the gradient of the dual generalized power cone ∇f*(z) at z, 
# and stores the result at g


# Returns true if s is primal feasible
function is_primal_feasible(
    K::GenPowerCone{T},
    s::AbstractVector{T},
) where {T}

    dim1 = Clarabel.dim1(K)
    α = K.α

    if (all(s[1:dim1] .> zero(T)))
        res = zero(T)
        @inbounds for i = 1:dim1
            res += 2*α[i]*logsafe(s[i])
        end
        res = exp(res) - sumsq(@view s[dim1+1:end])
        if res > zero(T)
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
function is_dual_feasible(
    K::GenPowerCone{T},
    z::AbstractVector{T},
) where {T}

    dim1 = Clarabel.dim1(K)
    α = K.α

    if (all(z[1:dim1] .> zero(T)))
        res = zero(T)
        @inbounds for i = 1:dim1
            res += 2*α[i]*logsafe(z[i]/α[i])
        end
        res = exp(res) - sumsq(@view z[dim1+1:end])
        if res > zero(T)
            return true
        end
    end
    
    return false
end

@inline function barrier_primal(
    K::GenPowerCone{T},
    s::AbstractVector{T}, 
) where {T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -(dim1(K)+1) = - ν

    # can't use "work" here because it was already
    # used to construct the argument s in some cases
    g = K.data.work_pb

    gradient_primal!(K,g,s)      
    g .= -g                 #-g(s)

    return -barrier_dual(K,g) - degree(K)
end 


@inline function barrier_dual(
    K::GenPowerCone{T},
    z::AbstractVector{T}, 
) where {T}

    # Dual barrier
    α = K.α

    res = zero(T)
    @inbounds for i = 1:dim1(K)
        res += 2*α[i]*logsafe(z[i]/α[i])
    end
    res = exp(res) - sumsq(@view z[dim1(K)+1:end])
    barrier = -logsafe(res) 
    @inbounds for i = 1:dim1(K)
        barrier -= (one(T)-α[i])*logsafe(z[i])
    end

    return barrier

end


# update gradient and Hessian at dual z = (u,w)
function update_dual_grad_H(
    K::GenPowerCone{T},
    z::AbstractVector{T}
) where {T}
    
    α = K.α
    data = K.data
    p = data.p
    q = data.q
    r = data.r 
    d1 = data.d1

    # ϕ = ∏_{i ∈ dim1}(ui/αi)^(2*αi), ζ = ϕ - ||w||^2
    phi = one(T)
    @inbounds for i = 1:dim1(K)
        phi *= (z[i]/α[i])^(2*α[i])
    end
    norm2w = sumsq(@view z[dim1(K)+1:end])
    ζ = phi - norm2w
    @assert ζ > zero(T)

    # compute the gradient at z
    grad = data.grad
    τ = q           # τ shares memory with q

    @inbounds for i = 1:dim1(K)
        τ[i] = 2*α[i]/z[i]
        grad[i] = -τ[i]*phi/ζ - (1-α[i])/z[i]
    end
    @inbounds for i = (dim1(K)+1):dim(K)
        grad[i] = 2*z[i]/ζ
    end

    # compute Hessian information at z 
    p0 = sqrt(phi*(phi+norm2w)/2)
    p1 = -2*phi/p0
    q0 = sqrt(ζ*phi/2)
    r1 = 2*sqrt(ζ/(phi+norm2w))

    # compute the diagonal d1,d2
    @inbounds for i = 1:dim1(K)
        d1[i] = τ[i]*phi/(ζ*z[i]) + (1-α[i])/(z[i]*z[i])
    end   
    data.d2 = 2/ζ

    # compute p, q, r where τ shares memory with q
    p[1:dim1(K)] .= p0*τ/ζ
    @views p[(dim1(K)+1):end] .= p1*z[(dim1(K)+1):end]/ζ

    q .*= q0/ζ      #τ is abandoned
    @views r .= r1*z[(dim1(K)+1):end]/ζ

end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function gradient_primal!(
    K::GenPowerCone{T},
    g::AbstractVector{T},
    s::AbstractVector{T},
) where {T}

    α = K.α
    data = K.data

    # unscaled phi
    phi = one(T)
    @inbounds for i = 1:dim1(K)
        phi *= s[i]^(2*α[i])
    end


    # obtain g1 from the Newton-Raphson method
    p = @view s[1:dim1(K)]
    r = @view s[dim1(K)+1:end]
    gp = @view g[1:dim1(K)]
    gr = @view g[dim1(K)+1:end]
    norm_r = norm(r)

    if norm_r > eps(T)
        g1 = _newton_raphson_genpowcone(norm_r,p,phi,α,data.ψ)
        @. gr = g1*r/norm_r
        @. gp = -(1+α+α*g1*norm_r)/p
    else
        @. gr = zero(T)
        @. gp = -(1+α)/p
    end

    return nothing
end

# ----------------------------------------------
#  internal operations for generalized power cones

# Newton-Raphson method:
# solve a one-dimensional equation f(x) = 0
# x(k+1) = x(k) - f(x(k))/f'(x(k))
# When we initialize x0 such that 0 < x0 < x* and f(x0) > 0, 
# the Newton-Raphson method converges quadratically

function _newton_raphson_genpowcone(
    norm_r::T,
    p::AbstractVector{T},
    phi::T,
    α::AbstractVector{T},
    ψ::T
) where {T}

    # init point x0: f(x0) > 0
    x0 = -one(T)/norm_r + (ψ*norm_r + sqrt((phi/norm_r/norm_r + ψ*ψ - one(T))*phi))/(phi - norm_r*norm_r)

    # # additional shift due to the choice of dual barrier
    # t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)   

    # function for f(x) = 0
    function f0(x)
        f0 = -logsafe(2*x/norm_r + x*x);
        @inbounds for i in eachindex(α)
            f0 += 2*α[i]*(logsafe(x*norm_r+(1+α[i])/α[i]) - logsafe(p[i]))
        end

        return f0
    end

    # first derivative
    function f1(x)
        f1 = -(2*x + 2/norm_r)/(x*x + 2*x/norm_r);
        @inbounds for i in eachindex(α)
            f1 += 2*α[i]*norm_r/(norm_r*x + (1+α[i])/α[i])
        end

        return f1
    end
    
    return _newton_raphson_onesided(x0,f0,f1)
end
