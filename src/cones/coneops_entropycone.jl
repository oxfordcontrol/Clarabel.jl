# ----------------------------------------------------
# Relative Entropy Cone
# ----------------------------------------------------

# degree of the cone is the dim of power vector + 1
dim(K::EntropyCone{T}) where {T} = K.dim
degree(K::EntropyCone{T}) where {T} = 3*K.d
numel(K::EntropyCone{T}) where {T} = dim(K)

is_symmetric(::EntropyCone{T}) where {T} = false

function shift_to_cone!(
    K::EntropyCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

# Generate an initial point following Hypatia
# primal variable s0
function get_central_ray_epirelentropy(w_dim::Int)
    if w_dim <= 10
        return central_rays_epirelentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    rtw_dim = sqrt(w_dim)
    if w_dim <= 20
        u = 1.2023 / rtw_dim - 0.015
        v = 0.432 / rtw_dim + 1.0125
        w = -0.3057 / rtw_dim + 0.972
    else
        u = 1.1513 / rtw_dim - 0.0069
        v = 0.4873 / rtw_dim + 1.0008
        w = -0.4247 / rtw_dim + 0.9961
    end
    return [u, v, w]
end

const central_rays_epirelentropy = [
    0.827838399 1.290927714 0.805102005
    0.708612491 1.256859155 0.818070438
    0.622618845 1.231401008 0.829317079
    0.558111266 1.211710888 0.838978357
    0.508038611 1.196018952 0.847300431
    0.468039614 1.183194753 0.854521307
    0.435316653 1.172492397 0.860840992
    0.408009282 1.163403374 0.866420017
    0.38483862 1.155570329 0.871385499
    0.364899122 1.148735192 0.875838068
]


function unit_initialization!(
    K::EntropyCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
     d = K.d
     dim  = K.dim
 
    # initialization from Hypatia
    (s[1], v, w) = get_central_ray_epirelentropy(dim)
    @views s[2:d+1] .= v
    @views s[d+2:end] .= w
    # find z such that z = -g*(s)

    minus_gs = minus_gradient_primal(K,s)     #YC: may have memory issue
    @. z = minus_gs

    return nothing
 end

function set_identity_scaling!(
    K::EntropyCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::EntropyCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    _update_dual_grad_H(K,z)

    # update the scaling matrix Hs
    # YC: dual-scaling at present; we time μ to the diagonal here,
    # but this could be implemented elsewhere; μ is also used later 
    # when updating the off-diagonal terms of Hs; Recording μ is redundant 
    # for the dual scaling as it is a global parameter
    K.μ = μ

    # K.z .= z
    dim = K.dim
    @inbounds for i = 1:dim
        K.z[i] = z[i]
    end
end

# YC: may need to _allocate_kkt_Hsblocks()
# Stop it here
function Hs_is_diagonal(
    K::EntropyCone{T}
) where{T}
    return false
end

# return μH*(z) for generalized power cone
function get_Hs!(
    K::EntropyCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal dd and offd
    μ = K.μ
    dim = K.dim
    d = K.d
    Hsblock[1:dim]    .= μ*K.dd
    Hsblock[dim+1:dim+2*d] .= μ*K.u
    Hsblock[2*dim:end]    .= μ*K.offd

end

# compute the product y = Hs*x = μH(z)x
function mul_Hs!(
    K::EntropyCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    d = K.d

    dot_1 = dot(K.u,x[2:end])

    x1 = @view x[2:d+1]
    x2 = @view x[d+2:end]
    y1 = @view y[2:d+1]
    y2 = @view y[d+2:end]
    y3 = @view y[2:end]
    
    @. y = K.dd*x
    y[1] += dot_1
    @. y1 += K.offd*x2
    @. y2 += K.offd*x1
    @. y3 += K.u*x[1]
    
    @. y *= K.μ

end

function affine_ds!(
    K::EntropyCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:K.dim
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::EntropyCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}
    
    #YC: No 3rd order correction at present

    # #3rd order correction requires input variables z
    # η = _higher_correction!(K,step_s,step_z)     

    @inbounds for i = 1:K.dim
        shift[i] = K.grad[i]*σμ # - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::EntropyCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    @inbounds for i = 1:K.dim
        out[i] = ds[i]
    end

    return nothing
end

#return maximum allowable step length while remaining in the generalized power cone
function step_length(
    K::EntropyCone{T},
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
    is_primal_feasible_fcn = s -> _is_primal_feasible_entropycone(s,K.d)
    is_dual_feasible_fcn   = s -> _is_dual_feasible_entropycone(s,K.d)

    αz = _step_length_n_cone(K, dz, z, αmax, αmin, backtrack, is_dual_feasible_fcn)
    αs = _step_length_n_cone(K, ds, s, αmax, αmin, backtrack, is_primal_feasible_fcn)

    return (αz,αs)
end

function compute_barrier(
    K::EntropyCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    dim = K.dim

    barrier = zero(T)

    # we want to avoid allocating a vector for the intermediate 
    # sums, so the two barrier functions are written to accept 
    # both vectors and MVectors. 
    # wq = similar(K.grad)
    wq = K.work

    #primal barrier
    @inbounds for i = 1:dim
        wq[i] = s[i] + α*ds[i]
    end
    barrier += _barrier_primal(K, wq)

    #dual barrier
    @inbounds for i = 1:dim
        wq[i] = z[i] + α*dz[i]
    end
    barrier += _barrier_dual(K, wq)

    return barrier
end


# ----------------------------------------------
#  internal operations for relative entropy cones
#
# Primal relative entropy cone: u - ∑_{i ∈ [d]} w_i log(w_i/v_i) ≥ 0
# Dual relative entropy cone: w_i ≥ u(log(u/v_i)-1), ∀ i ∈ [d]
# We use the dual barrier function: 
# f*(z) = -∑_{i ∈ [d]}log(w_i - ulog(u/v_i) + u) - log(u) - ∑_{i ∈ [d]}log(v_i):
# Evaluates the gradient of the dual relative entropy cone ∇f*(z) at z, 
# and stores the result at g


@inline function _barrier_dual(
    K::EntropyCone{T},
    z::Union{AbstractVector{T}, NTuple{N,T}}
) where {N<:Integer,T}

    # Dual barrier
    d = K.d 
    v = @view z[2:d+1]
    w = @view z[d+2:end]

    barrier = -logsafe(z[1]) 
    @inbounds for i = 1:d
        barrier -= logsafe(v[i]) + logsafe(w[i]-z[1]*logsafe(z[1]/v[i]) + z[1])
    end

    return barrier

end

@inline function _barrier_primal(
    K::EntropyCone{T},
    s::Union{AbstractVector{T}, NTuple{N,T}}
) where {N<:Integer,T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = - K.dim = - ν

    minus_g = minus_gradient_primal(K,s)     #compute g(s)

    #YC: need to consider the memory issue later
    return -_barrier_dual(K,minus_g) - K.dim
end 



# Returns true if s is primal feasible
function _is_primal_feasible_entropycone(
    s::AbstractVector{T},
    d::Int
) where {T}

    v = @view s[2:d+1]
    w = @view s[d+2:end]

    if (all(v .> zero(T)) && all(w .> zero(T)))
        res = s[1]
        @inbounds for i = 1:d
            res -= w[i]*logsafe(w[i]/v[i])
        end
        if res > zero(T)
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
function _is_dual_feasible_entropycone(
    z::AbstractVector{T},
    d::Int
) where {T}

    v = @view z[2:d+1]
    w = @view z[d+2:end]

    if (z[1] > zero(T) && all(v .> zero(T)))
        @inbounds for i = 1:d
            res = w[i] - z[1]*logsafe(z[1]/v[i]) + z[1]
            if res > zero(T)
                continue
            else
                return false        #one of the dual part violates the constraint
            end
        end

    end
    
    return true
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function minus_gradient_primal(
    K::EntropyCone{T},
    s::Union{AbstractVector{T}, NTuple{N,T}},
) where {N<:Integer,T}

    d = K.d
    # minus_g = similar(K.grad)
    minus_g = K.work
    minus_gq = @view minus_g[2:d+1]
    minus_gr = @view minus_g[d+2:end]
    q = @view s[2:d+1]
    r = @view s[d+2:end]

    minus_gp_inv = _newton_raphson_entropycone(s,d)
    minus_g[1] = inv(minus_gp_inv)
    @inbounds for i = 1:d
        minus_gq[i] = (minus_g[1]*r[i] + one(T))/q[i]
        minus_gr[i] = one(T)/r[i] + minus_g[1]*logsafe(minus_g[1]/minus_gq[i]) - minus_g[1]
    end

    return minus_g
end

# Newton-Raphson method:
# solve a one-dimensional concave equation f(x) = 0
# x(k+1) = x(k) - f(x(k))/f'(x(k))
# When we initialize x0 such that 0 ≤ x0 < x* and f(x0) < 0, 
# the Newton-Raphson method converges quadratically

function _newton_raphson_entropycone(
    s::AbstractVector{T},
    d::Int,
) where {T}

    # init point x0=0: f(x0) < 0
    p = s[1]
    q = @view s[2:d+1]
    r = @view s[d+2:end]
    x0 = zero(T)
    offset = p
    @inbounds for i = 1:d
        offset += r[i]*logsafe(q[i])
    end

    # function for f(x) = 0
    function f0(x)
        f0 = x*d - offset;
        @inbounds for i = 1:d
            f0 += r[i]*logsafe(r[i]+x)
        end

        return f0
    end

    # first derivative
    function f1(x)
        f1 = T(d);
        @inbounds for i = 1:d
            f1 += r[i]/(r[i]+x)
        end

        return f1
    end
    
    return _newton_raphson_onesided(x0,f0,f1)

end


# update gradient and Hessian at dual z = (u,w)
function _update_dual_grad_H(
    K::EntropyCone{T},
    z::AbstractVector{T}
) where {T}

    d = K.d
    u = K.u 
    offd = K.offd
    dd = K.dd

    # compute the gradient at z
    grad = K.grad
    logdiv = @view K.z[1:d] #working space for logdiv
    γ = @view K.z[d+1:2*d]  #working space for γ
    # YC:maybe γinv is faster?

    grad[1] = -T(d)/z[1]
    @inbounds for i = 1:d
        logdiv[i] = log(z[1]/z[i+1])
        γ[i] = z[d+1+i] -z[1]*logdiv[i] + z[1]

        grad[i+1] = -z[1]/(γ[i]*z[i+1]) - one(T)/z[i+1]
        grad[d+1+i] = -one(T)/γ[i]
        grad[1] += logdiv[i]/γ[i]
    end
    
    # compute Hessian information at z 
    dd[1] = T(d)/(z[1]^2)
    @inbounds for i = 1:d
        u[d+i] = -logdiv[i]/(γ[i]^2)
        u[i] = u[d+i]*z[1]/z[i+1]-one(T)/(γ[i]*z[i+1])
        dd[1] += one(T)/(z[1]*γ[i]) - logdiv[i]*u[d+i]

        offd[i] = z[1]/(γ[i]^2*z[i+1])
        dd[i+1] = offd[i]*(γ[i]+z[1])/z[i+1] + one(T)/(z[i+1]^2)
        dd[d+1+i] = one(T)/(γ[i]^2)
    end    
end
