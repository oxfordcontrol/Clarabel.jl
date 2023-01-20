# ----------------------------------------------------
# Relative Entropy Cone
# ----------------------------------------------------

# degree of the cone is the dim of power vector + 1
dim(K::EntropyCone{T}) where {T} = K.dim
degree(K::EntropyCone{T}) where {T} = K.dim
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

    gs = _gradient_primal(K,s)     #YC: may have memory issue
    @. z = -gs

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
    @view Hsblock[1:dim]    .= μ*K.dd
    @view Hsblock[dim+1:dim+2*d] .= μ*K.u
    @view Hsblock[2*dim:end]    .= μ*K.offd

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
    x3 = @view x[2:end]
    y1 = @view y[2:d+2]
    y2 = @view y[d+2:end]
    y3 = @view y[2:end]
    
    @. y = K.dd*x
    y[1] += dot_1
    @. y1 += K.offd*x1
    @. y2 += K.offd*x2
    @. y3 += K.u*x3
    
    @. y *= K.μ

end

# function affine_ds!(
#     K::EntropyCone{T},
#     ds::AbstractVector{T},
#     s::AbstractVector{T}
# ) where {T}

#     # @. x = y
#     @inbounds for i = 1:K.dim
#         ds[i] = s[i]
#     end
# end

# function combined_ds_shift!(
#     K::EntropyCone{T},
#     shift::AbstractVector{T},
#     step_z::AbstractVector{T},
#     step_s::AbstractVector{T},
#     σμ::T
# ) where {T}
    
#     #YC: No 3rd order correction at present

#     # #3rd order correction requires input variables z
#     # η = _higher_correction!(K,step_s,step_z)     

#     @inbounds for i = 1:K.dim
#         shift[i] = K.grad[i]*σμ # - η[i]
#     end

#     return nothing
# end

# function Δs_from_Δz_offset!(
#     K::EntropyCone{T},
#     out::AbstractVector{T},
#     ds::AbstractVector{T},
#     work::AbstractVector{T}
# ) where {T}

#     @inbounds for i = 1:K.dim
#         out[i] = ds[i]
#     end

#     return nothing
# end

# #return maximum allowable step length while remaining in the generalized power cone
# function step_length(
#     K::EntropyCone{T},
#     dz::AbstractVector{T},
#     ds::AbstractVector{T},
#      z::AbstractVector{T},
#      s::AbstractVector{T},
#      settings::Settings{T},
#      αmax::T,
# ) where {T}

#     backtrack = settings.linesearch_backtrack_step
#     αmin      = settings.min_terminate_step_length

#     #need functions as closures to capture the power K.α
#     #and use the same backtrack mechanism as the expcone
#     is_primal_feasible_fcn = s -> _is_primal_feasible_genpowcone(s,K.α,K.dim1)
#     is_dual_feasible_fcn   = s -> _is_dual_feasible_genpowcone(s,K.α,K.dim1)

#     αz = _step_length_n_cone(K, dz, z, αmax, αmin, backtrack, is_dual_feasible_fcn)
#     αs = _step_length_n_cone(K, ds, s, αmax, αmin, backtrack, is_primal_feasible_fcn)

#     return (αz,αs)
# end

# function compute_barrier(
#     K::EntropyCone{T},
#     z::AbstractVector{T},
#     s::AbstractVector{T},
#     dz::AbstractVector{T},
#     ds::AbstractVector{T},
#     α::T
# ) where {T}

#     dim = K.dim

#     barrier = zero(T)

#     # we want to avoid allocating a vector for the intermediate 
#     # sums, so the two barrier functions are written to accept 
#     # both vectors and MVectors. 
#     wq = similar(K.grad)

#     #primal barrier
#     @inbounds for i = 1:dim
#         wq[i] = s[i] + α*ds[i]
#     end
#     barrier += _barrier_primal(K, wq)

#     #dual barrier
#     @inbounds for i = 1:dim
#         wq[i] = z[i] + α*dz[i]
#     end
#     barrier += _barrier_dual(K, wq)

#     return barrier
# end


# # ----------------------------------------------
# #  internal operations for generalized power cones
# #
# # Primal generalized power cone: ∏_{i ∈ [d1]}s[i]^{α[i]} ≥ ||s[d1+1:end]||, s[1:d1] ≥ 0
# # Dual generalized power cone: ∏_{i ∈ [d1]}(z[i]/α[i])^{α[i]} ≥ ||z[d1+1:end]||, z[1:d1] ≥ 0
# # We use the dual barrier function: 
# # f*(z) = -log((∏_{i ∈ [d1]}(z[i]/α[i])^{2*α[i]} - ||z[d1+1:end]||^2) - ∑_{i ∈ [d1]} (1-α[i])*log(z[i]):
# # Evaluates the gradient of the dual generalized power cone ∇f*(z) at z, 
# # and stores the result at g


# @inline function _barrier_dual(
#     K::EntropyCone{T},
#     z::Union{AbstractVector{T}, NTuple{N,T}}
# ) where {N<:Integer,T}

#     # Dual barrier
#     dim1 = K.dim1
#     α = K.α

#     res = zero(T)
#     @inbounds for i = 1:dim1
#         res += 2*α[i]*logsafe(z[i]/α[i])
#     end
#     res = exp(res) - dot(z[dim1+1:end],z[dim1+1:end])
#     barrier = -logsafe(res) 
#     @inbounds for i = 1:dim1
#         barrier -= (one(T)-α[i])*logsafe(z[i])
#     end

#     return barrier

# end

# @inline function _barrier_primal(
#     K::EntropyCone{T},
#     s::Union{AbstractVector{T}, NTuple{N,T}}
# ) where {N<:Integer,T}

#     # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
#     # NB: ⟨s,g(s)⟩ = -(dim1+1) = - ν

#     α = K.α
#     dim1 = K.dim1

#     g = _gradient_primal(K,s)     #compute g(s)

#     #YC: need to consider the memory issue later
#     return -_barrier_dual(K,-g) - (dim1+one(T))
# end 



# # Returns true if s is primal feasible
# function _is_primal_feasible_genpowcone(
#     s::AbstractVector{T},
#     α::AbstractVector{T},
#     dim1::Int
# ) where {T}

#     if (all(s[1:dim1] .> zero(T)))
#         res = zero(T)
#         @inbounds for i = 1:dim1
#             res += 2*α[i]*logsafe(s[i])
#         end
#         res = exp(res) - dot(s[dim1+1:end],s[dim1+1:end])
#         if res > zero(T)
#             return true
#         end
#     end

#     return false
# end

# # Returns true if z is dual feasible
# function _is_dual_feasible_genpowcone(
#     z::AbstractVector{T},
#     α::AbstractVector{T},
#     dim1::Int
# ) where {T}

#     if (all(z[1:dim1] .> zero(T)))
#         res = zero(T)
#         @inbounds for i = 1:dim1
#             res += 2*α[i]*logsafe(z[i]/α[i])
#         end
#         res = exp(res) - dot(z[dim1+1:end],z[dim1+1:end])
#         if res > zero(T)
#             return true
#         end
#     end
    
#     return false
# end

# # Compute the primal gradient of f(s) at s
# # solve it by the Newton-Raphson method
# function _gradient_primal(
#     K::EntropyCone{T},
#     s::Union{AbstractVector{T}, NTuple{N,T}},
# ) where {N<:Integer,T}

#     α = K.α
#     dim1 = K.dim1
#     g = similar(K.grad)

#     # unscaled ϕ
#     ϕ = one(T)
#     @inbounds for i = 1:dim1
#         ϕ *= s[i]^(2*α[i])
#     end


#     # obtain g3 from the Newton-Raphson method
#     p = @view s[1:dim1]
#     r = @view s[dim1+1:end]
#     gp = @view g[1:dim1]
#     gr = @view g[dim1+1:end]
#     norm_r = norm(r)

#     if norm_r > eps(T)
#         g1 = _newton_raphson_genpowcone(norm_r,dim1,p,ϕ,α)
#         @. gr = g1*r/norm_r
#         @. gp = -(1+α+α*g1*norm_r)/p
#     else
#         @. gr = zero(T)
#         @. gp = -(1+α)/p
#     end

#     return g

# end

# # Newton-Raphson method:
# # solve a one-dimensional equation f(x) = 0
# # x(k+1) = x(k) - f(x(k))/f'(x(k))
# # When we initialize x0 such that 0 < x0 < x* and f(x0) > 0, 
# # the Newton-Raphson method converges quadratically

# function _newton_raphson_genpowcone(
#     norm_r::T,
#     dim::Int,
#     p::AbstractVector{T},
#     ϕ::T,
#     α::AbstractVector{T}
# ) where {T}

#     # init point x0: f(x0) > 0
#     dim2 = dim*dim
#     x0 = -one(T)/norm_r + (dim*norm_r + sqrt((ϕ/norm_r/norm_r + dim2 -1)*ϕ))/(ϕ - norm_r*norm_r)

#     # # additional shift due to the choice of dual barrier
#     # t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)   

#     # function for f(x) = 0
#     function f0(x)
#         f0 = -logsafe(2*x/norm_r + x*x);
#         @inbounds for i = 1:dim
#             f0 += 2*α[i]*(logsafe(x*norm_r+(1+α[i])/α[i]) - logsafe(p[i]))
#         end

#         return f0
#     end

#     # first derivative
#     function f1(x)
#         f1 = -(2*x + 2/norm_r)/(x*x + 2*x/norm_r);
#         @inbounds for i = 1:dim
#             f1 += 2*α[i]*norm_r/(norm_r*x + (1+α[i])/α[i])
#         end

#         return f1
#     end
    
#     return _newton_raphson_onesided(x0,f0,f1)
# end


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

    grad[1] = -one(T)/z[1]
    @inbounds for i = 1:d
        logdiv[i] = log(z[1]/z[i+1])
        γ[i] = z[d+1+i] -z[1]*logdiv[i] + z[1]

        grad[i+1] = -z[1]/(γ[i]*z[i+1]) - one(T)/z[i+1]
        grad[d+1+i] = -one(T)/γ[i]
        grad[1] += logdiv[i]/γ[i]
    end
    
    # compute Hessian information at z 
    dd[1] = one(T)/(z[1]^2)
    @inbounds for i = 1:d
        u[d+i] = -logdiv[i]/(γ[i]^2)
        u[i] = u[d+i]*z[1]/z[i+1]-one(T)/(γ[i]*z[i+1])
        dd[1] += one(T)/(z[1]*γ[i]) - logdiv[i]*u[d+i]

        offd[i] = z[1]/(γ[i]^2*z[i+1])
        dd[i+1] = offd[i]*(γ[i]+z[1])/z[i+1] + one(T)/(z[i+1]^2)
        dd[d+1+i] = one(T)/(γ[i]^2)
    end    


end

# function compute_grad(z)

#     grad = similar(z)
#     T = Float64

#     d = 2
#     logdiv = rand(d) #working space for logdiv
#     γ = rand(d)  #working space for γ

#     grad[1] = -one(T)/z[1]
#     @inbounds for i = 1:d
#         logdiv[i] = log(z[1]/z[i+1])
#         γ[i] = z[d+1+i] -z[1]*logdiv[i] + z[1]

#         grad[i+1] = -z[1]/(γ[i]*z[i+1]) - one(T)/z[i+1]
#         grad[d+1+i] = -one(T)/γ[i]
#         grad[1] += logdiv[i]/γ[i]
#     end

#     u = rand(4) 
#     offd = rand(2)
#     dd = rand(5)
#     H = zeros(5,5)
#     # compute Hessian information at z 
#     dd[1] = one(T)/(z[1]^2)
#     @inbounds for i = 1:d
#         u[d+i] = -logdiv[i]/(γ[i]^2)
#         u[i] = u[d+i]*z[1]/z[i+1]-one(T)/(γ[i]*z[i+1])
#         dd[1] += one(T)/(z[1]*γ[i]) - logdiv[i]*u[d+i]

#         offd[i] = z[1]/(γ[i]^2*z[i+1])
#         dd[i+1] = offd[i]*(γ[i]+z[1])/z[i+1] + one(T)/(z[i+1]^2)
#         dd[d+1+i] = one(T)/(γ[i]^2)
#     end   

#     H[2:5,1] .= u 
#     for i = 1:5
#         H[i,i] = dd[i]
#     end
#     for i = 1:d
#         H[d+1+i,i+1] = offd[i]
#     end

#     return grad,H
# end
