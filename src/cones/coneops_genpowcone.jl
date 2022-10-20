# ----------------------------------------------------
# Generalized Power Cone
# ----------------------------------------------------

# degree of the cone is the dim of power vector + 1
dim(K::GenPowerCone{T}) where {T} = K.dim
degree(K::GenPowerCone{T}) where {T} = K.dim1 + 1
numel(K::GenPowerCone{T}) where {T} = dim(K)

is_symmetric(::GenPowerCone{T}) where {T} = false

function shift_to_cone!(
    K::GenPowerCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function unit_initialization!(
    K::GenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
     α = K.α
     dim1 = K.dim1
     dim  = K.dim
 
    # init u[i] = √(1+αi), i ∈ [dim1]
    @inbounds for i = 1:dim1
        s[i] = sqrt(one(T)+α[i])
    end
    # init w = 0
    s[dim1+1:end] .= zero(T)
 
     #@. z = s
     @inbounds for i = 1:dim
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
    _update_dual_grad_H(K,z)

    # update the scaling matrix Hs
    # YC: dual-scaling at present
    @. K.d1 *= μ
    K.d2 *= μ
    @. K.p *= sqrt(μ)
    @. K.q *= sqrt(μ)
    @. K.r *= sqrt(μ)

    # K.z .= z
    dim = K.dim
    @inbounds for i = 1:dim
        K.z[i] = z[i]
    end
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
    dim1 = K.dim1
    @. Hsblock[1:dim1]    = K.d1
    @. Hsblock[dim1+1:end] = K.d2

end

# # compute the product y = Hₛx = μH(z)x
# function mul_Hs!(
#     K::PowerCone{T},
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     workz::AbstractVector{T}
# ) where {T}

#     # mul!(ls,K.Hs,lz,-one(T),zero(T))
#     H = K.Hs
#     @inbounds for i = 1:3
#         y[i] =  H[i,1]*x[1] + H[i,2]*x[2] + H[i,3]*x[3]
#     end

# end

# function affine_ds!(
#     K::PowerCone{T},
#     ds::AbstractVector{T},
#     s::AbstractVector{T}
# ) where {T}

#     # @. x = y
#     @inbounds for i = 1:3
#         ds[i] = s[i]
#     end
# end

# function combined_ds_shift!(
#     K::PowerCone{T},
#     shift::AbstractVector{T},
#     step_z::AbstractVector{T},
#     step_s::AbstractVector{T},
#     σμ::T
# ) where {T}
    
#     #3rd order correction requires input variables z
#     η = _higher_correction!(K,step_s,step_z)     

#     @inbounds for i = 1:3
#         shift[i] = K.grad[i]*σμ - η[i]
#     end

#     return nothing
# end

# function Δs_from_Δz_offset!(
#     K::PowerCone{T},
#     out::AbstractVector{T},
#     ds::AbstractVector{T},
#     work::AbstractVector{T}
# ) where {T}

#     @inbounds for i = 1:3
#         out[i] = ds[i]
#     end

#     return nothing
# end

# #return maximum allowable step length while remaining in the Power cone
# function step_length(
#     K::PowerCone{T},
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
#     is_primal_feasible_fcn = s -> _is_primal_feasible_powcone(s,K.α)
#     is_dual_feasible_fcn   = s -> _is_dual_feasible_powcone(s,K.α)

#     αz = _step_length_3d_cone(K, dz, z, αmax, αmin, backtrack, is_dual_feasible_fcn)
#     αs = _step_length_3d_cone(K, ds, s, αmax, αmin, backtrack, is_primal_feasible_fcn)

#     return (αz,αs)
# end

# function compute_barrier(
#     K::PowerCone{T},
#     z::AbstractVector{T},
#     s::AbstractVector{T},
#     dz::AbstractVector{T},
#     ds::AbstractVector{T},
#     α::T
# ) where {T}

#     barrier = zero(T)

#     # we want to avoid allocating a vector for the intermediate 
#     # sums, so the two barrier functions are written to accept 
#     # both vectors and 3-element tuples. 
#     cur_z    = (z[1] + α*dz[1], z[2] + α*dz[2], z[3] + α*dz[3])
#     cur_s    = (s[1] + α*ds[1], s[2] + α*ds[2], s[3] + α*ds[3])

#     barrier += _barrier_dual(K, cur_z)
#     barrier += _barrier_primal(K, cur_s)

#     return barrier
# end


# # ----------------------------------------------
# #  internal operations for power cones
# #
# # Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# # Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# # We use the dual barrier function: 
# # f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# # Evaluates the gradient of the dual Power cone ∇f*(z) at z, 
# # and stores the result at g


# @inline function _barrier_dual(
#     K::PowerCone{T},
#     z::Union{AbstractVector{T}, NTuple{3,T}}
# ) where {T}

#     # Dual barrier
#     α = K.α
#     return -logsafe((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*logsafe(z[1]) - α*logsafe(z[2])

# end

# @inline function _barrier_primal(
#     K::PowerCone{T},
#     s::Union{AbstractVector{T}, NTuple{3,T}}
# ) where {T}

#     # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
#     # NB: ⟨s,g(s)⟩ = -3 = - ν

#     α = K.α

#     g = _gradient_primal(K,s)     #compute g(s)
#     return logsafe((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*logsafe(-g[1]) + α*logsafe(-g[2]) - 3
# end 



# # Returns true if s is primal feasible
# function _is_primal_feasible_powcone(s::AbstractVector{T},α::T) where {T}

#     if (s[1] > 0 && s[2] > 0)
#         res = exp(2*α*logsafe(s[1]) + 2*(1-α)*logsafe(s[2])) - s[3]*s[3]
#         if res > 0
#             return true
#         end
#     end

#     return false
# end

# # Returns true if s is dual feasible
# function _is_dual_feasible_powcone(z::AbstractVector{T},α::T) where {T}

#     if (z[1] > 0 && z[2] > 0)
#         res = exp(2*α*logsafe(z[1]/α) + 2*(1-α)*logsafe(z[2]/(1-α))) - z[3]*z[3]
#         if res > 0
#             return true
#         end
#     end

#     return false
# end

# # Compute the primal gradient of f(s) at s
# # solve it by the Newton-Raphson method
# function _gradient_primal(
#     K::PowerCone{T},
#     s::Union{AbstractVector{T}, NTuple{3,T}},
# ) where {T}

#     α = K.α;

#     # unscaled ϕ
#     ϕ = (s[1])^(2*α)*(s[2])^(2-2*α)
#     g = similar(K.grad)


#     # obtain g3 from the Newton-Raphson method
#     abs_s = abs(s[3])
#     if abs_s > eps(T)
#         g[3] = _newton_raphson_powcone(abs_s,ϕ,α)
#         if s[3] < zero(T)
#             g[3] = -g[3]
#         end
#         g[1] = -(α*g[3]*s[3] + 1 + α)/s[1]
#         g[2] = -((1-α)*g[3]*s[3] + 2 - α)/s[2]
#     else
#         g[3] = zero(T)
#         g[1] = -(1+α)/s[1]
#         g[2] = -(2-α)/s[2]
#     end
#     return SVector(g)

# end

# # Newton-Raphson method:
# # solve a one-dimensional equation f(x) = 0
# # x(k+1) = x(k) - f(x(k))/f'(x(k))
# # When we initialize x0 such that 0 < x0 < x*, 
# # the Newton-Raphson method converges quadratically

# function _newton_raphson_powcone(
#     s3::T,
#     ϕ::T,
#     α::T
# ) where {T}

#     # init point x0: since our dual barrier has an additional 
#     # shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x),
#     # the previous selection is still feasible, i.e. f(x0) > 0
#     x0 = -one(T)/s3 + 2*(s3 + sqrt(4*ϕ*ϕ/s3/s3 + 3*ϕ))/(4*ϕ - s3*s3)

#     # additional shift due to the choice of dual barrier
#     t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)   

#     # function for f(x) = 0
#     function f0(x)
#         t1 = x*x; t2 = 2*x/s3;
#         2*α*logsafe(2*α*t1 + (1+α)*t2) + 
#              2*(1-α)*logsafe(2*(1-α)*t1 + (2-α)*t2) - 
#              logsafe(ϕ) - logsafe(t1+t2) - 
#              2*logsafe(t2) + t0
#     end

#     # first derivative
#     function f1(x)
#         t1 = x*x; t2 = x*2/s3;
#         2*α*α/(α*x + (1+α)/s3) + 2*(1-α)*(1-α)/((1-α)*x + 
#              (2-α)/s3) - 2*(x + 1/s3)/(t1 + t2)
#     end
    
#     return _newton_raphson_onesided(x0,f0,f1)
# end

# function _newton_raphson_onesided(x0::T,f0::Function,f1::Function) where {T}

#     #implements NR method from a starting point assumed to be to the 
#     #left of the true value.   Once a negative step is encountered 
#     #this function will halt regardless of the calculated correction.

#     x = x0
#     iter = 0

#     while iter < 100

#         iter += 1
#         dfdx  =  f1(x)  
#         dx    = -f0(x)/dfdx

#         if (dx < eps(T)) ||
#             (abs(dx/x) < sqrt(eps(T))) ||
#             (abs(dfdx) < eps(T))
#             break
#         end
#         x += dx
#     end
#     return x
# end


# # 3rd-order correction at the point z.  Output is η.

# # 3rd order correction: 
# # η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + 
# #            dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - 
# #            dotψuv/ψ + dothuv]
# # where: 
# # Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
# #         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
# #         0                       0                          -2;]
# function _higher_correction!(
#     K::PowerCone{T},
#     ds::AbstractVector{T},
#     v::AbstractVector{T}
# ) where {T}

#     # u for H^{-1}*Δs
#     H = K.H_dual
#     z = K.z

#     #solve H*u = ds
#     cholH = similar(K.H_dual)
#     issuccess = cholesky_3x3_explicit_factor!(cholH,H)
#     if issuccess 
#         u = cholesky_3x3_explicit_solve!(cholH,ds)
#     else 
#         return SVector(zero(T),zero(T),zero(T))
#     end

#     α = K.α

#     ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
#     ψ = ϕ - z[3]*z[3]

#     # Reuse cholH memory for further computation
#     Hψ = cholH
    
#     η = similar(K.grad)
#     η[1] = 2*α*ϕ/z[1]
#     η[2] = 2*(1-α)*ϕ/z[2]
#     η[3] = -2*z[3]

#     Hψ[1,1] = 2*α*(2*α-1)*ϕ/(z[1]*z[1])
#     Hψ[1,2] = 4*α*(1-α)*ϕ/(z[1]*z[2])
#     Hψ[2,1] = Hψ[1,2]
#     Hψ[1,3] = 0
#     Hψ[3,1] = 0
#     Hψ[2,2] = 2*(1-α)*(1-2*α)*ϕ/(z[2]*z[2])
#     Hψ[2,3] = 0
#     Hψ[3,2] = 0
#     Hψ[3,3] = -2.

#     dotψu = dot(η,u)
#     dotψv = dot(η,v)

#     Hψv = similar(K.grad)
#     Hψv[1] = Hψ[1,1]*v[1]+Hψ[1,2]*v[2]
#     Hψv[2] = Hψ[2,1]*v[1]+Hψ[2,2]*v[2]
#     Hψv[3] = -2*v[3]

#     coef = (dot(u,Hψv)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
#     coef2 = 4*α*(2*α-1)*(1-α)*ϕ*(u[1]/z[1] - u[2]/z[2])*(v[1]/z[1] - v[2]/z[2])/ψ
#     inv_ψ2 = 1/ψ/ψ

#     η[1] = coef*η[1] - 2*(1-α)*u[1]*v[1]/(z[1]*z[1]*z[1]) + 
#            coef2/z[1] + Hψv[1]*dotψu*inv_ψ2

#     η[2] = coef*η[2] - 2*α*u[2]*v[2]/(z[2]*z[2]*z[2]) - 
#            coef2/z[2] + Hψv[2]*dotψu*inv_ψ2

#     η[3] = coef*η[3] + Hψv[3]*dotψu*inv_ψ2

#     # reuse vector Hψv
#     Hψu = Hψv
#     Hψu[1] = Hψ[1,1]*u[1]+Hψ[1,2]*u[2]
#     Hψu[2] = Hψ[2,1]*u[1]+Hψ[2,2]*u[2]
#     Hψu[3] = -2*u[3]

#     # @. η <= (η + Hψu*dotψv*inv_ψ2)/2
#     @inbounds for i = 1:3
#         η[i] = (η[i] + Hψu[i]*dotψv*inv_ψ2)/2
#     end
#     # coercing to an SArray means that the MArray we computed 
#     # locally in this function is seemingly not heap allocated 
#     SArray(η)
# end


# update gradient and Hessian at dual z
function _update_dual_grad_H(
    K::GenPowerCone{T},
    z::AbstractVector{T}
) where {T}
    
    α = K.α
    p = K.p
    q = K.q
    r = K.r 
    d1 = K.d1

    dim1 = K.dim1
    dim = K.dim

    # ϕ = (ui/αi)^(2*αi), ζ = ϕ - ||w||^2
    ϕ = one(T)
    @inbounds for i = 1:dim1
        ϕ *= (z[i]/α[i])^(2*α[i])
    end
    norm2w = dot(z[dim1+1:end],z[dim1+1:end])
    ζ = ϕ - norm2w
    @assert ζ > zero(T)

    # compute the gradient at z
    grad = K.grad
    τ = q           # τ shares memory with q
    @inbounds for i = 1:dim1
        τ[i] = 2*α[i]/z[i]
        grad[i] = -τ[i]*ϕ/ζ - (1-α[i])/z[i]
    end
    @inbounds for i = dim1+1:dim
        grad[i] = 2*z[i]/ζ
    end

    # compute Hessian information at z 
    p0 = sqrt(ϕ*(ϕ+norm2w)/2)
    p1 = -2*ϕ/p0
    q0 = sqrt(ζ*ϕ/2)
    r1 = 2*sqrt(ζ/(ϕ+norm2w))

    # compute the diagonal d1,d2
    @inbounds for i = 1:dim1
        d1[i] = τ[i]*ϕ/(ζ*z[i]) + (1-α[i])/(z[i]*z[i])
    end   
    K.d2 = 2/ζ

    # compute p, q, r where τ shares memory with q
    p[1:dim1] .= p0*τ/ζ
    p[dim1+1:end] .= p1*z[dim1+1:end]/ζ

    q .*= q0/ζ
    r .*= r1*z[dim1+1:end]/ζ

end

