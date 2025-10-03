# # ----------------------------------------------------
# # Power Cone
# # ----------------------------------------------------

# # degree of the cone is always 3 for PowerCone
# degree(K::PowerCone{T}) where {T} = 3
# numel(K::PowerCone{T}) where {T} = 3

# is_symmetric(::PowerCone{T}) where {T} = false

# unit initialization for asymmetric solves
function _kernel_unit_initialization_pow!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where{T}
 
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @views si = s[rng_cone_i] 
        si[1] = sqrt(one(T)+αp[i])
        si[2] = sqrt(one(T)+((one(T)-αp[i])))
        si[3] = zero(T)

        #@. z = s
        @inbounds for j = 1:3
            zi[j] = si[j]
        end
    end

    return nothing
end

@inline function unit_initialization_pow!(
    ::Val{false},
    z::AbstractVector{T},
    s::AbstractVector{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where{T}
    return nothing
end

@inline function unit_initialization_pow!(
    ::Val{true},
    z::AbstractVector{T},
    s::AbstractVector{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where{T}
 
    kernel = @cuda launch=false _kernel_unit_initialization_pow!(z, s, αp, rng_cones, n_shift, n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    kernel(z, s, αp, rng_cones, n_shift, n_pow; threads, blocks)
end

# update the scaling matrix Hs
@inline function update_Hs_pow(
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractVector{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    μ::T,
    scaling_strategy::ScalingStrategy,
    α::T
) where {T}

    # Choose the scaling strategy
    if(scaling_strategy == Dual::ScalingStrategy)
        # Dual scaling: Hs = μ*H
        use_dual_scaling_gpu(Hs,H_dual,μ)
    else
        # Primal-dual scaling
        use_primal_dual_scaling_pow(s,z,grad,Hs,H_dual,α)
    end 
end

function _kernel_update_scaling_pow!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        shift_exp = n_exp+i
        @views gradi = grad[:,shift_exp]
        @views Hsi = Hs[:,:,shift_exp]
        @views Hi = H_dual[:,:,shift_exp]
        # update both gradient and Hessian for function f*(z) at the point z
        update_dual_grad_H_pow(gradi,Hi,zi,αp[i])

        # update the scaling matrix Hs
        update_Hs_pow(si,zi,gradi,Hsi,Hi,μ,scaling_strategy,αp[i])
    end

    return nothing
end

@inline function update_scaling_pow!(
    ::Val{false},
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}
    return nothing
end

@inline function update_scaling_pow!(
    ::Val{true},
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    kernel = @cuda launch=false _kernel_update_scaling_pow!(s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_shift, n_exp, n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    kernel(s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_shift, n_exp, n_pow; threads, blocks)
end

# return μH*(z) for power cone
function _kernel_get_Hs_pow!(
    Hsblock::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        shift_exp = n_exp+i
        @views Hsi = Hs[:,:,shift_exp]
        @views Hsblocki = Hsblock[rng_i]

        
        @inbounds for j in 1:length(Hsblocki)
            Hsblocki[j] = Hsi[j]
        end
    end

    return nothing

end

@inline function get_Hs_pow!(
    ::Val{false},
    Hsblocks::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}
    return nothing
end

@inline function get_Hs_pow!(
    ::Val{true},
    Hsblocks::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    kernel = @cuda launch=false _kernel_get_Hs_pow!(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    kernel(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow; threads, blocks)
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

function _kernel_combined_ds_shift_pow!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        shift_exp = i + n_exp
        @views Hi = H_dual[:,:,shift_exp]
        @views gradi = grad[:,shift_exp]
        @views zi = z[rng_i]
        @views step_si = step_s[rng_i]
        @views step_zi = step_z[rng_i]
        @views shifti = shift[rng_i]

        η = @MVector T[0, 0, 0]

        #3rd order correction requires input variables z
        higher_correction_pow!(Hi,zi,η,step_si,step_zi,αp[i])             

        @inbounds for i = 1:3
            shifti[i] = gradi[i]*σμ - η[i]
        end
    end

    return nothing
end

@inline function combined_ds_shift_pow!(
    ::Val{false},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}
    return nothing
end

@inline function combined_ds_shift_pow!(
    ::Val{true},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint,
    n_pow::Cint
) where {T}

    kernel = @cuda launch=false _kernel_combined_ds_shift_pow!(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    kernel(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow; threads, blocks)
end

# function Δs_from_Δz_offset!(
#     K::PowerCone{T},
#     out::AbstractVector{T},
#     ds::AbstractVector{T},
#     work::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @inbounds for i = 1:3
#         out[i] = ds[i]
#     end

#     return nothing
# end

#return maximum allowable step length while remaining in the Power cone
function _kernel_step_length_pow(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αp::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_pow::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views dzi = dz[rng_i]
        @views dsi = ds[rng_i]
        @views zi = z[rng_i]
        @views si = s[rng_i]
        
        α[i]  = backtrack_search_pow(dzi, zi, dsi, si, αmax, αmin, step,αp[i])
    end
    
    return nothing
end

@inline function step_length_pow(
    ::Val{false},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αp::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_pow::Cint
) where {T}
    return αmax
end

@inline function step_length_pow(
    ::Val{true},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αp::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_pow::Cint
) where {T}

    kernel = @cuda launch=false _kernel_step_length_pow(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    CUDA.@sync kernel(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow; threads, blocks)
    @views αmax = min(αmax,minimum(α[1:n_pow]))

    if αmax < 0
        throw(DomainError("starting point of line search not in power cones"))
    end

    return αmax
end

@inline function backtrack_search_pow(
    dz::AbstractVector{T},
    z::AbstractVector{T},
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α_init::T,
    α_min::T,
    step::T,
    αp::T
) where {T}

    α = α_init
    work = @MVector T[0, 0, 0]
 
    while true
        #@. wq = q + α*dq
        @inbounds for i in eachindex(work)
            work[i] = z[i] + α*dz[i]
        end

        if is_dual_feasible_pow(work,αp)
            break
        end
        if (α *= step) < α_min
            α = zero(T)
            return α
        end
    end
    
    while true
        #@. wq = q + α*dq
        @inbounds for i in eachindex(work)
            work[i] = s[i] + α*ds[i]
        end

        if is_primal_feasible_pow(work,αp)
            break
        end
        if (α *= step) < α_min
            α = zero(T)
            return α
        end
    end

    return α
end


function _kernel_compute_barrier_pow(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where {T}

    # we want to avoid allocating a vector for the intermediate 
    # sums, so the two barrier functions are written to accept 
    # both vectors and 3-element tuples. 
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_pow
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views dzi = dz[rng_i]
        @views dsi = ds[rng_i]
        @views zi = z[rng_i]
        @views si = s[rng_i]
        
        cur_z    = (zi[1] + α*dzi[1], zi[2] + α*dzi[2], zi[3] + α*dzi[3])
        cur_s    = (si[1] + α*dsi[1], si[2] + α*dsi[2], si[3] + α*dsi[3])
    
        barrier_d = barrier_dual_pow(cur_z, αp[i])
        barrier_p = barrier_primal_pow(cur_s, αp[i])
        barrier[i] = barrier_d + barrier_p
    end

    return nothing
end

@inline function compute_barrier_pow(
    ::Val{false},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where {T}
    return zero(T)
end

@inline function compute_barrier_pow(
    ::Val{true},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    αp::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_pow::Cint
) where {T}
    kernel = @cuda launch=false _kernel_compute_barrier_pow(barrier,z,s,dz,ds,α,αp,rng_cones,n_shift,n_pow)
    config = launch_configuration(kernel.fun)
    threads = min(n_pow, config.threads)
    blocks = cld(n_pow, threads)

    CUDA.@sync kernel(barrier,z,s,dz,ds,α,αp,rng_cones,n_shift,n_pow; threads, blocks)

    return sum(@view barrier[1:n_pow])
end


# # ----------------------------------------------
# #  nonsymmetric cone operations for power cones
# #
# # Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
# # Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0
# # We use the dual barrier function: 
# # f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
# # Evaluates the gradient of the dual Power cone ∇f*(z) at z, 
# # and stores the result at g


@inline function barrier_dual_pow(
    z::Union{AbstractVector{T}, NTuple{3,T}},
    α::T
) where {T}

    # Dual barrier
    return -logsafe((z[1]/α)^(2*α) * (z[2]/(1-α))^(2-2*α) - z[3]*z[3]) - (1-α)*logsafe(z[1]) - α*logsafe(z[2])

end

@inline function barrier_primal_pow(
    s::Union{AbstractVector{T}, NTuple{3,T}},
    α::T
) where {T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    g = gradient_primal_pow(s, α)     #compute g(s)
    return logsafe((-g[1]/α)^(2*α) * (-g[2]/(1-α))^(2-2*α) - g[3]*g[3]) + (1-α)*logsafe(-g[1]) + α*logsafe(-g[2]) - 3
end 



# Returns true if s is primal feasible
@inline function is_primal_feasible_pow(
    s::AbstractVector{T}, 
    α::T
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
@inline function is_dual_feasible_pow(
    z::AbstractVector{T}, 
    α::T
) where {T}

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
function gradient_primal_pow(
    s::Union{AbstractVector{T}, NTuple{3,T}},
    α::T
) where {T}

    # unscaled ϕ
    ϕ = (s[1])^(2*α)*(s[2])^(2-2*α)

    # obtain g3 from the Newton-Raphson method
    abs_s = abs(s[3])
    if abs_s > eps(T)
        g3 = _newton_raphson_powcone(abs_s,ϕ,α)
        if s[3] < zero(T)
            g3 = -g3
        end
        g1 = -(α*g3*s[3] + 1 + α)/s[1]
        g2 = -((1-α)*g3*s[3] + 2 - α)/s[2]
    else
        g3 = zero(T)
        g1 = -(1+α)/s[1]
        g2 = -(2-α)/s[2]
    end
    return (g1,g2,g3)

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
@inline function higher_correction_pow!(
    H::AbstractArray{T},
    z::AbstractVector{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T},
    α::T
) where {T}

    #solve H*u = ds
    cholH = @MArray T[0.0 0.0 0.0;0.0 0.0 0.0;0.0 0.0 0.0]
    issuccess = cholesky_3x3_explicit_factor!(cholH,H)
    if issuccess 
        u = cholesky_3x3_explicit_solve!(cholH,ds)
    else 
        return SVector(zero(T),zero(T),zero(T))
    end

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

    dotψu = _dot_xy_gpu(η,u,1:3)
    dotψv = _dot_xy_gpu(η,v,1:3)

    Hψv = @MVector T[0, 0, 0]
    Hψv[1] = Hψ[1,1]*v[1]+Hψ[1,2]*v[2]
    Hψv[2] = Hψ[2,1]*v[1]+Hψ[2,2]*v[2]
    Hψv[3] = -2*v[3]

    coef = (_dot_xy_gpu(u,Hψv,1:3)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
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
    SVector(η)
end


# update gradient and Hessian at dual z
function update_dual_grad_H_pow(
    grad::AbstractVector{T},
    H::AbstractArray{T},
    z::AbstractVector{T},
    α::T
) where {T}

    ϕ = (z[1]/α)^(2*α)*(z[2]/(1-α))^(2-2*α)
    ψ = ϕ - z[3]*z[3]

    # use K.grad as a temporary workspace
    gψ = grad
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
    grad[1] = -2*α*ϕ/(z[1]*ψ) - (1-α)/z[1]
    grad[2] = -2*(1-α)*ϕ/(z[2]*ψ) - α/z[2]
    grad[3] = 2*z[3]/ψ
end


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

#     # init point x0: f(x0) > 0
#     x0 = -one(T)/s3 + (2*s3 + sqrt(ϕ*ϕ/s3/s3 + 3*ϕ))/(ϕ - s3*s3)

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
    
#      return _newton_raphson_onesided(x0,f0,f1)
# end
