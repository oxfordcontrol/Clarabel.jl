# # ----------------------------------------------------
# # Exponential Cone
# # ----------------------------------------------------

# # degree of the cone.  Always 3
# degree(K::ExponentialCone{T}) where {T} = 3
# numel(K::ExponentialCone{T}) where {T} = 3

# is_symmetric(::ExponentialCone{T}) where {T} = false

# unit initialization for asymmetric solves
function _kernel_unit_initialization_exp!(
   z::AbstractVector{T},
   s::AbstractVector{T},
   rng_cones::AbstractVector,
   n_shift::Cint,
   n_exp::Cint
) where{T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @views si = s[rng_cone_i] 
        si[1] = T(-1.051383945322714)
        si[2] = T(0.556409619469370)
        si[3] = T(1.258967884768947)
    
        #@. z = s
        @inbounds for j = 1:3
            zi[j] = si[j]
        end
    end

    return nothing
end

@inline function unit_initialization_exp!(
    ::Val{false},
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
 ) where{T}
    return nothing
end

@inline function unit_initialization_exp!(
    ::Val{true},
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
 ) where{T}
 
    kernel = @cuda launch=false _kernel_unit_initialization_exp!(z, s, rng_cones, n_shift, n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    kernel(z, s, rng_cones, n_shift, n_exp; threads, blocks)
end

 # update the scaling matrix Hs
@inline function update_Hs_exp(
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractVector{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # Choose the scaling strategy
    if(scaling_strategy == Dual::ScalingStrategy)
        # Dual scaling: Hs = μ*H
        use_dual_scaling_gpu(Hs,H_dual,μ)
    else
        # Primal-dual scaling
        use_primal_dual_scaling_exp(s,z,grad,Hs,H_dual)
    end 
end

function _kernel_update_scaling_exp!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        @views gradi = grad[:,i]
        @views Hsi = Hs[:,:,i]
        @views Hi = H_dual[:,:,i]

        update_dual_grad_H_exp(gradi,Hi,zi)
      
        # update the scaling matrix Hs
        update_Hs_exp(si,zi,gradi,Hsi,Hi,μ,scaling_strategy)
        
    end

    return nothing
end

@inline function update_scaling_exp!(
    ::Val{false},
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint
) where {T}
    return nothing
end

@inline function update_scaling_exp!(
    ::Val{true},
    s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    Hs::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    μ::T,
    scaling_strategy::ScalingStrategy,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    kernel = @cuda launch=false _kernel_update_scaling_exp!(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    kernel(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp; threads, blocks)
end

# return μH*(z) for exponential cone
function _kernel_get_Hs_exp!(
    Hsblock::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Hsi = Hs[:,:,i]
        @views Hsblocki = Hsblock[rng_i]

        
        @inbounds for j in 1:length(Hsblocki)
            Hsblocki[j] = Hsi[j]
        end
    end

    return nothing

end

@inline function get_Hs_exp!(
    ::Val{false},
    Hsblocks::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}
    return nothing
end

@inline function get_Hs_exp!(
    ::Val{true},
    Hsblocks::AbstractVector{T},
    Hs::AbstractArray{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    kernel = @cuda launch=false _kernel_get_Hs_exp!(Hsblocks, Hs, rng_blocks, n_shift, n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    kernel(Hsblocks, Hs, rng_blocks, n_shift, n_exp; threads, blocks)
end

function _kernel_combined_ds_shift_exp!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views Hi = H_dual[:,:,i]
        @views gradi = grad[:,i]
        @views zi = z[rng_i]
        @views step_si = step_s[rng_i]
        @views step_zi = step_z[rng_i]
        @views shifti = shift[rng_i]

        η = @MVector T[0, 0, 0]

        #3rd order correction requires input variables z
        higher_correction_exp!(Hi,zi,η,step_si,step_zi)             

        @inbounds for i = 1:3
            shifti[i] = gradi[i]*σμ - η[i]
        end
    end

    return nothing
end

@inline function combined_ds_shift_exp!(
    ::Val{false},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint
) where {T}
    return nothing
end

@inline function combined_ds_shift_exp!(
    ::Val{true},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    grad::AbstractArray{T},
    H_dual::AbstractArray{T},
    rng_cones::AbstractVector,
    σμ::T,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    kernel = @cuda launch=false _kernel_combined_ds_shift_exp!(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    kernel(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp; threads, blocks)
end

# function Δs_from_Δz_offset!(
#     K::ExponentialCone{T},
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

#return maximum step length while staying in exponential cone
function _kernel_step_length_exp(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_exp::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views dzi = dz[rng_i]
        @views dsi = ds[rng_i]
        @views zi = z[rng_i]
        @views si = s[rng_i]
        
        α[i] = backtrack_search_exp(dzi, zi, dsi, si, αmax, αmin, step)
    end

    return nothing
end

@inline function step_length_exp(
    ::Val{false},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_exp::Cint
) where {T}
    return αmax
end

@inline function step_length_exp(
    ::Val{true},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     αmax::T,
     αmin::T,
     step::T,
     n_shift::Cint,
     n_exp::Cint
) where {T}
    kernel = @cuda launch=false _kernel_step_length_exp(dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_shift, n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    CUDA.@sync kernel(dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_shift, n_exp; threads, blocks)
    @views αmax = min(αmax, minimum(α[1:n_exp]))

    if αmax < 0
        throw(DomainError("starting point of line search not in expotential cones"))
    end

    return αmax
end

@inline function backtrack_search_exp(
    dz::AbstractVector{T},
    z::AbstractVector{T},
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α_init::T,
    α_min::T,
    step::T
) where {T}

    α = α_init
    work = @MVector T[0, 0, 0]
    
    while true
        #@. wq = q + α*dq
        @inbounds for i in eachindex(work)
            work[i] = s[i] + α*ds[i]
        end

        if is_primal_feasible_exp(work)
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
            work[i] = z[i] + α*dz[i]
        end

        if is_dual_feasible_exp(work)
            break
        end
        if (α *= step) < α_min
            α = zero(T)
            return α
        end
    end

    return α
end

function _kernel_compute_barrier_exp(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}

    # we want to avoid allocating a vector for the intermediate 
    # sums, so the two barrier functions are written to accept 
    # both vectors and 3-element tuples. 
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_exp
        # update both gradient and Hessian for function f*(z) at the point z
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views dzi = dz[rng_i]
        @views dsi = ds[rng_i]
        @views zi = z[rng_i]
        @views si = s[rng_i]
        
        cur_z    = @MVector [zi[1] + α*dzi[1], zi[2] + α*dzi[2], zi[3] + α*dzi[3]]
        cur_s    = @MVector [si[1] + α*dsi[1], si[2] + α*dsi[2], si[3] + α*dsi[3]]
    
        barrier_d = barrier_dual_exp(cur_z)
        barrier_p = barrier_primal_exp(cur_s)
        barrier[i] = barrier_d + barrier_p
    end

    return nothing
end

@inline function compute_barrier_exp(
    ::Val{false},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}
    return zero(T)
end

@inline function compute_barrier_exp(
    ::Val{true},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_exp::Cint
) where {T}
    kernel = @cuda launch=false _kernel_compute_barrier_exp(barrier,z,s,dz,ds,α,rng_cones,n_shift,n_exp)
    config = launch_configuration(kernel.fun)
    threads = min(n_exp, config.threads)
    blocks = cld(n_exp, threads)

    CUDA.@sync kernel(barrier,z,s,dz,ds,α,rng_cones,n_shift,n_exp; threads, blocks)

    return sum(@view barrier[1:n_exp])
end


# # -----------------------------------------
# # nonsymmetric cone operations for exponential cones
# #
# # Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# # Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# # We use the dual barrier function: 
# # f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
# # -----------------------------------------


@inline function barrier_dual_exp(
    z::AbstractVector{T}
) where {T}

    # Dual barrier
    l = logsafe(-z[3]/z[1])
    return -logsafe(-z[3]*z[1]) - logsafe(z[2]-z[1]-z[1]*l) 

end 

@inline function barrier_primal_exp(
    s::AbstractVector{T}
) where {T}

    # Primal barrier: 
    # f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    #      = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3, 
    # where barω = ω(1 - s1/s2 - log(s2) - log(s3))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    ω = _wright_omega_gpu(1-s[1]/s[2]-logsafe(s[2]/s[3]))
    ω = (ω-1)*(ω-1)/ω
   return -logsafe(ω)-2*logsafe(s[2])-logsafe(s[3]) - 3
end 



# Returns true if s is primal feasible
@inline function is_primal_feasible_exp(
    s::AbstractVector{T}
) where {T}

    if (s[3] > 0 && s[2] > zero(T))   #feasible
        res = s[2]*logsafe(s[3]/s[2]) - s[1]
        if (res > zero(T))
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
@inline function is_dual_feasible_exp(
    z::AbstractVector{T}
) where {T}

    if (z[3] > 0 && z[1] < zero(T))
        res = z[2] - z[1] - z[1]*logsafe(-z[3]/z[1])
        if (res > zero(T))
            return true
        end
    end
    return false
end

# Compute the primal gradient of f(s) at s
function gradient_primal_exp(
    s::Union{AbstractVector{T}, NTuple{3,T}},
) where {T}

    ω = _wright_omega_gpu(1-s[1]/s[2]-logsafe(s[2]/s[3]))
    @assert isfinite(ω)

    g1 = one(T)/((ω-one(T))*s[2])
    g2 = g1 + g1*logsafe(ω*s[2]/s[3]) - one(T)/s[2]
    g3 = ω/((one(T) - ω)*s[3])

    return SVector(g1,g2,g3)

end

# # 3rd-order correction at the point z.  Output is η.
# #
# # η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + 
# #      dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
# #
# # where :
# # Hψ = [  1/z[1]    0   -1/z[3];
# #           0       0   0;
# #         -1/z[3]   0   z[1]/(z[3]*z[3]);]
# # dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]); 
# #            0; 
# #           (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
# #
# # dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]) ; 
# #            0; 
# #           -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
# # Hψv = Hψ*v
# # Hψu = Hψ*u
# #gψ is used inside η

@inline function higher_correction_exp!(
    H::AbstractArray{T},
    z::AbstractVector{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}
 
    #solve H*u = ds
    cholH = @MArray T[0.0 0.0 0.0;0.0 0.0 0.0;0.0 0.0 0.0]
    issuccess = cholesky_3x3_explicit_factor!(cholH,H)
    if issuccess 
        u = cholesky_3x3_explicit_solve!(cholH,ds)
    else 
        return SVector(zero(T),zero(T),zero(T))
    end
    
    η[2] = one(T)
    η[3] = -z[1]/z[3]    # gradient of ψ
    η[1] = logsafe(η[3])

    ψ = z[1]*η[1]-z[1]+z[2]

    dotψu = _dot_xy_gpu_3(η,u)
    dotψv = _dot_xy_gpu_3(η,v)

    coef = ((u[1]*(v[1]/z[1] - v[3]/z[3]) + u[3]*(z[1]*v[3]/z[3] - v[1])/z[3])*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
    @inbounds for i = 1:3
        η[i] *= coef
    end

    inv_ψ2 = one(T)/ψ/ψ

    # efficient implementation for η above
    η[1] += (1/ψ - 2/z[1])*u[1]*v[1]/(z[1]*z[1]) - u[3]*v[3]/(z[3]*z[3])/ψ + dotψu*inv_ψ2*(v[1]/z[1] - v[3]/z[3]) + dotψv*inv_ψ2*(u[1]/z[1] - u[3]/z[3])
    η[3] += 2*(z[1]/ψ-1)*u[3]*v[3]/(z[3]*z[3]*z[3]) - (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3])/ψ + dotψu*inv_ψ2*(z[1]*v[3]/(z[3]*z[3]) - v[1]/z[3]) + dotψv*inv_ψ2*(z[1]*u[3]/(z[3]*z[3]) - u[1]/z[3])

    @inbounds for i = 1:3
        η[i] /= 2
    end

    # coercing to an SArray means that the MArray we computed 
    # locally in this function is seemingly not heap allocated 
    SVector(η)
end


# update gradient and Hessian at dual z
@inline function update_dual_grad_H_exp(
    grad::AbstractVector{T},
    H::AbstractArray{T},
    z::AbstractVector{T}
) where {T}

    l = logsafe(-z[3]/z[1])
    r = -z[1]*l-z[1]+z[2]

    # compute the gradient at z
    c2 = one(T)/r

    grad[1] = c2*l - 1/z[1]
    grad[2] = -c2
    grad[3] = (c2*z[1]-1)/z[3]

    # compute the Hessian at z
    H[1,1] = ((r*r-z[1]*r+l*l*z[1]*z[1])/(r*z[1]*z[1]*r))
    H[1,2] = (-l/(r*r))
    H[2,1] = H[1,2]
    H[2,2] = (1/(r*r))
    H[1,3] = ((z[2]-z[1])/(r*r*z[3]))
    H[3,1] = H[1,3]
    H[2,3] = (-z[1]/(r*r*z[3]))
    H[3,2] = H[2,3]
    H[3,3] = ((r*r-z[1]*r+z[1]*z[1])/(r*r*z[3]*z[3])) 
    
    return nothing
end



# # ω(z) is the Wright-Omega function
# # Computes the value ω(z) defined as the solution y to
# # y+log(y) = z for reals z>=1.
# #
# # Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
# #  Algorithms for Unsymmetric Cone Optimization and an
# #  Implementation for Problems with the Exponential Cone 
# #  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

function _wright_omega_gpu(z::T) where {T}

    if z < zero(T)
        return Inf
    end

   if z < one(T) + T(π)     
        #Initialize with the taylor series
        zm1 = z - one(T)
        p = zm1            #(z-1)
        w = 1+0.5*p
        p *= zm1         #(z-1)^2
        w += (1/16.0)*p
        p *= zm1          #(z-1)^3
        w -= (1/192.0)*p
        p *= zm1          #(z-1)^4
        w -= (1/3072.0)*p
        p *= zm1         #(z-1)^5
        w += (13/61440.0)*p
    else
        # Initialize with:
        # w(z) = z - log(z) + 
        #        log(z)/z + 
        #        log(z)/z^2(log(z)/2-1) + 
        #        log(z)/z^3(1/3log(z)^2-3/2log(z)+1)

        logz = logsafe(z)
        zinv = inv(z)
        w = z - logz

        # add log(z)/z 
        q = logz*zinv  # log(z)/z 
        w += q

        # add log(z)/z^2(log(z)/2-1)
        q *= zinv      # log(z)/(z^2) 
        w += q * (logz/2 - one(T))

        # add log(z)/z^3(1/3log(z)^2-3/2log(z)+1)
        q * zinv       # log(z)/(z^3) 
        w += q * (logz*logz/3. - (3/2.)*logz + one(T))

    end

    # Initialize the residual
    r = z - w - logsafe(w)

    # Santiago suggests two refinement iterations only
    @inbounds for i = 1:2
        wp1 = (w + one(T))
        t = wp1 * (wp1 + (2. * r)/3.0 )
        w *= 1 + (r/wp1) * ( t - 0.5 * r) / (t - r)
        r = (2*w*w-8*w-1)/(72.0*(wp1*wp1*wp1*wp1*wp1*wp1))*r*r*r*r
    end 

    return w
end