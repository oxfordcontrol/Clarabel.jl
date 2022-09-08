# ----------------------------------------------------
# Exponential Cone
# ----------------------------------------------------

# degree of the cone.  Always 3
dim(K::ExponentialCone{T}) where {T} = 3
degree(K::ExponentialCone{T}) where {T} = dim(K)
numel(K::ExponentialCone{T}) where {T} = dim(K)

is_symmetric(::ExponentialCone{T}) where {T} = false

function shift_to_cone!(
    K::ExponentialCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function unit_initialization!(
   K::ExponentialCone{T},
   z::AbstractVector{T},
   s::AbstractVector{T}
) where{T}

    s[1] = T(-1.051383945322714)
    s[2] = T(0.556409619469370)
    s[3] = T(1.258967884768947)

    #@. z = s
    @inbounds for i = 1:3
        z[i] = s[i]
    end

   return nothing
end

function set_identity_scaling!(
    K::ExponentialCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    _update_dual_grad_H(K,z)

    # update the scaling matrix Hs
    _update_Hs(K,s,z,μ,scaling_strategy)

    # K.z .= z
    @inbounds for i = 1:3
        K.z[i] = z[i]
    end
end

function Hs_is_diagonal(
    K::ExponentialCone{T}
) where{T}
    return false
end

# return μH*(z) for exponential cone
function get_Hs!(
    K::ExponentialCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    # stores triu(K.Hs) into a vector
    _pack_triu(Hsblock,K.Hs)

end

# compute the product y = Hₛx = μH(z)x
function mul_Hs!(
    K::ExponentialCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    Hs = K.Hs
    @inbounds for i = 1:3
        y[i] =  Hs[i,1]*x[1] + Hs[i,2]*x[2] + Hs[i,3]*x[3]
    end

end

function affine_ds!(
    K::ExponentialCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:3
        ds[i] = s[i]
    end

end

function combined_ds_shift!(
    K::ExponentialCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    η = @MVector zeros(T,3)

    #3rd order correction requires input variables z
    _higher_correction!(K,η,step_s,step_z)             

    @inbounds for i = 1:3
        shift[i] = K.grad[i]*σμ - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::ExponentialCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    @inbounds for i = 1:3
        out[i] = ds[i]
    end

    return nothing
end

#return maximum step length while staying in exponential cone
function step_length(
    K::ExponentialCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T
) where {T}

    backtrack = settings.linesearch_backtrack_step
    αmin      = settings.min_terminate_step_length
    
    αz = _step_length_3d_cone(dz, z, αmax, αmin,  backtrack, _is_dual_feasible_expcone)
    αs = _step_length_3d_cone(ds, s, αmax, αmin,  backtrack, _is_primal_feasible_expcone)

    return (αz,αs)
end

function compute_barrier(
    K::ExponentialCone{T},
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


# -----------------------------------------
# internal operations for exponential cones
#
# Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# We use the dual barrier function: 
# f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
# -----------------------------------------


@inline function _barrier_dual(
    K::ExponentialCone{T},
    z::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Dual barrier
    l = logsafe(-z[3]/z[1])
    return -logsafe(-z[3]*z[1]) - logsafe(z[2]-z[1]-z[1]*l) 

end 

@inline function _barrier_primal(
    K::ExponentialCone{T},
    s::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Primal barrier: 
    # f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    #      = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3, 
    # where barω = ω(1 - s1/s2 - log(s2) - log(s3))
    # NB: ⟨s,g(s)⟩ = -3 = - ν

    ω = _wright_omega(1-s[1]/s[2]-logsafe(s[2]/s[3]))
    ω = (ω-1)*(ω-1)/ω
   return -logsafe(ω)-2*logsafe(s[2])-logsafe(s[3]) - 3
end 



# Returns true if s is primal feasible
function _is_primal_feasible_expcone(s::AbstractVector{T}) where {T}

    if (s[3] > 0 && s[2] > zero(T))   #feasible
        res = s[2]*logsafe(s[3]/s[2]) - s[1]
        if (res > zero(T))
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
function _is_dual_feasible_expcone(z::AbstractVector{T}) where {T}

    if (z[3] > 0 && z[1] < zero(T))
        res = z[2] - z[1] - z[1]*logsafe(-z[3]/z[1])
        if (res > zero(T))
            return true
        end
    end
    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function _gradient_primal(
    K::ExponentialCone{T},
    g::Union{AbstractVector{T}, NTuple{3,T}},
    s::Union{AbstractVector{T}, NTuple{3,T}},
) where {T}

    ω = _wright_omega(1-s[1]/s[2]-logsafe(s[2]/s[3]))

    g[1] = one(T)/((ω-one(T))*s[2])
    g[2] = g[1] + g[1]*logsafe(ω*s[2]/s[3]) - one(T)/s[2]
    g[3] = ω/((one(T) - ω)*s[3])

end

# ω(z) is the Wright-Omega function
# Computes the value ω(z) defined as the solution y to
# y+log(y) = z for reals z>=1.
#
# Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
#  Algorithms for Unsymmetric Cone Optimization and an
#  Implementation for Problems with the Exponential Cone 
#  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

function _wright_omega(z::T) where {T}

 	if(z< zero(T))
        throw(error("argument not in supported range : ", z)); 
    end

	if(z<one(T)+π)      
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

    return w;
end

# 3rd-order correction at the point z, 
# w.r.t. directions u,v and then save it to η

function _higher_correction!(
    K::ExponentialCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs
    H = K.H_dual
    u = @MVector zeros(T,3)
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
    

    η[2] = one(T)
    η[3] = -z[1]/z[3]    # gradient of ψ
    η[1] = logsafe(η[3])

    ψ = z[1]*η[1]-z[1]+z[2]

    dotψu = dot(η,u)
    dotψv = dot(η,v)

    # 3rd order correction: 
    # η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + 
    #      dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
    #
    # where :
    # Hψ = [  1/z[1]    0   -1/z[3];
    #           0       0   0;
    #         -1/z[3]   0   z[1]/(z[3]*z[3]);]
    # dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]); 
    #            0; 
    #           (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
    #
    # dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]) ; 
    #            0; 
    #           -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
    # Hψv = Hψ*v
    # Hψu = Hψ*u
    #gψ is used inside η

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

end


#-------------------------------------
# primal-dual scaling
#-------------------------------------

# Implementation sketch
# 1) only need to replace μH by W^TW, where
#    W^TW is the primal-dual scaling matrix 
#    generated by BFGS, i.e. W^T W*[z,\tilde z] = [s,\tile s]
#   \tilde z = -f'(s), \tilde s = - f*'(z)


# update the scaling matrix Hs
function _update_Hs(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    #Choose the scaling strategy
    if(scaling_strategy == Dual::ScalingStrategy)
        # Dual scaling: Hs = μ*H
        _use_dual_scaling(K,K.Hs,K.H_dual,μ)
    else
        # Primal-dual scaling
        _use_primal_dual_scaling(K,K.Hs,K.H_dual,s,z)
    end 

end

# update gradient and Hessian at dual z
function _update_dual_grad_H(
    K::ExponentialCone{T},
    z::AbstractVector{T}
) where {T}
    grad = K.grad
    H = K.H_dual

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

# use the dual scaling strategy
function _use_dual_scaling(
    K::ExponentialCone{T},
    Hs::AbstractMatrix{T},
    H_dual::AbstractMatrix{T},
    μ::T
) where {T}
    @inbounds for i = 1:9
        Hs[i] = μ*H_dual[i]
    end
end

# use the primal-dual scaling strategy
function _use_primal_dual_scaling(
    K::ExponentialCone{T},
    Hs::AbstractMatrix{T},
    H_dual::AbstractMatrix{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    zt = @MVector zeros(T,3)
    st = K.grad
    δs = @MVector zeros(T,3)
    tmp = @MVector zeros(T,3) #shared for δz, tmp, axis_z

    # compute zt,st,μt locally
    # NB: zt,st have different sign convention wrt Mosek paper
    _gradient_primal(K,zt,s)
    dot_sz = dot(z,s)
    μ = dot_sz/3
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
    de2 = dot(zt,H_dual,zt) - 3*μt*μt

    #if !(abs(de1) > eps(T) && abs(de2) > eps(T))
    # if !(min(abs(de1),abs(de2),abs(dot_sz),abs(dot_δsz)) > eps(T))
    if abs(de1) < sqrt(eps(T))
        # Hs = μ*H_dual when s,z are on the central path
        _use_dual_scaling(K,Hs,H_dual,μ)

        return nothing
    else
        # compute t
        # tmp = μt*st - H_dual*zt
        @inbounds for i = 1:3
            tmp[i] = μt*st[i] - H_dual[i,1]*zt[1] - H_dual[i,2]*zt[2] - H_dual[i,3]*zt[3]
        end

        # Hs as a workspace
        copyto!(Hs,H_dual)
        @inbounds for i = 1:3
            @inbounds for j = i:3
                Hs[i,j] -= st[i]*st[j]/3 + tmp[i]*tmp[j]/de2
            end
        end
        # symmetrize matrix
        Hs[2,1] = Hs[1,2]
        Hs[3,1] = Hs[1,3]
        Hs[3,2] = Hs[2,3]
        
        t = μ*norm(Hs)  #Frobenius norm

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

        # Hs = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
        @inbounds for i = 1:3
            @inbounds for j = i:3
                Hs[i,j] = s[i]*s[j]/dot_sz + δs[i]*δs[j]/dot_δsz + t*axis_z[i]*axis_z[j]
            end
        end
        # symmetrize matrix
        Hs[2,1] = Hs[1,2]
        Hs[3,1] = Hs[1,3]
        Hs[3,2] = Hs[2,3]

        return nothing
    end
end