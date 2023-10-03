# ----------------------------------------------------
# Exponential Cone
# ----------------------------------------------------

# degree of the cone.  Always 3
degree(K::ExponentialCone{T}) where {T} = 3
numel(K::ExponentialCone{T}) where {T} = 3

is_symmetric(::ExponentialCone{T}) where {T} = false

function margins(
    K::ExponentialCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone,
) where{T}

    # We should never end up computing margins for this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function scaled_unit_shift!(
    K::ExponentialCone{T},
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
    pack_triu(Hsblock,K.Hs)

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

    η = similar(K.grad); η .= zero(T)

    #3rd order correction requires input variables z
    higher_correction!(K,η,step_s,step_z)             

    @inbounds for i = 1:3
        shift[i] = K.grad[i]*σμ - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::ExponentialCone{T},
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

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    work = similar(K.grad); work .= zero(T)

    is_prim_feasible_fcn = s -> is_primal_feasible(K,s)
    is_dual_feasible_fcn = s -> is_dual_feasible(K,s)
    
    αz = backtrack_search(K, dz, z, αmax, αmin, step, is_dual_feasible_fcn, work)
    αs = backtrack_search(K, ds, s, αmax, αmin, step, is_prim_feasible_fcn, work)

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

    barrier += barrier_dual(K, cur_z)
    barrier += barrier_primal(K, cur_s)

    return barrier
end


# -----------------------------------------
# nonsymmetric cone operations for exponential cones
#
# Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# We use the dual barrier function: 
# f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
# -----------------------------------------


@inline function barrier_dual(
    ::ExponentialCone{T},
    z::Union{AbstractVector{T}, NTuple{3,T}}
) where {T}

    # Dual barrier
    l = logsafe(-z[3]/z[1])
    return -logsafe(-z[3]*z[1]) - logsafe(z[2]-z[1]-z[1]*l) 

end 

@inline function barrier_primal(
    ::ExponentialCone{T},
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
function is_primal_feasible(
    ::ExponentialCone{T},
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
function is_dual_feasible(
    ::ExponentialCone{T},
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
function gradient_primal(
    ::ExponentialCone{T},
    s::Union{AbstractVector{T}, NTuple{3,T}},
) where {T}

    ω = _wright_omega(1-s[1]/s[2]-logsafe(s[2]/s[3]))

    g1 = one(T)/((ω-one(T))*s[2])
    g2 = g1 + g1*logsafe(ω*s[2]/s[3]) - one(T)/s[2]
    g3 = ω/((one(T) - ω)*s[3])

    SVector(g1,g2,g3)

end

# 3rd-order correction at the point z.  Output is η.
#
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

function higher_correction!(
    K::ExponentialCone{T},
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
    
    η[2] = one(T)
    η[3] = -z[1]/z[3]    # gradient of ψ
    η[1] = logsafe(η[3])

    ψ = z[1]*η[1]-z[1]+z[2]

    dotψu = dot(η,u)
    dotψv = dot(η,v)

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
    SArray(η)
end


# update gradient and Hessian at dual z
function update_dual_grad_H(
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



# ω(z) is the Wright-Omega function
# Computes the value ω(z) defined as the solution y to
# y+log(y) = z for reals z>=1.
#
# Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
#  Algorithms for Unsymmetric Cone Optimization and an
#  Implementation for Problems with the Exponential Cone 
#  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

function _wright_omega(z::T) where {T}

    if z < zero(T)
        throw(error("argument not in supported range : ", z)); 
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