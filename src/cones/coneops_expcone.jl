# ----------------------------------------------------
# Exponential Cone
# ----------------------------------------------------

# degree of the cone.  Always 3
dim(K::ExponentialCone{T}) where {T} = 3
degree(K::ExponentialCone{T}) where {T} = dim(K)
numel(K::ExponentialCone{T}) where {T} = dim(K)

is_symmetric(::ExponentialCone{T}) where {T} = false

#exponential cone returns a dense WtW block
function WtW_is_diagonal(
    K::ExponentialCone{T}
) where{T}
    return false
end

function update_scaling!(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}
    # update both gradient and Hessian for function f*(z) at the point z
    # NB: the update order can't be switched as we reuse memory in the Hessian computation
    # Hessian update
    update_grad_HBFGS(K,s,z,μ,scaling_strategy)

    # K.z .= z
    @inbounds for i = 1:3
        K.z[i] = z[i]
    end
end

# return μH*(z) for exponetial cone
function get_WtW_block!(
    K::ExponentialCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.HBFGS)
    _pack_triu(WtWblock,K.HBFGS)

end

# return x = y for asymmetric cones
function affine_ds!(
    K::ExponentialCone{T},
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
   K::ExponentialCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    s[1] = one(T)*(-1.051383945322714)
    s[2] = one(T)*(0.556409619469370)
    s[3] = one(T)*(1.258967884768947)

    #@. z = s
    @inbounds for i = 1:3
        z[i] = s[i]
    end

   return nothing
end

# compute ds in the combined step where μH(z)Δz + Δs = - ds
function combined_ds!(
    K::ExponentialCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}
    η = K.grad_work
    higher_correction!(K,η,step_s,step_z)             #3rd order correction requires input variables.z

    @inbounds for i = 1:3
        dz[i] = K.grad[i]*σμ - η[i]
    end

    return nothing
end

# compute the generalized step ds
function Wt_λ_inv_circ_ds!(
    K::ExponentialCone{T},
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
    K::ExponentialCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # mul!(ls,K.HBFGS,lz,-one(T),zero(T))
    H = K.HBFGS
    @inbounds for i = 1:3
        ls[i] = - H[i,1]*lz[1] - H[i,2]*lz[2] - H[i,3]*lz[3]
    end

end

#return maximum allowable step length while remaining in the exponential cone
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

    αz = _step_length_expcone(K.vec_work, dz, z, αmax, backtrack, is_dual_feasible_expcone)
    αs = _step_length_expcone(K.vec_work, ds, s, αmax, backtrack, is_primal_feasible_expcone)

    return (αz,αs)
end

###############################################
# Basic operations for exponential Cones
# Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# As in ECOS, we use the dual barrier function: f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3):

function compute_centrality(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    barrier = zero(T)

    # Dual barrier
    l = logsafe(-z[3]/z[1])
    barrier += -logsafe(-z[3]*z[1]) - logsafe(z[2]-z[1]-z[1]*l) 

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # f(s) = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3, where barω = ω(1 - s1/s2 - log(s2) - log(s3))
    # NB: ⟨s,g(s)⟩ = -3 = - ν
    o = wright_omega(1-s[1]/s[2]-logsafe(s[2]/s[3]))
    o = (o-1)*(o-1)/o
    barrier += -logsafe(o)-2*logsafe(s[2])-logsafe(s[3]) - 3

    return barrier
end

# Returns true if s is primal feasible
function is_primal_feasible_expcone(s::AbstractVector{T}) where {T}

    if (s[3] > 0 && s[2] > 0)   #feasible
        res = s[2]*logsafe(s[3]/s[2]) - s[1]
        if (res > 0)
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
function is_dual_feasible_expcone(z::AbstractVector{T}) where {T}

    if (z[3] > 0 && z[1] < 0)
        res = z[2] - z[1] - z[1]*logsafe(-z[3]/z[1])
        if (res > 0)
            return true
        end
    end
    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function gradient_primal(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    g::AbstractVector{T}
) where {T}

    o = wright_omega(1-s[1]/s[2]-logsafe(s[2]/s[3]))

    g[1] = one(T)/((o-1)*s[2])
    g[2] = g[1] + g[1]*logsafe(o*s[2]/s[3]) - one(T)/s[2]
    g[3] = o/((one(T) - o)*s[3])

end

# ω(z) is the Wright-Omega function
# Computes the value ω(z) defined as the solution y to
# the equation y+log(y) = z ONLY FOR z real and z>=1.
# NB::the code follows the ECOS solver, which comes from Santiago's thesis,
# "Algorithms for Unsymmetric Cone Optimization and an Implementation for Problems with the Exponential Cone"
function wright_omega(z::T) where {T}
    w  = zero(T);
    r  = zero(T);
    q  = zero(T);
    zi = zero(T);

	if(z< zero(T))
        throw(error("β not in supported range", z)); #Fail if the input is not supported
    end

	if(z<one(T)+π)      #If z is between 0 and 1+π
        q = z-1;
        r = q;
        w = 1+0.5*r;
        r *= q;
        w += 1/16.0*r;
        r *= q;
        w -= 1/192.0*r;
        r *= q;
        w -= 1/3072.0*q;
        r *= q;                 #(z-1)^5
        w += 13/61440.0*q;
        #Initialize with the taylor series
    else
        r = logsafe(z);
        q = r;
        zi  = one(T)/z;
        w = z-r;
        q = r*zi;
        w += q;
        q = q*zi;
        w += q*(0.5*r-1);
        q = q*zi;
        w += q*(1/3.0*r*r-3.0/2.0*r+1);
        # Initialize with w(z) = z-r+r/z^2(r/2-1)+r/z^3(1/3ln^2z-3/2r+1)
    end

    # FSC iteration
    # Initialize the residual
    r = z-w-logsafe(w);

    z = (1+w);
    q = z+2/3.0*r;
    w *= 1+r/z*(z*q-0.5*r)/(z*q-r);
    r = (2*w*w-8*w-1)/(72.0*(z*z*z*z*z*z))*r*r*r*r;
    # Check residual
    # if(r<1.e-16) return w;
    # Just do two rounds
    z = (1+w);
    q = z+2/3.0*r;
    w *= 1+r/z*(z*q-0.5*r)/(z*q-r);
    r = (2*w*w-8*w-1)/(72.0*(z*z*z*z*z*z))*r*r*r*r;

    return w;
end

# 3rd-order correction at the point z, w.r.t. directions u,v and then save it to η
# NB: not so effective at present
function higher_correction!(
    K::ExponentialCone{T},
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
    

    η[2] = one(T)
    η[3] = -z[1]/z[3]    # gradient of ψ
    η[1] = logsafe(η[3])

    ψ = z[1]*η[1]-z[1]+z[2]

    dotψu = dot(η,u)
    dotψv = dot(η,v)

    # 3rd order correction: η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
    # where :
    # Hψ = [  1/z[1]    0   -1/z[3];
    #           0       0   0;
    #         -1/z[3]   0   z[1]/(z[3]*z[3]);]
    # dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]); 0; (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
    # dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]) ; 0; -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
    # Hψv = Hψ*v
    # Hψu = Hψ*u
    #gψ is used inside η

    coef = ((u[1]*(v[1]/z[1] - v[3]/z[3]) + u[3]*(z[1]*v[3]/z[3] - v[1])/z[3])*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)
    @inbounds for i = 1:3
        η[i] *= coef
    end

    inv_ψ2 = 1/ψ/ψ

    η[1] += (1/ψ - 2/z[1])*u[1]*v[1]/(z[1]*z[1]) - u[3]*v[3]/(z[3]*z[3])/ψ + dotψu*inv_ψ2*(v[1]/z[1] - v[3]/z[3]) + dotψv*inv_ψ2*(u[1]/z[1] - u[3]/z[3])
    η[3] += 2*(z[1]/ψ-1)*u[3]*v[3]/(z[3]*z[3]*z[3]) - (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3])/ψ + dotψu*inv_ψ2*(z[1]*v[3]/(z[3]*z[3]) - v[1]/z[3]) + dotψv*inv_ψ2*(z[1]*u[3]/(z[3]*z[3]) - u[1]/z[3])

    @inbounds for i = 1:3
        η[i] /= 2
    end

end


######################################
# primal-dual scaling
######################################

# Implementation sketch
# 1) only need to replace μH by W⊤W,
#   where W⊤W is the primal-dual scaling matrix generated by BFGS, i.e. W⊤W*[z,̃z] = [s,̃s]
#   ̃z = -f'(s), ̃s = - f*'(z)

# NB: better to create two matrix spaces, one for the Hessian at z, H*(z), and another one for BFGS (primal-dual scaling) matrix, H-BFGS(z,s)

function update_grad_HBFGS(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}
    # reuse memory
    st = K.grad
    zt = K.vec_work
    δs = K.grad_work
    tmp = K.z
    H = K.H
    HBFGS = K.HBFGS

    # Hessian computation, compute μ locally
    l = logsafe(-z[3]/z[1])
    r = -z[1]*l-z[1]+z[2]

    # compute the gradient at z
    # gradient_f(K,z,st)  #st (K.grad) is indeed the gradient at z
    c2 = one(T)/r

    st[1] = c2*l - 1/z[1]
    st[2] = -c2
    st[3] = (c2*z[1]-1)/z[3]

    # compute_Hessian(K,z,H)
    H[1,1] = ((r*r-z[1]*r+l*l*z[1]*z[1])/(r*z[1]*z[1]*r))
    H[1,2] = (-l/(r*r))
    H[2,1] = H[1,2]
    H[2,2] = (1/(r*r))
    H[1,3] = ((z[2]-z[1])/(r*r*z[3]))
    H[3,1] = H[1,3]
    H[2,3] = (-z[1]/(r*r*z[3]))
    H[3,2] = H[2,3]
    H[3,3] = ((r*r-z[1]*r+z[1]*z[1])/(r*r*z[3]*z[3]))    

    #Use the local mu with primal dual strategy.  Otherwise 
    #we use the global one 
    if(scaling_strategy == PrimalDual::ScalingStrategy)
        μ = dot(z,s)/3
    end 

    # HBFGS .= μ*H
    @inbounds for i = 1:3
        @inbounds for j = 1:3
            HBFGS[i,j] = μ*H[i,j]
        end
    end

    # compute zt,st,μt locally
    # NB: zt,st have different sign conventiion w.r.t to Mosek paper

    # use the dual scaling
    if scaling_strategy == Dual
        return nothing
    end
    gradient_primal(K,s,zt)
 
    
    μt = dot(zt,st)/3

    # δs = s + μ*st
    # δz = z + μ*zt
    @inbounds for i = 1:3
        δs[i] = s[i] + μ*st[i]
    end

    de1 = μ*μt-1
    de2 = dot(zt,H,zt) - 3*μt*μt

    if (de1 > eps(T) && de2 > eps(T))
        # tmp = μt*st - H*zt
        @inbounds for i = 1:3
            tmp[i] = μt*st[i] - H[i,1]*zt[1] - H[i,2]*zt[2] - H[i,3]*zt[3]
        end

        # store (s - μ*st + δs/de1) into zt
        @inbounds for i = 1:3
            zt[i] = s[i] - μ*st[i] + δs[i]/de1
        end

        # Hessian HBFGS:= μ*H + 1/(2*μ*3)*δs*(s - μ*st + δs/de1)' + 1/(2*μ*3)*(s - μ*st + δs/de1)*δs' - μ/de2*tmp*tmp'
        coef1 = 1/(2*μ*3)
        coef2 = μ/de2
        # HBFGS .+= coef1*δs*zt' + coef1*zt*δs' - coef2*tmp*tmp'
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                HBFGS[i,j] += coef1*δs[i]*zt[j] + coef1*zt[i]*δs[j] - coef2*tmp[i]*tmp[j]
            end
        end
    else
        return nothing
    end
end

