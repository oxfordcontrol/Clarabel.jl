# ----------------------------------------------------
# Exponential Cone
# ----------------------------------------------------

# degree of the cone
dim(K::ExponentialCone{T}) where {T} = K.dim
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
    μ::T
) where {T}
    #update both gradient and Hessian for function f*(z) at the point z
    muHessianF(K,z,K.μH,μ)
    GradF(K,z,K.grad)
    K.z .= z
end

# return μH*(z) for exponetial cone
function get_WtW_block!(
    K::ExponentialCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    _pack_triu(WtWblock,K.μH)

end

# return x = y for unsymmetric cones
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

#  unsymmetric initialization
function unsymmetric_init!(
   K::ExponentialCone{T},
   s::AbstractVector{T},
   z::AbstractVector{T}
) where{T}

    s[1] = one(T)*(-1.051383945322714)
    s[2] = one(T)*(0.556409619469370)
    s[3] = one(T)*(1.258967884768947)

    @. z = s

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
    # η = similar(dz)
    # higherCorrection!(K,η,step_s,step_z)             #3rd order correction requires input variables.z
    # @. dz = η + σμ*K.grad


    @inbounds for i = 1:3
        dz[i] = K.grad[i]*σμ                 #dz <- σμ*g(z)
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

    mul!(ls,K.μH,lz,-one(T),zero(T))

end

#return maximum allowable step length while remaining in the exponential cone
function unsymmetric_step_length(
    K::ExponentialCone{T},
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

    αz = _step_length_exp_dual(K.vecWork,dz,z,α,scaling)
    αs = _step_length_exp_primal(K.vecWork,ds,s,α,scaling)

    return min(αz,αs)
end


# find the maximum step length α≥0 so that
# s + α*ds stays in the exponential cone
function _step_length_exp_primal(
    ws::AbstractVector{T},
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α::T,
    scaling::T
) where {T}

    # @. ws = s + α*ds
    @inbounds for i = 1:3
        ws[i] = s[i] + α*ds[i]                
    end

    while !checkExpPrimalFeas(ws)
        # NB: need to be tackled in a smarter way
        if (α < 1e-4)
            error("Expcone's step size fails in primal feasibility check!")
        end
        α *= scaling    #backtrack line search
        # @. ws = s + α*ds
        @inbounds for i = 1:3
            ws[i] = s[i] + α*ds[i]                
        end
    end

    return α
end
# z + α*dz stays in the dual exponential cone
function _step_length_exp_dual(
    ws::AbstractVector{T},
    dz::AbstractVector{T},
    z::AbstractVector{T},
    α::T,
    scaling::T
) where {T}

    # NB: additional memory, may need to remove it later
    # @. ws = z + α*dz
    @inbounds for i = 1:3
        ws[i] = z[i] + α*dz[i]                
    end

    while !checkExpDualFeas(ws)
        if (α < 1e-4)
            error("Expcone's step size fails in dual feasibility check!")
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
# Basic operations for exponential Cones
# Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# As in ECOS, we use the dual barrier function: f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3):
# Evaluates the gradient of the dual exponential cone ∇f*(z) at z, and stores the result at g
function GradF(
    K::ExponentialCone{T},
    z::AbstractVector{T},
    g::AbstractVector{T}
) where {T}

    c1 = log(-z[3]/z[1])
    c2 = 1/(-z[1]*c1-z[1]+z[2])

    g[1] = c2*c1 - 1/z[1]
    g[2] = -c2
    g[3] = (c2*z[1]-1)/z[3]

end

# Evaluates the Hessian of the dual exponential cone barrier at z and stores the upper triangular part of the matrix μH*(z)
# NB:could reduce H to an upper triangular matrix later, remove duplicate updates
function muHessianF(
    K::ExponentialCone{T},
    z::AbstractVector{T},
    H::AbstractMatrix{T},
    μ::T
) where {T}
    # y = z1; z = z2; x = z3;
    # l = log(-z3/z1);
    # r = -z1*l-z1+z2;
    # Problematic Hessian
    # Hessian = [1/z1^2 - 1/(r*z1) + l^2/r^2     -l/r^2     1/(r*z3) + (l*z1)/(r^2*z3);
    #            -l/r^2                           1/r^2                  -z1/(r^2*z3);
    #            1/(r*z3) + (l*z1)/(r^2*z3)   -z1/(r^2*z3)   1/z3^2 - z1/(r*z3^2) + z1^2/(r^2*z3^2)]

    l = log(-z[3]/z[1])
    r = -z[1]*l-z[1]+z[2]

    H[1,1] = ((r*r-z[1]*r+l*l*z[1]*z[1])/(r*z[1]*z[1]*r))
    H[1,2] = (-l/(r*r))
    H[2,1] = H[1,2]
    H[2,2] = (1/(r*r))
    H[1,3] = ((z[2]-z[1])/(r*r*z[3]))
    H[3,1] = H[1,3]
    H[2,3] = (-z[1]/(r*r*z[3]))
    H[3,2] = H[2,3]
    H[3,3] = ((r*r-z[1]*r+z[1]*z[1])/(r*r*z[3]*z[3]))

    H .*= μ
end

function f_sum(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    barrier = T(0)

    # Dual barrier
    l = log(-z[3]/z[1])
    barrier += -log(z[2]-z[1]-z[1]*l)-log(-z[1])-log(z[3])

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # f(s) = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3, where barω = ω(1 - s1/s2 - log(s2) - log(s3))
    # NB: ⟨s,g(s)⟩ = -3 = - ν 
    o = WrightOmega(1-s[1]/s[2]-log(s[2]/s[3]))
    o = (o-1)*(o-1)/o
    barrier += -log(o)-2*log(s[2])-log(s[3]) - 3

    return barrier
end

# Returns true if s is primal feasible
function checkExpPrimalFeas(s::AbstractVector{T}) where {T}

    if (s[3] > 0 && s[2] > 0)   #feasible
        res = s[2]*log(s[3]/s[2]) - s[1]
        if (res > 0)
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function checkExpDualFeas(z::AbstractVector{T}) where {T}

    if (z[3] > 0 && z[1] < 0)
        res = z[2] - z[1] - z[1]*log(-z[3]/z[1])
        if (res > 0)
            return true
        end
    end

    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function GradPrim(
    K::ExponentialCone{T},
    g::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    o = WrightOmega(1-s[1]/s[2]-log(s[2]/s[3]))
    
    g[1] = one(T)/((o-1)*s[2])
    g[2] = g[1] + g[1]*log(o*s[2]/s[3]) - one(T)/s[2]
    g[3] = o/((one(T) - o)*s[3])

end

# ω(z) is the Wright-Omega function
# Computes the value ω(z) defined as the solution y to
# the equation y+log(y) = z ONLY FOR z real and z>=1.
# NB::the code is from ECOS solver, which comes from Santiago's thesis, "Algorithms for Unsymmetric Cone Optimization and an Implementation for Problems with the Exponential Cone"
function WrightOmega(z::T) where {T}
    w  = T(0);
    r  = T(0);
    q  = T(0);
    zi = T(0);

	if(z< T(0))
        throw(error("β not in supported range", z)); #Fail if the input is not supported
    end

	if(z<T(1)+π)      #If z is between 0 and 1+π
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
        r = log(z);
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
    r = z-w-log(w);

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


######################################
# May need to be removed later
######################################


# 3rd-order correction at the point z, w.r.t. directions u,v and then save it to η
# NB: not so effective at present
function higherCorrection!(
    K::ExponentialCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    v::AbstractVector{T}
) where {T}

    # u for H^{-1}*Δs
    μH = K.μHWork
    u = K.vecWork
    F = K.FWork
    z = K.z

    # recompute Hessian
    muHessianF(K,z,μH,one(T))

    if F === nothing
        F = lu(μH, check= false)
    else
        F = lu!(μH, check= false)
    end

    if !issuccess(F)
        increase_diag!(μH)
        F = lu!(μH)
    end

    ldiv!(u,F,ds)    #equivalent to Hinv*ds

    l = log(-z[3]/z[1])
    ψ = -z[1]*l-z[1]+z[2]

    # memory allocation
    gψ = K.gradWork
    Hψ = K.μHWork

    gψ[1] = -l
    gψ[2] = 1
    gψ[3] = -z[1]/z[3]    # gradient of ψ

    # Hψ = [  1/z[1]    0   -1/z[3];
    #           0       0   0;
    #         -1/z[3]   0   z[1]/(z[3]*z[3]);]
    Hψ[1,1] = 1/z[1]
    Hψ[1,2] = 0
    Hψ[2,1] = Hψ[1,2]
    Hψ[1,3] = -1/z[3]
    Hψ[3,1] = Hψ[1,3]
    Hψ[2,2] = 0
    Hψ[2,3] = 0
    Hψ[3,2] = Hψ[2,3]
    Hψ[3,3] = z[1]/(z[3]*z[3])

    dotψu = dot(gψ,u)
    dotψv = dot(gψ,v)

    dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]); 0; (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
    dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]); 0; -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
    Hψv = Hψ*v
    Hψu = Hψ*u

    η .= (dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ + dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv
    η ./= -2

end

# check neighbourhood
function _check_neighbourhood(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    η::T
) where {T}

    # to = TimerOutput()

    grad = K.gradWork
    H = K.μHWork
    F = K.FWork
    tmp = K.vecWork

    # compute gradient at z
    l = log(-z[3]/z[1])
    r = -z[1]*l-z[1]+z[2]

    grad[1] = r*l - 1/z[1]
    grad[2] = -r
    grad[3] = (r*z[1]-1)/z[3]

    # compute inverse Hessian 
    H[1,1] = z[1]^2*(z[1]-r)
    H[1,2] = z[1]^2*(z[1] - l*r + l*z[1])
    H[2,1] = H[1,2]
    H[2,2] = -l^2*r*z[1]^2 + (l+2)*l*z[1]^3 - r^3 + 2*r^2*z[1] - r*z[1]^2 + z[1]^3
    H[1,3] = z[1]^2*z[3]
    H[3,1] = H[1,3]
    H[2,3] = z[1]*z[3]*(z[1] - r + l*z[1])
    H[3,2] = H[2,3]
    H[3,3] = z[3]^2*(z[1] - r)
    H ./= (2*z[1] - r)

    # grad as a workspace for s/μ + grad
    axpy!(1/μ, s, grad)
    if (dot(grad,H,grad) < η)
        return true
    end

    # NB::Currently, direct Hinv is horrible due to potential numerical errors,
    # YC:: 1) it implies that we'd better to use only first-order information for centrality check
    #      2) also, we should allow larger neighbourhood for a shorter iteration number

    # # grad as a workspace for s + μ*grad
    # axpby!(one(T), s, μ, grad)
    # if (dot(grad,H,grad)/μ^2 < η)
        # return true
    # end

    # println("away from central path due to cone with ", dot(grad,H,grad))

    # # compute gradient and Hessian at z
    # GradF(K, z, grad)
    # muHessianF(K, z, H, μ)

    # if F === nothing
    #     F = lu(H, check= false)
    # else
    #     F = lu!(H, check= false)
    # end

    # if !issuccess(F)
    #     increase_diag!(H)
    #     F = lu!(H)
    # end

    # # grad as a workspace for s + μ*grad
    # axpby!(one(T), s, μ, grad)

    # ldiv!(tmp,F,grad)
    # if (dot(tmp,grad)/μ < η)
    #     println("away from central path due to cone with ", norm(dot(tmp,grad)/μ))
    #     return true
    # end

    # println("away from central path due to cone with ", norm(dot(tmp,grad)/μ))

    return false
end