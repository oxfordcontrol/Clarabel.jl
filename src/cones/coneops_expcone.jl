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
    muHessianF(z,K.μH,K.μHinv,μ)
    GradF(z,K.grad)
end

# return μH*(z) for exponetial cone
function get_WtW_block!(
    K::ExponentialCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #Vectorize triu(K.μH)
    WtWblock .= _pack_triu(WtWblock,K.μH)

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

    z[1] = one(T)*(-1.051383945322714)
    z[2] = one(T)*(0.556409619469370)
    z[3] = one(T)*(1.258967884768947)

   return nothing
end

# add gradient to x
function add_grad!(
    K::ExponentialCone{T},
    x::AbstractVector{T},
    α::T
) where {T}

    #e is a vector of ones, so just shift
    @. x += α*K.grad

    return nothing
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
    
    αz = _step_length_exp_dual(dz,z,α,scaling)
    αs = _step_length_exp_primal(ds,s,α,scaling)

    return min(αz,αs)
end


# find the maximum step length α≥0 so that
# s + α*ds stays in the exponential cone
function _step_length_exp_primal(
    ds::AbstractVector{T},
    s::AbstractVector{T},
    α::T,
    scaling::T
) where {T}

    # NB: additional memoory, may need to remove it later
    ws = s + α*ds

    while !checkExpPrimalFeas(ws)
        α *= scaling    #backtrack line search
        @. ws = s + α*ds
    end

    return α
end
# z + α*dz stays in the dual exponential cone
function _step_length_exp_dual(
    dz::AbstractVector{T},
    z::AbstractVector{T},
    α::T,
    scaling::T
) where {T}

    # NB: additional memoory, may need to remove it later
    ws = z + α*dz

    while !checkExpDualFeas(ws)
        α *= scaling    #backtrack line search
        @. ws = z + α*dz
    end

    return α
end



###############################################
# Basic operations for exponential Cones
# Follwing ECOS solver and Santiago's thesis
# Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
# Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
# As in ECOS, we use the dual barrier function: f*(z) = -log(z2 - z1 - z1log(z3/-z1)) - log(-z1) - log(z3):
# Evaluates the gradient of the dual exponential cone ∇f*(z) at z, and stores the result at g
function GradF(z::AbstractVector{T}, g::AbstractVector{T}) where {T}
    z1 = z[1]               #z1
    z2 = z[2]               #z2
    z3 = z[3]               #z3
    c1 = log(-z3/z1)
    c2 = 1/(-z1*c1-z1+z2)

    g[1] = c2*c1 - 1/z1
    g[2] = -c2
    g[3] = (c2*z1-1)/z3

end

# Evaluates the Hessian of the exponential dual cone barrier at z and stores the upper triangular part of the matrix μH*(z)
# NB:could reduce H to an upper triangular matrix later, remove duplicate updates
function muHessianF(z::AbstractVector{T}, H::AbstractMatrix{T}, Hinv::AbstractMatrix{T}, μ::T) where {T}
    # y = z1; z = z2; x = z3;
    # l = log(-z3/z1);
    # r = -z1*l-z1+z2;
    # Hessian = [[1/z1^2 - 1/(r*z1) + l^2/r^2,     -l/r^2,     1/(r*z3) + (l*z1)/(r^2*z3)];
    #            [-l/r^2,                           1/r^2],                  -z1/(r^2*z3)];
    #            [1/(r*z3) + (l*z1)/(r^2*z3),   -z1/(r^2*z3),  1/z3^2 - z1/(r*z3^2) + z1^2/(r^2*z3^2)]]

    z1 = z[1]               #z1
    z2 = z[2]               #z2
    z3 = z[3]               #z3
    l = log(-z3/z1)
    r = -z1*l-z1+z2

    H[1,1] = ((r*r-z1*r+l*l*z1*z1)/(r*z1*z1*r))
    H[1,2] = (-l/(r*r))
    H[2,1] = H[1,2]
    H[2,2] = (1/(r*r))
    H[1,3] = ((z2-z1)/(r*r*z3))
    H[3,1] = H[1,3]
    H[2,3] = (-z1/(r*r*z3))
    H[3,2] = H[2,3]
    H[3,3] = ((r*r-z1*r+z1*z1)/(r*r*z3*z3))

    H .*= μ

    # compute (μH)^{-1}, 3x3 inverse is easy
    Hinv .= inv(H) 
end

# # f(s) = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3, where barω = ω(1 - s1/s2 - log(s2) - log(s3))
# function f_sum(
#     K::ExponentialCone{T}, 
#     s::AbstractVector{T}, 
#     z::AbstractVector{T}
# ) where {T}

#     z1 = z[1]
#     z2 = z[2]
#     z3 = z[3]
#     s1 = s[1]
#     s2 = s[2]
#     s3 = s[3]

#     barrier = T(0)

#     # Dual barrier
#     l = log(-z3/z1)
#     barrier += -log(z2-z1-z1*l)-log(-z1)-log(z3)

#     # Primal barrier
#     o = WrightOmega(1-s1/s2-log(s2)+log(s3))
#     o = (o-1)*(o-1)/o
#     barrier += -log(o)-2*log(s2)-log(s3)-3

#     return barrier
# end

# check neighbourhood
function _check_neighbourhood(
    K::ExponentialCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    η::T
) where {T}

    grad = zeros(T,3)
    μH = zeros(T,3,3)
    μHinv = zeros(T,3,3)

    # compute gradient and Hessian at z
    GradF(z, grad)
    muHessianF(z, μH, μHinv, μ)
    
    # grad as a workspace for s + μ*grad
    grad .*= μ
    grad .+= s

    if (norm(dot(grad,μHinv,grad)/μ) < η^2)
        return true
    end

    return false
end


# Hessian operator δ = μH*(z)[δz], where v contains μH*.
function compute_muHessianF(δ::AbstractVector{T}, H::AbstractMatrix{T}, δz::AbstractVector{T}) where {T}
    mul!(δ,H,δz)
end

# Returns true if s is primal feasible
function checkExpPrimalFeas(s::AbstractVector{T}) where {T}
    s1 = s[1]
    s2 = s[2]
    s3 = s[3]

    if (s3>0. && s2>0.)   #feasible
        res = s2*log(s3/s2) - s1
        if (res>0.)
            return true
        end
    end

    return false
end

# Returns true if s is dual feasible
function checkExpDualFeas(z::AbstractVector{T}) where {T}
    z1 = z[1]
    z2 = z[2]
    z3 = z[3]

    if (z3>0. && z1<0.)
        res = z2 - z1 - z1*log(-z3/z1)
        if (res>0.)
            return true
        end
    end

    return false
end


# ω(z) is the Wright-Omega function
# Computes the value ω(z) defined as the solution y to
# the equation y+log(y) = z ONLY FOR z real and z>=1.
# NB::copy from ECOS solver, which comes from Santiago's thesis, "Algorithms for Unsymmetric Cone Optimization and an Implementation for Problems with the Exponential Cone"
function WrightOmega(z::T) where {T}
    w  = T(0);
    r  = T(0);
    q  = T(0);
    zi = T(0);

	if(z<0.0)
        throw(error("β not in supported range", z)); #Fail if the input is not supported
    end

	if(z<1.0+π)      #If z is between 0 and 1+π
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
        zi  = 1.0/z;
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

