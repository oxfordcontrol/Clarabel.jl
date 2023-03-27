######################################################################
# YC: to do: redefine the operator A
######################################################################
function pcg_solve_x1!(
    cones::CompositeCone{T},
    x1::Vector{T}, 
    b1::Vector{T}, 
    b2::ConicVector{T}, 
    work1::ConicVector{T}, 
    work2::ConicVector{T}, 
    work::Vector{T},
    P::AbstractMatrix{T}, 
    A::AbstractMatrix{T}, 
    H::Vector{Vector{T}},
    M::Vector{T}, 
    maxiter::Int, 
    tol::T
) where {T}
    
    # Initialize
    n = length(b1)
    Ap = view(work,1:n)
    z = view(work,(n+1):2*n)
    r = view(work,(2*n+1):3*n)
    r .= b1
    p = view(work,(3*n+1):4*n)
    
    mul_Hinv!(cones,work1,b2)
    mul!(r,A',work1,one(T),one(T))

    # diagonal preconditioning
    preconditioning!(cones,M,A,Ap,work1,work2)  #Ap as a temporary workspace

    # YC: no warmstart
    @. x1 = r

    #Start pcg
    mul_schur!(cones, Ap, P, A, H, x1, work1, work2)
    @. r -= Ap
    @. z = r/M
    p .= z
    
    # Iterate 
    for k = 1:maxiter
        mul_schur!(cones, Ap, P, A, H, p, work1, work2)
        coef = dot(r, z)
        alpha =  coef/ dot(p, Ap)
        @. x1 += alpha * p

        @. r -= alpha * Ap
        
        # Check convergence
        res = norm(r,Inf)
        if res < tol
            println("Terminate at iter ", k, " with residual ", res)
            return x1
        end
        # println("res is ", res, " at iter ", k)
        
        # Precondition
        @. z = r/M
        beta = dot(z, r) / coef
        @. p = z + beta * p
    end
    
    # Maximum number of iterations reached
    return x1
end

# solve x2
function pcg_solve_x2!(
    cones::CompositeCone{T},
    x1::AbstractVector{T},
    x2::ConicVector{T}, 
    b2::ConicVector{T},
    A::AbstractMatrix{T},
    work::ConicVector{T}
) where {T}
    mul!(work,A,x1)
    @. work -= b2
    mul_Hinv!(cones,x2,work)
end

# preconditioning
function preconditioning!(
    cones::CompositeCone{T},
    M::AbstractVector{T},
    A::AbstractMatrix{T},
    work::AbstractVector{T},
    work1::ConicVector{T},
    work2::ConicVector{T}
) where {T}
    n = length(work)
    @inbounds for i = 1:n
        @. work1.vec = A[:,i]
        mul_Hinv!(cones,work2,work1)
        M[i] = dot(work1.vec,work2.vec)
    end
end

function _indirect_solve_kkt(
    cones::CompositeCone{T},
    x1::Vector{T},
    x2::ConicVector{T},
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    H::Vector{Vector{T}},
    b1::Vector{T},
    b2::ConicVector{T},
    work1::ConicVector{T},
    work2::ConicVector{T},
    work::Vector{T},
    M::Vector{T}
) where {T}
    """
    Solve the linear system
    [P     A'][x1] = [b1]
    [A    -H ][x2]   [b2]
    using the Preconditioned Conjugate Gradient Descent (PCG) method.

    # 1) we want to solve by condensing to a PSD form 
    # and doing an indirect solve of (P + A'*H^{-1}*A)x = r

    Arguments:
    A: the system matrix (square and symmetric positive definite)
    b: the right-hand side vector
    M: the preconditioner matrix (square and symmetric positive definite)
    x0: the initial guess
    maxiter: the maximum number of iterations
    tol: the tolerance for the stopping criterion

    Returns:
    x: the solution
    """

    maxiter = 1000 
    tol = T(1e-10)

    # Left and right hand sides for solves
    pcg_solve_x1!(cones, x1, b1, b2, work1, work2, work, P, A, H, M, maxiter, tol)
    pcg_solve_x2!(cones, x1, x2, b2, A, work1)


    #YC: need to define function mul_Hinv! for each cone via mul_Winv! for symmetric cones

end

# multiplication (P + A'*H^{-1}*A)x
function mul_schur!(
    cones::CompositeCone{T},
    Ap::AbstractVector{T},
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    H::Vector{Vector{T}},
    x::AbstractVector{T},
    work1::ConicVector{T},
    work2::ConicVector{T}
) where {T}

    mul!(Ap,P,x)
    mul!(work1,A,x)
    mul_Hinv!(cones,work2,work1)        #We need H for SOCP
    mul!(Ap, A', work2, one(T), one(T))

end









# ######################################################################
# function pcg(A, b, M, x0, maxiter, tol)
#     """
#     Solve the linear system Ax = b using the Preconditioned Conjugate Gradient Descent (PCG) method.
    
#     Arguments:
#     A: the system matrix (square and symmetric positive definite)
#     b: the right-hand side vector
#     M: the preconditioner matrix (square and symmetric positive definite)
#     x0: the initial guess
#     maxiter: the maximum number of iterations
#     tol: the tolerance for the stopping criterion
    
#     Returns:
#     x: the solution
#     """
    
#     # Initialize
#     Ap = similar(b)
#     z = similar(b)
    
#     r = b - A * x0
#     z .= M \ r
#     p = deepcopy(z)
#     x = deepcopy(x0)

    
#     # Iterate
#     for k = 1:maxiter
#         mul!(Ap, A, p)
#         coef = dot(r, z)
#         alpha =  coef/ dot(p, Ap)
#         @. x += alpha * p

#         @. r -= alpha * Ap
        
#         # Check convergence
#         res = norm(r)
#         if res < tol
#             println("Terminate at iter ", k)
#             return x
#         end
#         println("res is ", res, " at iter ", k)
        
#         # Precondition
#         z .= M \ r
#         beta = dot(z, r) / coef
#         @. p = z + beta * p
#     end
    
#     # Maximum number of iterations reached
#     return x
# end


