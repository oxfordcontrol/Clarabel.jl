######################################################################
# YC: to do: redefine the operator A

######################################################################
function pcg(A, b, M, x0, maxiter, tol)
    """
    Solve the linear system Ax = b using the Preconditioned Conjugate Gradient Descent (PCG) method.
    
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
    
    # Initialize
    Ap = similar(b)
    z = similar(b)
    
    r = b - A * x0
    z .= M \ r
    p = deepcopy(z)
    x = deepcopy(x0)

    
    # Iterate
    for k = 1:maxiter
        mul!(Ap, A, p)
        coef = dot(r, z)
        alpha =  coef/ dot(p, Ap)
        @. x += alpha * p

        @. r -= alpha * Ap
        
        # Check convergence
        res = norm(r)
        if res < tol
            println("Terminate at iter ", k)
            return x
        end
        println("res is ", res, " at iter ", k)
        
        # Precondition
        z .= M \ r
        beta = dot(z, r) / coef
        @. p = z + beta * p
    end
    
    # Maximum number of iterations reached
    return x
end


