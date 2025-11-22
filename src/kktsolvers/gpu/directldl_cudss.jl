using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolver
struct CUDSSDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKT::CuSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{T}
    x::CuVector{T}
    b::CuVector{T}
    

    function CUDSSDirectLDLSolver{T}(KKT::CuSparseMatrix{T},x,b) where {T}

        LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular

        cudssSolver = CUDSS.CudssSolver(KKT, "S", 'F')

        # allow hybrid use of GPU and CPU
        cudss_set(cudssSolver, "hybrid_execute_mode", 1)

        cudss("analysis", cudssSolver, x, b)
        cudss("factorization", cudssSolver, x, b)

        return new(KKT,cudssSolver,x,b)
    end

end

ldlsolver_constructor(::Val{:cudss}) = CUDSSDirectLDLSolver
ldlsolver_matrix_shape(::Val{:cudss}) = :full
ldlsolver_is_available(::Val{:cudss}) = CUDA.has_cuda_gpu()

function linear_solver_info(ldlsolver::CUDSSDirectLDLSolver{T}) where{T}
    name = :cudss;
    threads = 0;        #Not available for GPU solvers
    direct = true;
    LinearSolverInfo(name, threads, direct, 0, 0)
end

#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolver{T}) where{T}

    # Update the KKT matrix in the cudss solver
    cudss_update(ldlsolver.cudssSolver.matrix,ldlsolver.KKT)

    # Refactorization
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.x, ldlsolver.b)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end


#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    x::CuVector{T},
    b::CuVector{T}
) where{T}
    
    #solve on GPU
    ldiv!(x,ldlsolver.cudssSolver,b)

end
