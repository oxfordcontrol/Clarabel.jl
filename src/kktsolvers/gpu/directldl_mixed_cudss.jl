using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolverMixed
struct CUDSSDirectLDLSolverMixed{T} <: AbstractDirectLDLSolver{T}

    KKT::CuSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{Float32}

    KKTFloat32::CuSparseMatrix{Float32}
    xFloat32::CuVector{Float32}
    bFloat32::CuVector{Float32}
    

    function CUDSSDirectLDLSolverMixed{T}(KKT::CuSparseMatrix{T},x,b) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular

        val = CuVector{Float32}(KKT.nzVal)
        KKTFloat32 = CuSparseMatrixCSR(KKT.rowPtr,KKT.colVal,val,size(KKT))
        cudssSolver = CUDSS.CudssSolver(KKTFloat32, "S", 'F')

        xFloat32 = CUDA.zeros(Float32,dim)
        bFloat32 = CUDA.zeros(Float32,dim)

        cudss("analysis", cudssSolver, xFloat32, bFloat32)
        cudss("factorization", cudssSolver, xFloat32, bFloat32)


        return new(KKT,cudssSolver,KKTFloat32,xFloat32,bFloat32)
    end

end

ldlsolver_constructor(::Val{:cudssmixed}) = CUDSSDirectLDLSolverMixed
ldlsolver_matrix_shape(::Val{:cudssmixed}) = :full
ldlsolver_is_available(::Val{:cudssmixed}) = CUDA.has_cuda_gpu()

function linear_solver_info(ldlsolver::CUDSSDirectLDLSolverMixed{T}) where{T}
    name = :cudssmixed;
    threads = 0;        #Not available for GPU solvers
    direct = true;
    LinearSolverInfo(name, threads, direct, 0, 0)
end

#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolverMixed{T}) where{T}

    #YC: Copy data from a Float64 matrix to Float32 matrix
    copyto!(ldlsolver.KKTFloat32.nzVal,ldlsolver.KKT.nzVal)

    # Update the KKT matrix in the cudss solver
    cudss_update(ldlsolver.cudssSolver.matrix,ldlsolver.KKTFloat32)

    # Refactorization
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.xFloat32, ldlsolver.bFloat32)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end

#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolverMixed{T},
    x::CuVector{T},
    b::CuVector{T}
) where{T}

    xFloat32 = ldlsolver.xFloat32
    bFloat32 = ldlsolver.bFloat32

    #convert b to Float32
    copyto!(bFloat32, b)

    #solve on GPU
    ldiv!(xFloat32, ldlsolver.cudssSolver, bFloat32)

    #convert to Float64, copy to x 
    copyto!(x, xFloat32)

end
