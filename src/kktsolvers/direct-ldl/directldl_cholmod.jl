using SuiteSparse
mutable struct CholmodDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    F::SuiteSparse.CHOLMOD.Factor
    nnzA::Int

    function CholmodDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because Cholmod doesn't
        #use information about the expected signs

        #There is no obvious way to force cholmod to make
        #an initial symbolic factorization only
        F = ldlt(Symmetric(KKT); check = false)

        return new(F, nnz(KKT))
    end
end

ldlsolver_constructor(::Val{:cholmod}) = CholmodDirectLDLSolver
ldlsolver_matrix_shape(::Val{:cholmod}) = :triu
ldlsolver_is_available(::Val{:cholmod}) = true

function linear_solver_info(ldlsolver::CholmodDirectLDLSolver{T}) where{T}

    name = :cholmod;
    threads = 0;   #unknown 
    direct = true;
    LD = sparse(ldlsolver.F.LD)
    nnzA = ldlsolver.nnzA
    nnzL = nnz(LD) - size(LD,1)
    LinearSolverInfo(name, threads, direct, nnzA, nnzL)
end

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end


#refactor the linear system
function refactor!(ldlsolver::CholmodDirectLDLSolver{T}, K::SparseMatrixCSC{T}) where{T}

    #this reuses the symbolic factorization
    ldlt!(ldlsolver.F, Symmetric(K))

    return issuccess(ldlsolver.F)
end


#solve the linear system
function solve!(
    ldlsolver::CholmodDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    x .= ldlsolver.F\b
end
