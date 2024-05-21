using SuiteSparse
mutable struct CholmodDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    F::Option{SuiteSparse.CHOLMOD.Factor}

    function CholmodDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because Cholmod doesn't
        #use information about the expected signs

        #There is no obvious way to force cholmod to make
        #an initial symbolic factorization only, set
        #set F to nothing and create F as needed on the
        #first refactorization
        F = nothing

        return new(F)
    end
end

DirectLDLSolversDict[:cholmod] = CholmodDirectLDLSolver
required_matrix_shape(::Type{CholmodDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end


#refactor the linear system
function refactor!(ldlsolver::CholmodDirectLDLSolver{T}, K::SparseMatrixCSC{T}) where{T}

    if ldlsolver.F === nothing
        #initial symbolic and numeric factor since
        #we can't do symbolic on its own
        ldlsolver.F = ldlt(Symmetric(K))

    else
        #this reuses the symbolic factorization
        ldlt!(ldlsolver.F, Symmetric(K))
    end

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
