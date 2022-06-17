import QDLDL

struct QDLDLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    #KKT matrix and its LDL factors
    KKT::SparseMatrixCSC{T}
    factors::QDLDL.QDLDLFactorisation{T, Int}

    function QDLDLDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        factors = QDLDL.qdldl(
            KKT;
            Dsigns = Dsigns,
            regularize_eps   = settings.dynamic_regularization_eps,
            regularize_delta = settings.dynamic_regularization_delta,
            logical          = true
        )

        return new(KKT,factors)
    end

end

DirectLDLSolversDict[:qdldl] = QDLDLDirectLDLSolver
required_matrix_shape(::Type{QDLDLDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::Vector{Ti},
    values::Vector{T}
) where{T,Ti}

    #Update values that are stored within
    #the reordered copy held internally by QDLDL.

    #PJG: an alternative implementation would be
    #to just overwrite the complete KKT data
    #upon a call to refactor, which would avoid
    #this step and make the QDLDL implementation
    #much simpler (i.e. no update or offset methods
    #would be needed).   Need to test how slow a
    #complete permuted updated would be though
    QDLDL.update_values!(ldlsolver.factors,index,values)

end

#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function offset_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::Vector{Int},
    offset::Union{T,Vector{T}},
    signs::Union{Int,Vector{Int}} = 1
) where{T}

    QDLDL.offset_values!(ldlsolver.factors, index, offset, signs)

end

#refactor the linear system
function refactor!(ldlsolver::QDLDLDirectLDLSolver{T}, K::SparseMatrixCSC) where{T}

    #PJG: K is not used because QDLDL is maintaining
    #the update matrix entries for itself using the
    #offset/update methods implemented above.
    QDLDL.refactor!(ldlsolver.factors)
end


#solve the linear system
function solve!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #make an initial solve (solves in place)
    x .= b
    QDLDL.solve!(ldlsolver.factors,x)

    return nothing
end
