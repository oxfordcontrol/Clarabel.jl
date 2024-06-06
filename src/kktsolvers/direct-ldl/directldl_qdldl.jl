import QDLDL

struct QDLDLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    factors::QDLDL.QDLDLFactorisation{T, DefaultInt}

    function QDLDLDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        # occasionally we find that the default AMD parameters give a bad ordering, particularly 
        # for some big matrices.  In particular, KKT conditions for QPs are sometimes worse 
        # than their SOC counterparts for very large problems.   This is because the SOC form
        # is artificially "big", with extra rows, so the dense row threshold is effectively a 
        # different value.   We fix a bit more generous AMD_DENSE here, which should perhaps 
        # be user-settable.  

        #make a logical factorization to fix memory allocations
        factors = QDLDL.qdldl(
            KKT;
            Dsigns = Dsigns,
            regularize_eps   = settings.dynamic_regularization_eps,
            regularize_delta = settings.dynamic_regularization_delta,
            logical          = true,
            amd_dense_scale  = T(1.5),
        )

        return new(factors)
    end

end

DirectLDLSolversDict[:qdldl] = QDLDLDirectLDLSolver
required_matrix_shape(::Type{QDLDLDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where{T}

    #Update values that are stored within
    #the reordered copy held internally by QDLDL.
    QDLDL.update_values!(ldlsolver.factors,index,values)

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where{T}

    QDLDL.scale_values!(ldlsolver.factors,index,scale)

end


#refactor the linear system
function refactor!(ldlsolver::QDLDLDirectLDLSolver{T}, K::SparseMatrixCSC) where{T}

    # K is not used because QDLDL maintains
    # the update matrix entries for itself using the
    # offset/update methods implemented above.
    QDLDL.refactor!(ldlsolver.factors)

    return all(isfinite, ldlsolver.factors.Dinv.diag)

end


#solve the linear system
function solve!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    K::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #solve in place 
    @. x = b
    QDLDL.solve!(ldlsolver.factors,x)

end
