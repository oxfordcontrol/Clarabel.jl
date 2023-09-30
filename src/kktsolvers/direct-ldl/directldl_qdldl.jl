import QDLDL

struct QDLDLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    factors::QDLDL.QDLDLFactorisation{T, Int}

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
    index::AbstractVector{Int},
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
    index::AbstractVector{Int},
    scale::T
) where{T}

    QDLDL.scale_values!(ldlsolver.factors,index,scale)

end


#refactor the linear system
function refactor!(ldlsolver::QDLDLDirectLDLSolver{T}, K::SparseMatrixCSC) where{T}


    Dsigns = ldlsolver.factors.workspace.Dsigns

    #disable Dsigns in the factorisation 
    ldlsolver.factors.workspace.Dsigns = nothing

    # K is not used because QDLDL maintains
    # the update matrix entries for itself using the
    # offset/update methods implemented above.
    QDLDL.refactor!(ldlsolver.factors)

    # apply a regulization to the diagonal here instead 
    D = ldlsolver.factors.workspace.D
    regularize_eps = ldlsolver.factors.workspace.regularize_eps
    regularize_delta = ldlsolver.factors.workspace.regularize_delta

    for k = 1:length(Dsigns)
        if Dsigns[k]*D[k] < regularize_eps
            D[k] = regularize_delta * Dsigns[k]
        end
    end
    ldlsolver.factors.Dinv.diag .= 1 ./ D

    #restore Dsigns so we can read them again 
    #here later 
    ldlsolver.factors.workspace.Dsigns = Dsigns

    return all(isfinite, ldlsolver.factors.Dinv.diag)

end


#solve the linear system
function solve!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #solve in place 
    @. x = b
    QDLDL.solve!(ldlsolver.factors,x)

end
