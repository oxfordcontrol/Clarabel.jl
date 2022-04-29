import QDLDL

struct QDLDLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    #KKT matrix and its LDL factors
    KKT::SparseMatrixCSC{T}
    factors::QDLDL.QDLDLFactorisation{T, Int}

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int}}

    # internal workspace for IR scheme
    work::Vector{T}

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

        #KKT will be triu data only, but we will want
        #the following to allow products like KKT*x
        KKTsym = Symmetric(KKT)
        work = Vector{T}(undef,dim)

        return new(KKT,factors,KKTsym,work)
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

    #Updating values in both the KKT matrix and
    #in the reordered copy held internally by QDLDL.
    #The former is needed for iterative refinement since
    #QDLDL does not have internal iterative refinement
    QDLDL.update_values!(ldlsolver.factors,index,values)
    ldlsolver.KKT.nzval[index] .= values

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
    @. ldlsolver.KKT.nzval[index] += offset*signs

end

#refactor the linear system
function refactor!(ldlsolver::QDLDLDirectLDLSolver{T}) where{T}
    QDLDL.refactor!(ldlsolver.factors)
end


#solve the linear system
function solve!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T},
    settings
) where{T}

    #make an initial solve (solves in place)
    x .= b
    QDLDL.solve!(ldlsolver.factors,x)

    if(settings.iterative_refinement_enable)
        iterative_refinement(ldlsolver,x,b,settings)
    end

    return nothing
end


function iterative_refinement(ldlsolver::QDLDLDirectLDLSolver{T},x,b,settings) where{T}

    work = ldlsolver.work

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    #Note that K is only triu data, so need to
    #be careful when computing the residual here
    K      = ldlsolver.KKT
    KKTsym = ldlsolver.KKTsym
    lastnorme = Inf

    normb = norm(b,Inf)

    for i = 1:IR_maxiter

        #this is work = error = b - Kξ
        work .= b
        mul!(work,KKTsym,x,-1.,1.)
        norme = norm(work,Inf)

        # test for convergence before committing
        # to a refinement step
        if(norme <= IR_abstol + IR_reltol*normb)
            break
        end

        #if we haven't improved by at least the halting
        #ratio since the last pass through, then abort
        if(lastnorme/norme < IR_stopratio)
            println("IR error: ", norme)
            break
        end

        #make a refinement and continue
        QDLDL.solve!(ldlsolver.factors,work)     #this is Δξ
        x .+= work

        lastnorme = norme
    end

    println("IR error: ", lastnorme)

    return nothing
end
