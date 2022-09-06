import Pardiso
using AMD

struct PardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    ps::Pardiso.MKLPardisoSolver

    function PardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because pardiso doesn't
        #use information about the expected signs

        #make our AMD ordering outside of the solver
        perm = amd(KKT)

        #make a pardiso object and perform logical factor
        ps = Pardiso.MKLPardisoSolver()
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.pardisoinit(ps)
        Pardiso.fix_iparm!(ps, :N)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
        Pardiso.set_perm!(ps, perm)
        Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS

        return new(ps)
    end
end

DirectLDLSolversDict[:mkl] = PardisoDirectLDLSolver
required_matrix_shape(::Type{PardisoDirectLDLSolver}) = :tril

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function refactor!(ldlsolver::PardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # MKL is quite robust and will usually produce some 
    # kind of factorization unless there is an explicit 
    # zero pivot or some other nastiness.   "success" 
    # here just means that it didn't fail outright, although 
    # the factorization could still be garbage 

    # Recompute the numeric factorization susing fake RHS
    try 
        Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ldlsolver.ps, K, [1.])
        return is_success = true
    catch 
        return is_success = false
    end
     
end


#solve the linear system
function solve!(
    ldlsolver::PardisoDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps

    #Bug here: I am not storing a local reference
    #to KKT, and I don't think I actually need it anyway
    #since this should be solve phase.   Can I pass a dummy?
    #KKT = ldlsolver.KKT
    KKTdummy = sparse([],[],T[],length(x),length(x))

    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKTdummy, b)

end
