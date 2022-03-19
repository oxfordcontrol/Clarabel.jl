using Pardiso, AMD

struct PardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKT::SparseMatrixCSC{T}
    ps::MKLPardisoSolver

    function PardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because pardiso doesn't
        #use information about the expected signs

        #make our AMD ordering outside of the solver
        perm = amd(KKT)

        #make a pardiso object and perform logical factor
        ps = MKLPardisoSolver()
        set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        pardisoinit(ps)
        fix_iparm!(ps, :N)
        set_phase!(ps, Pardiso.ANALYSIS)
        set_perm!(ps, perm)
        pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSISs

        #we might (?) need to register a finalizer for the pardiso
        #object to free internal structures
        finalizer(ps -> set_phase!(ps, Pardiso.RELEASE_ALL), ps )

        return new(KKT,ps)
    end
end

required_matrix_shape(::Type{PardisoDirectLDLSolver}) = :tril

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::Vector{Integer},
    values::Vector{T}
) where{T}

    ldlsolver.KKT.nzval[index] .= values
end

#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function offset_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::Vector{Integer},
    offset::Union{T,Vector{T}},
    signs::Union{Integer,Vector{Integer}} = 1
) where{T}

    @. ldlsolver.KKT.nzval[index] = offset*signs
end


#refactor the linear system
function refactor!(ldlsolver::PardisoDirectLDLSolver{T}) where{T}

    # Recompute the numeric factorization susing fake RHS
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, KKT, [1.])
end


#solve the linear system
function solve!(
    ldlsolver::PardisoDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T},
    settings
) where{T}

    ps  = ldlsolver.ps
    KKT = ldlsolver.KKT

    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, x, KKT, b)
end
