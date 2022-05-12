import Pardiso
using AMD

struct PardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKT::SparseMatrixCSC{T}
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
        Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSISs

        return new(KKT,ps)
    end
end

DirectLDLSolversDict[:mkl] = PardisoDirectLDLSolver
required_matrix_shape(::Type{PardisoDirectLDLSolver}) = :tril

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::Vector{Int},
    values::Vector{T}
) where{T}

    ldlsolver.KKT.nzval[index] .= values
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::Vector{Int},
    scale::T
) where{T}

    ldlsolver.KKT.nzval[index] .*= scale
end

#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function offset_values!(
    ldlsolver::PardisoDirectLDLSolver{T},
    index::Vector{Int},
    offset::Union{T,Vector{T}},
    signs::Union{Int,Vector{Int}} = 1
) where{T}

    @. ldlsolver.KKT.nzval[index] += offset*signs
end


#refactor the linear system
function refactor!(ldlsolver::PardisoDirectLDLSolver{T}) where{T}

    # Recompute the numeric factorization susing fake RHS
    Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
    Pardiso.pardiso(ldlsolver.ps, ldlsolver.KKT, [1.])
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

    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)
end
