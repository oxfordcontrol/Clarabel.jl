using Pardiso, SparseArrays, Clarabel
import Clarabel: DefaultInt, AbstractDirectLDLSolver, ldlsolver_constructor, ldlsolver_matrix_shape
import Clarabel: update_values!, scale_values!, refactor!, solve!

abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.MKLPardisoSolver

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        Pardiso.mkl_is_available() || error("MKL Pardiso is not available")
        ps = Pardiso.MKLPardisoSolver()
        solver = new(ps)
        pardiso_init(ps,pardiso_kkt(solver,KKT),Dsigns,settings)
        return solver
    end
end

# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.PardisoSolver

    #Pardiso wants 32 bit CSC indices
    colptr32::Vector{Int32}
    rowval32::Vector{Int32}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        Pardiso.panua_is_available() || error("Panua Pardiso is not available")
        ps = Pardiso.PardisoSolver()
        colptr32 = Int32.(KKT.colptr)
        rowval32 = Int32.(KKT.rowval)
        ps.iparm[8]=-99 # No IR
        solver = new(ps,colptr32,rowval32)
        pardiso_init(ps,pardiso_kkt(solver,KKT),Dsigns,settings)
        return solver
    end
end

function pardiso_init(ps,KKT,Dsigns,settings)

        #NB: ignore Dsigns here because pardiso doesn't
        #use information about the expected signs

        #perform logical factor
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.pardisoinit(ps)
        Pardiso.fix_iparm!(ps, :N)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
        Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS
end 

ldlsolver_constructor(::Val{:mkl}) = MKLPardisoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:mkl}) = :tril

ldlsolver_constructor(::Val{:panua}) = PanuaPardisoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:panua}) = :tril

function pardiso_kkt(
    ::MKLPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
) where{T}
    # MKL allows for 64bit CSC indices, so just pass through
    return KKT
end

function pardiso_kkt(
    ldlsolver::PanuaPardisoDirectLDLSolver{Tf},
    KKT::SparseMatrixCSC{Tf, Tv},
) where{Tf, Tv}
    # Panua wants 32bit CSC indices, so make a new 
    # KKT matrix from the input KKT values and 
    # internally copied 32 bit versions.  The sparsity
    # pattern on the KKT matrix should not change
    # between updates.   There is minimal allocation here
    return SparseMatrixCSC{Tf,Int32}(
        KKT.m,
        KKT.n,
        ldlsolver.colptr32,
        ldlsolver.rowval32,
        KKT.nzval
    )
end

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function refactor!(ldlsolver::AbstractPardisoDirectLDLSolver{T},KKT::SparseMatrixCSC{T}) where{T}

    # Pardiso is quite robust and will usually produce some 
    # kind of factorization unless there is an explicit 
    # zero pivot or some other nastiness.   "success" 
    # here just means that it didn't fail outright, although 
    # the factorization could still be garbage 

    ps  = ldlsolver.ps
    KKT = pardiso_kkt(ldlsolver,KKT)

    # Recompute the numeric factorization using fake RHS
    try 
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ps, KKT, [1.])
        return is_success = true
    catch 
        return is_success = false
    end
     
end


#solve the linear system
function solve!(
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps
    KKT = pardiso_kkt(ldlsolver,KKT)

    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)

end
