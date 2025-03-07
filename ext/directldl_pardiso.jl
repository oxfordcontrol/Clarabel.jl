using Pardiso, SparseArrays, Clarabel
import Clarabel: DefaultInt, AbstractDirectLDLSolver, LinearSolverInfo
import Clarabel: ldlsolver_constructor, ldlsolver_matrix_shape, ldlsolver_is_available
import Clarabel: linear_solver_info, update_values!, scale_values!, refactor!, solve!

abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.MKLPardisoSolver
    nnzA::DefaultInt

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        ldlsolver_is_available(:mkl) || error("MKL Pardiso is loaded but not working or unlicensed")

        ps = Pardiso.MKLPardisoSolver()

        solver = new(ps,nnz(KKT))
        pardiso_init(ps,pardiso_kkt(solver,KKT),Dsigns,settings)

        Pardiso.set_nprocs!(ps, settings.max_threads) 

        return solver
    end
end

# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.PardisoSolver
    nnzA::DefaultInt

    #Pardiso wants 32 bit CSC indices
    colptr32::Vector{Int32}
    rowval32::Vector{Int32}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        ldlsolver_is_available(:panua) || error("Panua Pardiso is loaded but not working or unlicensed")

        ps = Pardiso.PardisoSolver()
        colptr32 = Int32.(KKT.colptr)
        rowval32 = Int32.(KKT.rowval)
        ps.iparm[8]=-99 # No IR

        solver = new(ps,nnz(KKT),colptr32,rowval32)
        pardiso_init(ps,pardiso_kkt(solver,KKT),Dsigns,settings)

        #Note : Panua doesn't support setting the number of threads
        #Always reads instead from ENV["OMP_NUM_THREADS"] before loading

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
ldlsolver_is_available(::Val{:mkl}) = Pardiso.mkl_is_available() 

ldlsolver_constructor(::Val{:panua}) = PanuaPardisoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:panua}) = :tril
ldlsolver_is_available(::Val{:panua}) = Pardiso.panua_is_available()

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

function linear_solver_info(ldlsolver::AbstractPardisoDirectLDLSolver{T}) where{T}

    if isa(ldlsolver, MKLPardisoDirectLDLSolver)
        name = :mkl
    elseif isa(ldlsolver, PanuaPardisoDirectLDLSolver)  
        name = :panua
    else 
        name = :unknown
    end 
    threads = Pardiso.get_nprocs(ldlsolver.ps)
    direct = true
    nnzA = ldlsolver.nnzA
    ncols = length(ldlsolver.ps.perm) #length of permutation vector
    nnzL = ldlsolver.ps.iparm[18] - ncols
    LinearSolverInfo(name, threads, direct, nnzA, nnzL)

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
