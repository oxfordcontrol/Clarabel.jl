import Pardiso

abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.MKLPardisoSolver
    nnzA::DefaultInt

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        Pardiso.mkl_is_available() || error("MKL Pardiso is not available")
        ps = Pardiso.MKLPardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)

        Pardiso.set_nprocs!(ps, settings.max_threads) 

        return new(ps, nnz(KKT))
    end
end

# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.PardisoSolver
    nnzA::DefaultInt

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        Pardiso.panua_is_available() || error("Panua Pardiso is not available")
        ps = Pardiso.PardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)
        ps.iparm[8]=-99 # No IR

        #Note : Panua doesn't support setting the number of threads
        #Always reads instead from ENV["OMP_NUM_THREADS"] before loading

        return new(ps,nnz(KKT))
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



DirectLDLSolversDict[:mkl]   = MKLPardisoDirectLDLSolver
DirectLDLSolversDict[:panua] = PanuaPardisoDirectLDLSolver
required_matrix_shape(::Type{PanuaPardisoDirectLDLSolver}) = :tril
required_matrix_shape(::Type{MKLPardisoDirectLDLSolver}) = :tril

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
function refactor!(ldlsolver::AbstractPardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # Pardiso is quite robust and will usually produce some 
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
    ldlsolver::AbstractPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)

end
