using Pardiso, SparseArrays, Clarabel
import Clarabel: Option, DefaultInt, AbstractDirectLDLSolver, LinearSolverInfo
import Clarabel: ldlsolver_constructor, ldlsolver_matrix_shape, ldlsolver_is_available
import Clarabel: linear_solver_info, update_values!, scale_values!, refactor!, solve!

MklInt = Pardiso.MklInt
PanuaInt = Int32 # always int32?

# on some platforms MKL or Panua might require a different 
# index type than we use for SparseMatrixCSC.  Sparse indices 
# are created once if necessary  
struct PardisoSparseIndex{Ti <: Integer} 
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    function PardisoSparseIndex{Ti}(A::SparseMatrixCSC) where{Ti}
        colptr = Ti.(A.colptr)
        rowval = Ti.(A.rowval)
        new{Ti}(colptr,rowval)
    end
end 

abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.MKLPardisoSolver
    nnzA::DefaultInt
    nvars::DefaultInt

    pardiso_indices::Option{PardisoSparseIndex{MklInt}}

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        ldlsolver_is_available(:mkl) || error("MKL Pardiso is loaded but not working or unlicensed")

        ps = Pardiso.MKLPardisoSolver()

        if MklInt !== typeof(KKT.colptr)
            pardiso_indices = PardisoSparseIndex{MklInt}(KKT)
        else
            pardiso_indices = nothing
        end 

        ldlsolver = new(ps,nnz(KKT),size(KKT)[1],pardiso_indices)
        pardiso_init(ldlsolver,KKT,Dsigns,settings)

        Pardiso.set_nprocs!(ps, settings.max_threads) 

        return ldlsolver
    end
end

# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: AbstractPardisoDirectLDLSolver{T}
    
    ps::Pardiso.PardisoSolver
    nnzA::DefaultInt
    nvars::DefaultInt

    pardiso_indices::Option{PardisoSparseIndex{PanuaInt}}

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        ldlsolver_is_available(:panua) || error("Panua Pardiso is loaded but not working or unlicensed")

        ps = Pardiso.PardisoSolver()

        if PanuaInt !== typeof(KKT.colptr)
            pardiso_indices = PardisoSparseIndex{PanuaInt}(KKT)
        else
            pardiso_indices = nothing
        end 

        ldlsolver = new(ps,nnz(KKT),size(KKT)[1],pardiso_indices)
        pardiso_init(ldlsolver,KKT,Dsigns,settings)

        #Note : Panua doesn't support setting the number of threads
        #Always reads instead from ENV["OMP_NUM_THREADS"] before loading

        return ldlsolver
    end
end

function custom_iparm_initialize!(ps::Pardiso.PardisoSolver, settings)
    # disable internal iterative refinement if user enabled
    # iterative refinement is enabled in the settings.   It is
    # seemingly not possible to disable this completely within
    # MKL, and setting -99 there would mean "execute 99 high
    # accuracy refinements steps".   Not good.
    if settings.iterative_refinement_enable 
        set_iparm!(ps, 8, -99); # NB: 1 indexed
    end
    # request count of non-zeros in the factorization
    set_iparm!(ps, 18, -1);  
end

function custom_iparm_initialize!(ps::Pardiso.MKLPardisoSolver, settings)
    # request count of non-zeros in the factorization
    set_iparm!(ps, 18, -1);  
end

function pardiso_init(ldlsolver,KKT,Dsigns,settings) 

    # NB: ignore Dsigns here because pardiso doesn't
    # use information about the expected signs

    KKT = pardiso_kkt(ldlsolver,KKT)
    ps = ldlsolver.ps

    # matrix is quasidefinite
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)

    #init here gets the defaults
    Pardiso.pardisoinit(ps)

    # overlay custom iparm initializations that might
    # be specific to MKL or Panua
    custom_iparm_initialize!(ps, settings);

    # now apply user defined iparm settings if they exist.
    # Check here first for failed solves, because misuse of 
    # this setting would likely be a disaster.
    for (i,iparm) in enumerate(settings.pardiso_iparm) 
        if iparm != typemin(Int32) 
            set_iparm!(ps, i, iparm);
        end
    end

    if settings.pardiso_verbose 
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    else 
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_OFF)
    end

    # perform logical factorization
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS phase

end 

ldlsolver_constructor(::Val{:mkl}) = MKLPardisoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:mkl}) = :tril
ldlsolver_is_available(::Val{:mkl}) = Pardiso.mkl_is_available() 

ldlsolver_constructor(::Val{:panua}) = PanuaPardisoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:panua}) = :tril
ldlsolver_is_available(::Val{:panua}) = Pardiso.panua_is_available()


function pardiso_kkt(
    ldlsolver::AbstractPardisoDirectLDLSolver{Tf},
    KKT::SparseMatrixCSC{Tf, Ti},
) where{Tf, Ti}

    # if ldlsolver carries its own indices, its because there 
    # is a mismatch between the SparseMatrixCSC int type and 
    # pardisos int type.   In that case, construct a shallow 
    # KKT with substitute indices 
    if isnothing(ldlsolver.pardiso_indices)
        return KKT
    else 
        return SparseMatrixCSC(
            KKT.m,
            KKT.n,
            ldlsolver.pardiso_indices.colptr,
            ldlsolver.pardiso_indices.rowval,
            KKT.nzval
        )
    end 
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
    nnzL = ldlsolver.ps.iparm[18] - ldlsolver.nvars
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
