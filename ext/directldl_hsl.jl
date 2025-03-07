using HSL, SparseArrays, Clarabel
import Clarabel: DefaultInt, AbstractDirectLDLSolver, LinearSolverInfo
import Clarabel: ldlsolver_constructor, ldlsolver_matrix_shape, ldlsolver_is_available
import Clarabel: linear_solver_info, update_values!, scale_values!, refactor!, solve!

abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end

mutable struct HSLMA57DirectLDLSolver{T} <: HSLDirectLDLSolver{T}

    F::Ma57
    work::Vector{T}

    function HSLMA57DirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        
        ldlsolver_is_available(:ma57)  || error("HSL package is loaded but not working or unlicensed")

        # work vector for solves
        Fcontrol = Ma57_Control{Float64}(; sqd = true)
        Finfo = Ma57_Info{Float64}()

        #Best guess at settings that will force LDL with diagonal D
        Fcontrol.icntl[5] = 0   # printing level.  verbose = 3
        Fcontrol.icntl[6] = 5   # ordering.  0 = AMD, 1 = AMD with dense rows, 5 automatic (default)
        Fcontrol.icntl[7] = 3   # do not perform pivoting 
        Fcontrol.icntl[9] = 0   # zero iterative refinement steps
    
        Fcontrol.cntl[1] = 0.0  # threshold for pivoting (should not be used anyway)
        Fcontrol.cntl[2] = 0.0  # pivots below this treated as zero 
        Fcontrol.cntl[4] = 0.0  # no static pivoting 

        F = HSL.Ma57(KKT,Fcontrol,Finfo); # performs static analysis 

        # work vector for solves.
        work = zeros(T,KKT.n)

        return new(F,work)
    end
end

ldlsolver_constructor(::Val{:ma57}) = HSLMA57DirectLDLSolver
ldlsolver_matrix_shape(::Val{:ma57}) = :tril
ldlsolver_is_available(::Val{:ma57}) = HSL.LIBHSL_isfunctional()


function linear_solver_info(ldlsolver::HSLMA57DirectLDLSolver{T}) where{T}

    name = :ma57
    threads = 1   #always single threaded 
    direct = true
    nnzA = length(ldlsolver.F.vals)
    nnzL = ldlsolver.F.info.info[5] # MA57: Forecast number of reals in factors.
    nnzL = nnzL - ldlsolver.F.n # subtract diagonal terms
    LinearSolverInfo(name, threads, direct, nnzA, nnzL)

end


#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::HSLDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where{T}

    # MA57 stores the values of the matrix originally 
    # passed to it in their original order 
    @views ldlsolver.F.vals[index] .= values

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::HSLDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where{T}

    # MA57 stores the values of the matrix originally 
    # passed to it in their original order 
    @views ldlsolver.F.vals[index] .*= scale

end


#refactor the linear system
function refactor!(ldlsolver::HSLMA57DirectLDLSolver{T}, K::SparseMatrixCSC{T}) where{T}

    ma57_factorize!(ldlsolver.F)

    #info[1] is negative on failure, and positive 
    #for a warning.  Zero is success.
    return ldlsolver.F.info.info[1] == 0
end

#solve the linear system
function solve!(
    ldlsolver::HSLMA57DirectLDLSolver{T},
    K::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    F    = ldlsolver.F
    work = ldlsolver.work

    x .= b #solves in place
    x .= ma57_solve!(F, x, work, job = :A) 
end




