using HSL, AMD

abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end

mutable struct HSLMA57DirectLDLSolver{T} <: HSLDirectLDLSolver{T}

    F::Ma57
    work::Vector{T}

    function HSLMA57DirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        
        HSL.LIBHSL_isfunctional() || error("HSL is not available")

        # work vector for solves
        Fcontrol = Ma57_Control{Float64}(; sqd = true)
        Finfo = Ma57_Info{Float64}()

        #Best guess at settings that will force LDL with diagonal D
        Fcontrol.icntl[5] = 0   # printing level.  verbose = 3
        Fcontrol.icntl[6] = 0   # ordering.  0 = AMD, 1 = AMD with dense rows, 5 automatic (default)
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


DirectLDLSolversDict[:ma57] = HSLMA57DirectLDLSolver
required_matrix_shape(::Type{HSLMA57DirectLDLSolver}) = :tril


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




