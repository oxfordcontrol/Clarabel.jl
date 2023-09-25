using HSL, AMD

abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end

mutable struct HSLMA57DirectLDLSolver{T} <: HSLDirectLDLSolver{T}

    F::Union{Nothing,Ma57}
    work::Vector{T}

    function HSLMA57DirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        
        F = nothing

        # work vector for solves
        work = zeros(T,KKT.n)

        return new(F,work)
    end
end


DirectLDLSolversDict[:hsl_ma57] = HSLMA57DirectLDLSolver
required_matrix_shape(::Type{HSLMA57DirectLDLSolver}) = :tril


#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::HSLDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::HSLDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}
    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!

end


#refactor the linear system
function refactor!(ldlsolver::HSLMA57DirectLDLSolver{T}, K::SparseMatrixCSC{T}) where{T}

    if isnothing(ldlsolver.F)
        #instantiates the HSL MA57 solver
        F = Ma57(K; perm = invperm(amd(K)), sqd = true, print_level = 0, pivot_order = 1)

        #try to force LDL with diagonal D
        ma57_config(F)

        ldlsolver.F = F
    else
        ldlsolver.F.vals .= K.nzval
    end 
    ldlsolver.F.control.icntl[5] = 0  #3 -> force print
    ma57_factorize!(ldlsolver.F)
    ldlsolver.F.control.icntl[5] = 0  #disable print

    #info[1] is negative on failure, and positive 
    #for a warning.  Zero is success.
    return ldlsolver.F.info.info[1] == 0
end

#solve the linear system
function solve!(
    ldlsolver::HSLMA57DirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    F    = ldlsolver.F
    work = ldlsolver.work

    x .= b #solves in place
    x .= ma57_solve!(ldlsolver.F, x, work, job = :A) # solves without iterative refinement
end


function ma57_config(F::Ma57)

    #Best guess at settings that will force LDL with diagonal D
    F.control.cntl[1] = eps() 
    #F.control.cntl[2] = 0.0
    #F.control.icntl[7] = 3 
    #F.control.icntl[15] = 0 #no scaling
end 


# function _ma97_config(F::Ma97)

#     #threshhold pivoting.  Set to zero in attempt
#     #to force LDL with diagonal D.  doesnt seem to 
#     #work with ma97 though
#     F.control.u = 0.0  
#     F.control.small = 0.0
# end 