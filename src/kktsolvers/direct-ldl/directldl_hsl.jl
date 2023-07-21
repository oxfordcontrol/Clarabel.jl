using HSL

abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end

mutable struct HSLMA97DirectLDLSolver{T} <: HSLDirectLDLSolver{T}
    F::Union{Ma97,Nothing}
    function HSLMA97DirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        F = nothing
        return new(F)
    end
end

mutable struct HSLMA57DirectLDLSolver{T} <: HSLDirectLDLSolver{T}
    F::Union{Ma57,Nothing}
    function HSLMA57DirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        F = nothing
        return new(F)
    end
end


DirectLDLSolversDict[:hsl_ma57] = HSLMA57DirectLDLSolver
DirectLDLSolversDict[:hsl_ma97] = HSLMA97DirectLDLSolver
required_matrix_shape(::Type{HSLMA57DirectLDLSolver}) = :tril
required_matrix_shape(::Type{HSLMA97DirectLDLSolver}) = :tril


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

    if ldlsolver.F === nothing
        ldlsolver.F = Ma57(K)
        _ma57_config(ldlsolver.F)
    else
        ldlsolver.F.vals .= K.nzval
    end
    ma57_factorize!(ldlsolver.F)

    #info[1] is negative on failure, and positive 
    #for a warning.  Zero is success.
    return ldlsolver.F.info.info[1] == 0
end

function refactor!(ldlsolver::HSLMA97DirectLDLSolver{T}, K::SparseMatrixCSC{T}) where{T}

    if ldlsolver.F === nothing
        ldlsolver.F = Ma97(K)
        _ma97_config(ldlsolver.F)
    else
        ldlsolver.F.nzval .= K.nzval
    end
    ma97_factorize!(ldlsolver.F)

    return true
end


#solve the linear system
function solve!(
    ldlsolver::HSLMA57DirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    x .= ma57_solve(ldlsolver.F, b) # solves without iterative refinement
end

function solve!(
    ldlsolver::HSLMA97DirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    x .= ma97_solve(ldlsolver.F, b)
end


function _ma57_config(F::Ma57)

    #threshhold pivoting.  Set to zero in attempt
    #to force LDL with diagonal D
    F.control.cntl[1] = 0.0  
    F.control.icntl[1] = 1 #enable pivoting value set in cntl[1]
end 


function _ma97_config(F::Ma97)

    #threshhold pivoting.  Set to zero in attempt
    #to force LDL with diagonal D.  doesnt seem to 
    #work with ma97 though
    F.control.u = 0.0  
    F.control.small = 0.0
end 