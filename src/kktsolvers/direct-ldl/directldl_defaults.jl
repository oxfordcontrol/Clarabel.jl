abstract type AbstractDirectLDLSolver{T <: AbstractFloat} end

const DirectLDLSolversDict = Dict{Symbol, UnionAll}()

function _get_ldlsolver_type(s::Symbol)
    try
        return DirectLDLSolversDict[s]
    catch
        throw(error("Unsupported direct LDL linear solver :", s))
    end
end

# Any new LDL solver type should provide implementations of all
# of the following and add itself to the DirectLDLSolversDict

# register type, e.g.
# DirectLDLSolversDict[:qdldl] = QDLDLDirectLDLSolver

# return either :triu or :tril
function required_matrix_shape(::Type{AbstractDirectLDLSolver})
    error("function not implemented")
end


#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::AbstractVector{T}
) where{T}
    error("function not implemented")
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}
    error("function not implemented")
end


#refactor the linear system
function refactor!(ldlsolver::AbstractDirectLDLSolver{T}) where{T}
    error("function not implemented")
end


#solve the linear system
function solve!(
    ldlsolver::AbstractDirectLDLSolver{T},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}
    error("function not implemented")
end
