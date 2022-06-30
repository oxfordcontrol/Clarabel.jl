abstract type AbstractDirectLDLSolver{T <: AbstractFloat} end

const DirectLDLSolversDict = Dict{Symbol, UnionAll}()

# Any new LDL solver type should provide implementations of all
# of the following and add itself to the DirectLDLSolversDict

# register type, .e.g
# DirectLDLSolversDict[:qdldl] = QDLDLDirectLDLSolver

# return either :triu or :tril
function required_matrix_shape(::Type{AbstractDirectLDLSolver})
    error("function not implemented")
end


#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector,
    values::AbstractVector{T}
) where{T}
    error("function not implemented")
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector,
    scale::T
) where{T}
    error("function not implemented")
end


#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function offset_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector,
    offset::Union{T,AbstractVector{T}},
    signs::Union{Integer,AbstractArray{Integer}} = 1
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
