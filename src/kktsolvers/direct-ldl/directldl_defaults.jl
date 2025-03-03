abstract type AbstractDirectLDLSolver{T <: AbstractFloat} end

# Any new LDL solver type should provide implementations 
# of all of the following functions 

# register type, .e.g
# ldlsolver_matrix_shape(::Val{:qdldl}) = QDLDLDirectLDLSolver
# ldlsolver_constructor(::Val{:qdldl}) = :triu

# return a concrete subtype of AbstractDirectLDLSolver
function ldlsolver_constructor(::Val{T}) where T
    error("No solver found for option ", T)
end

# return either :triu or :tril
function ldlsolver_matrix_shape(::Val{T}) where T
    error("No solver found for option ", T)
end

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::AbstractVector{T}
) where{T}
    error("function not implemented")
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
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
