abstract type AbstractDirectLDLSolver{T <: AbstractFloat} end

# Any new LDL solver type should provide implementations 
# of all of the following functions 

# register type, .e.g
# ldlsolver_constructor(::Val{:qdldl}) = QDLDLDirectLDLSolver
# ldlsolver_matrix_shape(::Val{:qdldl}) = :triu
# ldlsolver_is_available(::Val{:qdldl}) = true

# return a concrete subtype of AbstractDirectLDLSolver
function ldlsolver_constructor(::Val{T}) where T
    error(ldlsolver_default_error(T))
end

# return either :triu or :tril
function ldlsolver_matrix_shape(::Val{T}) where T
    error(ldlsolver_default_error(T))
end

# return true if the solver is licensed and operational
function ldlsolver_is_available(::Val{T}) where T
    # unconfigured solvers report unavailable
    # so that the :auto option can select packages
    # without having to catch errors 
    false 
end

# provide information about the linear solver via 
# a LinearSolverInfo object
function linear_solver_info(
    ldlsolver::AbstractDirectLDLSolver{T},
) where{T}
    error("function not implemented")
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
