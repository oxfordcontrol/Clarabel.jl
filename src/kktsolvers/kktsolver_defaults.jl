abstract type AbstractKKTSolver{T <: AbstractFloat} end

#update matrix data and factor
function kktsolver_update!(linsys::AbstractKKTSolver{T},cones::ConeSet{T}) where{T}
    error("function not implemented")
end

#set the RHS (do not solve)
function kktsolver_setrhs!(
    kktsolver::AbstractKKTSolver{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where{T}
    error("function not implemented")
end


#solve and assign LHS
function kktsolver_solve!(
    kktsolver::AbstractKKTSolver{T},
    x::Union{Nothing,AbstractVector{T}},
    z::Union{Nothing,AbstractVector{T}}
) where{T}
    error("function not implemented")
end

# check whether the factorization is successful
function kktsolver_checkfact!(
    kktsolver::AbstractKKTSolver{T}
) where{T}
    error("function not implemented")
end