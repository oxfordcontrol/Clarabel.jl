import QDLDL

abstract type AbstractLinearSolver{T <: AbstractFloat} end

#update matrix data and factor
function linsys_update!(linsys::AbstractLinearSolver{T},cones::ConeSet{T}) where{T}
    error("function not implemented")
end

#set the RHS (do not solve)
function linsys_setrhs!(
    linsys::AbstractLinearSolver{T},
    x::AbstractVector{T},
    z::AbstractVector
) where{T}
    error("function not implemented")
end

#solve and assign LHS
function linsys_solve!(
    linsys::AbstractLinearSolver{T},
    x::Union{Nothing,AbstractVector{T}},
    z::Union{Nothing,AbstractVector{T}}
) where{T}
    error("function not implemented")
end
