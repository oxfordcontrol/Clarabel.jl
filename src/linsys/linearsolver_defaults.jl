import QDLDL

abstract type AbstractLinearSolver{T <: AbstractFloat} end

# throw errors for unimplemented subtype behaviours

function linsys_is_soc_sparse_format(linsys::AbstractLinearSolver{T}) where{T}
    error("function not implemented")
end

function linsys_soc_sparse_variables(linsys::AbstractLinearSolver{T}) where{T}
    error("function not implemented")
end

#solves in place
function linsys_update!(linsys::AbstractLinearSolver{T},cones::ConeSet{T}) where{T}
    error("function not implemented")
end

#solves in place
function linsys_solve!(linsys::AbstractLinearSolver{T},x::AbstractArray{T}) where{T}
    error("function not implemented")
end
