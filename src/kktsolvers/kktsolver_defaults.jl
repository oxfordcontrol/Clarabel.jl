
#update matrix data and factor
function kktsolver_update!(linsys::AbstractKKTSolver{T},cones::CompositeCone{T}) where{T}
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
    x::Option{AbstractVector{T}},
    z::Option{AbstractVector{T}}
) where{T}
    error("function not implemented")
end

# update methods for P and A 
function kktsolver_update_P!(
    kktsolver::AbstractKKTSolver{T},
    P::SparseMatrixCSC{T}
) where{T}
    error("function not implemented")
end

function kktsolver_update_A!(
    kktsolver::AbstractKKTSolver{T},
    A::SparseMatrixCSC{T}
) where{T}
    error("function not implemented")
end