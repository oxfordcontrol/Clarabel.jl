const KKTSolversDict = Dict{Symbol, UnionAll}()

function _get_kktsolver_type(s::Symbol)
    try
        return KKTSolversDict[s]
    catch
        throw(error("Unsupported kkt solver method:", s))
    end
end

# Any new AbstractKKTSolver sub type should provide implementations of 
# of the following and add itself to the KKTSolversDict

# register type, e.g.
# KKTSolversDict[:directldl] = DirectLDLKKTSolver

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
    cones::CompositeCone{T},
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


# check whether the condition number is poors
function kktsolver_is_ill_conditioned!(
    kktsolver::AbstractKKTSolver{T}
) where{T}
    error("function not implemented")
end

