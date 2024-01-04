"""
	update_data!(solver,P,q,A,b)

Overwrites internal problem data structures in a solver object with new data, avoiding new memory 
allocations.   See [`update_P!`](@ref), [`update_q!`](@ref), [`update_A!`](@ref), [`update_b!`](@ref) for allowable input types.

"""

function update_data!(
    s::Solver{T},
    P::Union{SparseMatrixCSC{T},Vector{T},Nothing},
    q::Union{Vector{T},Nothing},
    A::Union{SparseMatrixCSC{T},Vector{T},Nothing},
    b::Union{Vector{T},Nothing}
) where{T}

    update_P!(s,P)
    update_q!(s,q)
    update_A!(s,A)
    update_b!(s,b)

    return nothing
end 


"""
	update_P!(solver,P)

Overwrites the `P` matrix data in an existing solver object.  The input `P` can be:

    - a nonempty Vector, in which case the nonzero values of the original `P` are overwritten, preserving the sparsity pattern, or

    - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the upper triangular part of the original `P`.   

    - an empty vector or `nothing`, in which case no action is taken.

"""

function update_P!(
    s::Solver{T},
    P::SparseMatrixCSC{T}
) where{T}

    is_equal_sparsity(P,s.data.P) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))

    update_P!(s,P.nzval)

    return nothing
end 

function update_P!(
    s::Solver{T},
    v::Union{Vector{T},Nothing}
) where{T}

    _check_presolve_disabled(s)

    isnothing(v) && return
    isempty(v) && return
    
    length(v) == length(s.data.P.nzval) || throw(DimensionMismatch("Input must match length of original data."))

    s.data.P.nzval .= v

    # reapply original equilibration 
    scale_P!(s.data.P,s.data.equilibration.d)

    # overwrite KKT data 
    kkt_update_P!(s.kktsystem,s.data.P)

    return nothing
end 

"""
	update_A!(solver,A)

Overwrites the `A` matrix data in an existing solver object.  The input `A` can be:

    - a nonempty Vector, in which case the nonzero values of the original `A` are overwritten, preserving the sparsity pattern, or

    - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the original `A`.   

    - an empty vector or `nothing`, in which case no action is taken.

"""

function update_A!(
    s::Solver{T},
    A::SparseMatrixCSC{T}
) where{T}

    isequal_sparsity(A,s.data.A) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))

    update_A!(s,A.nzval)

    return nothing
end 

function update_A!(
    s::Solver{T},
    v::Union{Vector{T},Nothing}
) where{T}

    _check_presolve_disabled(s)

    isnothing(v) && return
    isempty(v) && return
    
    length(v) == length(s.data.A.nzval) || throw(DimensionMismatch("Input must match length of original data."))

    s.data.A.nzval .= v

    # reapply original equilibration 
    scale_A!(s.data.A,s.data.equilibration.e,s.data.equilibration.d)

    # overwrite KKT data 
    kkt_update_A!(s.kktsystem,s.data.A)

    return nothing
end 

"""
	update_q!(solver,q)

Overwrites the `q` vector data in an existing solver object.  No action is taken if 'q' is an empty vector or `nothing`.

"""

function update_q!(
    s::Solver{T},
    v::Union{Vector{T},Nothing}
) where{T}

    _check_presolve_disabled(s)

    isnothing(v) && return
    isempty(v) && return
    
    length(v) == length(s.data.q) || throw(DimensionMismatch("Input must match length of original data."))

    s.data.q .= v

    #recompute unscaled norm 
    s.data.normq = norm(s.data.q,Inf)

    # reapply original equilibration 
    scale_q!(s.data.q,s.data.equilibration.d)

    return nothing
end 

"""
	update_b!(solver,b)

Overwrites the `b` vector data in an existing solver object.  No action is taken if 'b' is an empty vector or `nothing`.

"""

function update_b!(
    s::Solver{T},
    v::Union{Vector{T},Nothing}
) where{T}

    _check_presolve_disabled(s)

    isnothing(v) && return
    isempty(v) && return
    
    length(v) == length(s.data.b) || throw(DimensionMismatch("Input must match length of original data."))

    s.data.b .= v

    #recompute unscaled norm 
    s.data.normb = norm(s.data.b,Inf)

    # reapply original equilibration 
    scale_b!(s.data.b,s.data.equilibration.e)

    return nothing
end 



function _check_presolve_disabled(s)
    # Fail if presolve is enabled even if the sparsity is the same.
    # Not strictly necessary but may avoid confusion about expectations.
    if s.settings.presolve_enable 
        error("Disable presolve to allow data updates.")
    end
end 