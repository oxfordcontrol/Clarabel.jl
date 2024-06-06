
# Supported matrix and vector updating input types
# NB: this a trait in Rust, not a type. 
const MatrixProblemDataUpdate{T} = Union{
    Nothing,
    SparseMatrixCSC{T},
    AbstractVector{T},
    Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}}
} where {T} 

const VectorProblemDataUpdate{T} = Union{
    Nothing,
    AbstractVector{T},
    Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}}
} where {T} 

"""
	update_data!(solver,P,q,A,b)

Overwrites internal problem data structures in a solver object with new data, avoiding new memory 
allocations.   See [`update_P!`](@ref), [`update_q!`](@ref), [`update_A!`](@ref), [`update_b!`](@ref) 
for allowable input types.

"""

function update_data!(
    s::Solver{T},
    P::VectorProblemDataUpdate{T} ,
    q::Option{Vector{T}},
    A::MatrixProblemDataUpdate{T},
    b::VectorProblemDataUpdate{T} 
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
    data::MatrixProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    _check_update_allowed(s)
    d = s.data.equilibration.d
    _update_matrix(data,s.data.P,d,d)
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
    data::MatrixProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    _check_update_allowed(s)
    d = s.data.equilibration.d
    e = s.data.equilibration.e 
    _update_matrix(data,s.data.A,e,d)
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
    data::VectorProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    _check_update_allowed(s)
    d    = s.data.equilibration.d
    dinv = s.data.equilibration.dinv
    _update_vector(data,s.data.q,d)

    # flush unscaled norm.   Will be recalculated during solve
    data_clear_normq!(s.data)

    return nothing
end 

"""
	update_b!(solver,b)

Overwrites the `b` vector data in an existing solver object.  No action is taken if 'b' is an empty vector or `nothing`.

"""

function update_b!(
    s::Solver{T},
    data::VectorProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    _check_update_allowed(s)
    e    = s.data.equilibration.e     
    einv = s.data.equilibration.einv
    _update_vector(data,s.data.b,e)

    # flush unscaled norm.   Will be recalculated during solve
    data_clear_normb!(s.data)

    return nothing
end 

function _check_update_allowed(s)

    # Fail if presolve / chordal decomp is enabled.  
    # Not strictly necessary since the presolve and chordal decomp 
    # might not do anything, but may avoid confusion about expectations.

    # checks both settings and existence of presolve objects, since otherwise 
    # it would be possible to presolve and then disable the settings. 

    if s.settings.presolve_enable || 
       s.settings.chordal_decomposition_enable || 
       !isnothing(s.data.presolver) ||
       !isnothing(s.data.chordal_info)

        error("Disable presolve and chordal decomposition to allow data updates.")
    end

end 

function _update_matrix(
    data::SparseMatrixCSC{T},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T}
) where{T}
    
    isequal_sparsity(data,M) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))
    _update_matrix(data.nzval,M,lscale,rscale)
end

function _update_matrix(
    data::AbstractVector{T},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T}
) where{T}
    
    length(data) == 0 && return
    length(data) == nnz(M) || throw(DimensionMismatch("Input must match length of original data."))
    M.nzval .= data
    lrscale!(lscale,M,rscale)
end

function _update_matrix(
    data::Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T}
) where{T}
    
    for (idx,value) in data
        idx ∈ 0:nnz(M) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))
        (row,col) = index_to_coord(M,idx)
        M.nzval[idx] = lscale[row] * rscale[col] * value
    end
end


function _update_vector(
    data::AbstractVector{T},
    v::AbstractVector{T},
    scale::AbstractVector{T}
) where{T}
    
    length(data) == 0 && return
    length(data) == length(v) || throw(DimensionMismatch("Input must match length of original data."))
    
    @. v= data*scale
end


function _update_vector(
    data::Base.Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}},
    v::AbstractVector{T},
    scale::AbstractVector{T}
) where{T}
    for (idx,value) in data
        v[idx] = value*scale[idx]
    end
end



