# -------------------------------------
# Chordal Decomposition Information
# -------------------------------------
struct ConeMapEntry
    orig_index::Int 
    tree_and_clique::Option{Tuple{Int,Int}}
end 

mutable struct ChordalInfo{T}

    # sketch of the original problem
    init_dims::Tuple{Int64, Int64}       # (n,m) dimensions of the original problem
    init_cones::Vector{SupportedCone}    # original cones of the problem
    
    # decomposed problem data 
    spatterns::Vector{SparsityPattern}   # sparsity patterns for decomposed cones
    
    # "H" matrix for the standard chordal problem transformation 
    # remains as nothing if the "compact" transform is used 
    H::Option{SparseMatrixCSC{T}}

    # mapping from each generated cone to its original cone 
    # index, plus its tree and clique information if it has 
    # been generated as part of a chordal decomposition
    # remains as nothing if the "standard" transform is used
    cone_maps::Option{Vector{ConeMapEntry}}

    function ChordalInfo(
        A::SparseMatrixCSC{T}, 
        b::Vector{T}, 
        cones::Vector{SupportedCone}, 
        settings::Clarabel.Settings{T}
    ) where {T}
    
        # initial problem data
        init_dims  = (size(A,2),size(A,1))
        
        chordal_info = new{T}(
            init_dims, 
            SupportedCone[], 
            SparsityPattern[], 
            nothing,   # no H to start
            nothing    # no cone_maps to start
        )

        find_sparsity_patterns!(
            chordal_info, 
            A, b, 
            cones, 
            settings.chordal_decomposition_merge_method)

        # Only copy the generating cones if we have decomposition,  
        # since otherwise this object is going to be dropped anyway 
        if is_decomposed(chordal_info)
            chordal_info.init_cones = deepcopy(cones)  
        end

        return chordal_info

    end
end

  
function find_sparsity_patterns!(
    chordal_info::ChordalInfo,
    A::SparseMatrixCSC{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone},
    merge_method::Symbol
) where {T}

    rng_cones = rng_cones_iterator(cones)

    # aggregate sparsity pattern across the rows of [A;b]
    nz_mask = find_aggregate_sparsity_mask(A, b)

    # find the sparsity patterns of the PSD cones
    for (coneidx, (cone,rowrange)) in enumerate(zip(cones,rng_cones))

        if !isa(cone, PSDTriangleConeT)
            continue
        end

        analyse_sparsity_pattern!(
            chordal_info, 
            view(nz_mask,rowrange), 
            cone, 
            coneidx,
            merge_method)
    end
end


function find_aggregate_sparsity_mask(
    A::SparseMatrixCSC{T}, 
    b::Vector{T},
) where {T <: AbstractFloat}

    # returns true in every row in which [A;b] has a nonzero

    active = falses(length(b))
    active[A.rowval] .= true
    @. active |= (b != 0)
    active
end 


function analyse_sparsity_pattern!(
    chordal_info::ChordalInfo, 
    nz_mask::AbstractVector{Bool}, 
    cone::SupportedCone, 
    coneidx::Int,
    merge_method::Symbol
) 

    @assert length(nz_mask) == nvars(cone)
    if all(nz_mask) 
        return #dense / decomposable
    end
    
    L, ordering = find_graph!(nz_mask)
    spattern = SparsityPattern(L, ordering, coneidx, merge_method)

    if spattern.sntree.n_cliques == 1
        return #not decomposed, or everything re-merged
    end 

    push!(chordal_info.spatterns, spattern)
  
end


# did any PSD cones get decomposed?
function is_decomposed(chordal_info::ChordalInfo)
    return !isempty(chordal_info.spatterns)
end 


# total number of cones we started with 
function init_cone_count(chordal_info::ChordalInfo)
    length(chordal_info.init_cones)
end


"Determine the total number of sets `num_total` after decomposition and the number of new psd cones `num_new_psd_cones`."
function post_cone_count(chordal_info::ChordalInfo)

    # sum the number of cliques in each spattern
    npatterns = length(chordal_info.spatterns)
    ncliques  = sum([spattern.sntree.n_cliques for spattern in chordal_info.spatterns])

    # subtract npatterns to avoid double counting the 
    # original decomposed cones 
    return init_cone_count(chordal_info) - npatterns + ncliques
end


function get_decomposed_dim_and_overlaps(chordal_info::ChordalInfo{T}) where{T}

    cones = chordal_info.init_cones
    sum_cols     = 0
    sum_overlaps = 0
  
    for pattern in chordal_info.spatterns
      cone = cones[pattern.orig_index]
      @assert isa(cone, PSDTriangleConeT)
      cols, overlap = get_decomposed_dim_and_overlaps(pattern.sntree)
      sum_cols     += cols 
      sum_overlaps += overlap
    end 
  
    sum_cols, sum_overlaps
  end 
  


# -------------------------------------
# FUNCTION DEFINITIONS
# -------------------------------------

# compute a logical Cholesky decomposition and associated 
# ordering a for matrix with nonzero entries in the upper 
# triangular part, provided as a vector

function find_graph!(nz_mask::AbstractVector{Bool}) 
	
    nz = count(nz_mask)
    rows = sizehint!(Int[], nz)
    cols = sizehint!(Int[], nz)

    # check final row/col to get matrix dimension
    (m,n) = upper_triangular_index_to_coord(length(nz_mask))
    @assert m == n
    
    for (linearidx, isnonzero) in enumerate(nz_mask)
        if isnonzero
            (row,col) = upper_triangular_index_to_coord(linearidx)
            push!(rows, row)
            push!(cols, col)
        end
    end
    
    # PJG: QDLDL doesn't currently allow for logical-only decomposition 
    # on a matrix of Bools, so pattern must be a Float64 matrix here
    pattern = sparse(rows, cols, ones(length(rows)), n, n)

	F = QDLDL.qdldl(pattern, logical = true)

    L = F.L
    ordering = F.perm

	# this takes care of the case that QDLDL returns an unconnected adjacency matrix L
	connect_graph!(L)

    return L, ordering 
end


# ----------------------------------------------
# Iterator for the range of indices of the cones

# PJG: something similar could be done for the internal cones,
# to generate cone and block ranges, but need to make sure it 
# won't generate rust borrow conflicts

struct SupportedConeRangeIterator
    cones::Vector{SupportedCone}
end

function rng_cones_iterator(cones::Vector{SupportedCone})
    SupportedConeRangeIterator(cones)
end

Base.length(C::SupportedConeRangeIterator) = length(C.cones)

function Base.iterate(C::SupportedConeRangeIterator, state=(1, 1)) 
    (coneidx, start) = state 
    if coneidx > length(C.cones)
        return nothing 
    else 
        stop  = start + nvars(C.cones[coneidx]) - 1
        state = (coneidx + 1, stop + 1)
        return (start:stop, state)
    end 
end 

"""
	find_graph!(ci, linearidx::Array{Int, 1}, n::Int)

Given the indices of non-zero matrix elements in `linearidx`:
- Compute the sparsity pattern and find a chordal extension using `QDLDL` with AMD ordering `F.perm`.
- If unconnected, connect the graph represented by the cholesky factor `L`
"""

function connect_graph!(L::SparseMatrixCSC{T}) where{T}
 	# unconnected blocks don't have any entries below the diagonal in their right-most columns
	m = size(L, 1)
	row_val = L.rowval
	col_ptr = L.colptr
	for j = 1:m-1
		connected = false
		for k in col_ptr[j]:col_ptr[j+1]-1
			if row_val[k] > j
				connected  = true
				break
			end
		end
        #PJG: this insertion can happen in a midrange column, as long as 
        #that column is the last one for a given block 
		if !connected
			L[j+1, j] = 1
		end
	end
end
