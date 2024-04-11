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
        init_cones = deepcopy(cones)   #PJG: copy here potentially very bad.   Only do if needed

        # decomposition patterns.  length 
        # will be the same as the number
        # of cones eventually decomposed 
        spatterns = SparsityPattern[]

        chordal_info = new{T}(
            init_dims, 
            init_cones, 
            spatterns, 
            nothing,   # no H to start
            nothing    # no cone_maps to start
        )

        find_sparsity_patterns!(
            chordal_info, 
            A, b, 
            cones, 
            settings.chordal_decomposition_merge_method)

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

    rng_cones = _make_rng_conesT(cones)

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

    # linear indices of the nonzeros within this cone
    agg_sparsity = findall(nz_mask)

    if length(agg_sparsity) == nvars(cone)
        return #not decomposable
    end 
    
    L, ordering = find_graph!(agg_sparsity, cone.dim)
    spattern = SparsityPattern(L, ordering, coneidx, merge_method)

    if num_cliques(spattern.sntree) == 1
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
    ncliques  = sum([num_cliques(spattern.sntree) for spattern in chordal_info.spatterns])

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

#PJG: if this is implemented using a boolean vector, then the n can be dropped 
#since it should be inferable from the length of the vector.
function find_graph!(linearidx::Vector{Int}, n::Int) 
	
    rows,cols = upper_triangular_index_to_coords(linearidx)
    
    #PJG: probably the "ones" aren't needed at all here, but 
    #need to check if QDLDL will support that.   Otherwise 
    #need to go into lower level functions 
    #PJG: at the very least, the ones should be a vector of
    #integers or bools
    pattern = sparse(rows, cols, ones(length(rows)), n, n)

	F = QDLDL.qdldl(pattern, logical = true)

    L = F.L
    ordering = F.perm

	# this takes care of the case that QDLDL returns an unconnected adjacency matrix L
	connect_graph!(L)

    return L, ordering 
end


# PJG An identical function is used in compositecone_type.jl, but implemented 
# over the internal cone types rather then these user facing ones.   
# the only difference is that the function internally is "numel", and 
# the one here is "nvars" for the API types.   I can't just use this 
# one in both places, though, because there is also _make_range_blocks
# that uses internally information about whether Hs will be diagonal 
# or not.   Will leave this here for how, but they need to be consolidated

# PJG: this allocates, but I think it could be made some kind of iterator 
# over the cones with its own internal rng count state.   That would avoid 
# allocating memory.
function _make_rng_conesT(cones::Vector{SupportedCone})

    rngs = sizehint!(UnitRange{Int64}[],length(cones))

    if !isempty(cones)
        start = 1
        for cone in cones
            stop = start + nvars(cone) - 1
            push!(rngs,start:stop)
            start = stop + 1
        end
    end
    return rngs
end


"""
	find_graph!(ci, linearidx::Array{Int, 1}, n::Int)

Given the indices of non-zero matrix elements in `linearidx`:
- Compute the sparsity pattern and find a chordal extension using `QDLDL` with AMD ordering `F.perm`.
- If unconnected, connect the graph represented by the cholesky factor `L`
"""


# this assumes a sparse lower triangular matrix L
# PJG: not clear what the type T will be here.  Maybe Int / isize
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




#PJG: where do these functions want to live?

#PJG: first function below seems unnecssary, like it could just be a broadcast
#doing that gives me a vector of tuples though, which is not
#convenient for sparse matrix

#PJG fcn name sucks and is confusing given similar plural to next function

# given an array "linearidx" that represent the nonzero entries of the vectorized 
# upper triangular part of an nxn matrix,
# return the rows and columns of the nonzero entries of the original matrix

function upper_triangular_index_to_coords(linearidx::Vector{Int})
	rows = similar(linearidx)
	cols = similar(linearidx)
	for (i, idx) in enumerate(linearidx)
		(rows[i],cols[i]) = upper_triangular_index_to_coord(idx)
	end
	(rows,cols)
end

function upper_triangular_index_to_coord(linearidx::Int)
    col = (isqrt(8 * linearidx) + 1) >> 1 
    row = linearidx - triangular_number(col - 1)
    (row,col)
end

function coord_to_upper_triangular_index(coord::Tuple{Int, Int})
    (i,j) = coord
    if i <= j
        return triangular_number(j-1) + i
    else
        return triangular_number(i-1) + j
    end
end

