# -------------------------------------
# Chordal Decomposition Information
# -------------------------------------
mutable struct ChordalInfo{T}

    # sketch of the original problem
    init_dims::Tuple{Int64, Int64}       # (n,m) dimensions of the original problem
    init_cones::Vector{SupportedCone}    # original cones of the problem
    
    # PJG: do I care about this?
    init_num_psd_cones::Int              # number of psd cones in the original problem

    # decomposed problem data 
    spatterns::Vector{SparsityPattern}   # sparsity patterns for decomposed cones
    
    # map every cone in the decomposed problem to the equivalent
    # or undecomposed cone in the original problem. As post->init
    # PJG: not currently used?  Maybe for compact transform?
    post_to_init_map::Dict{Int, Int}      

    # "H" matrix for the standard chordal problem transformation 
    # remains as nothing if the "compact" transform is used 
    # PJG: I really don't like that this forces the whole 
    # type to be parametric
    H::Option{SparseMatrixCSC{T}}

    function ChordalInfo(A::SparseMatrixCSC{T}, b::Vector{T}, cones::Vector{SupportedCone}, settings::Clarabel.Settings) where {T <: AbstractFloat}
    
        # initial problem data
        init_dims  = size(A)
        init_cones = deepcopy(cones)   #PJG: copy here potentially very bad.   Only do if needed

        # decomposed problem data. Not initialized yet
        #PJG: no idea if this should be here or how / when 
        #to initialize it 
        init_num_psd_cones = 0   # PJG: not used?
        spatterns = Vector{SparsityPattern}[]

        post_to_init_map = Dict{Int, Int}()   # PJG: not used?
  
        chordal_info = new{T}(
            init_dims, 
            init_cones, 
            init_num_psd_cones, 
            spatterns, 
            post_to_init_map,
            nothing
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

    # find the sparsity patterns of the PSD cones
    for (coneidx, cone) in enumerate(cones)
        rowrange = rng_cones[coneidx]   # PJG: should be a zip

        if !isa(cone, PSDTriangleConeT)
            continue
        end

        agg_sparsity = find_aggregate_sparsity(A, b, rowrange)

        analyse_sparsity_pattern!(
            chordal_info, 
            agg_sparsity, 
            cone, 
            coneidx,
            merge_method)
    end
end


# PJG: not clear why I am passing the cone here, since I only want the 
# dimension and it is guaranteed to be a PSD triangle

function analyse_sparsity_pattern!(
    chordal_info::ChordalInfo, 
    agg_sparsity::Vector{Int}, 
    cone::SupportedCone, 
    coneidx::Int,
    merge_method::Symbol
) 

    @assert length(agg_sparsity) <= nvars(cone)

    if length(agg_sparsity) == nvars(cone)
        return #not decomposable
    end 
    
    L, ordering = find_graph!(chordal_info, agg_sparsity, cone.dim)
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



#PJG: it might be faster to just find all rows of the full A that have nonzeros.   That would avoid
#traversing A.rowval multiple times, once for each cone. 
function find_aggregate_sparsity(
    A::SparseMatrixCSC{T}, 
    b::Vector{T}, 
    rowrange::UnitRange{Int}
) where {T <: AbstractFloat}

    active = falses(length(rowrange))
    # mark all rows of A with nonzero entries in this range 
    for r in A.rowval
        if in(r, rowrange)
            active[r - rowrange.start + 1] = true
        end
    end
    active .= active .|| view(b,rowrange) .!= 0
    return findall(active)
end 



function get_decomposed_dim_and_overlaps(chordal_info::ChordalInfo{T}) where{T}

    cones = chordal_info.init_cones
    sum_cols     = 0
    sum_overlaps = 0
  
    for pattern in chordal_info.spatterns
      cone = cones[pattern.coneidx]
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

# PJG: partitioning into trait and non-trait like functions is a mess here 


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

#PJG: if this is implemented using a boolean vector, then the n can be dropped 
#since it should be inferable from the length of the vector.
function find_graph!(chordal_info::ChordalInfo, linearidx::Vector{Int}, n::Int) 
	
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

# this assumes a sparse lower triangular matrix L
# PJG: not clear what the type T will be here.  Maybe Int / isize
function connect_graph!(L::SparseMatrixCSC{T}) where{T}
 	# unconnected blocks don't have any entries below the diagonal in their right-most column
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
        #PJG: problem here since L[j+1,j] assignment is non-obvious in Rust
		if !connected
			L[j+1, j] = 1
		end
	end
end




#PJG: where do these functions want to live?

#PJG: first function below seems unnecssary, like it could just be a broadcast
#doing that gives me a vector of tuples though, which is not
#convenient for sparse matrix

#PJG fcn number sucks and is confusing given similar plural to next function

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

