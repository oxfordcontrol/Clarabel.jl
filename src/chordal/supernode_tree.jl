
# A structure to represent and analyse the sparsity pattern of the input matrix L.
mutable struct SuperNodeTree

  	# vertices of supernodes stored in one array (also called residuals)
	snode::Vector{VertexSet} 
  	# post order of supernodal elimination tree
  	snode_post::Vector{Int} 
  	# parent of each supernodes
	snode_parent::Vector{Int}  
  	# children of each supernode 
	snode_children::Vector{VertexSet}
  	# post ordering of the vertices in elim tree σ(j) = v
	post::Array{Int} 
  	# parent of each vertex in the elimination tree
	parent::Array{Int}
  	# vertices of clique seperators
	separators::Vector{VertexSet} 

  	# sizes of submatrices defined by each clique, sorted by post-ordering, 
	# e.g. size of clique with order 3 => nblk[3].   Only populated 
	# after a post-merging call to `calculate_block_dimensions!`
	nblk::Option{Vector{Int}}

	# number of nonempty supernodes / cliques in tree  
	n_cliques::Int 

	function SuperNodeTree(L::SparseMatrixCSC{T}) where {T}

		parent   = parent_from_L(L)
		children = children_from_parent(parent)
		post 	 = zeros(Int, length(parent))
		post_order!(post, parent, children, length(parent))   
		
    	degree   = higher_degree(L)
		snode, snode_parent = find_supernodes(parent, post, degree)

		snode_children = children_from_parent(snode_parent)
		snode_post     = zeros(Int,length(snode_parent)) 
    	post_order!(snode_post, snode_parent, snode_children, length(snode_parent))   

		# Her we find separators in all cases, unlike COSMO which defers until
		# after merging for the clique graph merging case.  These are later 
		# modified in the clique-graph merge case in the call to add_separators
		separators = find_separators(L, snode)

		# nblk will be allocated to the length of the *post-merging*
		# supernode count in calculate_block_dimensions!
		nblk = nothing

		# number of cliques / nonempty supernodes.  decrements as supernodes are merged
		n_cliques = length(snode)

		new(snode, snode_post, snode_parent, snode_children, 
			post, parent, separators, nblk, n_cliques)

	end

end

# ------------------------------------------
# public interface function to SuperNodeTree
# ------------------------------------------

function get_post_order(sntree::SuperNodeTree, i::Int)
	return sntree.snode_post[i]
end

function get_snode(sntree::SuperNodeTree, i::Int)
	return sntree.snode[sntree.snode_post[i]]
end

function get_separators(sntree::SuperNodeTree, i::Int)
	return sntree.separators[sntree.snode_post[i]]
end

function get_clique_parent(sntree::SuperNodeTree, clique_index::Int)
	return sntree.snode_parent[sntree.snode_post[clique_index]]
end

# the block sizes are stored in post order, e.g. if clique 4 (stored in pos 4) 
# has order 2, then nblk[2] represents the cardinality of clique 4
function get_nblk(sntree::SuperNodeTree, i::Int)
	return sntree.nblk[i]
end 

function get_overlap(sntree::SuperNodeTree, i::Int)
	return length(sntree.separators[sntree.snode_post[i]])
end

function get_clique(sntree::SuperNodeTree, ind::Int)
	c = sntree.snode_post[ind]
	return union(sntree.snode[c], sntree.separators[c])
end

function get_decomposed_dim_and_overlaps(sntree::SuperNodeTree)
	dim = 0
	overlaps = 0
	for i = 1:sntree.n_cliques
	  dim      += triangular_number(get_nblk(sntree, i))
	  overlaps += triangular_number(get_overlap(sntree, i))
	end
	(dim, overlaps)
end 


# ---------------------------
# utility functions for SuperNodeTree


function parent_from_L(L::SparseMatrixCSC{T}) where {T}

	parent = zeros(Int, L.n)
	# loop over Vertices of graph
	for i = 1:L.n
		parent[i] = find_parent_direct(L, i)
	end
	return parent
end

function find_parent_direct(L::SparseMatrixCSC{T}, v::Int) where{T}
	v == size(L, 1) && return 0
	return L.rowval[L.colptr[v]]
end 


function children_from_parent(parent::Vector{Int})

	children = [VertexSet() for i = 1:length(parent)]
	for (i,pi) = enumerate(parent)
		pi != 0 && push!(children[pi], i)
	end
	return children
end


# This could be faster for the case that merges happened, i.e. nc != length(parent)

function post_order!(post::Vector{Int}, parent::Vector{Int}, children::Vector{VertexSet}, nc::Int)

	order = (nc + 1) * ones(Int, length(parent))
	root  = findfirst(x -> x == 0, parent)

	stack = sizehint!(Int[], length(parent))
	push!(stack, root)

	resize!(post, length(parent))
	post .= 1:length(parent)

	i = nc

	while !isempty(stack)
		v = pop!(stack)
		order[v] = i
		i -= 1

		# maybe faster to append to the stack vector and then
		# sort a view of what was added, but this way gets
		# the children sorted and keeps everything consistent
		# with the COSMO implementation for reference
		append!(stack, sort!(children[v]))
	end

	sort!(post, by = x-> order[x])

	# if merges happened, remove the entries pointing to empty arrays / cliques
	nc != length(parent) && resize!(post, nc)
	return post
end


# findall the cardinality of adj+(v) for all v in V
function higher_degree(L::SparseMatrixCSC{T}) where{T}
	
	degree = zeros(Int, L.n)
	for v = 1:(L.n-1)
		degree[v] = L.colptr[v + 1] - L.colptr[v]
	end
	return degree
end



function find_supernodes(parent::Vector{Int}, post::Vector{Int}, degree::Vector{Int})
	
	snode = [VertexSet() for i = 1:length(parent)]

 	snode_parent, snode_index = pothen_sun(parent, post, degree)
	
	for i = 1:length(parent)
		f = snode_index[i]
		if f < 0
			push!(snode[i], i)
		else
			push!(snode[f], i)
		end
	end
	filter!(x -> !isempty(x), snode)
	return snode, snode_parent

end


# Algorithm from A. Poten and C. Sun: Compact Clique Tree Data Structures in Sparse Matrix Factorizations (1989)

function pothen_sun(parent::Vector{Int}, post::Vector{Int}, degree::Vector{Int})
	
	N = length(parent)

	# if snode_index[v] < 0 then v is a rep vertex, otherwise v ∈ supernode[snode_index[v]]
	snode_index  = -ones(Int, N) 
	snode_parent = -ones(Int, N)

	# This also works as array of Int[], which might be faster
	# note this arrays is local to the function, not the one 
	# contained in the SuperNodeTree
	children = [VertexSet() for i = 1:length(parent)]

	root_index = findfirst(x -> x == 0, parent)

	# go through parents of vertices in post_order
	for v in post

		if parent[v] == 0
			push!(children[root_index], v)
		else
			push!(children[parent[v]], v)
		end

		# parent is not the root
		if parent[v] != 0
			if degree[v] - 1 == degree[parent[v]] && snode_index[parent[v]] == -1
				# Case A: v is a representative vertex
				if snode_index[v] < 0
					snode_index[parent[v]] = v
					snode_index[v] -= 1
				# Case B: v is not representative vertex, add to sn_ind[v] instead
				else
					snode_index[parent[v]] = snode_index[v]
					snode_index[snode_index[v]] -= 1
				end
			else
				if snode_index[v] < 0
					snode_parent[v] = v
				else
					snode_parent[snode_index[v]] = snode_index[v]
				end
			end
		end

		# k: rep vertex of the snd that v belongs to
		if snode_index[v] < 0
			k = v
		else
			k = snode_index[v]
		end
		# loop over v's children
		v_children = children[v]
		if !isempty(v_children)
			for w in v_children
				if snode_index[w] < 0
					l = w
				else
					l = snode_index[w]
				end
				if l != k
					snode_parent[l] = k
				end
			end
		end
	end # loop over vertices

	# representative vertices
	repr_vertex = findall(x-> x < 0, snode_index)
	# vertices that are the parent of representative vertices
	repr_parent = snode_parent[repr_vertex]

	# resize and reset snode_parent to take into account that all non-representative 
	# arrays are removed from the parent structure
	resize!(snode_parent, length(repr_vertex))
	snode_parent .= 0

	for (i, rp) in enumerate(repr_parent)
		ind = findfirst(x -> x == rp, repr_vertex)
		isnothing(ind) && (ind = 0)
		snode_parent[i] = ind
	end

	return snode_parent, snode_index
end


function find_separators(
	L::SparseMatrixCSC, 
	snode::Vector{VertexSet}
)
	separators = sizehint!(VertexSet[], length(snode))

	for sn in snode 
		vrep    = minimum(sn)
		adjplus = find_higher_order_neighbors(L, vrep)

		sep = VertexSet()
		for neighbor in adjplus
			if neighbor ∉ sn
				push!(sep, neighbor)
			end
		end
		push!(separators, sep)
	end

	return separators

end



function find_higher_order_neighbors(L::SparseMatrixCSC, v::Int)
	v == size(L, 1) && return 0
	col_ptr = L.colptr
	row_val = L.rowval
	return view(row_val, col_ptr[v]:col_ptr[v + 1] - 1)
end


# The the block dimension vector is the size of the remaining 
# (unemptied) supernodes.   Before this call, t.nblk should be 
# nothing 

function calculate_block_dimensions!(t::SuperNodeTree)
  n = t.n_cliques
  t.nblk = zeros(Int,n)
  for iii = 1:n
    c = t.snode_post[iii]
    t.nblk[iii] = length(t.separators[c]) + length(t.snode[c])
  end
end



# Takes a SuperNodeTree and reorders the vertices in each supernode (and separator) to have consecutive order.

# The reordering is needed to achieve equal column structure for the psd completion of the dual variable `Y`. 
# This also modifies `ordering` which maps the vertices in the `sntree` back to the actual location in the 
# not reordered data, i.e. the primal constraint variable `S` and dual variables `Y`.


function reorder_snode_consecutively!(t::SuperNodeTree, ordering::Vector{Int})

	# determine permutation vector p and permute the vertices in each snd
	p = zeros(Int,length(t.post))

	k = 1
	for i in t.snode_post

	  snode = t.snode[i]
	  n = length(snode)
	  viewp  = view(p, k:k+n-1)
	  viewp .= t.snode[i]
	  sort!(viewp)

	  #assign k:(k+n)-1 to the OrderedSet snode,
	  #dropping the previous values
	  empty!(snode)
	  foreach(v->push!(snode,v), k:(k+n)-1)

	  k += n
	end

	# permute the separators as well
	p_inv = invperm(p)
	for sp in t.separators

		# use here the permutation vector p as scratch before flushing
		# the separator set and repopulating.  Assumes that the permutation
		# will be at least as long as the largest separator set

		@assert length(p) >= length(sp)
		tmp = view(p,1:length(sp))

		tmp .= Iterators.map(x -> p_inv[x], sp)  
		empty!(sp)
		foreach(v->push!(sp,v), tmp)
	end

	# because I used 'p' as scratch space, I will 
	# ipermute using pinv rather than permute using p

	invpermute!(ordering, p_inv)
	return nothing
end

