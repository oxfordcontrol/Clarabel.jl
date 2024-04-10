
"""
	SuperNodeTree

A structure to represent and analyse the sparsity pattern of the input matrix L.

# PJG: I think the following note always applies now, since the merge strategy is always graph-based
# confusing though, since add_separators is called, which definitely populates the separators.  It 
# looks to me like only the snode_children is empty? Basically no idea WTF.


# PJG: note following might not remain correct if I push the treatment 
# of separators into the clique merging initialization functions. 

### Note:
Based on the `merge_strategy` in the constructor, SuperNodeTree might be initialised
as a graph, i.e. seperators `sep` are left empty as well as `snd_child` and `snd_post`.

After cliques have been merged, a valid clique tree will be recomputed from the
consolidated clique graph.
"""

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


	# PJG: not clear that nblk is actually required.  It is not 
	# clear if nblk and n_snode actually belong here if th is 
	# meant to be a standalone analytical object 

  	# sizes of submatrices defined by each clique, sorted by post-ordering, e.g. size of clique with order 3 => nblk[3]
	nblk::Vector{Int}
	# number of supernodes / cliques in tree  
	# PJG: Needed because the node count is decreased on merge, but 
	# the `snode` vector of sets is not resized.  Instead, one of the 
	# sets is just drained.   Maybe should be labelled as "nonempty" cliques
	# for the documentation line above.
	n_snode::Int 

  # PJG: not clear that log and strategy are actually required, unless maybe 
  # I want to allow for a NoMerge method
	#merge_log::MergeLog
	#strategy::AbstractMergeStrategy

	# PJG: do I want to be prescriptive about the type T here?

	function SuperNodeTree(L::SparseMatrixCSC{T}) where {T}

		parent   = etree(L)
		children = children_from_parent(parent)
		post     = post_order(parent, children, length(parent))   
		
    	degree   = higher_degree(L)
		snode, snode_parent = find_supernodes(parent, post, degree)

		snode_children = children_from_parent(snode_parent)
    	snode_post     = post_order(snode_parent, snode_children, length(snode_parent))   

    	#PJG: maybe parent + children + post should be its own data structure?

		# PJG: here, I will find separators in all cases, unlike COSMO
		# I will then do the "add separators" step inside the initializer
		# for the graph merge strategy
		separators = find_separators(L, snode)


		# PJG: do I need to initialize nblk?  It is currently [0]
		# not sure where it is computed

		# PJG: removed the separate treatment of separators 
		# based on the merge strategy.   First initialization / 
		# reworking of separators is not tasked to the initialise
		# function of the merge strategy.     I now don't need 
		# the merge strategy any more, and SnTree is standalone

   	 	#PJG: temporarily (?) removed MergeLog from struct and constructor

    	#PJG: not clear why nblk is initialized to length 1, instead of length(snode) or something
		new(snode, snode_post, snode_parent, snode_children, post, parent, separators, [0], length(snode_post))

	end

end

# ------------------------------------------
# public interface function to SuperNodeTree
# ------------------------------------------

#PJG: don't make SuperNodeTree internals public, because 
#the indexing is also passing through snode_post

function num_cliques(sntree::SuperNodeTree)
	return sntree.n_snode
end

function get_post_order(sntree::SuperNodeTree)
	return sntree.snode_post
end

#PJG: problem here, two functions of the same name 
function get_post_order(sntree::SuperNodeTree, i::Int)
	return sntree.snode_post[i]
end
# Using the post order ensures that no empty arrays from the clique merging are returned
function get_snode(sntree::SuperNodeTree, i::Int)
	return sntree.snode[sntree.snode_post[i]]
end

function get_separator(sntree::SuperNodeTree, i::Int)
	return sntree.separators[sntree.snode_post[i]]
end

function get_clique_parent(sntree::SuperNodeTree, clique_index::Int)
	return sntree.snode_parent[sntree.snode_post[clique_index]]
end
# the block sizes are stored in post order, e.g. if clique 4 (stored in pos 4) 
# has order 2, then nblk[2] represents the cardinality of clique 4
function get_nblk(sntree::SuperNodeTree, i::Int)
	return sntree.nblk[i]::Int
end

function get_overlap(sntree::SuperNodeTree, i::Int)
	return length(sntree.separators[sntree.snode_post[i]])
end

function get_decomposed_dim_and_overlaps(sntree::SuperNodeTree)
	dim = 0
	overlaps = 0
	for i = 1:num_cliques(sntree)
	  dim      += triangular_number(get_nblk(sntree, i))
	  overlaps += triangular_number(get_overlap(sntree, i))
	end
	(dim, overlaps)
end 

" Return clique with post order `ind` (prevents returning empty arrays due to clique merging)"
# PJG: this is taken from the "tree based" version from COSMO.   I think, but 
# am not certain, that it is impossible to execute the "graph based" version
# which is only called if the tree has not been recomputed 
function get_clique(sntree::SuperNodeTree, ind::Int)
	c = sntree.snode_post[ind]
	return union(sntree.snode[c], sntree.separators[c])
end

function get_clique_by_index(sntree::SuperNodeTree, i::Int)
	return union(sntree.snode[i], sntree.separators[i])
end



# ---------------------------
# PJG: private / crate level utility functions for SuperNodeTree

# PJG : not clear how etree here differs from etree in QDLDL 
# Maybe should be something like "parents_from_graph" or something

function etree(L::SparseMatrixCSC{T}) where {T}
	parent = zeros(Int, L.n)
	# loop over Vertices of graph
	for i = 1:L.n
		parent[i] = find_parent_direct(L, i)
	end
	return parent
end


#PJG : this is only used in etree above?   Inline somehow?  
#Not clear I am really making a etree here in the first place
function find_parent_direct(L::SparseMatrixCSC{T}, v::Int) where{T}
	v == size(L, 1) && return 0
	return L.rowval[L.colptr[v]]
end 


function children_from_parent(parent::Vector{Int})

	# PJG: why is constructor not used here?
	children = [VertexSet() for i = 1:length(parent)]
	for (i,pi) = enumerate(parent)
		pi != 0 && push!(children[pi], i)
	end
	return children
end


# PJG: MG says that could be faster for the case that merges happened, i.e. nc != length(parent)

function post_order(parent::Vector{Int}, children::Vector{VertexSet}, nc::Int)

	order = (nc + 1) * ones(Int, length(parent))
	root = findfirst(x -> x == 0, parent)
	stack = Int[root]

	i = nc

	while !isempty(stack)
		v = pop!(stack)
		order[v] = i
		i -= 1
		#PJG: sort/collect here does not belong, but 
		#introduced to ensure consistent testing 
		#when comparing to COSMO
		push!(stack, sort(collect(children[v]))...)
	end

	post = collect(1:length(parent))

  	#PJG: is this the same as invperm(order)?
  	#depends on whether order is a permutation or not
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
	
  	# PJG: could be removed once issue described in next function below 
  	# for different types of initialization is resolved 
	snode = initialise_sets(length(parent))

 	 snode_parent, snode_index = pothen_sun(parent, post, degree)
	
	for iii = 1:length(parent)
		f = snode_index[iii]
		if f < 0
			push!(snode[iii], iii)
		else
			push!(snode[f], iii)
		end
	end
	filter!(x -> !isempty(x), snode)
	return snode, snode_parent

end


#PJG: this function is probably not needed since it just makes a bunch of empty 
#sets.   The issue is that the COSMO code has another "initialise_sets" that 
#accepts strategy of other "AbstractMergeStrategy" instead, which returns instead 
#a vector of vectors, rather than a vector of sets.   This perhaps explains 
#why argument types in COSMO were defined in terms of unions over vectors of 
#vectors and vectors of sets.   Need to look into which situations use each of 
#these two different abstract merge types 

#I think that the "GraphBasedMerge" is the only one we care about, and that the 
#other one was used for testing against things like parent-child merge methods

# function initialise_sets(N::Int, strategy::AbstractGraphBasedMerge)
# 	return [Set{Int}() for i = 1:N]
# end

function initialise_sets(n::Int)
	return [VertexSet() for i = 1:n]
end


# Algorithm from A. Poten and C. Sun: Compact Clique Tree Data Structures in Sparse Matrix Factorizations (1989)
function pothen_sun(parent::Vector{Int}, post::Vector{Int}, degree::Vector{Int})
	N = length(parent)
	snode_index  = -1 * ones(Int, N) # if snode_index[v] < 0 then v is a rep vertex, otherwise v ∈ supernode[snode_index[v]]
	snode_parent = -1 * ones(Int, N)

	# PJG : sets here Vector, not Set?
	children = 	[Int[] for i = 1:length(parent)]

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



# PJG: this function is used in COSMO during SuperNodeTree initialization,
# but the purpose is to add separators to the supernodes as a initialization
# step for the graph-based merge strategy.   I will not use this, and 
# will instead take the separators as they are found in the initialization
# (i.e. tree-based) and simply add separators to the supernodes in the
# initialize! of the clique graph merge strategy.   The graph merge 
# strategy will then only mess up the tree internally, and rebuilds it 
# before exiting 
function add_separators!(
	L::SparseMatrixCSC, 
	snodes::Vector{VertexSet}, 
	separators::Vector{VertexSet}, 
	snode_parent::Vector{Int}
)
	
	error("should not be used anymore")
		
	# for i = 1:length(snode_parent)
	# 	snode = snodes[i]
	# 	sep = separators[i]
	# 	vrep = minimum(snode)

	# 	adjplus = find_higher_order_neighbors(L, vrep)
	# 	for neighbor in adjplus
	# 		if !in(neighbor, snode)
	# 			push!(snode, neighbor)
	# 			push!(sep, neighbor)
	# 		end
	# 	end
	# end
end


# PJG: I changed the argument types to support Sets instead 
# of a union of Vector{Vector} and Vector{Set}.   This 
# function in COSMO is only called in the case of a tree-based
# merge, but I will try to get Sets to work for both tree
# and graph methods 
function find_separators(
	L::SparseMatrixCSC, 
	snode::Vector{VertexSet}
)

	# PJG: this fcn reimplemented completely
	# so bugs are possible / likely 

	separators = sizehint!(VertexSet[], length(snode))

	for sn in snode 
		vrep    = minimum(sn)
		adjplus = find_higher_order_neighbors(L, vrep)

		#PJG: this way is maybe too slow 
		#push!(separators, setdiff(Set(adjplus), sn))

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
  # PJG: needless copy here.  Should be a view?
	return row_val[col_ptr[v]:col_ptr[v + 1] - 1]
end



# calculate block sizes (notice: the block sizes are stored in post order)
# PJG: need global search and replace for "Nc"

#PJG: here the block dimension vector is the size of the remaining 
#(uncleared) supernodes.   This maybe explains why it was set to [0]
# in the constructor.   When does it get assigned to be the right 
#size?   Maybe initialize it as zeros(length(snode)) in the constructor
#before merging, and then mark all clear nodes as -1 dimension or similar?

function calculate_block_dimensions!(t::SuperNodeTree)
  Nc = t.n_snode
  t.nblk = zeros(Nc)
  for iii = 1:Nc
    c = t.snode_post[iii]
    t.nblk[iii] = length(t.separators[c]) + length(t.snode[c])
  end
end



"""
		reorder_snpde_consecutively!(sntree, ordering)

Takes a SuperNodeTree and reorders the vertices in each supernode (and separator) to have consecutive order.

The reordering is needed to achieve equal column structure for the psd completion of the dual variable `Y`. 
This also modifies `ordering` which maps the vertices in the `sntree` back to the actual location in the 
not reordered data, i.e. the primal constraint variable `S` and dual variables `Y`.
"""

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

		# use here the permutation vector p as temporary 
		# storage before flushing the separator set 
		# an repopulating.  Assumes that the permutation
		# p will be at least as long as the largest 
		# separator set
		@assert length(p) >= length(sp)
		tmp = view(p,1:length(sp))

		tmp .= Iterators.map(x -> p_inv[x], sp)  # PJG: Base.map fails here?
		empty!(sp)
		foreach(v->push!(sp,v), tmp)
	end

	#PJG: because I used 'p' as scratch space, I will 
	#ipermute using pinv rather than permute using p
	#Bug city maybe
	invpermute!(ordering, p_inv)
	return nothing
end

