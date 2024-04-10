# PJG: edge weight method is currently an enum and kind of sucks.   Need to 
#replace this with something more flexible.  Also can't be used in a 
#default constructor configuration like this 

"""
    CliqueGraphMergeStrategy(edge_weight::EdgeWeightMethod) <: AbstractMergeStrategy

The (default) merge strategy based on the *reduced* clique graph ``\\mathcal{G}(\\mathcal{B}, \\xi)``, for a set of cliques ``\\mathcal{B} = \\{ \\mathcal{C}_1, \\dots, \\mathcal{C}_p\\}`` where the edge set ``\\xi`` is obtained by taking the edges of the union of clique trees.

Moreover, given an edge weighting function ``e(\\mathcal{C}_i,\\mathcal{C}_j) = w_{ij}``, we compute a weight for each edge that quantifies the computational savings of merging the two cliques.
After the initial weights are computed, we merge cliques in a loop:

**while** clique graph contains positive weights:
- select two permissible cliques with the highest weight ``w_{ij}``
- merge cliques ``\\rightarrow`` update clique graph
- recompute weights for updated clique graph

Custom edge weighting functions can be used by defining your own `CustomEdgeWeight <: AbstractEdgeWeight` and a corresponding `edge_metric` method. By default, the `ComplexityWeight <: AbstractEdgeWeight` is used which computes the weight based
on the cardinalities of the cliques: ``e(\\mathcal{C}_i,\\mathcal{C}_j)  = |\\mathcal{C}_i|^3 + |\\mathcal{C}_j|^3 - |\\mathcal{C}_i \\cup \\mathcal{C}_j|^3``.

See also: *Garstka, Cannon, Goulart - A clique graph based merging strategy for decomposable SDPs (2019)*
"""
mutable struct CliqueGraphMergeStrategy <: AbstractMergeStrategy
  stop::Bool                                  # a flag to indicate that merging should be stopped
  edges::SparseMatrixCSC{Float64, Int}        # the edges and weights of the reduced clique graph
  p::Array{Int, 1}                            # as a workspace variable to store the sorting of weights
  adjacency_table::Dict{Int, VertexSet}       # a double structure of edges, to allow fast lookup of neighbors
  edge_weight::EdgeWeightMethod               # used to dispatch onto the correct scoring function
  clique_tree_recomputed::Bool                # a flag to indicate whether a final clique tree has been recomputed from the clique graph
  function CliqueGraphMergeStrategy(; edge_weight = CUBIC::EdgeWeightMethod)
    new(false, spzeros(0, 0),  Int[], Dict{Int, VertexSet}(), edge_weight, false)
  end
end


function is_done(strategy::CliqueGraphMergeStrategy)
  strategy.stop
end


function initialise!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

		# this merge strategy is clique graph-based, we give up the tree structure and add 
		# the seperators to the supernodes.  The supernodes then represent the full clique.
		# after clique merging a new clique tree will be computed in post_process_merge!
    # for this type 

		for (snode,separator) in zip(t.snode, t.separators)
      union!(snode, separator)
    end

    # PJG: wipe the parent and child sets as in COSMO SuperNodeTree constructor.
    # Here I do it exactly as in COSMO, but probably better to resize will deallocation,
    # or even just leave them untouched .

    # PJG: instead of marking the parent as -1, I could mark it as nothing and 
    # use a vector of Options in Rust maybe.  Possibly overkill
    for i in eachindex(t.snode_parent)
      t.snode_parent[i] = -1
      t.snode_children[i] = VertexSet()
    end

    # compute the edges and intersections of cliques in the reduced clique graph
    rows, cols = compute_reduced_clique_graph!(t.separators, t.snode)
  
    weights = compute_weights!(rows, cols, t.snode, strategy.edge_weight)
  
    strategy.edges = sparse(rows, cols, weights, t.n_snode, t.n_snode)
    strategy.p     = zeros(Int, length(strategy.edges.nzval))
    strategy.adjacency_table = compute_adjacency_table(strategy.edges, t.n_snode)
    return nothing
  end


"Find the next two cliques in the clique graph `t` to merge."
function traverse(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

   p = strategy.p
   # find edge with highest weight, if permissible return cliques
   edge = max_elem(strategy.edges)
   ispermissible(edge, strategy.adjacency_table, t.snode) && return [edge[1]; edge[2]]
   # else: sort the weights in edges.nzval to find the permutation p
   sortperm!(view(p, 1:length(strategy.edges.nzval)), strategy.edges.nzval, alg = QuickSort, rev = true)

   # try edges with decreasing weight and check if the edge is permissible
  for k = 2:length(strategy.edges.nzval)
    edge = edge_from_index(strategy.edges, p[k])
    if ispermissible(edge, strategy.adjacency_table, t.snode)
      # PJG: this should be a tuple, not a 2 element vector
      return [edge[1]; edge[2]]
    end
  end

end


function evaluate(strategy::CliqueGraphMergeStrategy, _t::SuperNodeTree, cand::Vector{Int})
  
  do_merge = (strategy.edges[cand[1], cand[2]] >= 0)

  if !do_merge
    strategy.stop = true
  end
  return do_merge
end


function merge_two_cliques!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree, cand::Vector{Int})
  c_1 = cand[1]
  c_2 = cand[2]
  # merge clique c_2 into c_1
  union!(t.snode[c_1], t.snode[c_2])

  #PJG: don't understand why it is drained here instead of dropped, 
  #but I guess it would mess up the size of all of the other arrays
  empty!(t.snode[c_2])

  # PJG: decrement number of mergeable cliques in graph
  t.n_snode -= 1

  return nothing
end




  # PJG: not sure yet if I care about logging 
#   "Store information about the merge of the two merge candidates `cand`."
# function log_merge!(t::SuperNodeTree, do_merge::Bool, cand::Array{Int, 1})
#   t.merge_log.clique_pairs = vcat(t.merge_log.clique_pairs, cand')
#   push!(t.merge_log.decisions, do_merge)
#   do_merge && (t.merge_log.num += 1)
#   return nothing
# end


"After a merge happened, update the reduced clique graph."
function update_strategy!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree, cand::Vector{Int}, do_merge::Bool)

  # After a merge operation update the information of the strategy
  if do_merge

    c_1_ind = cand[1]
    c_removed = cand[2]
    edges = strategy.edges
    n = size(edges, 2)
    adjacency_table = strategy.adjacency_table

    c_1 = t.snode[c_1_ind]
    neighbors = adjacency_table[c_1_ind]
    # neighbors exclusive to the removed clique (and not c1)
    new_neighbors = setdiff(adjacency_table[c_removed], neighbors, c_1_ind)

    # recalculate edge values of all of c_1's neighbors
    for n_ind in neighbors
        if n_ind != c_removed
          neighbor = t.snode[n_ind]
          edges[max(c_1_ind, n_ind), min(c_1_ind, n_ind)] = edge_metric(c_1, neighbor, strategy.edge_weight)
        end
    end

    # point edges exclusive to removed clique to "surviving" clique 1
    for n_ind in new_neighbors
        neighbor = t.snode[n_ind]
        edges[max(c_1_ind, n_ind), min(c_1_ind, n_ind)]  = edge_metric(c_1, neighbor, strategy.edge_weight)
    end

    # overwrite the weight to any "deleted" edges that still contain a link to c_removed
    strategy.edges[c_removed+1:n, c_removed] .= 0
    strategy.edges[c_removed, 1:c_removed] .= 0
    dropzeros!(edges)

    # update adjacency table in a similar manner
    union!(adjacency_table[c_1_ind], new_neighbors)
    for new_neighbor in new_neighbors
      push!(adjacency_table[new_neighbor], c_1_ind)
    end
    delete!(adjacency_table, c_removed)
    for (key, set) in adjacency_table
      delete!(set, c_removed)
    end
  end

  return nothing
end


function post_process_merge!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

  # since for now we have a graph, not a tree, a post ordering or a parent structure does not make sense. Therefore just number
  # the non-empty supernodes in t.snd
  t.snode_post = findall(x -> !isempty(x), t.snode)
  t.snode_parent = -ones(Int, length(t.snode))

  # recompute a clique tree from the clique graph
  t.n_snode > 1 && clique_tree_from_graph!(strategy, t)

  #turn the unordered sets for t.snd and t.sep into sorted versions
  # PJG : danger here.   This was where COSMO was converting from 
  # sets to arrays. 

  # PJG: It is not clear to me whether this step is still 
  # necessary.   The next operation on these sets is in 
  # reorder_snode_consecutively!, and that set also does 
  # some manipulation of the ordering and does so through
  # intermediate arrays.   It might be better to handle 
  # the sorting there instead.   I will leave this in for 
  # now so that comparisons with COSMO are easier.
  foreach(s->sort!(s), t.snode)
  foreach(s->sort!(s), t.separators)

  return nothing
end


# PJG: functions from here down are implemented as trait-like methods 
# for the clique graph strategy only.   So not part of of the general 
# strategy trait, but also not low level utilities.  

"""
    clique_tree_from_graph!(strategy::CliqueGraphMergeStrategy, clique_graph::SuperNodeTree)

Given the cliques and edges of a clique graph, this function computes a valid clique tree.

This is necessary to perform the psd completion step of the dual variable after solving the problem.
"""
function clique_tree_from_graph!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)
  # a clique tree is a maximum weight spanning tree of the clique graph where the edge weight is the 
  # cardinality of the intersection between two cliques compute intersection value for each edge 
  # in clique graph
  clique_intersections!(strategy.edges, t.snode)

  # find a maximum weight spanning tree of the clique graph using Kruskal's algorithm
  kruskal!(strategy.edges, t.n_snode)

  # determine the root clique of the clique tree (it can be any clique, but we use the 
  # clique that contains the vertex with the highest order)
  determine_parent_cliques!(t.snode_parent, t.snode_children, t.snode, t.post, strategy.edges)

  # recompute a postorder for the supernodes
  t.snode_post = post_order(t.snode_parent, t.snode_children, t.n_snode)

  # PJG: here I just hose the separators.  They are reconstructed
  # in the split_cliques! call, so not clear why the separators 
  # need to be flushed here.   Maybe move this flush to the 
  # split_cliques function 

  t.separators = [VertexSet() for i = 1:length(t.snode)]

  # split clique sets back into separators and supernodes
  split_cliques!(t.snode, t.separators, t.snode_parent, t.snode_post, t.n_snode)

  # PJG : how does this get used?
  strategy.clique_tree_recomputed = true
  return nothing

end



