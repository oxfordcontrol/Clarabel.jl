# The (default) merge strategy based on the *reduced* clique graph ``\\mathcal{G}(\\mathcal{B}, \\xi)``, 
# for a set of cliques ``\\mathcal{B} = \\{ \\mathcal{C}_1, \\dots, \\mathcal{C}_p\\}``, where the edge 
# set ``\\xi`` is obtained by taking the edges of the union of clique trees.

# Moreover, given an edge weighting function ``e(\\mathcal{C}_i,\\mathcal{C}_j) = w_{ij}``, we compute a 
# weight for each edge that quantifies the computational savings of merging the two cliques.
#
# After the initial weights are computed, we merge cliques in a loop:
#
# **while** clique graph contains positive weights:
# - select two permissible cliques with the highest weight ``w_{ij}``
# - merge cliques ``\\rightarrow`` update clique graph
# - recompute weights for updated clique graph
#
# See also: *Garstka, Cannon, Goulart - A clique graph based merging strategy for decomposable SDPs (2019)*
#
# NB: edges is currently an integer valued matrix since the weights are taken as 
# powers of the cardinality of the intersection of the cliques. This needs to change 
# to floats if empirical edge weight functions are to be supported.

mutable struct CliqueGraphMergeStrategy <: AbstractMergeStrategy
  stop::Bool                                  # a flag to indicate that merging should be stopped
  edges::SparseMatrixCSC{Int,Int}             # the edges and weights of the reduced clique graph
  p::Vector{Int}                              # as a workspace variable to store the sorting of weights
  adjacency_table::Dict{Int, VertexSet}       # a double structure of edges, to allow fast lookup of neighbors
  edge_weight::EdgeWeightMethod               # used to dispatch onto the correct scoring function

  function CliqueGraphMergeStrategy(; edge_weight = CUBIC::EdgeWeightMethod)
    new(false, spzeros(Int, 0, 0),  Int[], Dict{Int, VertexSet}(), edge_weight)
  end
end


function initialise!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

  # this merge strategy is clique-graph based, we give up the tree structure and add 
  # the seperators to the supernodes.  The supernodes then represent the full clique.
  # after clique merging a new clique tree will be computed in post_process_merge!
  # for this type 

  for (snode,separator) in zip(t.snode, t.separators)
    union!(snode, separator)
  end

  for i in eachindex(t.snode_parent)
    t.snode_parent[i] = INACTIVE_NODE
    t.snode_children[i] = VertexSet()
  end

  # compute the edges and intersections of cliques in the reduced clique graph
  rows, cols = compute_reduced_clique_graph!(t.separators, t.snode)

  weights = compute_weights!(rows, cols, t.snode, strategy.edge_weight)

  strategy.edges = sparse(rows, cols, weights, t.n_cliques, t.n_cliques)
  strategy.p     = zeros(Int, length(strategy.edges.nzval))
  strategy.adjacency_table = compute_adjacency_table(strategy.edges, t.n_cliques)
  return nothing

end

function is_done(strategy::CliqueGraphMergeStrategy)
  strategy.stop
end
  

# Find the next two cliques in the clique graph `t` to merge.
function traverse(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

   p = strategy.p
   # find edge with highest weight, if permissible return cliques
   edge = max_elem(strategy.edges)

   if ispermissible(edge, strategy.adjacency_table, t.snode)
      return edge
   end 

   # sort the weights in edges.nzval to find the permutation p
   sortperm!(view(p, 1:length(strategy.edges.nzval)), strategy.edges.nzval, alg = QuickSort, rev = true)

   # try edges with decreasing weight and check if the edge is permissible
  for k = 2:length(strategy.edges.nzval)
    edge = edge_from_index(strategy.edges, p[k])

    if ispermissible(edge, strategy.adjacency_table, t.snode)
      return (edge[1], edge[2])
    end
  end

  # PJG: it seems possible to exit here and return "nothing".   Handled 
  # as an option in Rust.  Is it actually possible to reach this point?
  # Maybe try debugging by using a crazy scoring function that is always 
  # terrible value

end


function evaluate(strategy::CliqueGraphMergeStrategy, _t::SuperNodeTree, cand::Tuple{Int, Int})
  
  (c1,c2) = cand
  
  do_merge = (strategy.edges[c1, c2] >= 0)

  if !do_merge
    strategy.stop = true
  end
  return do_merge
end


function merge_two_cliques!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree, cand::Tuple{Int, Int})
  
  (c1,c2) = cand
  
  # merge clique c2 into c1
  union!(t.snode[c1], t.snode[c2])
  empty!(t.snode[c2])

  # decrement number of mergeable / nonempty cliques in graph
  t.n_cliques -= 1

  return nothing
end


# After a merge happened, update the reduced clique graph. 

function update_strategy!(
  strategy::CliqueGraphMergeStrategy, 
  t::SuperNodeTree, 
  cand::Tuple{Int, Int}, 
  do_merge::Bool
)
  do_merge || return;

  # After a merge operation update the information of the strategy
  c_1_ind, c_removed = cand

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
        row = max(c_1_ind, n_ind)
        col = min(c_1_ind, n_ind)
        val = edge_metric(c_1, neighbor, strategy.edge_weight)
        edges[row, col] = val 
      end 
  end

  # point edges exclusive to removed clique to surviving clique 1
  for n_ind in new_neighbors
      neighbor = t.snode[n_ind]
      row = max(c_1_ind, n_ind)
      col = min(c_1_ind, n_ind)
      val = edge_metric(c_1, neighbor, strategy.edge_weight)
      edges[row, col] = val
  end

  # overwrite the weight to any removed edges that still contain a link to c_removed
  edges[c_removed+1:n, c_removed] .= 0
  edges[c_removed, 1:c_removed] .= 0

  dropzeros!(edges)

  # update adjacency table in a similar manner
  union!(adjacency_table[c_1_ind], new_neighbors)
  for new_neighbor in new_neighbors
    push!(adjacency_table[new_neighbor], c_1_ind)
  end
  delete!(adjacency_table, c_removed)
  for (_, set) in adjacency_table
    delete!(set, c_removed)
  end

  return nothing
end


function post_process_merge!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

  # since for now we have a graph, not a tree, a post ordering or a parent structure 
  #does not make sense. Therefore just number the non-empty supernodes in t.snd
  
  t.snode_post = findall(x -> !isempty(x), t.snode)
  t.snode_parent = fill(INACTIVE_NODE, length(t.snode))

  # recompute a clique tree from the clique graph
  t.n_cliques > 1 && clique_tree_from_graph!(strategy, t)

  # PJG: This seems unnecessary because the next operation on this
  # object is the call to reorder_snode_consecutively, which overwrites 
  # the snode anyway.  Treatment of separators possibly ends up different.
  # Seems to work without, but keep for now for consistency with COSMO.
  
  foreach(s->sort!(s), t.snode)
  foreach(s->sort!(s), t.separators)

  return nothing
end


# Given the cliques and edges of a clique graph, this function computes a valid clique tree.
# This is necessary to perform the psd completion step of the dual variable after solving the problem.

function clique_tree_from_graph!(strategy::CliqueGraphMergeStrategy, t::SuperNodeTree)

  # a clique tree is a maximum weight spanning tree of the clique graph, where the edge weight is the 
  # cardinality of the intersection between two cliques compute intersection value for each edge 
  # in the clique graph
  clique_intersections!(strategy.edges, t.snode)

  # find a maximum weight spanning tree of the clique graph using Kruskal's algorithm
  kruskal!(strategy.edges, t.n_cliques)

  # determine the root clique of the clique tree (it can be any clique, but we use the 
  # clique that contains the vertex with the highest order)
  determine_parent_cliques!(t.snode_parent, t.snode_children, t.snode, t.post, strategy.edges)

  # recompute a postorder for the supernodes (NB: snode_post will shrink
  # to the possibly reduced length n_cliques after the merge)
  post_order!(t.snode_post, t.snode_parent, t.snode_children, t.n_cliques)

  # Clear the (graph) separators.  They will be rebuilt in the split_cliques
  map(set -> empty!(set), t.separators)

  # split clique sets back into separators and supernodes
  split_cliques!(t.snode, t.separators, t.snode_parent, t.snode_post, t.n_cliques)

  return nothing

end




#------------------- internal utilities -------------------#

# Compute the reduced clique graph (union of all clique trees) given an initial clique tree defined by its 
# supernodes and separator sets.

# We are using the algorithm described in **Michel Habib and Juraj Stacho - Polynomial-time algorithm for the 
# leafage of chordal graphs (2009)**, which
# computes the reduced clique graph in the following way:
# 1. Sort all minimal separators by size
# 2. Initialise graph CG(R) with cliques as nodes and no edges
# 3. for largest unprocessed separator S and
#     |  add an edge between any two cliques C1 and C2 if they both contain S and are in different connected   
#        components of CG(R) and store in `edges`.
#     |  Compute an edge weight used for merge decision and store in `val`.
#     |  Store the index of the separator which is the intersection C1 ∩ C2 in `iter`
#    end

function compute_reduced_clique_graph!(separators::Vector{VertexSet}, snode::Vector{VertexSet})

  # loop over separators by decreasing cardinality
  sort!(separators, by = x -> length(x), rev = true)

  rows = Int[]
  cols = Int[]

  for separator in separators
      # find cliques that contain the separator
      clique_indices = findall(x -> separator ⊆ x, snode)

      # Compute the separator graph (see Habib, Stacho - Reduced clique graphs of chordal graphs) 
      # to analyse connectivity.  We represent the separator graph H by a hashtable
      H = separator_graph(clique_indices, separator, snode)
      # find the connected components of H
      components = find_components(H, clique_indices)

      # for each pair of cliques that contain the separator, add an edge to the reduced 
      # clique tree if they are in unconnected components

      ncliques = length(clique_indices)
      for i in 1:ncliques, j in (i+1):ncliques
          pair = (clique_indices[i], clique_indices[j])
          if is_unconnected(pair, components)
              push!(rows, max(pair...))
              push!(cols, min(pair...))
          end
      end  
  end

  return rows, cols

end 

# Find the separator graph H given a separator and the relevant index-subset of cliques.

function separator_graph(clique_ind::Vector{Int}, separator::VertexSet, snd::Vector{VertexSet})

    # make the separator graph using a hash table
    # key: clique_ind --> edges to other clique indices
    H = Dict{Int, Array{Int, 1}}()

    nindex = length(clique_ind)
    for i in 1:nindex, j in (i+1):nindex
        ca = clique_ind[i]
        cb = clique_ind[j]
          # if intersect_dim(snd[ca], snd[cb]) > length(separator)
        if !inter_equal(snd[ca], snd[cb], separator)
            if haskey(H, ca)
                push!(H[ca], cb)
            else
                H[ca] = [cb]
            end
            if haskey(H, cb)
                push!(H[cb], ca)
            else
                H[cb] = [ca]
            end
        end
    end
    # add unconnected cliques
    for v in clique_ind
        !haskey(H, v) && (H[v] = Int[])
    end
    return H
end


# Find connected components in undirected separator graph represented by `H`.
function find_components(H::Dict{Int, Vector{Int}}, clique_ind::Vector{Int})
    visited = Dict{Int, Bool}(v => false for v in clique_ind)
    components = Vector{VertexSet}()
    for v in clique_ind
        if visited[v] == false
            component = VertexSet()
            DFS_hashtable!(component, v, visited, H)
            push!(components, component)
        end
    end
    return components
end


# Check whether the `pair` of cliques are in different `components`.
function is_unconnected(pair::Tuple{Int, Int}, components::Vector{VertexSet})
    component_ind = findfirst(x -> pair[1] ∈ x, components)
    return pair[2] ∉ components[component_ind]
end


# Depth first search on a hashtable `H`.
function DFS_hashtable!(
  component::VertexSet, 
  v::Int, 
  visited::Dict{Int, Bool}, 
  H::Dict{Int, Vector{Int}}
)

    visited[v] = true
    push!(component, v)
    for n in H[v]
        if visited[n] == false
            DFS_hashtable!(component, n, visited, H)
        end
    end
end


# Check if s ∩ s2 == s3.
function inter_equal(s1::T, s2::T, s3::T) where {T <: AbstractSet}
    dim = 0
  
    len_s1 = length(s1)
    len_s2 = length(s2)
    len_s3 = length(s3)

    # maximum possible intersection size 
    max_intersect = len_s1 + len_s2

    # abort if there's no way the intersection can be the same
    if max_intersect < len_s3
      return false
    end
  
    if len_s1 < len_s2
        sa = s1
        sb = s2
    else
        sa = s2
        sb = s1
    end
    for e in sa
        if e in sb
            dim += 1
            dim > len_s3 && return false
            e in s3 || return false
        end
        max_intersect -= 1 
        max_intersect < len_s3 && return false
    end
    return dim == len_s3
end


# Given a list of edges, return an adjacency hash-table `table` with nodes from 1 to `num_vertices`.

function compute_adjacency_table(edges::SparseMatrixCSC{T}, num_vertices::Int) where {T}

    table = Dict(i => VertexSet() for i = 1:num_vertices)
    r = edges.rowval
    c = edges.colptr
     for col = 1:num_vertices
         for i in c[col]:c[col+1]-1
             row = r[i]
             push!(table[row], col)
            push!(table[col], row)
        end
     end
     return table
end

# Check whether `edge` is permissible for a merge. An edge is permissible if for every common neighbor N, 
# C_1 ∩ N == C_2 ∩ N or if no common neighbors exist.

function ispermissible(
  edge::Tuple{Integer, Integer}, 
  adjacency_table::Dict{Int, VertexSet}, 
  snode::Vector{VertexSet}
)

    c_1 = edge[1]
    c_2 = edge[2]
    common_neighbors = intersect(adjacency_table[c_1], adjacency_table[c_2])

    # N.B. This is allocating and could be made more efficient
    # Here OrderedSets return equality if they have the same 
    # elements, regardless of insertion order.
    for neighbor in common_neighbors
        intersect(snode[c_1], snode[neighbor]) != intersect(snode[c_2], snode[neighbor]) && return false
    end
    return true
end


# Find the matrix indices (i, j) of the first maximum element among the elements stored in A.nzval

function max_elem(A::SparseMatrixCSC{T}) where {T}
  length(A.nzval) == 0 && throw(DomainError("Sparse matrix A doesn't contain any entries"))
  n = size(A, 2)

  ~, ind = findmax(A.nzval)
  row = A.rowval[ind]

  col = 0
  for c = 1:n
    col_indices = A.colptr[c]:A.colptr[c+1]-1
    if in(ind, col_indices)
      col = c
      break;
    end
  end
  return (row, col)
end

function edge_from_index(A::SparseMatrixCSC{T, Int}, ind::Int) where {T}
    index_to_coord(A,ind)
  end
  


function clique_intersections!(E::SparseMatrixCSC{T}, snd::Vector{VertexSet}) where {T}
    # iterate over the nonzeros of the connectivity matrix E which represents the 
    # clique graph and replace the value by |C_i ∩ C_j|
    rows = rowvals(E)
    for col in 1:size(E, 2)
      for j in nzrange(E, col)
        row = rows[j]
        E.nzval[j] = intersect_dim(snd[row], snd[col])
      end
    end
    return nothing
  end

# Return the number of elements in s ∩ s2.
function intersect_dim(s1::AbstractSet, s2::AbstractSet)
  if length(s1) < length(s2)
        sa = s1
        sb = s2
    else
        sa = s2
        sb = s1
    end
    dim = 0
    for e in sa
        e in sb && (dim += 1)
    end
    return dim
end

# Find the size of the set `A ∪ B` under the assumption that `A` and `B` only have unique elements.
function union_dim(s1::T, s2::T)  where {T <: AbstractSet}

  length(s1) + length(s2) - intersect_dim(s1, s2)

end


# Kruskal's algorithm to find a maximum weight spanning tree from the clique intersection graph.
#
#  `E[i,j]` holds the cardinalities of the intersection between two cliques (i, j). Changes the entries in the 
#   connectivity matrix `E` to a negative balue if an edge between two cliques is included in the max spanning tree.
#
#  This is a modified version of https://github.com/JuliaGraphs/LightGraphs.jl/blob/master/src/spanningtrees/kruskal.jl


function kruskal!(E::SparseMatrixCSC{T}, num_cliques::Int) where{T}

  num_initial_cliques = size(E, 2)

  connected_c = DataStructures.IntDisjointSets(num_initial_cliques)

  I, J, V = findnz(E)

  # sort the weights and edges from maximum to minimum value
  p = sortperm(V, rev = true)
  I = I[p]                                            
  J = J[p]
  num_edges_found = 0

  # iterate through edges (I -- J) with decreasing weight
  for k in eachindex(I)
    row = I[k]
    col = J[k]

    if !in_same_set(connected_c, row, col)
        union!(connected_c, row, col)
        # indicate an edge in the MST with a negative value in E (all other values are >= 0)
        # PJG: I think here E[row,col] == E.nzval[p[k]] and could be assigned that way
        E[row, col] = -1.0
        num_edges_found += 1
        # break when all cliques are connected in one tree
        num_edges_found >= num_cliques - 1 && break
    end
  end
  return nothing
end

# Given the maximum weight spanning tree represented by `E`, determine a parent 
# structure `snd_par` for the clique tree.

function determine_parent_cliques!(
    snode_parent::Vector{Int}, 
    snode_children::Vector{VertexSet}, 
    cliques::Vector{VertexSet}, 
    post::Vector{Int}, 
    E::SparseMatrixCSC{T}
  ) where {T}

    # vertex with highest order
    v = post[end]
    c = 0
    # Find clique that contains that vertex
    for (k, clique) in enumerate(cliques)
      if v ∈ clique
        # set that clique to the root
        snode_parent[k] = NO_PARENT
        c = k
        break
      end
    end

    # assign children to cliques along the MST defined by E
    assign_children!(snode_parent, snode_children, c, E)

    return nothing
end


# function assign_children!(
#   snode_parent::Vector{Int}, 
#   snode_children::Vector{VertexSet},
#   c::Int,
#   edges::SparseMatrixCSC{T}
# ) where {T}

#   # determine neighbors
#   neighbors = find_neighbors(edges, c)

#   for n in neighbors
#     # conditions that there is a edge in the MST and that n is not the parent of c
#     if edges[max(c, n), min(c, n)] == -1.0 && snode_parent[c] != n
#       snode_parent[n] = c
#       push!(snode_children[c], n)
#       assign_children!(snode_parent, snode_children, n, edges)
#     end
#   end
#   return nothing
# end

#NB: non-recursive version of the COSMO.jl implementation 
function assign_children!(
  snode_parent::Vector{Int}, 
  snode_children::Vector{VertexSet},
  c::Int,
  edges::SparseMatrixCSC{T}
) where {T}

  stack = [c]  
  while !isempty(stack)
      c = pop!(stack)
      neighbors = find_neighbors(edges, c)

      for n in neighbors
          # conditions that there is an edge in the MST and that n is not the parent of c
          if edges[max(c, n), min(c, n)] == -1.0 && snode_parent[c] != n
              snode_parent[n] = c
              push!(snode_children[c], n)
              push!(stack, n)  # Add child to stack for processing
          end
      end
  end
end


# Find all the cliques connected to `c` which are given by the nonzeros in `(c, 1:c-1)` and `(c+1:n, c)`.

function find_neighbors(edges::SparseMatrixCSC, c::Int)
  neighbors = zeros(Int, 0)
  _, n = size(edges)
  # find all nonzero columns in row c up to column c
  if c > 1
    neighbors = vcat(neighbors, findall(x -> x != 0, edges[c, 1:c-1]))
  end
  # find all nonzero entries in column c below c
  if c < n
    rows = @view edges.rowval[edges.colptr[c]:edges.colptr[c+1]-1]
    if edges.colptr[c] <= edges.colptr[c+1] - 1
      neighbors = vcat(neighbors, rows)
    end
  end
  return neighbors
end


# Traverse the clique tree in descending topological order and split the clique sets into supernodes and separators.

function split_cliques!(
  snode::Vector{VertexSet}, 
  separators::Vector{VertexSet}, 
  snode_parent::Vector{Int}, 
  snode_post::Vector{Int}, 
  num_cliques::Int
)

  # travese in topological decending order through the clique tree and split the clique 
  # into supernodes and separators
  for j = 1:1:(num_cliques - 1)
      c_ind = snode_post[j]
      p_ind = snode_parent[c_ind]

      # find intersection of clique with parent
      separators[c_ind] = intersect(snode[c_ind], snode[p_ind])
      snode[c_ind]      = filter!(x -> x ∉ separators[c_ind], snode[c_ind])
  end
  return nothing
end


# -------------------
# functions relating to edge weights 
# -------------------


# Compute the edge weight between all cliques specified by the edges (rows, cols).
# weights on the edges currently defined as integer values, but could be changed 
# to floats to allow emperical edge weight functions.

function compute_weights!(
  rows::Vector{Int}, 
  cols::Vector{Int}, 
  snode::Vector{VertexSet}, 
  edge_weight::EdgeWeightMethod
)

  weights = zeros(Int, length(rows))

  for k = 1:length(rows)
    c_1 = snode[rows[k]]
    c_2 = snode[cols[k]]
    weights[k] = edge_metric(c_1, c_2, edge_weight)
  end
  return weights

end



#Given two cliques `c_a` and `c_b` return a value for their edge weight.

function edge_metric(c_a::VertexSet, c_b::VertexSet, edge_weight::EdgeWeightMethod) 
  n_1 = length(c_a)
  n_2 = length(c_b)

  # merged block size
  n_m = union_dim(c_a, c_b)

  if edge_weight == CUBIC::EdgeWeightMethod
    return (n_1^3 + n_2^3 - n_m^3)::Int
  else
    throw(ArgumentError("Unknown weight metric not implemented"))
  end
end


