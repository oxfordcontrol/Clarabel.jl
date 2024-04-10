# PJG : nothing done here yet other than fixing some 
# type and some fixing up of Array to Vector.   Everything 
# in this file was previously in clique_graph.jl in COSMO,
# but I think all of this code amounts to utility functions 
# for the graph-based merge strategy 

using IterTools

"""
    compute_reduced_clique_graph!(sep::Array{Set{Int}, 1}, snd::Array{Set{Int}, 1})

Compute the reduced clique graph (union of all clique trees) given an initial clique tree defined by its supernodes `snd` and separator `sep` sets.

We are using the algorithm described in **Michel Habib and Juraj Stacho - Polynomial-time algorithm for the leafage ofchordal graphs (2009)**, which
computes the reduced clique graph in the following way:
1. Sort all minimal separators by size
2. Initialise graph CG(R) with cliques as nodes and no edges
3. for largest unprocessed separator S and
    |  add an edge between any two cliques C1 and C2 if they both contain S and are in different connected   components   of CG(R) and store in `edges`.
    |  Compute an edge weight used for merge decision and store in `val`.
    |  Store the index of the separator which is the intersection C1 ∩ C2 in `iter`
   end
"""
function compute_reduced_clique_graph!(separators::Vector{VertexSet}, snode::Vector{VertexSet})
    # loop over separators by decreasing cardinality
    sort!(separators, by = x -> length(x), rev = true)

    # edges = Array{Tuple{Int, Int}, 1}()    # a list of edges in the reduced clique graph, higher clique index first
    rows = Int[]
    cols = Int[]
    # inter = Array{Int, 1}()                  # the index of the separator which corresponds to the intersection of the two cliques

    for (k, separator) in enumerate(separators)
        # find cliques that contain the separator
        clique_indices = findall(x -> separator ⊆ x, snode)

        # we compute the separator graph (see Habib, Stacho - Reduced clique graphs of chordal graphs) to analyse connectivity
        # we represent the separator graph H by a hashtable
        H = separator_graph(clique_indices, separator, snode)
        # find the connected components of H
        components = find_components(H, clique_indices)
        # for each pair of cliques that contain the separator, add an edge to the reduced clique tree if they are 
        # in unconnected components

        # PJG: here, the "subsets" call is generating all possible pairs 
        # of elements from clique_indices.   I think that this is the only 
        # place in the code where IterTools is required.  An identical call
        # to the same function, again over pairs, in separator_graph.
        # In rust, maybe this: https://docs.rs/itertools/latest/itertools/structs/struct.Combinations.html
        for pair in IterTools.subsets(clique_indices, Val{2}())
            if is_unconnected(pair, components)
                # push!(edges, (max(pair...), min(pair...))) #add edge
                push!(rows, max(pair...))
                push!(cols, min(pair...))
                # push!(inter, k) # store intersection
            end
        end

    end
    return rows, cols#, inter
end


"Find the separator graph H given a separator and the relevant index-subset of cliques."
function separator_graph(clique_ind::Vector{Int}, separator::VertexSet, snd::Vector{VertexSet})

    # make the separator graph using a hash table
    # key: clique_ind --> edges to other clique indices
    H = Dict{Int, Array{Int, 1}}()

    for pair in IterTools.subsets(clique_ind, Val{2}())
        ca = pair[1]
        cb = pair[2]
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


"Find connected components in undirected separator graph represented by `H`."
function find_components(H::Dict{Int, Vector{Int}}, clique_ind::Vector{Int})
    visited = Dict{Int, Bool}(v => false for v in clique_ind)
    components = Vector{VertexSet}()
    for v in clique_ind
        if visited[v] == false
            component = VertexSet()
            push!(components, DFS_hashtable!(component, v, visited, H))
        end
    end
    return components
end


"Check whether the `pair` of cliques are in different `components`."
function is_unconnected(pair::Tuple{Int, Int}, components::Vector{VertexSet})
    component_ind = findfirst(x -> pair[1] ∈ x, components)
    return pair[2] ∉ components[component_ind]
end


"Depth first search on a hashtable `H`."
function DFS_hashtable!(component::VertexSet, v::Int, visited::Dict{Int, Bool}, H::Dict{Int, Vector{Int}})
    visited[v] = true
    push!(component, v)
    for n in H[v]
        if visited[n] == false
            DFS_hashtable!(component, n, visited, H)
        end
    end
    return component
end


"Check if s ∩ s2 == s3."
function inter_equal(s1::T, s2::T, s3::T) where {T <: AbstractSet}
    dim = 0
  
    len_s1 = length(s1)
    len_s2 = length(s2)
    len_s3 = length(s3)

    # maximum possible intersection size 
    max_intersect = len_s1 + len_s2

    # abort if there's no way the intersection can be the same
    if max_intersect < len_s3
      println("bail")
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


"Given a list of edges, return an adjacency hash-table `table` with nodes from 1 to `num_vertices`."
# PJG: not clear to me why {T} was Float64 before.   Is that from a sparsity matrix, or 
# from the original data matrix?  Why isn't it SparseMatrixCSC{bool,Int}, for example?
# Maybe it should be a matrix of Int64 / isize, since we assign -1 entries in functions below
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

"Check whether `edge` is permissible for a merge. An edge is permissible if for every common neighbor N, C_1 ∩ N == C_2 ∩ N or if no common neighbors exist."
function ispermissible(edge::Tuple{Integer, Integer}, adjacency_table::Dict{Int, VertexSet}, snd::Vector{VertexSet})
    c_1 = edge[1]
    c_2 = edge[2]
    common_neighbors = intersect(adjacency_table[c_1], adjacency_table[c_2])
    # N.B. This can be made faster by first checking whether the sizes of the intersection are the same before allocating anything
    for neighbor in common_neighbors
        intersect(snd[c_1], snd[neighbor]) != intersect(snd[c_2], snd[neighbor]) && return false
    end
    return true
end


"""
    max_elem(A::SparseMatrixCSC)

Find the matrix indices (i, j) of the first maximum element among the elements stored in A.nzval
"""
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
    # find the edge for that value
    row = A.rowval[ind]
    n = size(A, 2)
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
  


function clique_intersections!(E::SparseMatrixCSC{T}, snd::Vector{VertexSet}) where {T}
    # iterate over the nonzeros of the connectivity matrix E which represents the clique graph and replace the value by
    # |C_i ∩ C_j|
    rows = rowvals(E)
    for col in 1:size(E, 2)
      for j in nzrange(E, col)
        row = rows[j]
        E[row, col] = intersect_dim(snd[row], snd[col])
      end
    end
    return nothing
  end

"Return the number of elements in s ∩ s2."
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


#PJG: redefined this function to try to make it faster by always 
#taking the sets in a favorable order.   Be careful of bugs here 
" Find the size of the set `A ∪ B` under the assumption that `A` and `B` only have unique elements."
function union_dim(s1::T, s2::T)  where {T <: AbstractSet}

  # dim1 = length(s1)
  # dim2 = length(s2) 

  # if dim1 < dim2
  #   sa = s1
  #   sb = s2
  #   udim = dim2
  # else
  #   sa = s2
  #   sb = s1
  #   udim = dim1
  # end

  # for e in sa
  #   !in(e, sb) && (udim += 1)
  # end
  # return udim

  length(s1) + length(s2) - intersect_dim(s1, s2)

end


"""
    kruskal!(E::SparseMatrixCSC, num_cliques::Int)

Kruskal's algorithm to find a maximum weight spanning tree from the clique intersection graph.

 `E[i,j]` holds the cardinalities of the intersection between two cliques (i, j). Changes the entries in the connectivity matrix `E` to a negative
 value if an edge between two cliques is included in the max spanning tree.

 This is a modified version of https://github.com/JuliaGraphs/LightGraphs.jl/blob/master/src/spanningtrees/kruskal.jl
 """
function kruskal!(E::SparseMatrixCSC{T}, num_cliques::Int) where{T}
  num_initial_cliques = size(E, 2)

  # PJG: this is currently the only place that requires DataStructures 
  # as a package.   What does this do and do I need it.
  # oh fuck it's really complicated 
  connected_c = DataStructures.IntDisjointSets(num_initial_cliques)

  I, J, V = findnz(E)
  # sort the weights and edges from maximum to minimum value
  p = sortperm(V, rev = true)
  I = I[p]
  J = J[p]
  V = V[p]
  num_edges_found = 0
  # iterate through edges (I -- J) with decreasing weight
  for k = 1:length(V)
    row = I[k]
    col = J[k]
    if !in_same_set(connected_c, row, col)
      union!(connected_c, row, col)
      # we indicate an edge in the MST with a positive value in E (all other values are >= 0)
      E[row, col] = -1.0
      num_edges_found += 1
      # break when all cliques are connected in one tree
      num_edges_found >= num_cliques - 1 && break
    end
  end
  return nothing
end

" Given the maximum weight spanning tree represented by `E`, determine a parent structure `snd_par` for the clique tree."
function determine_parent_cliques!(snd_par::Vector{Int}, snd_child::Vector{VertexSet}, cliques::Vector{VertexSet}, post::Vector{Int}, E::SparseMatrixCSC{T}) where {T}
  # vertex with highest order
  v = post[end]
  c = 0
  # Find clique that contains that vertex
  for (k, clique) in enumerate(cliques)
    if v ∈ clique
      # set that clique to the root
      snd_par[k] = 0
      c = k
      break
    end
  end

  # recursively assign children to cliques along the MST defined by E
  assign_children!(snd_par, snd_child, c, E)
  return nothing
end


function assign_children!(snode_parent::Vector{Int}, snode_children::Vector{VertexSet}, c::Int, edges::SparseMatrixCSC{T}) where {T}
    # determine neighbors
    neighbors = find_neighbors(edges, c)
    for n in neighbors
      # conditions that there is a edge in the MST and that n is not the parent of c
      if edges[max(c, n), min(c, n)] == -1.0 && snode_parent[c] != n
        snode_parent[n] = c
        push!(snode_children[c], n)
        assign_children!(snode_parent, snode_children, n, edges)
      end
    end
    return nothing
  end


"""
find_neighbors(edges::SparseMatrixCSC, c::Int)

Find all the cliques connected to `c` which are given by the nonzeros in `(c, 1:c-1)` and `(c+1:n, c)`.
"""
#PJG: this function is only called from assign_children, which itself 
#is never called?
function find_neighbors(edges::SparseMatrixCSC, c::Int)
  neighbors = zeros(Int, 0)
  m, n = size(edges)
  # find all nonzero columns in row c up to column c
  if c > 1
    neighbors = vcat(neighbors, findall(x -> x != 0, edges[c, 1:c-1]))
  end
  # find all nonzero entries in column c below c
  if c < n
    rows = edges.rowval[edges.colptr[c]:edges.colptr[c+1]-1]
    if edges.colptr[c] <= edges.colptr[c+1] - 1
      neighbors = vcat(neighbors, rows)
    end
  end
  return neighbors
end


" Traverse the clique tree in descending topological order and split the clique sets into supernodes and separators."
function split_cliques!(snd::Vector{VertexSet}, sep::Vector{VertexSet}, snd_par::Vector{Int}, snd_post::Vector{Int}, num_cliques::Int)

  # travese in topological decending order through the clique tree and split the clique in snd and sep
  for j = 1:1:(num_cliques - 1)
    c_ind = snd_post[j]
    par_ind = snd_par[c_ind]

    # find intersection of clique with parent
    # FIXME: Can be made a lot faster by using non-standard functions
    sep[c_ind] = intersect(snd[c_ind], snd[par_ind])
    snd[c_ind] = filter!(x -> x ∉ sep[c_ind], snd[c_ind])
  end
  return nothing
end


# -------------------
# functions relating to edge weights 

"Compute the edge weight between all cliques specified by the edges (rows, cols)."
function compute_weights!(rows::Vector{Int}, cols::Vector{Int}, snd::Vector{VertexSet}, edge_weight::EdgeWeightMethod)
  weights = zeros(Float64, length(rows))
  for k = 1:length(rows)
    c_1 = snd[rows[k]]
    c_2 = snd[cols[k]]
    weights[k] = edge_metric(c_1, c_2, edge_weight)
  end
  return weights
end


"""
    edge_metric(c_a::AbstractVector, c_b::AbstractVector, edge_weight::EdgeWeightMethod)

Given two cliques `c_a` and `c_b` return a value for their edge weight.
"""
function _edge_metric(c_a::T, c_b::T, edge_weight::EdgeWeightMethod) where {T <: Union{AbstractVector, AbstractSet}}
  n_1 = length(c_a)
  n_2 = length(c_b)

  # merged block size
  n_m = union_dim(c_a, c_b)

  if edge_weight == CUBIC::EdgeWeightMethod
    return n_1^3 + n_2^3 - n_m^3
  else
    throw(ArgumentError("Unknown weight metric not implemented"))
  end
end


function edge_metric(c_a::T, c_b::T, edge_weight::EdgeWeightMethod) where {T <: Union{AbstractVector, AbstractSet}}
  
  n_1 = triangular_number(length(c_a))
  n_2 = triangular_number(length(c_b))

  # merged block size
  n_m = triangular_number(union_dim(c_a, c_b))

  if edge_weight == CUBIC::EdgeWeightMethod
    return n_1^4 + n_2^4 - n_m^4
  else
    throw(ArgumentError("Unknown weight metric not implemented"))
  end
end


