# ---------------------------
# Struct to hold clique and sparsity data for a constraint
# ---------------------------
mutable struct SparsityPattern
  sntree::SuperNodeTree
  ordering::Array{DefaultInt}
  orig_index::DefaultInt # original index of the cone being decomposed

  # constructor for sparsity pattern
  function SparsityPattern(
    L::SparseMatrixCSC{T}, 
    ordering::Array{DefaultInt, 1}, 
    orig_index::DefaultInt,
    merge_method::Symbol,

  ) where {T}

    sntree = SuperNodeTree(L)

    # clique merging only if more than one clique present

    if sntree.n_cliques > 1

      if merge_method == :none 
        merge_cliques!(NoMergeStrategy(), sntree)

      elseif merge_method == :parent_child
        merge_cliques!(ParentChildMergeStrategy(), sntree)

      elseif merge_method == :clique_graph
        merge_cliques!(CliqueGraphMergeStrategy(), sntree)

      else
        error("Unknown merge strategy: ", merge_method)
      end

    end 

    # reorder vertices in supernodes to have consecutive order
    # necessary for equal column structure for psd completion
    reorder_snode_consecutively!(sntree, ordering)

    # for each clique determine the number of entries of the block 
    #represented by that clique
    calculate_block_dimensions!(sntree)

    return new(sntree, ordering, orig_index)
  end
end 