# ---------------------------
# Struct to hold clique and sparsity data for a constraint
# ---------------------------

#PJG: rowrange and coneidx don't seem to belong here ?
#when / how to ordering reverse_ordering her used?
#is the reverse ordering actually required?  Seems like 
#I could just use the ordering anywhere it is needed

# PJG: looking at COSMO code, it seems like the reverse 
# ordering is not used anywhere.   If I save the ordering 
# inside the SuperNodeTree (or externally), I might not 
# need this type at all?

# PJG: same for rowrange.  I don't think it is ever
# used in COSMO anywhere.  Same for coneidx too.  WTF?

mutable struct SparsityPattern
  sntree::SuperNodeTree
  ordering::Array{Int}
  #reverse_ordering::Array{Int}
  #rowrange::UnitRange{Int} # rows originally occupied by the cone being decomposed
  coneidx::Int # original index of the cone being decomposed

  #PJG: do I want to keep the L that generated this, just for debugging?
  

  #PJG: I think that SparsityPattern coul just be dropped.  The only field 
  #other than the SuperNodeTree is the ordering, and the two fields are 
  #never required independently of each other.   

  # constructor for sparsity pattern
  function SparsityPattern(
    L::SparseMatrixCSC, 
    ordering::Array{Int, 1}, 
    coneidx::Int,
    merge_method::Symbol,
    #PJG: these probably don't really belong as part of the pattern
    #rowrange::UnitRange{Int}, 
  )

    sntree = SuperNodeTree(L)

    # clique merging only if more than one clique present
    # PJG: "merge_strategy" now the first argument after rewrite
    # PJG: using n_snode here since it decreases as snodes drain 
    # (but are not deleted)
    if sntree.n_snode > 1

      if merge_method == :none 
        merge_strategy = NoMergeStrategy()
        merge_cliques!(merge_strategy, sntree)

      elseif merge_method == :parent_child
        merge_strategy = ParentChildMergeStrategy()
        merge_cliques!(merge_strategy, sntree)

      elseif merge_method == :clique_graph
        merge_strategy = CliqueGraphMergeStrategy()
        merge_cliques!(merge_strategy, sntree)

      else
        error("Unknown merge strategy: ", merge_method)
      end

    end 

    # reorder vertices in supernodes to have consecutive order
    # necessary for equal column structure for psd completion
    # PJG: change name to ..._snode_... when finished
    reorder_snd_consecutively!(sntree, ordering)

    # for each clique determine the number of entries of the block represented by that clique
    calculate_block_dimensions!(sntree)

    return new(sntree, ordering, coneidx)
  end
end 