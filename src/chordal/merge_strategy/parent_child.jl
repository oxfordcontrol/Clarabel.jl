mutable struct ParentChildMergeStrategy <: AbstractMergeStrategy
  stop::Bool
  clique_index::Int
  t_fill::Int
  t_size::Int

  # PJG: fill and size need to be settable 
  function ParentChildMergeStrategy(; t_fill = 8, t_size = 8)
    new(false, 0, t_fill, t_size)
  end
end

function initialise!(strategy::ParentChildMergeStrategy, t::SuperNodeTree)
  # start with node that has second highest order
  strategy.clique_index = length(t.snode) - 1
end

function is_done(strategy::ParentChildMergeStrategy)
  strategy.stop
end

# Traverse tree `t` in descending topological order and return parent and clique (root has highest order).

function traverse(strategy::ParentChildMergeStrategy, t::SuperNodeTree)

  c = t.snode_post[strategy.clique_index]

  return (t.snode_parent[c], c)
end



# Decide whether to merge the two clique candidates.

function evaluate(strategy::ParentChildMergeStrategy, t::SuperNodeTree, cand::Tuple{Int, Int})
  
  strategy.stop && return false

  (parent, child) = cand
  
  dim_parent_snode, dim_parent_sep = clique_dim(t, parent)
  dim_clique_snode, dim_clique_sep = clique_dim(t, child)

  fill      = fill_in(dim_clique_snode, dim_clique_sep, dim_parent_snode, dim_parent_sep)
  max_snode = max(dim_clique_snode, dim_parent_snode)

  return fill <= strategy.t_fill || max_snode <= strategy.t_size
end

# Given the clique tree `t` merge the two cliques with indices in `cand` as parent-child.

function merge_two_cliques!(
  strategy::ParentChildMergeStrategy, 
  t::SuperNodeTree, 
  cand::Tuple{Int, Int}
) 
  
  # determine which clique is the parent
  (p, ch) = determine_parent(t, cand...);

  # merge child's vertex sets into parent's vertex set
  union!(t.snode[p], t.snode[ch])
  empty!(t.snode[ch])
  empty!(t.separators[ch])

  # update parent structure
  for grandch in t.snode_children[ch]
    t.snode_parent[grandch] = p
  end
  t.snode_parent[ch] = INACTIVE_NODE 

  # update children structure
  delete!(t.snode_children[p], ch)
  union!(t.snode_children[p], t.snode_children[ch])
  empty!(t.snode_children[ch])

  # decrement number of cliques in tree
  t.n_cliques -= 1
  return nothing

end


# After a merge attempt, update the strategy information.
function update_strategy!(
  strategy::ParentChildMergeStrategy, 
  t::SuperNodeTree, 
  cand::Tuple{Int, Int}, 
  do_merge::Bool
)
  # try to merge last node of order 1, then stop
  if strategy.clique_index == 1
     strategy.stop = true
    # otherwise decrement node index
  else
    strategy.clique_index -= 1
  end
end



function post_process_merge!(strategy::ParentChildMergeStrategy, t::SuperNodeTree)

  # the merging creates empty supernodes and seperators, recalculate a 
  # post order for the supernodes (shrinks to length t.n_cliques )
  post_order!(t.snode_post, t.snode_parent, t.snode_children, t.n_cliques)

  return nothing
end



#-------------------- utilities --------------------


# Given two cliques `c1` and `c2` in the tree `t`, return the parent clique first.

# Not implemented as part of the general SuperNodeTree interface 
# since this should only be called when we can guarantee that we
# are acting on a parent-child pair.

function determine_parent(t::SuperNodeTree, c1::Int, c2::Int)
  if in(c2, t.snode_children[c1])
    return c1, c2
  else
    return c2, c1
  end
end

# not implemented as part of the main SuperNodeTree interface since the 
# index is not passed through the post ordering 
function clique_dim(t::SuperNodeTree, i::Int)
  return length(t.snode[i]), length(t.separators[i])
end


# Compute the amount of fill-in created by merging two cliques with the 
# respective supernode and separator dimensions.

function fill_in(
  dim_clique_snode::Int, 
  dim_clique_sep::Int, 
  dim_parent_snode::Int, 
  dim_parent_sep::Int
)
  dim_parent = dim_parent_snode + dim_parent_sep
  dim_clique = dim_clique_snode + dim_clique_sep

  return (dim_parent - dim_clique_sep) * (dim_clique - dim_clique_sep)
end