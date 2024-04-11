mutable struct ParentChildMergeStrategy <: AbstractMergeStrategy
  stop::Bool
  clique_ind::Int
  t_fill::Int
  t_size::Int

  function ParentChildMergeStrategy(; t_fill = 8, t_size = 8)
    new(false, 2, t_fill, t_size)
  end
end


function is_done(strategy::ParentChildMergeStrategy)
  strategy.stop
end

function initialise!(strategy::ParentChildMergeStrategy, t::SuperNodeTree)
  # start with node that has second highest order
  strategy.clique_ind = length(t.snode) - 1
end



" Traverse tree `t` in descending topological order and return parent and clique (root has highest order)."
function traverse(strategy::ParentChildMergeStrategy, t::SuperNodeTree)
  #PJG: probably should return a tuple, not a two element array.
  c = t.snode_post[strategy.clique_ind]
  return [t.snode_parent[c]; c]
end



"Decide whether to merge the two cliques with clique indices `cand`."
#PJG: variable names suck in this function 
function evaluate(strategy::ParentChildMergeStrategy, t::SuperNodeTree, cand::Vector{Int})
  strategy.stop && return false
  par = cand[1]
  c = cand[2]
  dim_par_snd, dim_par_sep = clique_dim(t, par)
  dim_clique_snd, dim_clique_sep = clique_dim(t, c)

  #PJG: this line way too long
  return fill_in(dim_clique_snd, dim_clique_sep, dim_par_snd, dim_par_sep) <= strategy.t_fill || max(dim_clique_snd, dim_par_snd) <= strategy.t_size
end

#PJG: this function previously was a one-line call to merge_child! in the case 
#of dispatch on ParentChildMergeStrategy.   I have therefore deleted merge_child! and 
#just folded it into the implementation here. 
"Given the clique tree `t` merge the two cliques with indices in `cand` as parent-child."
function merge_two_cliques!(strategy::ParentChildMergeStrategy, t::SuperNodeTree, cand::Vector{Int}) 
  
  # determine which clique is the parent
  p, ch = determine_parent(t, cand[1], cand[2])

  # merge child's vertex sets into parent's vertex set
  union!(t.snode[p], t.snode[ch])
  empty!(t.snode[ch])
  empty!(t.separators[ch])

  # update parent structure
  for grandch in t.snode_children[ch]
    t.snode_parent[grandch] = p
  end
  t.snode_parent[ch] = -1 #-1 instead of NaN, effectively remove that entry from the parent list

  # update children structure
  delete!(t.snode_children[p], ch)
  union!(t.snode_children[p], t.snode_children[ch])
  empty!(t.snode_children[ch])

  # decrement number of cliques in tree
  t.n_snode -= 1
  return nothing

end


" After a merge attempt, update the strategy information."
function update_strategy!(strategy::ParentChildMergeStrategy, t::SuperNodeTree, cand::Vector{Int}, do_merge::Bool)
  # try to merge last node of order 1, then stop
  if strategy.clique_ind == 1
     strategy.stop = true
    # otherwise decrement node index
  else
    strategy.clique_ind -= 1
  end
end



function post_process_merge!(strategy::ParentChildMergeStrategy, t::SuperNodeTree)

  # the merging creates empty supernodes and seperators, recalculate a post order for the supernodes
  # PJG: possibly allocating operation
  t.snode_post = post_order(t.snode_parent, t.snode_children, t.n_snode)

  return nothing
end



#-------------------- utilities --------------------

#PJG: this is only used in the tree merge, but it seems like a general
#SuperNodeTree fcn.   One problem though is that it's not clear what 
#happens if there is no parent-child relationship at all.
#maybe needs a different name if this is a niche function for the tree merge
" Given two cliques `c1` and `c2` in the tree `t`, return the parent clique first."
function determine_parent(t::SuperNodeTree, c1::Int, c2::Int)
  if in(c2, t.snode_children[c1])
    return c1, c2
  else
    return c2, c1
  end
end


"Compute the amount of fill-in created by merging two cliques with the respective supernode and separator dimensions."
function fill_in(dim_clique_snd::Int, dim_clique_sep::Int, dim_par_snd::Int, dim_par_sep::Int)
  dim_par = dim_par_snd + dim_par_sep
  dim_clique = dim_clique_snd + dim_clique_sep
  return ((dim_par - dim_clique_sep) * (dim_clique - dim_clique_sep))::Int
end


#PJG: this function only used in evaluate and seems totally unnecessary
clique_dim(t, c_ind) = length(t.snode[c_ind]), length(t.separators[c_ind])
