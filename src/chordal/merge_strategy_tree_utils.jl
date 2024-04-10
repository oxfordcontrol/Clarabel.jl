
#PJG: this is only used in the tree merge, but it seems like a general
#SuperNodeTree fcn.   One probably though is that it's not clear what 
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
