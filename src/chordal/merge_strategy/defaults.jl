# Main clique merging routine, common to all strategies 

function merge_cliques!(strategy::AbstractMergeStrategy, t::SuperNodeTree)
    
  initialise!(strategy, t)

  while !is_done(strategy)
    
    # find merge candidates
    cand = traverse(strategy, t)

    # evaluate wether to merge the candidates
    do_merge = evaluate(strategy, t, cand)
    if do_merge
      merge_two_cliques!(strategy, t, cand)
    end

    # update strategy information after the merge
    update_strategy!(strategy, t, cand, do_merge)
    t.n_cliques == 1 && break

  end

  post_process_merge!(strategy, t)
  return nothing
end

#All strategies must then implement the following interface: 

# initialise! - tree and strategy
# is_done - merging complete, so can stop the merging process
# traverse - find the next merge candidates
# evalute - evaluate wether to merge them or not
# merge_two_cliques! - execute a merge 
# update_strategy! - update the tree/graph and strategy
# post_process_merge! - do any post-processing of the tree/graph

function is_done(strategy::AbstractMergeStrategy)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing
end

function initialise!(strategy::AbstractMergeStrategy, t::SuperNodeTree)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing
end

function traverse(strategy::AbstractMergeStrategy, t::SuperNodeTree)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing
end

function evaluate(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Tuple{Int, Int})
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return bool
end

function merge_two_cliques!(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Tuple{Int, Int}) 
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 

function update_strategy!(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Tuple{Int, Int}, do_merge::Bool)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 

function post_process_merge!(strategy::AbstractMergeStrategy, t::SuperNodeTree)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 

