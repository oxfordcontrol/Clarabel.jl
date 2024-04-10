# Main clique merging routine, common to all strategies 

function merge_cliques!(strategy::AbstractMergeStrategy, t::SuperNodeTree)
    
  #PJG: so far have grabbed the high level functions in here 
  #and some of their callees, but no editing yet

  initialise!(strategy, t)

  while !is_done(strategy)
    # find merge candidates

    cand = traverse(strategy, t)
    # evaluate wether to merge the candidates
    do_merge = evaluate(strategy, t, cand)
    if do_merge
      merge_two_cliques!(strategy, t, cand)
    end

    #PJG: no logging for now
    # log_merge!(t, do_merge, cand)

    # update strategy information after the merge

    #PJG: update strategy appears to be a no-op
    #if do_merge is false, at least for the graph 
    #method.   I don't see any reason to call it
    #instead of merging with the if block above 
    # PJG: possible bug because if do_merge ends 
    # up being false when n_snode is > 1, we 
    # have an infinite loop
    update_strategy!(strategy, t, cand, do_merge)

    # PJG: this probably needs to be a function 
    # since it implicitly assumes that the same 
    # field will appear for both tree and graph 
    # methods 
    t.n_snode == 1 && break
    #PJG: dropping because this appears to 
    #just repeat the same condition as the while 
    # strategy.stop && break   
  end

  post_process_merge!(strategy, t)
  return nothing
end

#All implemented strategies must then implement the following interface: 

# 0. is_done() - merging complete, so can stop the merging process
# 1. initialise!() - tree and strategy
# 2. traverse() - find the next merge candidates
# 3. evalute() - evaluate wether to merge them or not
# 4. merge_two_cliques!() - execute a merge 
# PJG (possibly removed): log_merge!() - log the merge
# 5. update_strategy!() - update the tree/graph and strategy
# 6. post_process_merge!() - do any post-processing of the tree/graph

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

function evaluate(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Vector{Int})
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return bool
end

#PJG: log template goes here if one is implemented 

function merge_two_cliques!(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Vector{Int}) 
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 


function update_strategy!(strategy::AbstractMergeStrategy, t::SuperNodeTree, cand::Vector{Int}, do_merge::Bool)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 


function post_process_merge!(strategy::AbstractMergeStrategy, t::SuperNodeTree)
  error("Incomplete merge strategy specification: ",typeof(strategy))
  #return nothing 
end 

