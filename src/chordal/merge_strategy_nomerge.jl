struct NoMergeStrategy <: AbstractMergeStrategy end

function is_done(strategy::NoMergeStrategy)
    true 
end

function initialise!(strategy::NoMergeStrategy, t::SuperNodeTree)
    return nothing
end

function post_process_merge!(strategy::NoMergeStrategy, t::SuperNodeTree)
    return nothing
end


#PJG: all other functions should be unreachable
#this is not that nice, since Rust will have to 
#actually implement a load of unreachable functions 
#would be better to short-circuit the merge 
#process in a different way 