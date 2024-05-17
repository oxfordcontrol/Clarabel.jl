struct NoMergeStrategy <: AbstractMergeStrategy end

function initialise!(strategy::NoMergeStrategy, t::SuperNodeTree)
    return nothing
end

function is_done(strategy::NoMergeStrategy)
    true 
end

function post_process_merge!(strategy::NoMergeStrategy, t::SuperNodeTree)
    return nothing
end


# All other functions should be unreachable this is not that nice, 
# and should revert the the default / error implementation if called 