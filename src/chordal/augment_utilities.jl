# Here should go all common utilities for transforms.   But try to assign 
# utilities as methods on the inferior data types.

function getindex(vals::Vector{Int}, set::VertexSet)
    return [vals[i] for i in set]
end