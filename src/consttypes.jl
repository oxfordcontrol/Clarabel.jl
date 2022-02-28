# -------------------------------------
# parametric view type.   Needs to be
# defined seperately and first in the
# module to avoid weird circular
# dependencies
const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
