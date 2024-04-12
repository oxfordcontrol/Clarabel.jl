# some types need to be defined here since Julia 
# is senstive to order of inclusion

# PJG : DataStructures provides OrderedSet, but only because it 
# itself includes OrderedCollections.  I could just use OrderedCollections
# directly, but the kruskal! function also uses a function from DataStructures.
# If that can be removed then use the lighter weight option here 
using DataStructures

abstract type AbstractMergeStrategy end

# PJG: this sucks.  Switch to symbols as in the merge 
# method.  Not consistent with Rust, but at least 
# internally consistent with all other settings.  How 
# are those handled?

@enum EdgeWeightMethod begin 
  CUBIC = 1
end


# PJG: OrderedSet appears to be equivalent to rust indexmap crate, 
# (type IndexSet), but be careful about the removal of elements 
# because it is not order preserving.  Need to use something 
# like "shift_remove" to maintain order in Rust.   

VertexSet = OrderedSet{Int}