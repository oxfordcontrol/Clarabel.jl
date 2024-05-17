# DataStructures provides OrderedSet, but only because it itself includes 
# OrderedCollections.  Could just use OrderedCollections directly, 
# but the kruskal! function also uses a function from DataStructures.
# If that is removed then use the lighter weight option here 

using DataStructures

abstract type AbstractMergeStrategy end
VertexSet = OrderedSet{Int}

#PJG: make a settable option
@enum EdgeWeightMethod begin 
  CUBIC = 1
end

