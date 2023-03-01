# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct CompositeCone{T} <: AbstractCone{T}

    cones::Vector{AbstractCone{T}}

    #API type specs and count of each cone type
    types::Vector{Type}
    type_counts::Dict{Type,Int}

    #overall size of the composite cone
    numel::DefaultInt
    degree::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    # the flag for symmetric cone check
    _is_symmetric::Bool

    function CompositeCone{T}(cone_specs::Vector{<:SupportedCone}) where {T}

        #make copy to protect from user interference after setup,
        #and explicitly cast as the abstract type.  This prevents
        #errors arising from input vectors that are all the same cone,
        #and therefore more concretely typed than we want
        cone_specs = Vector{SupportedCone}(cone_specs)

        ncones = length(cone_specs)
        cones  = Vector{AbstractCone{T}}(undef,ncones)
        types = Vector{DataType}(undef,ncones)

        #count the number of each cone type
        type_counts = Dict{Type,Int}()
        for (key, val) in ConeDict
            type_counts[val] = count(C->isa(C,key), cone_specs)
        end

        #assumed symmetric to start
        _is_symmetric = true

        #create cones with the given dims
        for i in eachindex(cone_specs)
            types[i] = typeof(cone_specs[i])
            if types[i] == ExponentialConeT
                cones[i] = ConeDict[typeof(cone_specs[i])]{T}()
                _is_symmetric = false
            elseif types[i] == PowerConeT
                cones[i] = ConeDict[typeof(cone_specs[i])]{T}(T(cone_specs[i].α))
                _is_symmetric = false
            else
                cones[i] = ConeDict[typeof(cone_specs[i])]{T}(cone_specs[i].dim)
            end
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        #headidx gives the index of the first element
        #of each constituent cone
        headidx = Vector{Int}(undef,length(cones))
        _make_headidx!(headidx,cones)

        return new(cones,types,type_counts,numel,degree,headidx,_is_symmetric)
    end
end

CompositeCone(args...) = CompositeCone{DefaultFloat}(args...)


# partial implementation of AbstractArray behaviours
function Base.getindex(S::CompositeCone{T}, i::Int) where {T}
    @boundscheck checkbounds(S.cones,i)
    @inbounds S.cones[i]
end

Base.getindex(S::CompositeCone{T}, b::BitVector) where {T} = S.cones[b]
Base.iterate(S::CompositeCone{T}) where{T} = iterate(S.cones)
Base.iterate(S::CompositeCone{T}, state) where{T} = iterate(S.cones, state)
Base.length(S::CompositeCone{T}) where{T} = length(S.cones)
Base.eachindex(S::CompositeCone{T}) where{T} = eachindex(S.cones)
Base.IndexStyle(S::CompositeCone{T}) where{T} = IndexStyle(S.cones)


function _make_headidx!(headidx,cones)

    if(length(cones) > 0)
        #index of first element in each cone
        headidx[1] = 1
        for i = 2:length(cones)
            headidx[i] = headidx[i-1] + numel(cones[i-1])
        end
    end
    return nothing
end
