# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct ConeSet{T}

    cones::Vector{AbstractCone{T}}

    #Type tags and count of each cone
    types::Vector{SupportedCones}
    type_counts::Dict{SupportedCones,Int}

    #overall size of the composite cone
    numel::DefaultInt
    degree::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    function ConeSet{T}(types,dims) where {T}

        length(types) == length(dims) || throw(DimensionMismatch())

        #make copy to protect from user interference after setup
        types = copy(types)

        ncones = length(types)
        cones  = Vector{AbstractCone{T}}(undef,ncones)

        #create cones with the given dims
        for i in eachindex(dims)
            cones[i] = ConeDict[types[i]]{T}(dims[i])
        end

        #count the number of each cone type
        type_counts = Dict{SupportedCones,Int}()
        for coneT in instances(SupportedCones)
            type_counts[coneT] = count(==(coneT), types)
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        #headidx gives the index of the first element 
        #of each constituent cone
        headidx = Vector{Int}(undef,length(cones))
        _coneset_make_headidx!(headidx,cones)

        return new(cones,types,type_counts,numel,degree,headidx)
    end
end

ConeSet(args...) = ConeSet{DefaultFloat}(args...)


# partial implementation of AbstractArray behaviours
function Base.getindex(S::ConeSet{T}, i::Int) where {T}
    @boundscheck checkbounds(S.cones,i)
    @inbounds S.cones[i]
end

Base.getindex(S::ConeSet{T}, b::BitVector) where {T} = S.cones[b]
Base.iterate(S::ConeSet{T}) where{T} = iterate(S.cones)
Base.iterate(S::ConeSet{T}, state) where{T} = iterate(S.cones, state)
Base.length(S::ConeSet{T}) where{T} = length(S.cones)
Base.eachindex(S::ConeSet{T}) where{T} = eachindex(S.cones)
Base.IndexStyle(S::ConeSet{T}) where{T} = IndexStyle(S.cones)


function _coneset_make_headidx!(headidx,cones)

    if(length(cones) > 0)
        #index of first element in each cone
        headidx[1] = 1
        for i = 2:length(cones)
            headidx[i] = headidx[i-1] + numel(cones[i-1])
        end
    end
    return nothing
end
