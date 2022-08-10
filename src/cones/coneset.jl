# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct ConeSet{T}

    cones::Vector{AbstractCone{T}}

    #API type specs and count of each cone type
    cone_specs::Vector{SupportedCone}
    types::Vector{DataType}
    type_counts::Dict{DataType,Int}

    #overall size of the composite cone
    numel::DefaultInt
    degree::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    # the scaling for asymmetric backtrack
    backtrack::T              #backtracking parameter

    # the flag for symmetric cone check
    sym_flag::Bool

    function ConeSet{T}(cone_specs::Vector{<:SupportedCone}) where {T}

        #make copy to protect from user interference after setup,
        #and explicitly cast as the abstract type.  This prevents
        #errors arising from input vectors that are all the same cone,
        #and therefore more concretely typed than we want
        cone_specs = Vector{SupportedCone}(cone_specs)

        ncones = length(cone_specs)
        cones  = Vector{AbstractCone{T}}(undef,ncones)
        types = Vector{DataType}(undef,ncones)

        #count the number of each cone type
        type_counts = Dict{DataType,Int}()
        for coneT in keys(ConeDict)
            type_counts[coneT] = count(C->isa(C,coneT), cone_specs)
        end

        # parameter for asymmetric cones
        backtrack = T(0.8)

        #create cones with the given dims
        for i in eachindex(cone_specs)
            types[i] = typeof(cone_specs[i])
            if types[i] == ExponentialConeT
                cones[i] = ConeDict[typeof(cone_specs[i])]{T}()
            elseif types[i] == PowerConeT
                cones[i] = ConeDict[typeof(cone_specs[i])]{T}(cone_specs[i].α)
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
        _coneset_make_headidx!(headidx,cones)

        #check whether the problem only contains symmetric cones
        if (type_counts[ZeroConeT] + type_counts[NonnegativeConeT] +
            type_counts[SecondOrderConeT] + type_counts[PSDTriangleConeT] == ncones)
            sym_flag = true
        else
            sym_flag = false
        end

        return new(cones,cone_specs,types,type_counts,numel,degree,headidx,backtrack,sym_flag)
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
