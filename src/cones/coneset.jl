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

    # the scaling for unsymmetric backtrack
    scaling::T
    minDist::T
    ind_exp::Vector{Int}    #index for exponential cones
    ind_pow::Vector{Int}    #index for power cones
    η::T                    #centrality

    # the flag for symmetric cone check
    symFlag::Bool

    function ConeSet{T}(types,dims,α) where {T}

        length(types) == length(dims) || throw(DimensionMismatch())

        #make copy to protect from user interference after setup
        types = copy(types)

        ncones = length(types)
        cones  = Vector{AbstractCone{T}}(undef,ncones)

        #count the number of each cone type
        type_counts = Dict{SupportedCones,Int}()
        for coneT in instances(SupportedCones)
            type_counts[coneT] = count(==(coneT), types)
        end

        # parameter for unsymmetric cones
        scaling = T(0.8)
        minDist = T(0.1)
        ind_exp = Vector{Int}(undef,type_counts[ExponentialConeT])
        ind_pow = Vector{Int}(undef,type_counts[PowerConeT])
        η = T(0.99)     #should be less than 1

        cur_exp = 0
        cur_pow = 0
        # create cones with the given dims
        # store indexing of exponential and power cones
        for i in eachindex(dims)
            if types[i] == PowerConeT
                cur_pow += 1
                ind_pow[cur_pow] = i
                cones[i] = ConeDict[types[i]]{T}(α[i])
            elseif types[i] == ExponentialConeT
                cur_exp += 1
                ind_exp[cur_exp] = i
                cones[i] = ConeDict[types[i]]{T}()
            else
                cones[i] = ConeDict[types[i]]{T}(dims[i])
            end
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        #make head index
        headidx = Vector{Int}(undef,length(cones))
        _coneset_make_headidx!(headidx,cones)

        #check whether we only have symmetric cones
        if (type_counts[ZeroConeT] + type_counts[NonnegativeConeT] +
            type_counts[SecondOrderConeT] + type_counts[PSDTriangleConeT] == ncones)
            symFlag = true
        else
            symFlag = false
        end

        return new(cones,types,type_counts,numel,degree,headidx,scaling,minDist,ind_exp,ind_pow,η,symFlag)
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
