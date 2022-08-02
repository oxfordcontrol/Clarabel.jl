using InteractiveUtils:subtypes

# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct ConeSet{T}

    cones::Vector{AbstractCone{T}}

    #API type specs and count of each cone type
    cone_specs::Vector{SupportedCone}
    type_counts::Dict{DataType,Int}

    #overall size of the composite cone
    numel::DefaultInt
    degree::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    # the scaling for unsymmetric backtrack
    scaling::T              #backtracking parameter
    minDist::T
    ind_exp::Vector{Int}    #index for exponential cones
    ind_pow::Vector{Int}    #index for power cones
    η::T                    #upper centrality parameter

    # the flag for symmetric cone check
    symFlag::Bool

    # function ConeSet{T}(types,dims,α) where {T}

    #     length(types) == length(dims) || throw(DimensionMismatch())
    function ConeSet{T}(cone_specs::Vector{<:SupportedCone}) where {T}

        #make copy to protect from user interference after setup,
        #and explicitly cast as the abstract type.  This prevents
        #errors arising from input vectors that are all the same cone,
        #and therefore more concretely typed than we want
        cone_specs = Vector{SupportedCone}(cone_specs)

        ncones = length(cone_specs)
        cones  = Vector{AbstractCone{T}}(undef,ncones)

        #create cones with the given dims
        for i in eachindex(cone_specs)
            cones[i] = ConeDict[typeof(cone_specs[i])]{T}(cone_specs[i].dim)
        end

        #count the number of each cone type
        type_counts = Dict{DataType,Int}()
        for coneT in subtypes(SupportedCone)
            type_counts[coneT] = count(C->isa(C,coneT), cone_specs)
        end

        # parameter for unsymmetric cones
        scaling = T(0.8)
        minDist = T(0.1)
        ind_exp = Vector{Int}(undef,type_counts[ExponentialConeT])
        ind_pow = Vector{Int}(undef,type_counts[PowerConeT])
        η = T(1)     #should be less than 1 theoretically; but we could set it larger empirically

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

        #headidx gives the index of the first element
        #of each constituent cone
        headidx = Vector{Int}(undef,length(cones))
        _coneset_make_headidx!(headidx,cones)

        #check whether the problem only contains symmetric cones
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
