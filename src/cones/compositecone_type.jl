# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct CompositeCone{T} <: AbstractCone{T}

    cones::Vector{AbstractCone{T}}

    #count of each cone type
    type_counts::Dict{Type,Int}

    #overall size of the composite cone
    numel::DefaultInt
    degree::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    # the flag for symmetric cone check
    _is_symmetric::Bool

    function CompositeCone{T}(cone_specs::Vector{SupportedCone}) where {T}

        ncones = length(cone_specs)
        cones  = AbstractCone{T}[]
        sizehint!(cones,ncones)

        type_counts = Dict{Type,Int}()

        #assumed symmetric to start
        _is_symmetric = true

        #create cones with the given dims
        for coneT in cone_specs
            #make a new cone
            cone = make_cone(T, coneT);

            #update global problem symmetry
            _is_symmetric = _is_symmetric && is_symmetric(cone)

            #increment type counts 
            key = ConeDict[typeof(coneT)]
            haskey(type_counts,key) ? type_counts[key] += 1 : type_counts[key] = 1
            
            push!(cones,cone)
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        #headidx gives the index of the first element
        #of each constituent cone
        headidx = Vector{Int}(undef,length(cones))
        _make_headidx!(headidx,cones)

        return new(cones,type_counts,numel,degree,headidx,_is_symmetric)
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

function get_type_count(cones::CompositeCone{T}, type::Type) where {T}
    if haskey(cones.type_counts,type)
        return cones.type_counts[type]
    else
        return 0
    end
end

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
