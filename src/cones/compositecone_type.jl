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

    #ranges for the indices of the constituent cones
    rng_cones::Vector{UnitRange{Int64}}

    #ranges for the indices of the constituent Hs blocks
    #associated with each cone
    rng_blocks::Vector{UnitRange{Int64}}

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

        #ranges for the subvectors associated with each cone,
        #and the range for the corresponding entries
        #in the Hs sparse block

        rng_cones = _make_rng_cones(cones);
        rng_blocks = _make_rng_blocks(cones);

        return new(cones,type_counts,numel,degree,rng_cones,rng_blocks,_is_symmetric)
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

function _make_rng_cones(cones)

    rngs = UnitRange{Int64}[]
    sizehint!(rngs,length(cones))

    if !isempty(cones)
        start = 1
        for cone in cones
            stop = start + numel(cone) - 1
            push!(rngs,start:stop)
            start = stop + 1
        end
    end
    return rngs
end

function _make_rng_blocks(cones)

    rngs = UnitRange{Int64}[]
    sizehint!(rngs,length(cones))

    if !isempty(cones)
        start = 1
        for cone in cones
            nvars = numel(cone)
            if Hs_is_diagonal(cone)
                stop = start + nvars - 1
            else
                stop = start + triangular_number(nvars) - 1
            end
            push!(rngs,start:stop)
            start = stop + 1
        end
    end
    return rngs
end 