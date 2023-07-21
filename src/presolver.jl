
struct PresolverRowReductionIndex 

    # vector of length = original RHS.   Entries are false
    # for those rows that should be eliminated before solve
    keep_logical::Vector{Bool}

    # vector of length = reduced RHS, taking values
    # that map reduced b back to their original index
    # This is just findall(keep_logical) and is held for
    # efficient solution repopulation
    keep_index::Vector{Int64}

end

struct Presolver{T}

    # possibly reduced internal copy of user cone specification
    cone_specs::Vector{SupportedCone}

    # record of reduced constraints for NN cones with inf bounds
    reduce_map::Union{Nothing,PresolverRowReductionIndex}

    # size of original and reduced RHS, respectively 
    mfull::Int64 
    mreduced::Int64

    # inf bound that was taken from the module level 
    # and should be applied throughout.   Held here so 
    # that any subsequent change to the module's state 
    # won't mess up our solver mid-solve 
    infbound::Float64 

    function Presolver{T}(
        A::AbstractMatrix{T},
        b::Vector{T},
        cone_specs::Vector{<:SupportedCone},
        settings::Settings{T}
    ) where {T}

        infbound = Clarabel.get_infinity()

        # make copy of cone_specs to protect from user interference after
        # setup and explicitly cast as the abstract type.  This also prevents
        # errors arising from input vectors that are all the same cone, and
        # therefore more concretely typed than we want
        cone_specs = Vector{SupportedCone}(cone_specs)
        mfull = length(b)

        (reduce_map, mreduced) = let
            if settings.presolve_enable 
                reduce_cones!(cone_specs,b,T(infbound))
            else 
                (nothing,mfull)
            end 
        end
    
        return new(cone_specs,reduce_map,mfull, mreduced, infbound)

    end

end

Presolver(args...) = Presolver{DefaultFloat}(args...)

is_reduced(ps::Presolver{T})    where {T} = !isnothing(ps.reduce_map)
count_reduced(ps::Presolver{T}) where {T} = ps.mfull  - ps.mreduced


function reduce_cones!(
    cone_specs::Vector{<:SupportedCone}, 
    b::Vector{T},
    infbound::T) where {T}

    keep_logical = trues(length(b))
    mreduced     = length(b)

    # we loop through b and remove any entries that are both infinite
    # and in a nonnegative cone

    is_reduced = false
    bptr = 1   # index into the b vector 

    for (cidx,cone) in enumerate(cone_specs)  

        numel_cone = nvars(cone)

        # only try to reduce nn cones. Make a slight contraction
        # so that we are firmly "less than" here
        infbound *= (1-10*eps(T))

        if isa(cone, NonnegativeConeT)
            num_finite = 0
            for i in bptr:(bptr + numel_cone - 1)
                if b[i] < infbound
                    num_finite += 1 
                else 
                    keep_logical[i] = false
                    mreduced -= 1
                end
            end
            if num_finite < numel_cone 
                # contract the cone to a smaller size
                cone_specs[cidx] = NonnegativeConeT(num_finite)
                is_reduced = true
            end
        end   
        
        bptr += numel_cone
    end

    outoption = 
    let
        if is_reduced
            keep_index = findall(keep_logical)
            PresolverRowReductionIndex(keep_logical, keep_index)
        else 
            nothing
        end
    end

    (outoption, mreduced)
end 
