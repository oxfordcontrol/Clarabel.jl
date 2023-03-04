mutable struct Presolver{T}

    # possibly reduced internal copy of user cone specification
    cone_specs::Vector{SupportedCone}

    # vector of length = original RHS.   Entries are false
    # for those rows that should be eliminated before solve
    reduce_idx::Union{BitVector,Nothing}

    # vector of length = reduced RHS, taking values 
    #that map reduced b back to their original index
    lift_map::Union{Vector{Int64},Nothing}

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

        if !settings.presolve_enable 
            reduce_idx = nothing 
            lift_map   = nothing 
            mreduced   = mfull 

        else 
            (reduce_idx, lift_map) = reduce_cones!(cone_specs,b,T(infbound))
            mreduced = isnothing(reduce_idx) ? mfull : length(lift_map)

        end 

        return new(cone_specs,reduce_idx,lift_map, mfull, mreduced, infbound)

    end

end

Presolver(args...) = Presolver{DefaultFloat}(args...)

is_reduced(ps::Presolver{T})    where {T} = ps.mfull != ps.mreduced
count_reduced(ps::Presolver{T}) where {T} = ps.mfull  - ps.mreduced


function reduce_cones!(
    cone_specs::Vector{<:SupportedCone}, 
    b::Vector{T},
    infbound::T) where {T}

    reduce_idx = trues(length(b))

    # we loop through the finite_idx and shrink any nonnegative 
    # cones that are marked as having infinite right hand sides.   
    # Mark the corresponding entries as zero in the reduction index

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
                    reduce_idx[i] = false
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

    #if we reduced anything then return the reduce_idx and a 
    #map of the entries to keep back into the original vector 
    if is_reduced
        return (reduce_idx, findall(reduce_idx))

    else 
        return (nothing, nothing)
    end

end 
