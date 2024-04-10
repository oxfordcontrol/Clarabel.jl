
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

   # original cones of the problem
    init_cones::Vector{SupportedCone}

    # record of reduced constraints for NN cones with inf bounds
    reduce_map::Option{PresolverRowReductionIndex}

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
        cones::Vector{SupportedCone},
        settings::Settings{T}
    ) where {T}

        infbound = Clarabel.get_infinity()

        # make copy of cones to protect from user interference
        init_cones = Vector{SupportedCone}(cones)

        mfull = length(b)

        (reduce_map, mreduced) = make_reduction_map(cones,b,T(infbound))

        return new(init_cones, reduce_map, mfull, mreduced, infbound)

    end

end

Presolver(args...) = Presolver{DefaultFloat}(args...)

is_reduced(ps::Presolver{T})    where {T} = !isnothing(ps.reduce_map)
count_reduced(ps::Presolver{T}) where {T} = ps.mfull  - ps.mreduced

function make_reduction_map(
    cones::Vector{SupportedCone}, 
    b::Vector{T},
    infbound::T
) where {T}

    keep_logical = trues(length(b))
    mreduced     = length(b)

    # only try to reduce nn cones. Make a slight contraction
    # so that we are firmly "less than" here
    infbound *= (1-10*eps(T))

    idx = 1

    for cone in cones
        numel_cone = nvars(cone)

        if isa(cone, NonnegativeConeT)
            for _ in 1:numel_cone
                if b[idx] > infbound
                    keep_logical[i] = false
                    mreduced -= 1
                end
                idx += 1
            end
        else 
            # skip this cone 
            idx += numel_cone
        end   
    end

    outoption = 
    let
        if mreduced < length(b)  
            keep_index = findall(keep_logical)
            PresolverRowReductionIndex(keep_logical, keep_index)
        else 
            nothing
        end
    end

    (outoption, mreduced)
end 


function presolve(
    presolver::Presolver{T}, 
    A::AbstractMatrix{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone}
) where {T}

    A_new, b_new = reduce_A_b(presolver,A,b)
    cones_new    = reduce_cones(presolver, cones)

    return A_new, b_new, cones_new 

end

function reduce_A_b(
    presolver::Presolver{T}, 
    A::AbstractMatrix{T}, 
    b::Vector{T}
) where{T}

    @assert !isnothing(presolver.reduce_map)
    map = presolver.reduce_map
    A = A[map.keep_logical,:]
    b = b[map.keep_logical]

end 

function reduce_cones(
    presolver::Presolver{T},
    cones::Vector{SupportedCone}, 
) where {T}

    @assert !isnothing(presolver.reduce_map)
    map = presolver.reduce_map

    # assume that we will end up with the same 
    # number of cones, despite small possibility 
    # that some will be completely eliminated

    cones_new = sizehint!(SupportedCone[],length(cones))
    keep_iter = Iterators.Stateful(map.keep_logical)

    for cone in cones 

        numel_cone = nvars(cone)
        markers    = Iterators.take(keep_iter,numel_cone)
        
        if isa(cone, NonnegativeConeT)
            nkeep = count(markers)
            if nkeep > 0
                push!(cones_new, NonnegativeConeT(nkeep))
            end 
        else 
            push!(cones_new, deepcopy(cone))
        end         
    end

    return cones_new

end 

function presolve(
    A::AbstractMatrix{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone}, 
    settings::Settings{T}
) where {T}

    if(!settings.presolve_enable)
        return nothing
    end

    presolver = Presolver{T}(A,b,cones,settings)

    if !is_reduced(presolver)
        return nothing 
    end 

    return presolver 
end