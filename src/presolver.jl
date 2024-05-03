
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

    return Presolver{T}(init_cones, reduce_map, mfull, mreduced, infbound)

end


is_reduced(ps::Presolver{T})    where {T} = !isnothing(ps.reduce_map)
count_reduced(ps::Presolver{T}) where {T} = ps.mfull  - ps.mreduced


function presolve(
    presolver::Presolver{T}, 
    A::SparseMatrixCSC{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone}
) where {T}

    A_new, b_new = reduce_A_b(presolver,A,b)
    cones_new    = reduce_cones(presolver, cones)

    return A_new, b_new, cones_new 

end

function reduce_A_b(
    presolver::Presolver{T}, 
    A::SparseMatrixCSC{T}, 
    b::Vector{T}
) where{T}

    @assert !isnothing(presolver.reduce_map)
    map = presolver.reduce_map
    A = A[map.keep_logical,:]
    b = b[map.keep_logical]
    
    A, b

end 

function reduce_cones(
    presolver::Presolver{T},
    cones::Vector{SupportedCone}, 
) where {T}

    @assert !isnothing(presolver.reduce_map)
    map = presolver.reduce_map

    cones_new = sizehint!(SupportedCone[],length(cones))
    keep_iter = Iterators.Stateful(map.keep_logical)

    for cone in cones 
        numel_cone = nvars(cone)
        
        if isa(cone, NonnegativeConeT)
            nkeep = count(Iterators.take(keep_iter,numel_cone))
            if nkeep > 0
                push!(cones_new, NonnegativeConeT(nkeep))
            end 
        else 
            keep_iter = Iterators.drop(keep_iter,numel_cone)
            push!(cones_new, deepcopy(cone))
        end         
    end

    return cones_new

end 

function reverse_presolve!(
    presolver::Presolver{T}, 
    solution::DefaultSolution{T},
    variables::DefaultVariables{T}
) where {T}

    @. solution.x = variables.x

    map = presolver.reduce_map
    ctr = 1  

    for (idx,keep) in enumerate(map.keep_logical)
        if keep
            solution.s[idx] = variables.s[ctr] 
            solution.z[idx] = variables.z[ctr] 
            ctr += 1
        else 
            solution.s[idx] = T(presolver.infbound)
            solution.z[idx] = zero(T)
        end
    end

end


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

    idx = 1 # index into the b vector

    for cone in cones
        numel_cone = nvars(cone)

        if isa(cone, NonnegativeConeT)
            for _ in 1:numel_cone
                if b[idx] > infbound
                    keep_logical[idx] = false
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
            PresolverRowReductionIndex(keep_logical)
        else 
            nothing
        end
    end

    (outoption, mreduced)
end 
