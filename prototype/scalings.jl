function DefaultScalings{T}(
    cone_info::ConeInfo
) where {T}

    #look up constructors for every cone type and
    #build an array of appropriately sized cones.
    cones = AbstractCone{T}[]

    view_start_idx = 1
    for i = eachindex(cone_info.types)
        dim   = cone_info.dims[i]
        type  = cone_info.types[i]
        push!(cones, ConeDict[type](dim))
    end

    # total cone degree (not the same as dimension for SOC and zero cone)
    totaldegree = sum(cone -> degree(cone), cones)

    λ = SplitVector{T}(cone_info)

    return DefaultScalings(cone_info,cones,λ,totaldegree)
end

function scaling_update!(
    scalings::DefaultScalings{T},
    variables::DefaultVariables{T}
) where {T}

    # we call via this function instead of calling
    # the operation on the Vector{AbstractCones{T}} directly
    # so that we can isolate the top level solver from
    # our default implementation of the scaling update
    cones_update_scaling!(scalings.cones,variables.s,variables.z,scalings.λ)
    return nothing
end


#set all scalings to identity (or zero for the zero cone)
function scaling_identity!(
    scalings::DefaultScalings{T}
) where {T}

    cones_set_identity_scaling!(scalings.cones)
    return nothing
end


function scaling_get_diagonal!(
    scalings::DefaultScalings{T},
    diagW2::SplitVector{T}
) where {T}

    cones_get_diagonal_scaling!(scalings.cones,diagW2)

end
