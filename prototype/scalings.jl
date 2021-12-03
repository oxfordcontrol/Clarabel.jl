function DefaultConeScalings{T}(
    cone_info::ConeInfo) where {T}

    #look up constructors for every cone type and
    #build an array of appropriately sized cones.
    cones = AbstractCone{T}[]

    view_start_idx = 1
    for i = eachindex(cone_info.types)
        dim   = cone_info.dims[i]
        type  = cone_info.types[i]
        push!(cones, ConeDict[type](dim))
    end

    # total cone order (not the same as dimension for SO and zero)
    totalorder = sum(cone -> order(cone), cones)

    λ = SplitVector{T}(cone_info)

    DefaultConeScalings(cone_info,cones,λ,totalorder)

end


function UpdateScalings!(
    scalings::DefaultConeScalings{T},
    variables::DefaultVariables{T}) where {T}

    cones  = scalings.cones
    sviews = variables.s.views
    zviews = variables.z.views
    λviews = scalings.λ.views

    # update scalings by passing subview to each of
    # the appropriate cone types.   
    foreach(UpdateScaling!,cones,sviews,zviews,λviews)

end

function IdentityScalings!(
    scalings::DefaultConeScalings{T},
    variables::DefaultVariables{T}) where {T}

    foreach(IdentityScaling!,scalings.cones)

end
