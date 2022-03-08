using LinearAlgebra


function DefaultScalings{T}(
    nvars::Int,
    cone_info::ConeInfo,
    settings::Settings
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
    if isempty(cones)
        totaldegree = 0
    else
        totaldegree = sum(cone -> degree(cone), cones)
    end

    #scaled version s and z
    λ = ConicVector{T}(cone_info)

    #Left/Right diagonal scaling for problem data
    d    = Vector{T}(undef,nvars)
    dinv = Vector{T}(undef,nvars)
    D    = Diagonal(d)
    Dinv = Diagonal(dinv)

    e    = ConicVector{T}(cone_info)
    einv = ConicVector{T}(cone_info)
    E    = Diagonal(e)
    Einv = Diagonal(einv)

    c    = Ref(T(1.))

    return DefaultScalings(
            cone_info,cones,λ,totaldegree,
            d,dinv,D,Dinv,e,einv,E,Einv,c
           )
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
    diagW2::ConicVector{T}
) where {T}

    cones_get_diagonal_scaling!(scalings.cones,diagW2)

end
