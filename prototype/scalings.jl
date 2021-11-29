function DefaultConeScalings{T}(cone_info::ConeInfo) where {T}

    #look up constructors for every cone type and
    #build an array of appropriately sized cones.
    cones = AbstractCone{T}[]

    view_start_idx = 1
    for i = eachindex(cone_info.types)
        dim   = cone_info.dims[i]
        type  = cone_info.types[i]
        push!(cones, ConeDict[type](dim))
    end

    #count the number of each cone type
    #PJG: assumed to be properly ordered
    #e.g. SOCs come last
    k_zerocone = count(==(ZeroConeT),cone_info.types)
    k_nncone   = count(==(NonnegativeConeT),cone_info.types)
    k_socone   = count(==(SecondOrderT),cone_info.types)

    # total dimension and order (not the same for SOC)
    totaldim   = sum(cone -> dim(cone)  , cones)
    totalorder = sum(cone -> order(cone), cones)

    DefaultConeScalings(cone_info,cones,k_zerocone,k_nncone,k_socone,totaldim,totalorder)
end


function UpdateScalings!(scalings::DefaultConeScalings{T},
                         variables::DefaultVariables{T}) where {T}

    cones  = scalings.cones
    sviews = variables.s.views
    zviews = variables.z.views
    λviews = variables.λ.views

    # update scalings by passing subview to each of
    # the appropriate cone types.   Done here via
    # multiple dispatch, but in C by assuming that
    # the cones appear in the appropriate order and
    # according to the counts in cone_info
    foreach(UpdateScaling!,cones,sviews,zviews,λviews)

end

function IdentityScalings!(scalings::DefaultConeScalings{T},
                           variables::DefaultVariables{T}) where {T}

    foreach(IdentityScaling!,scalings.cones)

end

function make_scaling_matrix(scalings::DefaultConeScalings{T}) where {T}

    # PJG: This is super inefficient and only for
    # temporary use.  I am computing W^TW here.

    WtW = SparseMatrixCSC(zeros(0,0))

    for i = 1:length(scalings.cones)
        Wnext = make_WTW(scalings.cones[i])
        WtW = blockdiag(WtW,Wnext)
    end

    return WtW

end
