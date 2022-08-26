# -----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

function _conedispatch(x, call)
    thetypes = collect(values(ConeDict))
    foldr((t, tail) -> :(if $x isa $t; $call else $tail end), thetypes, init=Expr(:block))
end

macro conedispatch(call)
    esc(_conedispatch(:cone, call))
end

# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function cones_is_symmetric(cones::ConeSet{T}) where {T}
    #true if all pieces are symmetric.  
    #determined during obj construction
    return cones._is_symmetric
end

function cones_rectify_equilibration!(
    cones::ConeSet{T},
     δ::ConicVector{T},
     e::ConicVector{T}
) where{T}

    any_changed = false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    δ .= 1

    for (cone,δi,ei) in zip(cones,δ.views,e.views)
        @conedispatch any_changed |= rectify_equilibration!(cone,δi,ei)
    end

    return any_changed
end


function cones_update_scaling!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update cone scalings by passing subview to each of
    # the appropriate cone types.
    for (cone,si,zi) in zip(cones,s.views,z.views)
        @conedispatch update_scaling!(cone,si,zi,μ,scaling_strategy)
    end

    return nothing
end

function cones_set_identity_scaling!(
    cones::ConeSet{T}
) where {T}

    for cone in cones
        @conedispatch set_identity_scaling!(cone)
    end

    return nothing
end

# The WtW block for each cone.
function cones_get_WtW_blocks!(
    cones::ConeSet{T},
    WtWblocks::Vector{Vector{T}}
) where {T}

    for (cone, block) in zip(cones,WtWblocks)
        @conedispatch get_WtW_block!(cone,block)
    end
    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
function cones_affine_ds!(
    cones::ConeSet{T},
    x::ConicVector{T},
    s::ConicVector{T}
) where {T}

    for (cone,xi,si) in zip(cones,x.views,s.views)
        @conedispatch affine_ds!(cone,xi,si)
    end
    return nothing
end


# place a vector to some nearby point in the cone
function cones_shift_to_cone!(
    cones::ConeSet{T},
    z::ConicVector{T}
) where {T}

    for (cone,zi) in zip(cones,z.views)
        @conedispatch shift_to_cone!(cone,zi)
    end
    return nothing
end

# unit initialization for asymmetric solves
function cones_unit_initialization!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,si,zi) in zip(cones,s.views,z.views)
        @conedispatch unit_initialization!(cone,si,zi)
    end
    return nothing
end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function cones_combined_ds!(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    step_z::ConicVector{T},
    step_s::ConicVector{T},
    σμ::T
) where {T}

    for (cone,dzi,zi,si) in zip(cones,dz.views,step_z.views,step_s.views)

        # compute the centering and the higher order correction parts in ds and save it in dz
        @conedispatch combined_ds!(cone,dzi,zi,si,σμ)
    end

    #We are relying on d.s = λ ◦ λ (symmetric) or d.s = s (asymmetric) already from the affine step here
    ds .+= dz

    return nothing
end

# compute the generalized step Wᵀ(λ \ ds)
function cones_Wt_λ_inv_circ_ds!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    rz::ConicVector{T},
    rs::ConicVector{T},
    Wtlinvds::ConicVector
) where {T}

    for (cone,lzi,rzi,rsi,Wtlinvdsi) in zip(cones,lz.views,rz.views,rs.views,Wtlinvds.views)
        @conedispatch Wt_λ_inv_circ_ds!(cone,lzi,rzi,rsi,Wtlinvdsi)
    end

    return nothing
end

# compute the generalized step of -WᵀWΔz
function cones_WtW_Δz!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    ls::ConicVector{T},
    workz::ConicVector{T}
) where {T}

    for (cone,lzi,lsi,workzi) in zip(cones,lz.views,ls.views,workz.views)
        @conedispatch WtW_Δz!(cone,lzi,lsi,workzi)
    end

    return nothing
end

# maximum allowed step length over all cones
function cones_step_length(
     cones::ConeSet{T},
        dz::ConicVector{T},
        ds::ConicVector{T},
         z::ConicVector{T},
         s::ConicVector{T},
  settings::Settings{T},
         α::T,
         steptype::Symbol
) where {T}

    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views

    # Force symmetric cones first.   
    for (cone,type,dzi,dsi,zi,si) in zip(cones,cones.types,dz,ds,z,s)

        if !is_symmetric(cone) continue end
        @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si,settings,α)
        α = min(α,nextαz,nextαs)
    end
        
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(!cones_is_symmetric(cones))
        α = min(α,0.99)
    end

    # Force asymmetric cones last.  
    for (cone,type,dzi,dsi,zi,si) in zip(cones,cones.types,dz,ds,z,s)

        if is_symmetric(cone) continue end
        @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si,settings,α)
        α = min(α,nextαz,nextαs)
    end

    if(steptype == :combined)
        α *= settings.max_step_fraction
    end

    return α
end

# check the distance to the boundary for asymmetric cones
# PJG: This function should not take DefaultVariables as 
# input, since the definition is problem dependent.  Instead 
# it should take only ConicVectors like other functions in 
# this file
function check_μ_and_centrality(
    cones::ConeSet{T},
    step::DefaultVariables{T},
    variables::DefaultVariables{T},
    work::DefaultVariables{T},
    α::T,
    settings::Settings{T},
) where {T}

    dz    = step.z
    ds    = step.s
    dτ    = step.τ
    dκ    = step.κ
    z     = variables.z
    s     = variables.s
    τ     = variables.τ
    κ     = variables.κ
    cur_z = work.z
    cur_s = work.s

    central_coef = cones.degree + 1

    backtrack = settings.linesearch_backtrack_step

    for j = 1:50
        # current z,s
        @. cur_z = z + α*dz
        @. cur_s = s + α*ds
        cur_τ = τ + α*dτ
        cur_κ = κ + α*dκ

        # compute current μ
        μ = (dot(cur_s,cur_z) + cur_τ*cur_κ)/central_coef

        # check centrality
        barrier = central_coef*logsafe(μ) - logsafe(cur_τ) - logsafe(cur_κ)

        for (cone,cur_si,cur_zi) in zip(cones,cur_s.views, cur_z.views)
            @conedispatch barrier += compute_centrality(cone, cur_si, cur_zi)
        end

        if barrier < 1.
            return α
        else
            α = backtrack*α   #backtrack line search
        end

    end

    return α
end
