# -----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

function _conedispatch(x, call)

    # We do not set thetypes = subtypes(AbstractCone), but 
    # rather to entries in our dictionary of primitive cone
    # types.   This avoids adding CompositeCone itself to the
    # switchyard we construct here, but would also prevent 
    # the use of nested CompositeCones.  
    thetypes = collect(values(ConeDict))
    foldr((t, tail) -> :(if $x isa $t; $call else $tail end), thetypes, init=Expr(:block))
end

macro conedispatch(call)
    esc(_conedispatch(:cone, call))
end

dim(::CompositeCone{T}) where {T} = error("dim() not well defined for the CompositeCone");
degree(cones::CompositeCone{T}) where {T} = cones.degree
numel(cones::CompositeCone{T}) where {T}  = cones.numel

# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function is_symmetric(cones::CompositeCone{T}) where {T}
    #true if all pieces are symmetric.  
    #determined during obj construction
    return cones._is_symmetric
end

function rectify_equilibration!(
    cones::CompositeCone{T},
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

# place a vector to some nearby point in the cone
function shift_to_cone!(
    cones::CompositeCone{T},
    z::ConicVector{T}
) where {T}

    for (cone,zi) in zip(cones,z.views)
        @conedispatch shift_to_cone!(cone,zi)
    end
    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    cones::CompositeCone{T},
    z::ConicVector{T},
    s::ConicVector{T}
) where {T}

    for (cone,zi,si) in zip(cones,z.views,s.views)
        @conedispatch unit_initialization!(cone,zi,si)
    end
    return nothing
end

function set_identity_scaling!(
    cones::CompositeCone{T}
) where {T}

    for cone in cones
        @conedispatch set_identity_scaling!(cone)
    end

    return nothing
end

function update_scaling!(
    cones::CompositeCone{T},
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

# The WtW block for each cone.
function get_WtW!(
    cones::CompositeCone{T},
    WtWblocks::Vector{Vector{T}}
) where {T}

    for (cone, block) in zip(cones,WtWblocks)
        @conedispatch get_WtW!(cone,block)
    end
    return nothing
end

# compute the generalized product :
# c⋅ WᵀWx for symmetric cones 
# c⋅ μH(s)x for symmetric cones

function mul_WtW!(
    cones::CompositeCone{T},
    y::ConicVector{T},
    x::ConicVector{T},
    c::T,
    work::ConicVector{T}
) where {T}

    for (cone,yi,xi,worki) in zip(cones,y.views,x.views,work.views)
        @conedispatch mul_WtW!(cone,yi,xi,c,worki)
    end

    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
function affine_ds!(
    cones::CompositeCone{T},
    ds::ConicVector{T},
    s::ConicVector{T}
) where {T}

    for (cone,dsi,si) in zip(cones,ds.views,s.views)
        @conedispatch affine_ds!(cone,dsi,si)
    end
    return nothing
end

function combined_ds_shift!(
    cones::CompositeCone{T},
    shift::ConicVector{T},
    step_z::ConicVector{T},
    step_s::ConicVector{T},
    σμ::T
) where {T}

    for (cone,shifti,step_zi,step_si) in zip(cones,shift.views,step_z.views,step_s.views)

        # compute the centering and the higher order correction parts in ds and save it in dz
        @conedispatch combined_ds_shift!(cone,shifti,step_zi,step_si,σμ)
    end

    return nothing
end

function Δs_from_Δz_offset!(
    cones::CompositeCone{T},
    out::ConicVector{T},
    ds::ConicVector{T},
    work::ConicVector{T}
) where {T}

    for (cone,outi,dsi,worki) in zip(cones,out.views,ds.views,work.views)
        @conedispatch Δs_from_Δz_offset!(cone,outi,dsi,worki) 
    end

    return nothing
end

# maximum allowed step length over all cones
function step_length(
     cones::CompositeCone{T},
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
    if(!is_symmetric(cones))
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

# compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
function compute_barrier(
    cones::CompositeCone{T},
    z::ConicVector{T},
    s::ConicVector{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    α::T
) where {T}

    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views

    barrier = zero(T)

    for (cone,zi,si,dzi,dsi) in zip(cones,z,s,dz,ds)
        @conedispatch barrier += compute_barrier(cone,zi,si,dzi,dsi,α)
    end

    return barrier
end

