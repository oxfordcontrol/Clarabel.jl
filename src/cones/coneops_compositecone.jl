# -----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

# macro dispatch won't work unless each cone type is completely specified, i.e. 
# we can't dispatch statically on the non-concrete types PowerCone{T} or 
# ExponentialCone{T}.  So we first need a way to expand each AbstractCone{T} 
# to its complete type, including the extra parameters in the exp / power cones 

# None if this would be necessary if StaticArrays could write to MArrays 
# with non-isbits types.  See here:  
# https://github.com/JuliaArrays/StaticArrays.jl/pull/749
# If the PR is accepted then the type dependent vector and matrix types 
# defined in CONE3D_M3T_TYPE and CONE3D_V3T_TYPE could be dropped, 
# and ExponentialCone  and PowerCone would no longer need hidden 
#internal parameters with  outer-only constructors.

# turns PowerCone{T} to PowerCone{T,M3T,V3T}
function _make_conetype_concrete(::Type{PowerCone},T::Type) 
    return PowerCone{T,CONE3D_M3T_TYPE(T),CONE3D_V3T_TYPE(T)}
end
# turns ExponentialCone{T} to ExponentialCone{T,M3T,V3T}
function _make_conetype_concrete(::Type{ExponentialCone},T::Type) 
    return ExponentialCone{T,CONE3D_M3T_TYPE(T),CONE3D_V3T_TYPE(T)}
end
# turns any other AbstractCone{T} to itself
_make_conetype_concrete(conetype,T::Type) = conetype{T}

function _conedispatch(x, call)

    # We do not set thetypes = subtypes(AbstractCone), but 
    # rather to entries in our dictionary of primitive cone
    # types.   This avoids adding CompositeCone itself to the
    # switchyard we construct here, but would also prevent 
    # the use of nested CompositeCones.  
    thetypes = collect(values(ConeDict))
    foldr((t, tail) -> :(if $x isa _make_conetype_concrete($t,T); $call else $tail end), thetypes, init=Expr(:block))
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

# The Hs block for each cone.
function get_Hs!(
    cones::CompositeCone{T},
    Hsblocks::Vector{Vector{T}}
) where {T}

    for (cone, block) in zip(cones,Hsblocks)
        @conedispatch get_Hs!(cone,block)
    end
    return nothing
end

# compute the generalized product :
# WᵀWx for symmetric cones 
# μH(s)x for symmetric cones

function mul_Hs!(
    cones::CompositeCone{T},
    y::ConicVector{T},
    x::ConicVector{T},
    work::ConicVector{T}
) where {T}

    for (cone,yi,xi,worki) in zip(cones,y.views,x.views,work.views)
        @conedispatch mul_Hs!(cone,yi,xi,worki)
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
      αmax::T,
) where {T}

    α     = αmax
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
    #PJG: is this still necessary?
    if(!is_symmetric(cones))
        α = min(α,0.99)
    end

    # Force asymmetric cones last.  
    for (cone,type,dzi,dsi,zi,si) in zip(cones,cones.types,dz,ds,z,s)

        if is_symmetric(cone) continue end
        @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si,settings,α)
        α = min(α,nextαz,nextαs)
    end

    return (α,α)
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

