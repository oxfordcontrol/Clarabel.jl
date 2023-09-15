
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

function is_sparse_expandable(cones::CompositeCone{T}) where {T}
    
    #This should probably never be called
    #any(is_sparse_expandable, cones)
    ErrorException("This function should not be reachable")
    
end

function allows_primal_dual_scaling(cones::CompositeCone{T}) where {T}
    all(allows_primal_dual_scaling, cones)
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

function margins(
    cones::CompositeCone{T},
    z::ConicVector{T},
    pd::PrimalOrDualCone,
) where {T}
    α = typemax(T)
    β = zero(T)
    for (cone,zi) in zip(cones,z.views)
        @conedispatch (αi,βi) = margins(cone,zi,pd)
        α = min(α,αi)
        β += βi
    end

    return (α,β)
end

function scaled_unit_shift!(
    cones::CompositeCone{T},
    z::ConicVector{T},
    α::T,
    pd::PrimalOrDualCone
) where {T}

    for (cone,zi) in zip(cones,z.views)
        @conedispatch scaled_unit_shift!(cone,zi,α,pd)
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
        @conedispatch is_scaling_success = update_scaling!(cone,si,zi,μ,scaling_strategy)
        # YC: currently, only check whether SOC variables are in the interior;
        # we could extend the interior checkfor other cones
        if !is_scaling_success
            return is_scaling_success = false
        end
    end
    return is_scaling_success = true
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
    work::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,outi,dsi,worki,zi) in zip(cones,out.views,ds.views,work.views,z.views)
        @conedispatch Δs_from_Δz_offset!(cone,outi,dsi,worki,zi) 
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
    for (cone,dzi,dsi,zi,si) in zip(cones,dz,ds,z,s)

        if !is_symmetric(cone) continue end
        @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si,settings,α)
        α = min(α,nextαz,nextαs)
    end
        
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(!is_symmetric(cones))
        α = min(α,settings.max_step_fraction)
    end

    # Force asymmetric cones last.  
    for (cone,dzi,dsi,zi,si) in zip(cones,dz,ds,z,s)

        if @conedispatch(is_symmetric(cone)) continue end
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

