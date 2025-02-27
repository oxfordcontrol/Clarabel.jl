
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
     δ::Vector{T},
) where{T}

    any_changed = false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    δ .= 1

    for (cone,rng) in zip(cones,cones.rng_cones)
        δi = view(δ,rng)
        @conedispatch rectify_equilibration!(cone,δi)
    end

end

function margins(
    cones::CompositeCone{T},
    z::Vector{T},
    pd::PrimalOrDualCone,
) where {T}
    α = typemax(T)
    β = zero(T)
    for (cone,rng) in zip(cones,cones.rng_cones)
        @conedispatch (αi,βi) = margins(cone,view(z,rng),pd)
        α = min(α,αi)
        β += βi
    end

    return (α,β)
end

function scaled_unit_shift!(
    cones::CompositeCone{T},
    z::Vector{T},
    α::T,
    pd::PrimalOrDualCone
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        @conedispatch scaled_unit_shift!(cone,view(z,rng),α,pd)
    end

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    cones::CompositeCone{T},
    z::Vector{T},
    s::Vector{T}
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        @conedispatch unit_initialization!(cone,view(z,rng),view(s,rng))
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
    s::Vector{T},
    z::Vector{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        si = view(s,rng)
        zi = view(z,rng)
        @conedispatch is_scaling_success = update_scaling!(cone,si,zi,μ,scaling_strategy)
        if !is_scaling_success
            return is_scaling_success = false
        end
    end
    return is_scaling_success = true
end

# The Hs block for each cone.
function get_Hs!(
    cones::CompositeCone{T},
    Hsblock::Vector{T}
) where {T}

    for (cone, rng) in zip(cones,cones.rng_blocks)
        @conedispatch get_Hs!(cone,view(Hsblock,rng))
    end
    return nothing
end

# compute the generalized product :
# WᵀWx for symmetric cones 
# μH(s)x for symmetric cones

function mul_Hs!(
    cones::CompositeCone{T},
    y::Vector{T},
    x::Vector{T},
    work::Vector{T}
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        @conedispatch mul_Hs!(cone,view(y,rng),view(x,rng),view(work,rng))
    end

    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
function affine_ds!(
    cones::CompositeCone{T},
    ds::Vector{T},
    s::Vector{T}
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        dsi = view(ds,rng)
        si  = view(s,rng)
        @conedispatch affine_ds!(cone,dsi,si)
    end
    return nothing
end

function combined_ds_shift!(
    cones::CompositeCone{T},
    shift::Vector{T},
    step_z::Vector{T},
    step_s::Vector{T},
    σμ::T
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        shifti = view(shift,rng)
        step_zi = view(step_z,rng)
        step_si = view(step_s,rng)
        @conedispatch combined_ds_shift!(cone,shifti,step_zi,step_si,σμ)
    end

    return nothing
end

function Δs_from_Δz_offset!(
    cones::CompositeCone{T},
    out::Vector{T},
    ds::Vector{T},
    work::Vector{T},
    z::Vector{T}
) where {T}

    for (cone,rng) in zip(cones,cones.rng_cones)
        outi  = view(out,rng)
        dsi   = view(ds,rng)
        worki = view(work,rng)
        zi    = view(z,rng)
        @conedispatch Δs_from_Δz_offset!(cone,outi,dsi,worki,zi) 
    end

    return nothing
end

# maximum allowed step length over all cones
function step_length(
     cones::CompositeCone{T},
        dz::Vector{T},
        ds::Vector{T},
         z::Vector{T},
         s::Vector{T},
  settings::Settings{T},
      αmax::T,
) where {T}

    α = αmax

    function innerfcn(α,symcond)
        for (cone,rng) in zip(cones,cones.rng_cones)
            if @conedispatch is_symmetric(cone) == symcond
                continue 
            end
            (dzi,dsi) = (view(dz,rng),view(ds,rng))
            (zi,si)  = (view(z,rng),view(s,rng))
            @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si,settings,α)
            α = min(α,nextαz,nextαs)
        end
        return α
    end

    # Force symmetric cones first.   
    α = innerfcn(α,true)
        
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(!is_symmetric(cones))
        α = min(α,settings.max_step_fraction)
    end

    # now the nonsymmetric cones
    α = innerfcn(α,false)

    return (α,α)
end

# compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
function compute_barrier(
    cones::CompositeCone{T},
    z::Vector{T},
    s::Vector{T},
    dz::Vector{T},
    ds::Vector{T},
    α::T
) where {T}
    barrier = zero(T)

    for (cone,rng) in zip(cones,cones.rng_cones)
        zi = view(z,rng)
        si = view(s,rng)
        dzi = view(dz,rng)
        dsi = view(ds,rng)
        @conedispatch barrier += compute_barrier(cone,zi,si,dzi,dsi,α)
    end

    return barrier
end

