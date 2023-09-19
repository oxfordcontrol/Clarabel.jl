
function variables_calc_mu(
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    cones::CompositeCone{T}
) where {T}

  μ = (residuals.dot_sz + variables.τ * variables.κ)/(cones.degree + 1)

  return μ
end


function variables_calc_step_length(
    variables::DefaultVariables{T},
    step::DefaultVariables{T},
    cones::CompositeCone{T},
    settings::Settings{T},
    steptype::Symbol
) where {T}

    ατ    = step.τ < 0 ? -variables.τ / step.τ : floatmax(T)
    ακ    = step.κ < 0 ? -variables.κ / step.κ : floatmax(T)

    # Find a feasible step size for all cones
    α = min(ατ,ακ,one(T))
    (αz,αs) = step_length(cones, step.z, step.s, variables.z, variables.s, settings, α)

    # We have partly preserved the option of implementing 
    # split length steps, but at present step_length
    # itself only allows for a single maximum value.  
    # To enable split lengths, we need to also pass a 
    # tuple of limits to the step_length function of 
    # every cone 
    α = min(αz, αs)

    if(steptype == :combined)
        α *= settings.max_step_fraction
    end


    return α
end


function variables_barrier(
    variables::DefaultVariables{T},
    step::DefaultVariables{T},
    α::T,
    cones::CompositeCone{T},
) where {T}

    central_coef = cones.degree + 1

    cur_τ = variables.τ + α*step.τ
    cur_κ = variables.κ + α*step.κ

    # compute current μ
    sz = dot_shifted(variables.z,variables.s,step.z,step.s,α)
    μ = (sz + cur_τ*cur_κ)/central_coef

    # barrier terms from gap and scalars
    barrier = central_coef*logsafe(μ) - logsafe(cur_τ) - logsafe(cur_κ)

    # barriers from the cones
    ( z, s) = (variables.z, variables.s)
    (dz,ds) = (step.z, step.s)

    barrier += compute_barrier(cones, z, s, dz, ds, α)

    return barrier
end

function variables_copy_from(dest::DefaultVariables{T},src::DefaultVariables{T}) where {T}
    dest.x .= src.x
    dest.s .= src.s
    dest.z .= src.z
    dest.τ  = src.τ
    dest.κ  = src.κ
end 

function variables_scale_cones!(
    variables::DefaultVariables{T},
    cones::CompositeCone{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}
    return update_scaling!(cones,variables.s,variables.z,μ,scaling_strategy)
end


function variables_add_step!(
    variables::DefaultVariables{T},
    step::DefaultVariables{T}, α::T
) where {T}

    @. variables.x += α*step.x
    @. variables.s += α*step.s
    @. variables.z += α*step.z
    variables.τ    += α*step.τ
    variables.κ    += α*step.κ

    return nothing
end


function variables_affine_step_rhs!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    cones::CompositeCone{T}
) where{T}

    @. d.x    .=  r.rx
    @. d.z     =  r.rz
    affine_ds!(cones, d.s, variables.s)    # asymmetric cones need value of s
    d.τ        =  r.rτ
    d.κ        =  variables.τ * variables.κ

    return nothing
end


function variables_combined_step_rhs!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    cones::CompositeCone{T},
    step::DefaultVariables{T},
    σ::T,
    μ::T,
    m::T,
) where {T}

    dotσμ = σ*μ

    @. d.x  = (one(T) - σ)*r.rx
       d.τ  = (one(T) - σ)*r.rτ
       d.κ  = - dotσμ + m * step.τ * step.κ + variables.τ * variables.κ

    # ds is different for symmetric and asymmetric cones:
    # Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
    # Asymmetric cones: d.s = s + σμ*g(z)

    # we want to scale the Mehotra correction in the symmetric 
    # case by M, so just scale step_z by M.  This is an unnecessary
    # vector operation (since it amounts to M*z'*s), but it 
    # doesn't happen very often 
    if (m != one(T))
        step.z .*= m
    end

    combined_ds_shift!(cones,d.z,step.z,step.s,dotσμ)

    #We are relying on d.s = affine_ds already here
    d.s .+= d.z

    # now we copy the scaled res for rz and d.z is no longer work
    @. d.z .= (1 - σ)*r.rz

    return nothing
end

# Calls shift_to_cone on all conic variables and does not 
# touch the primal variables. Used for symmetric problems.

function variables_symmetric_initialization!(
    variables::DefaultVariables{T},
    cones::CompositeCone{T}
) where {T}

    _shift_to_cone_interior!(variables.s, cones, PrimalCone::PrimalOrDualCone)
    _shift_to_cone_interior!(variables.z, cones, DualCone::PrimalOrDualCone)

    variables.τ = 1
    variables.κ = 1
end


function _shift_to_cone_interior!(
    z::AbstractVector{T},
    cones::CompositeCone{T},
    pd::PrimalOrDualCone
) where {T}
    
    (min_margin, pos_margin) = margins(cones,z,pd)
    target  =  max(one(T),T(0.1) * pos_margin / degree(cones))

    if min_margin <= 0  #at least some component is outside its cone
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        scaled_unit_shift!(cones,z,-min_margin, pd)
        scaled_unit_shift!(cones,z, target, pd)

    elseif min_margin < target
        #margin is positive but too small
        scaled_unit_shift!(cones,z, target-min_margin, pd)

    else 
        #good margin, but still shift explicitly by 
        #zero to catch any elements in the zero cone 
        #that need to be forced to zero 
        scaled_unit_shift!(cones,z, zero(T), pd)
    end

    return nothing

end


# Calls unit initialization on all conic variables and zeros 
# the primal variables.   Used for nonsymmetric problems.
function variables_unit_initialization!(
    variables::DefaultVariables{T},
    cones::CompositeCone{T}
) where {T}

    #set conic variables to units and x to 0
    unit_initialization!(cones,variables.z,variables.s)

    variables.x .= zero(T)
    variables.τ = one(T)
    variables.κ = one(T)

    return nothing
end

function variables_rescale!(variables)

    scale = max(variables.τ,variables.κ)
    invscale = 1/scale;
    
    variables.x .*= invscale
    variables.z.vec .*= invscale
    variables.s.vec .*= invscale
    variables.τ *= invscale
    variables.κ *= invscale
    
end