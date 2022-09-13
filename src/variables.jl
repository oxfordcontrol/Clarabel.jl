
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

    update_scaling!(cones,variables.s,variables.z,μ,scaling_strategy)
    return nothing
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
    μ::T
) where {T}

    dotσμ = σ*μ

    @. d.x  = (one(T) - σ)*r.rx
       d.τ  = (one(T) - σ)*r.rτ
       d.κ  = - dotσμ + step.τ * step.κ + variables.τ * variables.κ

    # ds is different for symmetric and asymmetric cones:
    # Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
    # Asymmetric cones: d.s = s + σμ*g(z)
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

    shift_to_cone!(cones,variables.s)
    shift_to_cone!(cones,variables.z)

    variables.τ = 1
    variables.κ = 1
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

function variables_finalize!(
    variables::DefaultVariables{T},
    equil::DefaultEquilibration{T},
    status::SolverStatus
) where {T}

    #undo the homogenization
    #
    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if(status == PRIMAL_INFEASIBLE || status == DUAL_INFEASIBLE)
        scaleinv = one(T) / variables.κ
    else
        scaleinv = one(T) / variables.τ
    end

    @. variables.x *= scaleinv
    @. variables.z *= scaleinv
    @. variables.s *= scaleinv
       variables.τ *= scaleinv
       variables.κ *= scaleinv

    #undo the equilibration
    d = equil.d; dinv = equil.dinv
    e = equil.e; einv = equil.einv
    cscale = equil.c[]

    @. variables.x *=  d
    @. variables.z *=  e ./ cscale
    @. variables.s *=  einv

end


function variables_rescale!(variables)

    vars = variables
    τ     = vars.τ
    κ     = vars.κ
    scale = max(τ,κ)
    
    vars.x ./= scale
    vars.z.vec ./= scale
    vars.s.vec ./= scale
    vars.τ /= scale
    vars.κ /= scale
    
end