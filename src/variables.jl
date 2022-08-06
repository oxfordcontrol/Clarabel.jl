
function calc_mu(
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    cones::ConeSet{T}
) where {T}

  μ = (residuals.dot_sz + variables.τ * variables.κ)/(cones.degree + 1)

  return μ
end


function calc_step_length(
    variables::DefaultVariables{T},
    step::DefaultVariables{T},
    work_vars::DefaultVariables{T},
    cones::ConeSet{T},
    steptype::Symbol
) where {T}

    ατ    = step.τ < 0 ? -variables.τ / step.τ : floatmax(T)
    ακ    = step.κ < 0 ? -variables.κ / step.κ : floatmax(T)

    α = min(ατ,ακ,one(T))

    # Find a feasible step size for all cones
    α = cones_step_length(cones, step.z, step.s, variables.z, variables.s, α)


    #   Centrality check for unsymmetric cones
    #   balance global μ and local μ_i of each nonsymmetric cone;
    #   check centrality, ensure the update is close to the central path
    if (!cones.sym_flag && steptype == :combined)
        α = check_μ_and_centrality(cones,step,variables,work_vars,α)

        if (steptype == :combined && α < 1e-4)
            # error("get stalled with step size ", α)
            return α
        end
    end

    return α
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


function variables_scale_cones!(
    variables::DefaultVariables{T},
    cones::ConeSet{T},
	μ::T,
    scale_flag::Bool
) where {T}

    cones_update_scaling!(cones,variables.s,variables.z,μ,scale_flag)
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


function calc_affine_step_rhs!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    cones::ConeSet{T}
) where{T}

    @. d.x    .=  r.rx
    @. d.z     =  r.rz
    cones_affine_ds!(cones, d.s, variables.s)    # unsymmetric cones need value of s
    d.τ        =  r.rτ
    d.κ        =  variables.τ * variables.κ

    return nothing
end


function calc_combined_step_rhs!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    cones::ConeSet{T},
    step::DefaultVariables{T},
    σ::T,
    μ::T
) where {T}

    dotσμ = σ*μ

    @. d.x  = (one(T) - σ)*r.rx
       d.τ  = (one(T) - σ)*r.rτ
       d.κ  = - dotσμ + step.τ * step.κ + variables.τ * variables.κ

    # d.s must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.
    #
    # Will also use d.z as a temporary work vector here. Note that we don't
    # want to have aliasing vector arguments to gemv_W or gemv_Winv, so we
    # need to copy into a temporary variable to assign #Δz = WΔz and Δs = W⁻¹Δs
    #
    # ds is different for symmetric and unsymmetric cones:
    # Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
    # Unsymmetric cones: d.s = s + σμ*g(z)
    cones_combined_ds!(cones,d.z,d.s,step.z,step.s,dotσμ)

    # now we copy the scaled res for rz and d.z is no longer work
    @. d.z .= (1 - σ)*r.rz

    return nothing
end

function variables_shift_to_cone!(
    variables::DefaultVariables{T},
    cones::ConeSet{T}
) where {T}

    cones_shift_to_cone!(cones,variables.s)
    cones_shift_to_cone!(cones,variables.z)

    variables.τ = 1
    variables.κ = 1
end


# Set the initial point to the jordan algebra identity e times scaling (now is 1.) for the symmetric cones
# and the central ray for the exponential cone, scaled by scaling (now is 1.)

# For symmetric cones, e is the identity in the Jordan algebra where the cone
# is defined. This corresponds to the following:
# for the nonnegative cones, e is the vector of all ones;
# for the second-order cones, e = (1; 0; ... ; 0) where the 1 corresponds to the first variable;
# for semidefinite cones, e is the identity matrix.
function unsymmetricInit(
    variables::DefaultVariables{T},
    cones::ConeSet{T}
) where {T}
    #set conic variables to units and x to 0
    unit_initialization!(cones,variables.s,variables.z)

    variables.x .= T(0)
    variables.τ = T(1)
    variables.κ = T(1)

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
