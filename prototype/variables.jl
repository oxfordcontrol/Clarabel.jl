
function calc_mu(
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultScalings{T}
) where {T}

  μ = (residuals.dot_sz + variables.τ * variables.κ)/(scalings.total_degree + 1)

  return μ
end


function calc_step_length(
    variables::DefaultVariables{T},
    step::DefaultVariables{T},
    scalings::DefaultScalings{T}
) where {T}

    ατ    = step.τ < 0 ? -variables.τ / step.τ : 1/eps(T)
    ακ    = step.κ < 0 ? -variables.κ / step.κ : 1/eps(T)

    αcone = cones_step_length(
        scalings.cones, step.z, step.s,
        variables.z, variables.s, scalings.λ
    )

    return min(ατ,ακ,αcone,1.)
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
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T}
) where{T}

    cones = scalings.cones

    @. d.x    .=  r.rx
    @. d.z     =  r.rz
    cones_circle_op!(cones, d.s, scalings.λ, scalings.λ)
    d.τ        =  r.rτ
    d.κ        =  variables.τ * variables.κ

    return nothing
end


function calc_combined_step_rhs!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
    step::DefaultVariables{T},
    σ::T, μ::T
) where {T}

    cones = scalings.cones

    #PJG: Still not clear whether second order corrections
    #on the dτ variable make sense here or not.   Not used for now
    #tmp2 = symdot(q,data.Psym,q) / variables.τ
    tmp0 = 0.  #PJG no higher order correction

    @. d.x  = (1. - σ)*r.rx
       d.τ  = (1. - σ)*r.rτ + tmp0    #PJG: second order correction?
       d.κ  = - σ*μ + step.τ * step.κ + variables.τ * variables.κ

    # d.s must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.
    #
    # Will also use d.z as a temporary work vector here. Note that we don't
    # want to have aliasing vector arguments to gemv_W or gemv_Winv, so we
    # need to copy into a temporary variable to assign #Δz = WΔz and Δs = W⁻¹Δs

    tmp  = d.z     #alias
    tmp .= step.z  #copy for safe call to gemv_W
    cones_gemv_W!(cones, false, tmp, step.z,  1., 0.)       #Δz <- WΔz
    tmp .= step.s  #copy for safe call to gemv_Winv
    cones_gemv_Winv!(cones, false, tmp, step.s,  1., 0.)    #Δs <- W⁻¹Δs
    cones_circle_op!(cones, tmp, step.s, step.z)            #tmp = W⁻¹Δs ∘ WΔz
    cones_add_scaled_e!(cones,tmp,-σ*μ)                     #tmp = W⁻¹Δs ∘ WΔz - σμe

    #PJG: We are relying on d.s = λ ◦ λ from the affine step here
    @. d.s += d.z                                           #d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe

    # now we copy the scaled res for rz and d.z is no longer work
    @. d.z .= (1 - σ)*r.rz

    return nothing
end

function variables_shift_to_cone!(
    variables::DefaultVariables,
    scalings::DefaultScalings
)

    cones_shift_to_cone!(scalings.cones,variables.s)
    cones_shift_to_cone!(scalings.cones,variables.z)

    variables.τ = 1
    variables.κ = 1
end

function variables_finalize!(
    variables::DefaultVariables,
    scalings::DefaultScalings,
    status::SolverStatus
)

    #undo the homogenization
    #
    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if(status == PRIMAL_INFEASIBLE || status == DUAL_INFEASIBLE)
        scaleinv = 1. / variables.κ
    else
        scaleinv = 1. / variables.τ
    end

    @. variables.x *= scaleinv
    @. variables.z *= scaleinv
    @. variables.s *= scaleinv
       variables.τ *= scaleinv
       variables.κ *= scaleinv

    #undo the equilibration
    d = scalings.d; dinv = scalings.dinv
    e = scalings.e; einv = scalings.einv
    cscale = scalings.c[]

    @. variables.x *=     d
    @. variables.z *=     e ./ cscale
    @. variables.s *=  einv

end
