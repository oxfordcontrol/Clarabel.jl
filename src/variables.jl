
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

    ατ    = step.τ < 0 ? -variables.τ / step.τ : inv(eps(T))
    ακ    = step.κ < 0 ? -variables.κ / step.κ : inv(eps(T))

    (αz,αs) = cones_step_length(
        scalings.cones, step.z, step.s,
        variables.z, variables.s,
    )

    return min(ατ,ακ,αz,αs,one(T))
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
    cones_λ_circ_λ!(cones, d.s)
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
    tmp0 = zero(T)  #PJG no higher order correction

    @. d.x  = (one(T) - σ)*r.rx
       d.τ  = (one(T) - σ)*r.rτ + tmp0    #PJG: second order correction?
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
    cones_gemv_W!(cones, :N, tmp, step.z, one(T), zero(T))       #Δz <- WΔz
    tmp .= step.s  #copy for safe call to gemv_Winv
    cones_gemv_Winv!(cones, :T, tmp, step.s, one(T), zero(T))    #Δs <- W⁻¹Δs
    cones_circ_op!(cones, tmp, step.s, step.z)                   #tmp = W⁻¹Δs ∘ WΔz
    cones_add_scaled_e!(cones,tmp,-σ*μ)                          #tmp = W⁻¹Δs ∘ WΔz - σμe

    #PJG: We are relying on d.s = λ ◦ λ from the affine step here
    @. d.s += d.z                                                #d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe

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
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
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
    d = scalings.d; dinv = scalings.dinv
    e = scalings.e; einv = scalings.einv
    cscale = scalings.c[]

    @. variables.x *=     d
    @. variables.z *=     e ./ cscale
    @. variables.s *=  einv

end
