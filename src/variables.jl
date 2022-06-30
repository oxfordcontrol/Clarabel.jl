
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
    cones::ConeSet{T}
) where {T}

    ατ    = step.τ < 0 ? -variables.τ / step.τ : floatmax(T)
    ακ    = step.κ < 0 ? -variables.κ / step.κ : floatmax(T)

    (αz,αs) = cones_step_length(
        cones, step.z, step.s,
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


# PJG: Rust makes this variables_scale_cones
function scaling_update!(
    cones::ConeSet{T},
    variables::DefaultVariables{T},
) where {T}

    cones_update_scaling!(cones,variables.s,variables.z)
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
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::ConeSet{T}
) where{T}

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
    cones::ConeSet{T},
    step::DefaultVariables{T},
    σ::T, μ::T
) where {T}

    @. d.x  = (one(T) - σ)*r.rx
       d.τ  = (one(T) - σ)*r.rτ
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

    #We are relying on d.s = λ ◦ λ already from the affine step here
    @. d.s += d.z                                                #d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe

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
