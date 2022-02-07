
function calc_mu(
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultScalings{T}
) where {T}

  μ = (residuals.dot_sz + variables.τ * variables.κ)/(scalings.total_order + 1)

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

    variables.x     .+= α*step.x
    variables.s.vec .+= α*step.s.vec
    variables.z.vec .+= α*step.z.vec
    variables.τ      += α*step.τ
    variables.κ      += α*step.κ

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

    d.x     .=  r.rx
    d.z.vec .= -r.rz .+ variables.s.vec
    cones_circle_op!(cones, d.s, scalings.λ, scalings.λ)
    d.τ      =  r.rτ
    d.κ      =  variables.τ * variables.κ

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

    # NB: calc_combined_step_rhs! modifies the step values
    # in place to be economical with memory

    # PJG: Except for here.  Temporarily allocating
    # vectors here for QP implementation
    ξ   = variables.x./variables.τ
    q   = step.x - ξ.*step.τ

    #PJG: Mehrotra style higher order correction.

    #try to get all higher orders instead?
    tmph = 1/(variables.τ + step.τ) * dot(variables.x + step.x,data.P, variables.x + step.x)
    tmph -= dot(variables.x,data.P, variables.x)/variables.τ
    tmph -= 2*dot(variables.x,data.P,step.x)
    tmph += dot(ξ,data.P,ξ)*step.τ

    tmp2 = dot(q,data.P,q) / variables.τ   #PJG: second order approximation only
    tmp0 = 0.  #PJG no higher order correction

    # assume that the affine RHS currently occupies d,
    # so only applies incremental changes to get the
    # combined corrector step
    d.x .*= (1. - σ)
    d.τ  *= (1. - σ)
    d.τ  += tmp0
    d.κ  += - σ*μ + step.τ * step.κ

    # d.s and d.z are harder if we want to be
    # economical with allocated memory.  Modify the
    # step.z and step.s vectors in place since they
    # are from the affine step and not needed anymore.
    # Use d.z as temporary space to hold W⁻¹Δs ∘ WΔz
    cones_gemv_W!(cones,  false, step.z, step.z,  1., 0.)        #Δz = WΔz
    cones_gemv_Winv!(cones, false, step.s, step.s,  1., 0.)      #Δs = W⁻¹Δs
    cones_circle_op!(cones, d.z, step.s, step.z)                 #tmp = W⁻¹Δs ∘ WΔz
    cones_add_scaled_e!(cones,d.z,-σ*μ)                          #tmp = tmp -σμe
    d.s.vec .+= d.z.vec

    # now build d.z from scratch
    cones_inv_circle_op!(cones, d.z, scalings.λ, d.s)  #dz = λ \ ds
    cones_gemv_W!(cones, false, d.z, d.z, 1., 0.)      #dz = Wdz
    d.z.vec .+= -(1-σ).*r.rz

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
